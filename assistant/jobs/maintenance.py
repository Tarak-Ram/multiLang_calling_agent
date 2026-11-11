"""Scheduled maintenance jobs. Each takes explicit dependencies and `now`, so it is unit-testable
without a scheduler; jobs.worker wires them to Arq cron.

  expire_conversations   open conversations past `expires_at` -> EXPIRED
  send_reminders         customer silent for N hours mid-flow -> ONE nudge (free-form inside the
                         24h window, an approved TEMPLATE outside it)
  redeliver_pending      outbound messages whose send failed / was interrupted -> retried in order

Every per-conversation action takes the SAME Redis lock inbound processing uses (tenant+phone),
so a job can never interleave with a customer's message being handled.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from redis.asyncio import Redis
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import Option
from assistant.core.locks import LockNotAcquiredError, conversation_lock_key, redis_lock
from assistant.core.models import (
    Conversation,
    ConversationStatus,
    Direction,
    Message,
    Sender,
    Tenant,
)
from assistant.orchestrator.service import ConversationService, PendingSend
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.jobs.maintenance")
MAX_SEND_ATTEMPTS = 3
RETRY_AFTER = timedelta(seconds=20)  # don't race the worker that is still delivering


async def expire_conversations(factory: async_sessionmaker[AsyncSession], now: datetime) -> int:
    """Mark overdue open conversations EXPIRED. A later message starts a fresh conversation
    with a one-line recap (see orchestrator.ingest). HANDOFF conversations never expire."""
    async with factory() as s, s.begin():
        res = await s.execute(
            update(Conversation)
            .where(
                Conversation.status.in_([ConversationStatus.ACTIVE,
                                         ConversationStatus.AWAITING_CUSTOMER]),
                Conversation.expires_at.is_not(None), Conversation.expires_at < now,
            )
            .values(status=ConversationStatus.EXPIRED)
        )
    n = int(res.rowcount or 0)  # type: ignore[attr-defined]
    if n:
        log.info("conversations_expired", extra={"count": n})
    return n


# --------------------------------------------------------------------------------- reminders
async def send_reminders(
    factory: async_sessionmaker[AsyncSession], redis: Redis, service: ConversationService,
    now: datetime, batch: int = 100,
) -> dict[str, int]:
    stats = {"free_form": 0, "template": 0, "no_template": 0, "skipped": 0}
    async with factory() as s:
        rows = (await s.execute(
            select(Conversation.id, Conversation.tenant_id, Conversation.customer_phone)
            .where(
                Conversation.status == ConversationStatus.AWAITING_CUSTOMER,
                Conversation.reminder_sent_at.is_(None), Conversation.flow.is_not(None),
                Conversation.last_customer_msg_at <= now - timedelta(hours=1),
                or_(Conversation.expires_at.is_(None), Conversation.expires_at > now),
            ).limit(batch)
        )).all()
    for conv_id, tenant_id, phone in rows:
        outcome = await _remind_one(factory, redis, service, tenant_id, conv_id, phone, now)
        stats[outcome] += 1
    return stats


async def _remind_one(
    factory: async_sessionmaker[AsyncSession], redis: Redis, service: ConversationService,
    tenant_id: uuid.UUID, conv_id: uuid.UUID, phone: str, now: datetime,
) -> str:
    pending: list[PendingSend] = []
    outcome = "skipped"
    try:
        async with (
            redis_lock(redis, conversation_lock_key(tenant_id, phone), wait_s=0.5),
            factory() as s,
            s.begin(),
        ):
            conv = await s.get(Conversation, conv_id)
            tenant = await s.get(Tenant, tenant_id)
            if conv is None or tenant is None or conv.last_customer_msg_at is None:
                return "skipped"
            cfg = TenantConfig.model_validate(tenant.config)
            due = conv.last_customer_msg_at + timedelta(hours=cfg.timing.reminder_after_hours)
            # Re-check under the lock: the customer may have replied a moment ago.
            if (conv.status != ConversationStatus.AWAITING_CUSTOMER
                    or conv.reminder_sent_at is not None or now < due):
                return "skipped"
            conv.reminder_sent_at = now  # at most ONE reminder per silence, even on failure
            lang = conv.language or cfg.default_language
            account = tenant.whatsapp_phone_number_id or ""
            if conv.window_expires_at is not None and conv.window_expires_at > now:
                text = service.orch.industries[tenant.industry].text(cfg, "reminder", lang)
                row = _system_row(tenant_id, conv_id, text, lang, {"kind": "reminder"})
                s.add(row)
                await s.flush()
                pending.append(PendingSend(row.id, account, phone, text, [], "",
                                           conv.window_expires_at))
                outcome = "free_form"
            else:  # window closed: ONLY an approved template may be sent
                template = cfg.templates.get("reminder")
                if not template:
                    log.warning("reminder_skipped_no_template",
                                extra={"tenant": str(tenant_id)})
                    return "no_template"
                row = _system_row(tenant_id, conv_id, template, lang,
                                  {"kind": "reminder", "template": template, "lang": lang},
                                  msg_type="template")
                s.add(row)
                await s.flush()
                pending.append(PendingSend(row.id, account, phone, template, [], "", None,
                                           template=(template, lang, [])))
                outcome = "template"
    except LockNotAcquiredError:
        return "skipped"  # customer is mid-conversation: no reminder needed right now
    await service.deliver(factory, pending)
    return outcome


def _system_row(tenant_id: uuid.UUID, conv_id: uuid.UUID, text: str, lang: str,
                meta: dict[str, object], msg_type: str = "text") -> Message:
    return Message(
        tenant_id=tenant_id, conversation_id=conv_id, direction=Direction.OUTBOUND,
        sender=Sender.SYSTEM, msg_type=msg_type, text=text, language=lang,
        meta={**meta, "send_status": "pending"},
    )


# -------------------------------------------------------------------------------- outbox retry
async def redeliver_pending(
    factory: async_sessionmaker[AsyncSession], redis: Redis, service: ConversationService,
    now: datetime, batch: int = 200,
) -> dict[str, int]:
    """Retry outbound messages stuck in 'pending' (worker died between commit and send) or
    'failed'. Strictly in order per conversation so a customer never gets part 2 before part 1.
    A message is retried at most MAX_SEND_ATTEMPTS times; then it is marked 'gave_up'."""
    stats = {"retried": 0, "gave_up": 0}
    async with factory() as s:
        convs = (await s.execute(
            select(Conversation.id, Conversation.tenant_id, Conversation.customer_phone)
            .join(Message, Message.conversation_id == Conversation.id)
            .where(
                Message.direction == Direction.OUTBOUND,
                Message.meta["send_status"].astext.in_(["pending", "failed"]),
                Message.created_at < now - RETRY_AFTER,
                Message.created_at > now - timedelta(hours=24),
            ).distinct().limit(batch)
        )).all()
    for conv_id, tenant_id, phone in convs:
        try:
            async with redis_lock(redis, conversation_lock_key(tenant_id, phone), wait_s=0.2):
                done = await _redeliver_conversation(factory, service, conv_id, now)
        except LockNotAcquiredError:
            continue
        for k, v in done.items():
            stats[k] += v
    return stats


async def _redeliver_conversation(
    factory: async_sessionmaker[AsyncSession], service: ConversationService, conv_id: uuid.UUID,
    now: datetime,
) -> dict[str, int]:
    out = {"retried": 0, "gave_up": 0}
    pendings: list[PendingSend] = []
    async with factory() as s, s.begin():
        conv = await s.get(Conversation, conv_id)
        tenant = await s.get(Tenant, conv.tenant_id) if conv else None
        if conv is None or tenant is None:
            return out
        rows = (await s.execute(
            select(Message).where(
                Message.conversation_id == conv_id, Message.direction == Direction.OUTBOUND,
                Message.meta["send_status"].astext.in_(["pending", "failed"]),
                Message.created_at < now - RETRY_AFTER,  # per message: a worker may still be sending
            ).order_by(Message.created_at)
        )).scalars().all()
        broken = False
        for r in rows:
            if broken or int(r.meta.get("attempts", 0)) >= MAX_SEND_ATTEMPTS:
                r.meta = {**r.meta, "send_status": "gave_up"}  # keep order: stop the chain
                out["gave_up"] += 1
                broken = True
                continue
            tpl = r.meta.get("template")
            pendings.append(PendingSend(
                r.id, tenant.whatsapp_phone_number_id or "", conv.customer_phone, r.text or "",
                [Option(id=o["id"], title=o["title"]) for o in r.meta.get("options", [])],
                r.meta.get("list_button", ""), conv.window_expires_at,
                template=(str(tpl), str(r.meta.get("lang", r.language or "en")), [])
                if tpl else None,
            ))
    out["retried"] = len(pendings)
    if pendings:
        await service.deliver(factory, pendings)
    return out

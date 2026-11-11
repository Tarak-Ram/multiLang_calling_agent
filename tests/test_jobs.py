"""Delayed replies: expiry, recap, reminders (24h window vs template), outbox retry."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fakeredis import FakeAsyncRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import Option
from assistant.channels.whatsapp.parser import parse_webhook
from assistant.core.locks import conversation_lock_key, redis_lock
from assistant.core.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Direction,
    Message,
    Sender,
    Tenant,
)
from assistant.jobs.maintenance import expire_conversations, redeliver_pending, send_reminders
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.flows import load_industries
from assistant.orchestrator.ingest import ingest_inbound
from assistant.orchestrator.service import ConversationService
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from assistant.tools.registry import build_registry
from tests.engine_helpers import SAMPLE
from tests.whatsapp_fixtures import envelope, text_msg

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
PHONE = "+919876543210"


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = False

    async def _rec(self, kind: str, *a: Any) -> str:
        if self.fail:
            raise RuntimeError("meta down")
        self.sent.append((kind, a))
        return f"out{len(self.sent)}"

    async def send_text(self, account_id: str, to: str, body: str) -> str:
        return await self._rec("text", to, body)

    async def send_options(self, account_id: str, to: str, body: str, options: list[Option],
                           list_button: str) -> str:
        return await self._rec("options", to, body, [o.id for o in options])

    async def send_template(self, account_id: str, to: str, name: str, lang: str,
                            params: list[str]) -> str:
        return await self._rec("template", to, name, lang)


class Env:
    def __init__(self, session: AsyncSession) -> None:
        self.session, self.ch = session, Channel()
        self.redis = FakeAsyncRedis(decode_responses=True)
        self.now = NOW
        self.factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
        self.tenant: Tenant

    async def start(self, template: str | None = "clinic_reminder_v1") -> Env:
        data = parse_tenant_yaml(SAMPLE)
        data.slug = "jobs"
        data.whatsapp_phone_number_id = "pn-jobs"
        data.config.templates = {"reminder": template} if template else {}
        self.tenant = await import_tenant(self.session, data)
        await self.session.commit()
        orch = Orchestrator(load_industries(), build_registry(httpx.AsyncClient(), self.ch))  # type: ignore[arg-type]
        self.svc = ConversationService(orch, self.ch, clock=lambda: self.now)
        return self

    async def conv(self, *, status: ConversationStatus = ConversationStatus.AWAITING_CUSTOMER,
                   flow: str | None = "book_appointment", last_msg_h_ago: float = 5,
                   window_h_left: float = 19, expires_in_h: float = 19,
                   lang: str = "en", phone: str = PHONE) -> Conversation:
        cust = (await self.session.execute(select(Customer).where(Customer.phone == phone))
                ).scalar_one_or_none()
        if cust is None:
            cust = Customer(tenant_id=self.tenant.id, phone=phone, name="Ravi")
            self.session.add(cust)
            await self.session.flush()
        c = Conversation(
            tenant_id=self.tenant.id, customer_id=cust.id, customer_phone=phone, status=status,
            flow=flow, current_state="collect_slots", language=lang,
            last_customer_msg_at=self.now - timedelta(hours=last_msg_h_ago),
            window_expires_at=self.now + timedelta(hours=window_h_left),
            expires_at=self.now + timedelta(hours=expires_in_h),
        )
        self.session.add(c)
        await self.session.commit()
        return c


async def outbound(session: AsyncSession) -> list[Message]:
    return list((await session.execute(
        select(Message).where(Message.direction == Direction.OUTBOUND).order_by(Message.created_at)
    )).scalars().all())


# ------------------------------------------------------------------------------- expiry
async def test_expiry_job_only_touches_overdue_open_conversations(session: AsyncSession) -> None:
    e = await Env(session).start()
    overdue = await e.conv(expires_in_h=-1)
    fresh = await e.conv(expires_in_h=5, phone="+919000000001")
    handoff = await e.conv(status=ConversationStatus.HANDOFF, expires_in_h=-1,
                           phone="+919000000002")
    closed = await e.conv(status=ConversationStatus.CLOSED, expires_in_h=-1, phone="+919000000003")
    assert await expire_conversations(e.factory, NOW) == 1
    for c in (overdue, fresh, handoff, closed):
        await session.refresh(c)
    assert overdue.status == ConversationStatus.EXPIRED and fresh.status != ConversationStatus.EXPIRED
    assert handoff.status == ConversationStatus.HANDOFF  # a human owns it: never auto-expired
    assert closed.status == ConversationStatus.CLOSED
    assert await expire_conversations(e.factory, NOW) == 0  # idempotent


# ---------------------------------------------------------------------- late reply + recap
async def late_reply(e: Env, text: str, when: datetime) -> list[str]:
    ts = str(int(when.timestamp()))
    msg = parse_webhook(envelope([text_msg("late1", text, ts)]))[0]
    e.now = when
    ing = await ingest_inbound(e.session, e.tenant.id, msg)
    assert ing
    pend = await e.svc.process(e.session, e.tenant, ing, msg)
    await e.session.commit()
    e.last_ing = ing  # type: ignore[attr-defined]
    return [p.text for p in pend]


async def test_late_reply_after_expiry_starts_fresh_with_one_line_recap(
    session: AsyncSession,
) -> None:
    e = await Env(session).start()
    old = await e.conv(expires_in_h=-1, lang="en")
    old.collected_slots = {"service": "general", "date": "2026-11-13"}
    await session.commit()
    await expire_conversations(e.factory, NOW)
    texts = await late_reply(e, "hello", NOW + timedelta(hours=30))
    assert texts[0].startswith("Welcome back!") and "booking an appointment" in texts[0]
    assert len(texts) == 2 and "How can I help" in texts[1]  # recap line, then normal greeting
    new = e.last_ing.conversation  # type: ignore[attr-defined]
    assert new.id != old.id
    assert "_recap" not in new.collected_slots and "service" not in new.collected_slots
    assert new.window_expires_at == NOW + timedelta(hours=30 + 24)  # fresh 24h window


async def test_late_reply_before_job_runs_is_treated_as_expired(session: AsyncSession) -> None:
    e = await Env(session).start()
    old = await e.conv(expires_in_h=1)  # deadline passes, but the cron has not run yet
    texts = await late_reply(e, "hello", NOW + timedelta(hours=2))
    await session.refresh(old)
    assert old.status == ConversationStatus.EXPIRED and texts[0].startswith("Welcome back!")


async def test_recap_is_localized_and_skipped_when_old_or_flowless(session: AsyncSession) -> None:
    e = await Env(session).start()
    await e.conv(expires_in_h=-1, lang="hi")
    await expire_conversations(e.factory, NOW)
    texts = await late_reply(e, "नमस्ते", NOW + timedelta(hours=30))
    assert texts[0].startswith("वापसी पर स्वागत है") and "अपॉइंटमेंट" in texts[0]

    e2 = await Env(session).start()
    await e2.conv(expires_in_h=-1, phone="+919000000011")  # expired > 72h before the reply
    await expire_conversations(e2.factory, NOW)
    e2.tenant = e2.tenant
    ts = NOW + timedelta(hours=100)
    msg = parse_webhook(envelope([{"from": "919000000011", "id": "x1",
                                   "timestamp": str(int(ts.timestamp())), "type": "text",
                                   "text": {"body": "hello"}}]))[0]
    ing = await ingest_inbound(session, e2.tenant.id, msg)
    assert ing and "_recap" not in ing.conversation.collected_slots


# ---------------------------------------------------------------------------- reminders
async def test_reminder_inside_window_is_free_form_and_sent_once(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv(last_msg_h_ago=5, window_h_left=19, lang="te")
    stats = await send_reminders(e.factory, e.redis, e.svc, NOW)
    assert stats["free_form"] == 1 and stats["template"] == 0
    kind, args = e.ch.sent[0]
    assert kind == "text" and "కొనసాగించాలనుకుంటున్నారా" in args[1]  # Telugu nudge
    await session.refresh(c)
    assert c.reminder_sent_at == NOW and c.status == ConversationStatus.AWAITING_CUSTOMER
    [m] = await outbound(session)
    assert m.sender == Sender.SYSTEM and m.meta["kind"] == "reminder" and m.meta["send_status"] == "sent"
    assert (await send_reminders(e.factory, e.redis, e.svc, NOW + timedelta(hours=1)))[
        "free_form"] == 0  # never a second reminder for the same silence


async def test_reminder_outside_window_uses_approved_template(session: AsyncSession) -> None:
    e = await Env(session).start()
    await e.conv(last_msg_h_ago=26, window_h_left=-2, expires_in_h=30, lang="hi")
    stats = await send_reminders(e.factory, e.redis, e.svc, NOW)
    assert stats["template"] == 1
    assert e.ch.sent == [("template", (PHONE, "clinic_reminder_v1", "hi"))]  # no free-form text!
    assert all(k != "text" for k, _ in e.ch.sent)


async def test_no_template_configured_means_no_free_form_fallback(session: AsyncSession) -> None:
    e = await Env(session).start(template=None)
    c = await e.conv(last_msg_h_ago=26, window_h_left=-2, expires_in_h=30)
    assert (await send_reminders(e.factory, e.redis, e.svc, NOW))["no_template"] == 1
    assert e.ch.sent == []  # WhatsApp would reject it; we don't try
    await session.refresh(c)
    assert c.reminder_sent_at == NOW  # and we don't retry forever


async def test_reminder_respects_timing_status_and_customer_activity(session: AsyncSession) -> None:
    e = await Env(session).start()
    await e.conv(last_msg_h_ago=2)  # only 2h silent (< reminder_after_hours=4)
    await e.conv(status=ConversationStatus.ACTIVE, phone="+919000000021")  # customer just replied
    await e.conv(flow=None, phone="+919000000022")  # nothing in progress worth nudging
    await e.conv(last_msg_h_ago=5, expires_in_h=-1, phone="+919000000023")  # already overdue
    stats = await send_reminders(e.factory, e.redis, e.svc, NOW)
    # only the 2h-silent conversation is even a candidate, and it is correctly skipped
    assert stats == {"free_form": 0, "template": 0, "no_template": 0, "skipped": 1}
    assert e.ch.sent == []


async def test_reminder_skipped_while_customer_message_is_being_processed(
    session: AsyncSession,
) -> None:
    e = await Env(session).start()
    await e.conv(last_msg_h_ago=5)
    async with redis_lock(e.redis, conversation_lock_key(e.tenant.id, PHONE)):
        stats = await send_reminders(e.factory, e.redis, e.svc, NOW)  # lock held by "inbound"
    assert stats["skipped"] == 1 and e.ch.sent == []


# ------------------------------------------------------------------------------- outbox
async def add_outbound(e: Env, conv: Conversation, text: str, status: str, attempts: int = 0,
                       age_s: int = 120, **meta: Any) -> Message:
    m = Message(tenant_id=e.tenant.id, conversation_id=conv.id, direction=Direction.OUTBOUND,
                sender=Sender.BOT, text=text, meta={"send_status": status, "attempts": attempts,
                                                    **meta})
    e.session.add(m)
    await e.session.flush()
    m.created_at = NOW - timedelta(seconds=age_s)
    await e.session.commit()
    return m


async def test_failed_and_stuck_sends_are_retried_in_order(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv()
    m1 = await add_outbound(e, c, "first", "failed", attempts=1, age_s=200)
    m2 = await add_outbound(e, c, "second", "pending", age_s=100)  # worker died before sending
    await add_outbound(e, c, "recent", "pending", age_s=5)  # too fresh: worker may still send it
    stats = await redeliver_pending(e.factory, e.redis, e.svc, NOW)
    assert stats["retried"] == 2
    assert [a[1] for _, a in e.ch.sent] == ["first", "second"]  # in order
    await session.refresh(m1)
    await session.refresh(m2)
    assert m1.meta["send_status"] == "sent" and m1.meta["attempts"] == 2 and m1.wa_message_id
    assert m2.meta["send_status"] == "sent"
    assert (await redeliver_pending(e.factory, e.redis, e.svc, NOW))["retried"] == 0  # idempotent


async def test_retry_stops_the_chain_on_failure_to_preserve_order(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv()
    m1 = await add_outbound(e, c, "first", "failed", age_s=200)
    m2 = await add_outbound(e, c, "second", "pending", age_s=100)
    e.ch.fail = True
    await redeliver_pending(e.factory, e.redis, e.svc, NOW)
    await session.refresh(m1)
    await session.refresh(m2)
    assert m1.meta["send_status"] == "failed" and m2.meta["send_status"] == "pending"  # not sent early
    e.ch.fail = False
    await redeliver_pending(e.factory, e.redis, e.svc, NOW + timedelta(seconds=60))
    assert [a[1] for _, a in e.ch.sent] == ["first", "second"]


async def test_gives_up_after_max_attempts(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv()
    m1 = await add_outbound(e, c, "first", "failed", attempts=3)
    m2 = await add_outbound(e, c, "second", "pending", age_s=100)
    stats = await redeliver_pending(e.factory, e.redis, e.svc, NOW)
    await session.refresh(m1)
    await session.refresh(m2)
    assert stats["gave_up"] == 2 and e.ch.sent == []
    assert m1.meta["send_status"] == m2.meta["send_status"] == "gave_up"


async def test_retry_never_sends_free_form_after_window_closed(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv(window_h_left=-1)
    m = await add_outbound(e, c, "late", "failed", age_s=200)
    await redeliver_pending(e.factory, e.redis, e.svc, NOW)
    await session.refresh(m)
    assert e.ch.sent == [] and m.meta["send_status"] == "blocked_outside_window"
    assert (await redeliver_pending(e.factory, e.redis, e.svc, NOW))["retried"] == 0  # final


async def test_failed_template_reminder_is_retried_as_template(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv(window_h_left=-1)
    await add_outbound(e, c, "clinic_reminder_v1", "failed", age_s=200, template="clinic_reminder_v1",
                       lang="te")
    await redeliver_pending(e.factory, e.redis, e.svc, NOW)
    assert e.ch.sent == [("template", (PHONE, "clinic_reminder_v1", "te"))]

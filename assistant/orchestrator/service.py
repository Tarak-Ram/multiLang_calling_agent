"""Glue between storage, the orchestrator and the channel.

process(): runs inside the DB transaction (load state -> decide -> save state + outbound rows).
deliver(): runs AFTER commit, so a slow/failed send never holds a transaction or loses state.
Both run under the per-conversation Redis lock (see jobs/worker.py).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import ChannelAdapter, InboundMessage, Option
from assistant.core.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Direction,
    Message,
    Sender,
    Tenant,
)
from assistant.llm.trace import build_trace, emit_trace
from assistant.llm.usage import collect_usage, persist_usage
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.handoff import build_handoff_summary
from assistant.orchestrator.ingest import IngestResult
from assistant.orchestrator.types import ConvState, TurnInput
from assistant.tenants.schema import TenantConfig
from assistant.tools.base import ToolEnv

log = logging.getLogger("assistant.service")


@dataclass(slots=True)
class PendingSend:
    message_id: uuid.UUID
    account_id: str
    to_phone: str
    text: str
    options: list[Option]
    list_button: str
    window_expires_at: datetime | None
    # (template_name, language, params): sent as an approved template, which is allowed even
    # outside the 24h window. None = free-form text/options (window required).
    template: tuple[str, str, list[str]] | None = None


class ConversationService:
    def __init__(
        self,
        orchestrator: Orchestrator,
        channel: ChannelAdapter,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        trace_exporter: Any = None,
    ) -> None:
        self.orch = orchestrator
        self.channel = channel
        self.clock = clock
        self.trace_exporter = trace_exporter  # optional Langfuse exporter (llm.trace)
        self._bg: set[asyncio.Task[None]] = set()

    async def process(
        self, session: AsyncSession, tenant: Tenant, ing: IngestResult, msg: InboundMessage
    ) -> list[PendingSend]:
        """Decide the reply and persist everything; returns sends to perform after commit."""
        now = self.clock()
        cfg = TenantConfig.model_validate(tenant.config)
        conv: Conversation = ing.conversation
        customer = await session.get(Customer, conv.customer_id)
        assert customer is not None

        state = ConvState(
            tenant_id=tenant.id, conversation_id=conv.id, customer_phone=conv.customer_phone,
            customer_name=customer.name, language=conv.language or customer.language,
            status=conv.status, flow=conv.flow, state=conv.current_state,
            slots=dict(conv.collected_slots or {}), summary=conv.short_summary,
            recent=await self._recent_turns(session, conv.id, ing.message.id),
        )
        started = time.perf_counter()
        with collect_usage() as usage:
            result = await self.orch.handle(
                cfg, tenant.industry, state,
                TurnInput(text=msg.text, payload_id=msg.payload_id, msg_type=msg.msg_type), now,
                ToolEnv(session=session,
                        account_id=tenant.whatsapp_phone_number_id or msg.channel_account_id),
            )
        new = result.state
        trace = build_trace(
            tenant_id=tenant.id, conversation_id=conv.id, message_id=ing.message.id,
            route=result.route, language=new.language, needs_human=result.needs_human,
            latency_ms=int((time.perf_counter() - started) * 1000), usage=usage,
        )
        emit_trace(trace)  # one structured log line per turn (no message text / PII)
        if self.trace_exporter is not None:  # never on the reply path: fire and forget
            task = asyncio.create_task(self.trace_exporter.export(trace))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        # every LLM/embedding call of this turn, for the cost report
        await persist_usage(session, tenant.id, conv.id, ing.message.id, result.route, usage,
                            trace_id=trace["trace_id"])

        # Persist conversation state (assign new objects so JSONB changes are detected).
        conv.flow, conv.current_state = new.flow, new.state
        conv.collected_slots = dict(new.slots)
        conv.language, customer.language = new.language, new.language
        conv.status = new.status
        if new.status == ConversationStatus.ACTIVE and result.replies:
            conv.status = ConversationStatus.AWAITING_CUSTOMER  # we asked; reminder job watches
        if conv.status in (ConversationStatus.CLOSED, ConversationStatus.HANDOFF):
            conv.expires_at = None
        else:
            conv.expires_at = now + timedelta(hours=cfg.timing.expire_after_hours)
        if result.needs_human:
            industry = self.orch.industries[tenant.industry]
            labels = new.slots.get("_labels", {})
            conv.handoff_summary = build_handoff_summary(
                reason=result.handoff_reason, customer_name=customer.name, phone=conv.customer_phone,
                language=new.language,
                flow_name=industry.text(cfg, f"flow_{new.flow}", "en") if new.flow else None,
                slots={k: str(labels.get(k, v)) for k, v in new.slots.items()
                       if not k.startswith("_")},
                turns=[*state.recent, ("customer", msg.text or f"[{msg.msg_type}]"),
                       *(("bot", r.text) for r in result.replies)],
            )
            conv.handoff_notified_at = None

        ing.message.language = new.language
        ing.message.meta = {**ing.message.meta, "route": result.route,
                            "tool_calls": result.tool_calls}

        pendings: list[PendingSend] = []
        for reply in result.replies:
            row = Message(
                tenant_id=tenant.id, conversation_id=conv.id, direction=Direction.OUTBOUND,
                sender=Sender.BOT, channel=msg.channel, msg_type="interactive" if reply.options
                else "text", text=reply.text, language=new.language,
                meta={"route": result.route, "send_status": "pending",
                      "options": [o.model_dump() for o in reply.options],
                      "list_button": reply.list_button},
            )
            session.add(row)
            await session.flush()
            pendings.append(PendingSend(
                row.id, tenant.whatsapp_phone_number_id or msg.channel_account_id,
                conv.customer_phone, reply.text, reply.options, reply.list_button,
                conv.window_expires_at,
            ))
        await session.flush()
        return pendings

    @staticmethod
    async def _recent_turns(
        session: AsyncSession, conversation_id: uuid.UUID, current_id: uuid.UUID, n: int = 4
    ) -> list[tuple[str, str]]:
        """Last n earlier messages (oldest first). This is the ONLY history an LLM ever sees."""
        rows = (await session.execute(
            select(Message.direction, Message.text)
            .where(Message.conversation_id == conversation_id, Message.id != current_id,
                   Message.text.is_not(None))
            .order_by(Message.created_at.desc()).limit(n)
        )).all()
        return [("customer" if d == Direction.INBOUND else "bot", t or "")
                for d, t in reversed(rows)]

    async def deliver(
        self, factory: async_sessionmaker[AsyncSession], pendings: list[PendingSend]
    ) -> None:
        """Send replies in order; record the outcome on each outbound message row.

        24h rule: free-form only while the customer-service window is open; templates are
        always allowed. Failed/interrupted sends are retried by jobs.maintenance.redeliver_pending.
        """
        for p in pendings:
            status, wamid = "sent", None
            if (p.template is None and p.window_expires_at is not None
                    and p.window_expires_at <= self.clock()):
                status = "blocked_outside_window"
                log.warning("send_blocked_outside_window")
            else:
                try:
                    if p.template:
                        name, lang, params = p.template
                        wamid = await self.channel.send_template(
                            p.account_id, p.to_phone, name, lang, params)
                    elif p.options:
                        wamid = await self.channel.send_options(
                            p.account_id, p.to_phone, p.text, p.options, p.list_button)
                    else:
                        wamid = await self.channel.send_text(p.account_id, p.to_phone, p.text)
                except Exception:
                    log.exception("send_failed")
                    status = "failed"
            async with factory() as s, s.begin():
                row = await s.get(Message, p.message_id)
                if row is not None:
                    row.wa_message_id = wamid
                    row.meta = {**row.meta, "send_status": status,
                                "attempts": int(row.meta.get("attempts", 0)) + 1}
            if status != "sent":
                break  # keep reply order: later parts wait for the retry job

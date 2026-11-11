"""notify_handoff job: (optionally) polish the summary with a small LLM, then alert staff.

Offline by design: the customer was already told "connecting you to staff" instantly; nothing
here is on the live reply path, so latency doesn't matter and thinking-style models would be
allowed. We still use a small, cheap model with a hard output cap.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import ChannelAdapter
from assistant.core.models import Conversation, ConversationStatus, Tenant
from assistant.llm.base import LLMClient, LLMRequest, LLMUnavailableError
from assistant.llm.usage import collect_usage, persist_usage, record_response
from assistant.orchestrator.handoff import one_line
from assistant.rag.guards import numbers_grounded
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.jobs.handoff")

SYSTEM_PROMPT = (
    "You write a 2-3 line internal handoff note for customer-support staff, in English. Use ONLY "
    "facts present in the input. Say what the customer wants, what was already done, and what "
    "staff should do next. Never add numbers, names, prices or dates that are not in the input."
)


async def polish_summary(
    llm: LLMClient, model: str, summary: str, max_tokens: int = 120
) -> str | None:
    """LLM rewrite of the deterministic summary; None if unusable.

    Rejected if it contains any number that is not in the input: a polished note must not
    introduce facts (phone numbers, dates, amounts) the conversation never contained."""
    try:
        resp = await llm.complete(LLMRequest(
            model=model, system=SYSTEM_PROMPT, user=summary, max_tokens=max_tokens,
            json_mode=False))
    except LLMUnavailableError:
        return None
    record_response("summary", resp)
    text = resp.text.strip()
    if not text or not numbers_grounded(text, summary):
        log.warning("handoff_summary_rejected")
        return None
    return text


async def notify_handoff(
    factory: async_sessionmaker[AsyncSession],
    channel: ChannelAdapter,
    llm: LLMClient | None,
    fast_model: str,
    conversation_id: uuid.UUID,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> str:
    """Returns: notified | no_staff_channel | already_notified | not_handoff | send_failed."""
    async with factory() as s:
        conv = await s.get(Conversation, conversation_id)
        tenant = await s.get(Tenant, conv.tenant_id) if conv else None
        if conv is None or tenant is None or conv.status != ConversationStatus.HANDOFF:
            return "not_handoff"
        if conv.handoff_notified_at is not None:
            return "already_notified"
        cfg = TenantConfig.model_validate(tenant.config)
        base_summary = conv.handoff_summary or "Handed off to staff"
        tenant_id, phone = tenant.id, conv.customer_phone
        account = tenant.whatsapp_phone_number_id or ""

    final = base_summary
    with collect_usage() as usage:
        if llm is not None and fast_model:
            polished = await polish_summary(llm, cfg.models.fast or fast_model, base_summary)
            final = polished or base_summary

    template = cfg.templates.get("handoff_alert")
    staff = cfg.escalation.phone
    outcome = "no_staff_channel"
    if staff and template:
        try:
            await channel.send_template(account, staff, template, "en",
                                        [phone, one_line(final)])
            outcome = "notified"
        except Exception:
            log.exception("handoff_notify_failed")
            outcome = "send_failed"
    else:
        # Staff still see the conversation in the admin inbox (GET /admin/handoffs).
        log.warning("handoff_not_pushed", extra={"has_staff": bool(staff),
                                                 "has_template": bool(template)})

    async with factory() as s, s.begin():
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.handoff_summary = final
            if outcome == "notified":
                conv.handoff_notified_at = clock()
            await persist_usage(s, tenant_id, conv.id, None, "offline", usage)
    return outcome

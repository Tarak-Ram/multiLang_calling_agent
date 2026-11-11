from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import Conversation, ConversationStatus, Customer, UsageCost
from assistant.jobs.handoff import notify_handoff, polish_summary
from assistant.llm.base import LLMRequest, LLMResponse, LLMUnavailableError
from assistant.orchestrator.handoff import build_handoff_summary, humanize_reason, one_line
from tests.test_jobs import Env


def test_summary_is_complete_short_and_staff_readable() -> None:
    s = build_handoff_summary(
        reason="medical_question", customer_name="Ravi", phone="+919876543210", language="te",
        flow_name="booking an appointment", slots={"service": "General", "_labels": {}},  # type: ignore[dict-item]
        turns=[("customer", "hi"), ("bot", "menu"), ("customer", "book"), ("bot", "which dept"),
               ("customer", "నాకు జ్వరం ఉంది " + "x" * 400)],
    )
    assert s.startswith("Reason: Medical question: the bot gave NO advice")
    assert "[medical_question]" in s and "Ravi (+919876543210), language: te" in s
    assert "In progress: booking an appointment (service=General)" in s
    assert "- customer: hi" not in s and "- bot: which dept" in s  # only the last 4 turns
    assert "…" in s and len(s) < 900  # a long message is truncated, never dumped whole
    assert "_labels" not in s


def test_reason_codes_are_humanized_with_safe_fallback() -> None:
    assert "unhappy" in humanize_reason("extractor_needs_human")
    assert "could not understand" in humanize_reason("unclear:service")
    assert "not automated" in humanize_reason("flow_unavailable:cancel")
    assert humanize_reason("brand_new_code") == "Handed off (brand_new_code)"
    assert humanize_reason(None) == "Handed off to staff"


def test_one_line_for_template_parameters() -> None:
    out = one_line("Reason: x\nCustomer: y\n\nLast:   a   b")
    assert "\n" not in out and "  " not in out and out.startswith("Reason: x | Customer: y")
    assert len(one_line("x" * 5000)) == 900


class ScriptLLM:
    def __init__(self, out: str | Exception) -> None:
        self.out, self.reqs = out, []

    async def complete(self, r: LLMRequest) -> LLMResponse:
        self.reqs.append(r)
        if isinstance(self.out, Exception):
            raise self.out
        return LLMResponse(text=self.out, model=r.model, input_tokens=90, output_tokens=30,
                           cost_usd=0.0004)


async def test_polish_accepts_grounded_and_rejects_invented_numbers() -> None:
    src = "Reason: medical\nCustomer: Ravi (+919876543210)\n- customer: fever since 2 days"
    good = await polish_summary(ScriptLLM("Ravi has had a fever for 2 days; clinical staff should call."),
                                "fast", src)
    assert good and "2 days" in good
    bad = await polish_summary(ScriptLLM("Ravi has had fever for 5 days; call 9999999999."),
                               "fast", src)
    assert bad is None  # invented numbers are never trusted
    assert await polish_summary(ScriptLLM(LLMUnavailableError("down")), "fast", src) is None
    llm = ScriptLLM("ok")
    await polish_summary(llm, "fast", src)
    assert llm.reqs[0].json_mode is False and llm.reqs[0].max_tokens == 120  # capped, plain text


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[Any, ...]] = []
        self.fail = False

    async def send_template(self, account: str, to: str, name: str, lang: str,
                            params: list[str]) -> str:
        if self.fail:
            raise RuntimeError("meta down")
        self.sent.append((account, to, name, lang, params))
        return "wamid.alert"


async def handoff_conv(e: Env, template: bool = True, staff: bool = True) -> Conversation:
    cfg = dict(e.tenant.config)
    cfg["templates"] = {"handoff_alert": "staff_alert_v1"} if template else {}
    cfg["escalation"] = {"phone": "+910000000000" if staff else None, "message": {}}
    e.tenant.config = cfg
    c = await e.conv(status=ConversationStatus.HANDOFF)
    c.handoff_summary = "Reason: Customer asked to talk to a person [user_requested]\nCustomer: Ravi (+919876543210)"
    await e.session.commit()
    return c


async def test_staff_alert_sent_via_template_once(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await handoff_conv(e)
    ch = Channel()
    assert await notify_handoff(e.factory, ch, None, "", c.id) == "notified"  # type: ignore[arg-type]
    [(account, to, name, lang, params)] = ch.sent
    assert (account, to, name, lang) == ("pn-jobs", "+910000000000", "staff_alert_v1", "en")
    assert params[0] == "+919876543210" and "\n" not in params[1] and "user_requested" in params[1]
    await session.refresh(c)
    assert c.handoff_notified_at is not None
    assert await notify_handoff(e.factory, ch, None, "", c.id) == "already_notified"  # type: ignore[arg-type]
    assert len(ch.sent) == 1


async def test_no_template_or_staff_number_leaves_it_for_the_inbox(session: AsyncSession) -> None:
    e = await Env(session).start()
    c1 = await handoff_conv(e, template=False)
    ch = Channel()
    assert await notify_handoff(e.factory, ch, None, "", c1.id) == "no_staff_channel"  # type: ignore[arg-type]
    await session.refresh(c1)
    assert ch.sent == [] and c1.handoff_notified_at is None  # still visible in /admin/handoffs


async def test_failed_alert_is_not_marked_notified_so_it_can_retry(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await handoff_conv(e)
    ch = Channel()
    ch.fail = True
    assert await notify_handoff(e.factory, ch, None, "", c.id) == "send_failed"  # type: ignore[arg-type]
    await session.refresh(c)
    assert c.handoff_notified_at is None
    ch.fail = False
    assert await notify_handoff(e.factory, ch, None, "", c.id) == "notified"  # type: ignore[arg-type]


async def test_llm_polish_replaces_summary_and_usage_is_recorded_offline(
    session: AsyncSession,
) -> None:
    e = await Env(session).start()
    c = await handoff_conv(e)
    llm = ScriptLLM("Customer wants to talk to a person; please call Ravi back.")
    assert await notify_handoff(e.factory, Channel(), llm, "fast-model", c.id) == "notified"  # type: ignore[arg-type]
    await session.refresh(c)
    assert c.handoff_summary == "Customer wants to talk to a person; please call Ravi back."
    [u] = (await session.execute(select(UsageCost))).scalars().all()
    assert (u.step, u.route, u.model, u.input_tokens) == ("summary", "offline", "fast-model", 90)


async def test_invented_polish_falls_back_to_deterministic_summary(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await handoff_conv(e)
    original = c.handoff_summary
    llm = ScriptLLM("Customer wants a refund of 4500 rupees.")  # 4500 is not in the input
    await notify_handoff(e.factory, Channel(), llm, "fast-model", c.id)  # type: ignore[arg-type]
    await session.refresh(c)
    assert c.handoff_summary == original


async def test_not_handoff_conversation_is_ignored(session: AsyncSession) -> None:
    e = await Env(session).start()
    c = await e.conv()  # awaiting_customer
    ch = Channel()
    assert await notify_handoff(e.factory, ch, None, "", c.id) == "not_handoff"  # type: ignore[arg-type]
    assert ch.sent == []


async def test_service_builds_summary_with_progress_and_recent_turns(session: AsyncSession) -> None:
    """Customer is mid-booking, then asks for a person: staff sees what was already collected."""
    from tests.test_journey import Journey

    j = await Journey(session).start()
    await j.send("hi")
    await j.send(payload="menu:book_appointment")
    await j.send(payload=j.first_option())
    await j.send("talk to staff please")
    conv = (await session.execute(select(Conversation))).scalar_one()
    assert conv.status == ConversationStatus.HANDOFF and conv.handoff_notified_at is None
    s = conv.handoff_summary or ""
    assert "Customer asked to talk to a person" in s and "In progress: booking an appointment" in s
    assert "service=" in s and "- customer: talk to staff please" in s
    assert "+919876543210" in s
    cust = (await session.execute(select(Customer))).scalar_one()
    assert cust.phone in s
    _ = (async_sessionmaker, datetime, UTC)

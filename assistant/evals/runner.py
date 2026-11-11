"""Runs eval cases through the REAL orchestrator, rules, flows and validators.

Only the outside world is simulated: tools use an in-memory backend, and in stub mode the LLM
extractor / RAG answer replay what the case author scripted (with declared token usage), so
the suite is deterministic, free and fast. `--live` swaps the stub extractor for the real one.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from assistant.core.models import ConversationStatus
from assistant.evals import fakes
from assistant.evals.schema import EvalCase, Expect, Turn
from assistant.llm.usage import UsageRecord, collect_usage
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.flows import load_industries
from assistant.orchestrator.types import ConvState, Extraction, Extractor, TurnInput, TurnResult
from assistant.tenants.loader import parse_tenant_yaml
from assistant.tenants.schema import TenantConfig, TenantFile

TEMPLATES = Path(__file__).parents[1] / "tenants" / "templates"
PHONE = "+919876543210"
TENANT_FILES = {"clinic": "clinic_sample.yaml", "support": "support_sample.yaml"}


@dataclass
class TurnRecord:
    index: int
    said: str
    replies: list[str]
    problems: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class CaseResult:
    case: EvalCase
    turns: list[TurnRecord]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turns)

    @property
    def tokens(self) -> int:
        return sum(t.tokens_in + t.tokens_out for t in self.turns)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def llm_calls(self) -> int:
        return sum(t.llm_calls for t in self.turns)


def derive_intent(result: TurnResult) -> str | None:
    """Collapse a turn's outcome to one comparable label."""
    reason = result.handoff_reason or ""
    if reason:
        if reason == "user_requested":
            return "human"
        if reason == "medical_question":
            return "medical_advice"
        if reason.startswith("unclear"):
            return "unclear"
        return "handoff"
    if result.state.flow:
        return result.state.flow
    if result.route in ("faq", "cache", "rag"):
        return "faq"
    if result.route == "unknown":
        return "faq_unknown"
    if result.state.state == "identify_intent":
        return "menu"
    return None


def _action_tools(result: TurnResult) -> list[str]:
    """Tool names excluding the read-only `list_*` option lookups (they are plumbing)."""
    return [c["tool"] for c in result.tool_calls if not c["tool"].startswith("list_")]


def check(exp: Expect, result: TurnResult, be: fakes.Backend) -> list[str]:
    problems: list[str] = []
    text = "\n".join(r.text for r in result.replies)
    options = {o.id for r in result.replies for o in r.options}
    st = result.state
    pub = {k: v for k, v in st.slots.items() if not k.startswith("_")}

    def eq(name: str, want: object, got: object) -> None:
        if want is not None and want != got:
            problems.append(f"{name}: expected {want!r}, got {got!r}")

    eq("intent", exp.intent, derive_intent(result))
    eq("route", exp.route, result.route)
    eq("state", exp.state, st.state)
    eq("flow", exp.flow, st.flow)
    eq("status", exp.status, st.status.value)
    eq("language", exp.language, st.language)
    eq("needs_human", exp.needs_human, result.needs_human)
    eq("reply_count", exp.reply_count, len(result.replies))
    eq("bookings", exp.bookings, len(be.live(PHONE)))
    if exp.tool_calls is not None:
        eq("tool_calls", exp.tool_calls, _action_tools(result))
    if exp.handoff_reason is not None and not (result.handoff_reason or "").startswith(
            exp.handoff_reason):
        problems.append(f"handoff_reason: expected prefix {exp.handoff_reason!r}, "
                        f"got {result.handoff_reason!r}")
    for k, v in (exp.slots or {}).items():
        if pub.get(k) != v:
            problems.append(f"slot {k}: expected {v!r}, got {pub.get(k)!r}")
    for s in exp.replies_contain:
        if s not in text:
            problems.append(f"reply should contain {s!r}")
    for s in exp.replies_not_contain:
        if s in text:
            problems.append(f"reply must NOT contain {s!r}")
    for o in exp.options_include:
        if o not in options:
            problems.append(f"option {o!r} not offered (got {sorted(options)})")
    for o in exp.options_exclude:
        if o in options:
            problems.append(f"option {o!r} must NOT be offered")
    return problems


class Runner:
    def __init__(self, *, live_extractor: Extractor | None = None) -> None:
        self.industries = load_industries()
        self.tenants: dict[str, TenantFile] = {
            k: parse_tenant_yaml(TEMPLATES / v) for k, v in TENANT_FILES.items()}
        self.live_extractor = live_extractor  # None = stub mode

    async def run_case(self, case: EvalCase) -> CaseResult:
        tenant = self.tenants[case.industry]
        cfg: TenantConfig = tenant.config
        be = fakes.Backend(
            taken={tuple(t) for t in case.setup.taken}, orders=case.setup.orders,  # type: ignore[misc]
            race_on_book=case.setup.race_on_book, no_slots=case.setup.no_slots)
        for b in case.setup.bookings:
            bid = b.get("id") or be.next_id()
            be.bookings[bid] = {"id": bid, "status": "ok", "phone": PHONE, "service": "general",
                                "name": "Ravi", **b}
            be.taken.add((b["date"], b["time"]))
        script = fakes.ScriptedLLM()
        extractor: Extractor = self.live_extractor or fakes.StubExtractor(script)
        orch = Orchestrator(
            self.industries, fakes.build_registry(be), extractor,
            fakes.EvalAnswerer([(f.language, f.question, f.answer) for f in tenant.faqs], script))
        now = datetime.fromisoformat(case.setup.now) if case.setup.now else fakes.EVAL_NOW

        def fresh(lang: str | None) -> ConvState:
            return ConvState(tenant_id=uuid.uuid4(), conversation_id=uuid.uuid4(),
                             customer_phone=PHONE, language=lang)

        state = fresh(None)
        records: list[TurnRecord] = []
        for i, turn in enumerate(case.turns, 1):
            if state.status in (ConversationStatus.CLOSED, ConversationStatus.EXPIRED):
                state = fresh(state.language)  # a closed chat is followed by a new conversation
            self._script(script, turn)
            with collect_usage() as usage:
                result = await orch.handle(
                    cfg, case.industry, state,
                    TurnInput(text=turn.say, payload_id=turn.payload), now)
            state = result.state
            records.append(self._record(i, turn, result, usage, script, be))
        return CaseResult(case, records)

    def _script(self, script: fakes.ScriptedLLM, turn: Turn) -> None:
        script.calls, script.unexpected = 0, False
        e = turn.extraction
        script.extraction = (Extraction(intent=e.intent, slots=e.slots, confidence=e.confidence,
                                        language=e.language, needs_human=e.needs_human)
                             if e else None)
        if e:
            script.extraction_usage = (e.tokens_in, e.tokens_out, e.cost_usd)
        a = turn.answer
        script.answer_given = a is not None
        script.answer = a.text if a else None
        if a:
            script.answer_usage = (a.tokens_in, a.tokens_out, a.cost_usd)

    def _record(self, i: int, turn: Turn, result: TurnResult, usage: list[UsageRecord],
                script: fakes.ScriptedLLM, be: fakes.Backend) -> TurnRecord:
        rec = TurnRecord(
            index=i, said=turn.say or f"[payload:{turn.payload}]",
            replies=[r.text for r in result.replies],
            llm_calls=sum(1 for u in usage if u.step != "embed"),
            tokens_in=sum(u.input_tokens for u in usage),
            tokens_out=sum(u.output_tokens for u in usage),
            cost_usd=sum(u.cost_usd for u in usage),
        )
        rec.problems = check(turn.expect, result, be)
        if turn.expect.llm_calls is not None and turn.expect.llm_calls != rec.llm_calls:
            rec.problems.append(f"llm_calls: expected {turn.expect.llm_calls}, got {rec.llm_calls}")
        if self.live_extractor is None:  # stub mode keeps cases honest about LLM use
            if script.unexpected:
                rec.problems.append("unexpected LLM call: no extraction/answer was scripted "
                                    "for this turn (rules should have handled it)")
            if turn.extraction and not any(u.step == "extract" for u in usage):
                rec.problems.append("scripted extraction was never used")
            if turn.answer and not any(u.step == "answer" for u in usage):
                rec.problems.append("scripted answer was never used")
        return rec


async def run_all(cases: list[EvalCase], *, live_extractor: Extractor | None = None,
                  on_case: Callable[[CaseResult], None] | None = None) -> list[CaseResult]:
    runner = Runner(live_extractor=live_extractor)
    out = []
    for c in cases:
        r = await runner.run_case(c)
        if on_case:
            on_case(r)
        out.append(r)
    return out

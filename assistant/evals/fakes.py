"""Deterministic in-memory backends for the eval harness (no DB, no network).

Real tool logic is covered by its own DB-backed tests; evals test CONVERSATION behaviour:
routing, slot filling, confirmations, refusals, handoffs, languages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel

from assistant.orchestrator.types import Answer, ConvState, ExtractHints, Extraction
from assistant.rag.answerer import LEXICAL_THRESHOLD, lexical_score
from assistant.router.rules import normalize
from assistant.tenants.schema import TenantConfig
from assistant.tools.base import ToolContext, ToolRegistry, ToolResult
from assistant.tools.labels import date_label

EVAL_NOW = datetime.fromisoformat("2026-11-12T06:00:00+00:00")  # Thu 11:30 IST (clinic open)
DATES = ["2026-11-13", "2026-11-14", "2026-11-16"]  # Fri, Sat, Mon
TIMES = ["10:00", "11:00", "12:00"]


class OptIn(BaseModel):
    slots: dict[str, str] = {}


class AnyArgs(BaseModel):
    model_config = {"extra": "allow"}


@dataclass
class Backend:
    """Mutable world for one eval case; `setup:` in the YAML configures it."""

    taken: set[tuple[str, str]] = field(default_factory=set)
    bookings: dict[str, dict[str, str]] = field(default_factory=dict)  # id -> booking
    orders: dict[str, dict[str, str]] = field(default_factory=dict)
    race_on_book: bool = False  # booking fails with slot_taken although the slot was offered
    no_slots: bool = False
    _n: int = 0

    def live(self, phone: str) -> list[dict[str, str]]:
        return [b for b in self.bookings.values() if b["phone"] == phone and b["status"] == "ok"]

    def next_id(self) -> str:
        self._n += 1
        return f"BK{self._n}"


class _T:
    group: ClassVar[str] = "booking"

    def __init__(self, be: Backend) -> None:
        self.be = be


class ListServices(_T):
    name: ClassVar[str] = "list_services"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        return ToolResult(ok=True, data={"options": [
            {"id": s.id, "title": s.name(ctx.language)} for s in ctx.config.booking.services]})


class ListDates(_T):
    name: ClassVar[str] = "list_dates"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        return ToolResult(ok=True, data={"options": [
            {"id": d, "title": date_label(datetime.fromisoformat(d).date(), ctx.language)}
            for d in DATES]})


class ListSlots(_T):
    name: ClassVar[str] = "list_slots"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        day = args.slots.get("date", "")
        times = [] if self.be.no_slots else [t for t in TIMES if (day, t) not in self.be.taken]
        return ToolResult(ok=True, data={"options": [{"id": t, "title": t} for t in times]})


class ListMyBookings(_T):
    name: ClassVar[str] = "list_my_bookings"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        return ToolResult(ok=True, data={"options": [
            {"id": b["id"], "title": f"{b['date']} {b['time']}"}
            for b in self.be.live(ctx.customer_phone)]})


class BookAppointment(_T):
    name: ClassVar[str] = "book_appointment"
    input_model: ClassVar[type[BaseModel]] = AnyArgs

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        d, t = args.date, args.time
        if self.be.race_on_book or (d, t) in self.be.taken:
            return ToolResult(ok=False, error_code="slot_taken")
        bid = self.be.next_id()
        self.be.taken.add((d, t))
        self.be.bookings[bid] = {"id": bid, "date": d, "time": t, "service": args.service,
                                 "name": args.patient_name, "phone": ctx.customer_phone,
                                 "status": "ok"}
        return ToolResult(ok=True, data={"ref": bid})


class CancelBooking(_T):
    name: ClassVar[str] = "cancel_booking"
    input_model: ClassVar[type[BaseModel]] = AnyArgs

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        b = self.be.bookings.get(args.booking)
        if not b or b["phone"] != ctx.customer_phone or b["status"] != "ok":
            return ToolResult(ok=False, error_code="not_found")
        b["status"] = "cancelled"
        self.be.taken.discard((b["date"], b["time"]))
        return ToolResult(ok=True, data={"ref": b["id"]})


class RescheduleBooking(_T):
    name: ClassVar[str] = "reschedule_booking"
    input_model: ClassVar[type[BaseModel]] = AnyArgs

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        b = self.be.bookings.get(args.booking)
        if not b or b["phone"] != ctx.customer_phone or b["status"] != "ok":
            return ToolResult(ok=False, error_code="not_found")
        if (args.date, args.time) in self.be.taken:
            return ToolResult(ok=False, error_code="slot_taken")
        self.be.taken.discard((b["date"], b["time"]))
        b["date"], b["time"] = args.date, args.time
        self.be.taken.add((b["date"], b["time"]))
        return ToolResult(ok=True, data={"ref": b["id"]})


class LookupOrder(_T):
    name: ClassVar[str] = "lookup_order"
    group: ClassVar[str] = "orders"
    input_model: ClassVar[type[BaseModel]] = AnyArgs

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        o = self.be.orders.get(args.order_id)
        if not o:
            return ToolResult(ok=False, error_code="not_found")
        return ToolResult(ok=True, data={"status": o["status"], "eta": o.get("eta", "-")})


class CreateReturn(_T):
    name: ClassVar[str] = "create_return"
    group: ClassVar[str] = "orders"
    input_model: ClassVar[type[BaseModel]] = AnyArgs

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        o = self.be.orders.get(args.order_id)
        if not o:
            return ToolResult(ok=False, error_code="not_found")
        if o.get("returnable") == "no":
            return ToolResult(ok=False, error_code="not_eligible")
        return ToolResult(ok=True, data={"ref": f"RT-{args.order_id}"})


def build_registry(be: Backend) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (ListServices, ListDates, ListSlots, ListMyBookings, BookAppointment,
                 CancelBooking, RescheduleBooking, LookupOrder, CreateReturn):
        reg.register(tool(be))
    return reg


# ------------------------------------------------------------------ scripted LLM stand-ins
@dataclass
class ScriptedLLM:
    """Stub-mode stand-in for the extractor and RAG answerer: replays what the case author
    wrote for the current turn, and records the declared token usage. If the engine asks for an
    LLM on a turn where none was scripted, the runner flags it ('unexpected LLM call')."""

    extraction: Extraction | None = None
    extraction_usage: tuple[int, int, float] = (180, 25, 0.0002)
    answer: str | None = None
    answer_given: bool = False
    answer_usage: tuple[int, int, float] = (320, 40, 0.0004)
    calls: int = 0
    unexpected: bool = False


class StubExtractor:
    def __init__(self, script: ScriptedLLM) -> None:
        self.script = script

    async def extract(self, cfg: TenantConfig, state: ConvState, text: str,
                      hints: ExtractHints) -> Extraction | None:
        from assistant.llm.usage import UsageRecord, record

        self.script.calls += 1
        if self.script.extraction is None:
            self.script.unexpected = True
            return None
        tin, tout, cost = self.script.extraction_usage
        record(UsageRecord("extract", "stub-fast", tin, 0, tout, 0, cost))
        return self.script.extraction


class EvalAnswerer:
    """FAQ lookup mirrors the real answerer's exact + lexical tiers over the tenant's YAML FAQs;
    generate() replays the case's scripted RAG answer."""

    def __init__(self, faqs: list[tuple[str, str, str]], script: ScriptedLLM) -> None:
        self.faqs, self.script = faqs, script  # (language, question, answer)

    async def lookup(self, cfg: TenantConfig, state: ConvState, text: str, lang: str, *,
                     exact_only: bool = False) -> Answer | None:
        pool = [(q, a) for lg, q, a in self.faqs if lg == lang]
        norm = normalize(text)
        for q, a in pool:
            if normalize(q) == norm:
                return Answer(text=a, source="faq")
        if exact_only:
            return None
        best = max(((lexical_score(text, q), a) for q, a in pool), default=(0.0, ""))
        return Answer(text=best[1], source="faq") if best[0] >= LEXICAL_THRESHOLD else None

    async def generate(self, cfg: TenantConfig, state: ConvState, text: str, lang: str
                       ) -> Answer | None:
        from assistant.llm.usage import UsageRecord, record

        self.script.calls += 1
        if not self.script.answer_given:
            self.script.unexpected = True
            return None
        tin, tout, cost = self.script.answer_usage
        record(UsageRecord("answer", "stub-fast", tin, 0, tout, 0, cost))
        return Answer(text=self.script.answer, source="rag") if self.script.answer else None

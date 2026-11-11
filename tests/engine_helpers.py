"""Test doubles: an in-memory clinic backend exposed through the real ToolRegistry."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from assistant.core.models import ConversationStatus
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.flows import load_industries
from assistant.orchestrator.types import (
    Answer,
    ConvState,
    ExtractHints,
    Extraction,
    TurnInput,
    TurnResult,
)
from assistant.tenants.loader import parse_tenant_yaml
from assistant.tenants.schema import TenantConfig
from assistant.tools.base import ToolContext, ToolRegistry, ToolResult

SAMPLE = Path(__file__).parents[1] / "assistant/tenants/templates/clinic_sample.yaml"
CFG: TenantConfig = parse_tenant_yaml(SAMPLE).config
OPEN_NOW = datetime(2026, 11, 12, 6, 0, tzinfo=UTC)  # Thu 11:30 IST: clinic open
CLOSED_NOW = datetime(2026, 11, 15, 6, 0, tzinfo=UTC)  # Sunday: closed


class OptIn(BaseModel):
    slots: dict[str, str] = {}


class BookIn(BaseModel):
    service: str
    date: str
    time: str
    patient_name: str
    phone: str
    idempotency_key: str


class Backend:
    def __init__(self) -> None:
        self.taken: set[tuple[str, str]] = set()
        self.booked: list[BookIn] = []
        self.no_slots = False
        self.race = False  # booking fails even though list_slots showed the slot free


class _Base:
    group: ClassVar[str] = "booking"

    def __init__(self, be: Backend) -> None:
        self.be = be


class ListServices(_Base):
    name: ClassVar[str] = "list_services"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        return ToolResult(ok=True, data={"options": [
            {"id": "gen", "title": "General"}, {"id": "derm", "title": "Dermatology"}]})


class ListDates(_Base):
    name: ClassVar[str] = "list_dates"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        return ToolResult(ok=True, data={"options": [
            {"id": "2026-11-13", "title": "Fri 13 Nov"}, {"id": "2026-11-14", "title": "Sat 14 Nov"}]})


class ListSlots(_Base):
    name: ClassVar[str] = "list_slots"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        date = args.slots.get("date", "")
        opts = [] if self.be.no_slots else [
            {"id": t, "title": t} for t in ("10:00", "11:00") if (date, t) not in self.be.taken]
        return ToolResult(ok=True, data={"options": opts})


class Book(_Base):
    name: ClassVar[str] = "book_appointment"
    input_model: ClassVar[type[BaseModel]] = BookIn

    async def run(self, ctx: ToolContext, args: BookIn) -> ToolResult:
        if self.be.race or (args.date, args.time) in self.be.taken:
            return ToolResult(ok=False, error_code="slot_taken")
        self.be.taken.add((args.date, args.time))
        self.be.booked.append(args)
        return ToolResult(ok=True, data={"ref": "BK1"})


class FakeExtractor:
    def __init__(self, ext: Extraction | None) -> None:
        self.ext, self.calls = ext, 0
        self.hints: list[ExtractHints] = []

    async def extract(self, cfg: TenantConfig, state: ConvState, text: str,
                      hints: ExtractHints) -> Extraction | None:
        self.calls += 1
        self.hints.append(hints)
        return self.ext


class FakeAnswerer:
    """`answer_text` is returned by generate(); `lookup_text` (if set) by lookup()."""

    def __init__(self, answer: str | None, lookup: str | None = None) -> None:
        self.answer_text, self.lookup_text = answer, lookup
        self.calls = 0
        self.lookups = 0

    async def lookup(self, cfg: TenantConfig, state: ConvState, text: str, lang: str, *,
                     exact_only: bool = False) -> Answer | None:
        self.lookups += 1
        if exact_only:  # the real answerer's exact tier only matches verbatim FAQ questions
            return None
        return Answer(text=self.lookup_text, source="faq") if self.lookup_text else None

    async def generate(self, cfg: TenantConfig, state: ConvState, text: str, lang: str
                       ) -> Answer | None:
        self.calls += 1
        return Answer(text=self.answer_text, source="rag") if self.answer_text else None


def make(be: Backend | None = None, extractor: FakeExtractor | None = None,
         tools_enabled: list[str] | None = None, answerer: FakeAnswerer | None = None) -> tuple[Orchestrator, Backend, TenantConfig]:
    be = be or Backend()
    reg = ToolRegistry()
    for t in (ListServices, ListDates, ListSlots, Book):
        reg.register(t(be))  # type: ignore[arg-type]
    cfg = CFG.model_copy(deep=True)
    if tools_enabled is not None:
        cfg.tools_enabled = tools_enabled
    return Orchestrator(load_industries(), reg, extractor, answerer), be, cfg


class Chat:
    """Drives one conversation turn by turn, persisting state like the service would."""

    def __init__(self, orch: Orchestrator, cfg: TenantConfig, now: datetime = OPEN_NOW) -> None:
        self.orch, self.cfg, self.now = orch, cfg, now
        self.state = ConvState(tenant_id=uuid.uuid4(), conversation_id=uuid.uuid4(),
                               customer_phone="+919876543210")
        self.last: TurnResult | None = None

    async def say(self, text: str | None = None, payload: str | None = None) -> TurnResult:
        self.last = await self.orch.handle(
            self.cfg, "clinic", self.state, TurnInput(text=text, payload_id=payload), self.now)
        self.state = self.last.state
        return self.last

    @property
    def texts(self) -> list[str]:
        assert self.last
        return [r.text for r in self.last.replies]

    @property
    def option_ids(self) -> list[str]:
        assert self.last
        return [o.id for r in self.last.replies for o in r.options]

    @property
    def closed(self) -> bool:
        return self.state.status == ConversationStatus.CLOSED


def _unused(_: Any) -> None: ...

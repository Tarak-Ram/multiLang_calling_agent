"""Eval case format (YAML). Each case is a short conversation with per-turn expectations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taken: list[tuple[str, str]] = []  # (date, time) already booked by someone else
    bookings: list[dict[str, str]] = []  # this customer's existing bookings
    orders: dict[str, dict[str, str]] = {}
    race_on_book: bool = False
    no_slots: bool = False
    now: str | None = None  # ISO time override (e.g. a Sunday to test closed-hours handoff)


class ScriptedExtraction(BaseModel):
    """What the (stubbed) LLM extractor returns this turn; omitted = LLM must NOT be called."""

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    slots: dict[str, str] = {}
    confidence: float = 0.9
    language: str | None = None
    needs_human: bool = False
    tokens_in: int = 180
    tokens_out: int = 25
    cost_usd: float = 0.0002


class ScriptedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None  # None = the model declined (-> "I don't know")
    tokens_in: int = 320
    tokens_out: int = 40
    cost_usd: float = 0.0004


class Expect(BaseModel):
    """Every key is optional; only the keys present are checked."""

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None  # flow | human | medical_advice | faq | faq_unknown | menu | unclear
    route: str | None = None  # rule | extractor | faq | cache | rag | unknown | human_owned
    state: str | None = None
    flow: str | None = None
    status: str | None = None
    language: str | None = None
    slots: dict[str, str] | None = None  # subset of the public slots
    tool_calls: list[str] | None = None  # tool names called this turn, in order
    replies_contain: list[str] = []
    replies_not_contain: list[str] = []
    options_include: list[str] = []
    options_exclude: list[str] = []
    llm_calls: int | None = None  # LLM completions this turn (0 = rules/FAQ only)
    needs_human: bool | None = None
    handoff_reason: str | None = None  # prefix match
    reply_count: int | None = None
    bookings: int | None = None  # live bookings in the backend after this turn


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str | None = None
    payload: str | None = None
    extraction: ScriptedExtraction | None = None
    answer: ScriptedAnswer | None = None
    expect: Expect = Field(default_factory=Expect)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: str  # the language this case is reported under
    industry: str = "clinic"
    tags: list[str] = []
    setup: Setup = Field(default_factory=Setup)
    turns: list[Turn]


def load_cases(path: Path) -> list[EvalCase]:
    """Load one YAML file or every *.yaml in a directory. Duplicate ids are an error."""
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    cases: list[EvalCase] = []
    for f in files:
        raw: Any = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        cases.extend(EvalCase.model_validate(c) for c in raw)
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate eval case ids: {sorted(dupes)}")
    return cases

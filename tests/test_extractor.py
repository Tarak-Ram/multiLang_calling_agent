import json
import uuid

from assistant.channels.base import Option
from assistant.llm.base import LLMRequest, LLMResponse, LLMUnavailableError
from assistant.llm.usage import collect_usage
from assistant.orchestrator.types import ConvState, ExtractHints
from assistant.router.extractor import (
    SYSTEM_PROMPT,
    LLMExtractor,
    build_user_block,
    parse_extraction,
)
from tests.engine_helpers import CFG

HINTS = ExtractHints(intents=["book_appointment", "faq", "human", "medical_advice", "other"],
                     today="2026-11-12 Thu", slot_names=["service", "date"])


def state(**kw: object) -> ConvState:
    return ConvState(tenant_id=uuid.uuid4(), conversation_id=uuid.uuid4(),
                     customer_phone="+919876543210", **kw)  # type: ignore[arg-type]


def j(**kw: object) -> str:
    return json.dumps({"intent": "book_appointment", "slots": {}, "confidence": 0.9, **kw})


def test_parse_valid_and_fenced_json() -> None:
    e = parse_extraction("```json\n" + j(slots={"date": "2026-11-13"}, language="te") + "\n```", HINTS)
    assert e and e.intent == "book_appointment" and e.slots == {"date": "2026-11-13"}
    assert e.language == "te" and e.confidence == 0.9


def test_parse_rejects_garbage_and_sanitises() -> None:
    assert parse_extraction("I think they want an appointment", HINTS) is None
    assert parse_extraction("{not json}", HINTS) is None
    assert parse_extraction("[1,2]", HINTS) is None
    e = parse_extraction(j(intent="launch_rockets", slots={"service": "gen", "evil": "x", "date": None},
                           confidence=7), HINTS)
    assert e and e.intent is None  # intent outside the allowed set is dropped
    assert e.slots == {"service": "gen"}  # unknown slot 'evil' and null values dropped
    assert e.confidence == 1.0  # clamped


def test_user_block_is_compact_and_includes_only_needed_context() -> None:
    st = state(language="te", flow="book_appointment", slots={"service": "gen", "_labels": {}},
               summary="asked for dermatology", recent=[("customer", "hi"), ("bot", "hello!"),
                                                        ("customer", "a"), ("bot", "b"),
                                                        ("customer", "c")])
    h = HINTS.model_copy(update={"slot": "date", "options": [Option(id="2026-11-13", title="Fri 13 Nov")]})
    block = build_user_block(st, "kal", h)
    assert "choices[date]=2026-11-13=Fri 13 Nov" in block and 'filled={"service":"gen"}' in block
    assert "customer: hi" not in block  # only the LAST 4 turns are sent
    assert "customer: c" in block and "summary=asked for dermatology" in block
    assert block.rstrip().endswith("message: kal")
    assert len(block) < 700  # token budget guard (~175 tokens)
    assert len(SYSTEM_PROMPT) < 1100  # short, stable


class ScriptLLM:
    def __init__(self, by_model: dict[str, str | Exception]) -> None:
        self.by_model, self.reqs = by_model, []

    async def complete(self, r: LLMRequest) -> LLMResponse:
        self.reqs.append(r)
        out = self.by_model[r.model]
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=r.model, input_tokens=100, output_tokens=15, cost_usd=0.001)


async def test_confident_fast_answer_makes_exactly_one_call() -> None:
    llm = ScriptLLM({"fast": j(confidence=0.95)})
    ex = LLMExtractor(llm, "fast", "strong")
    with collect_usage() as usage:
        e = await ex.extract(CFG, state(), "book me", HINTS)
    assert e and e.intent == "book_appointment" and len(llm.reqs) == 1
    assert [u.step for u in usage] == ["extract"] and usage[0].input_tokens == 100
    assert llm.reqs[0].max_tokens == 150 and llm.reqs[0].json_mode


async def test_low_confidence_escalates_to_strong_model() -> None:
    llm = ScriptLLM({"fast": j(confidence=0.3), "strong": j(confidence=0.9, intent="faq")})
    with collect_usage() as usage:
        e = await LLMExtractor(llm, "fast", "strong").extract(CFG, state(), "hmm", HINTS)
    assert e and e.intent == "faq" and [u.step for u in usage] == ["extract", "escalate"]


async def test_no_escalation_without_strong_model_or_when_same() -> None:
    llm = ScriptLLM({"fast": j(confidence=0.3)})
    await LLMExtractor(llm, "fast", "").extract(CFG, state(), "hmm", HINTS)
    await LLMExtractor(llm, "fast", "fast").extract(CFG, state(), "hmm", HINTS)
    assert len(llm.reqs) == 2


async def test_unparseable_fast_output_escalates_once() -> None:
    llm = ScriptLLM({"fast": "sorry I cannot", "strong": j()})
    e = await LLMExtractor(llm, "fast", "strong").extract(CFG, state(), "x", HINTS)
    assert e and len(llm.reqs) == 2


async def test_llm_down_degrades_to_none_instead_of_crashing() -> None:
    llm = ScriptLLM({"fast": LLMUnavailableError("down")})
    assert await LLMExtractor(llm, "fast").extract(CFG, state(), "x", HINTS) is None


async def test_tenant_model_override_and_stable_system_prompt() -> None:
    cfg = CFG.model_copy(deep=True)
    cfg.models.fast = "tenant-fast"
    llm = ScriptLLM({"tenant-fast": j(), "fast": j()})
    ex = LLMExtractor(llm, "fast")
    await ex.extract(cfg, state(), "a", HINTS)
    await ex.extract(CFG, state(language="te"), "b", HINTS)
    assert [r.model for r in llm.reqs] == ["tenant-fast", "fast"]
    assert llm.reqs[0].system == llm.reqs[1].system == SYSTEM_PROMPT  # cacheable prefix

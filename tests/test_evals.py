"""The eval suite as a pytest gate, plus tests of the harness itself."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from assistant.evals.report import compute_metrics, render_text
from assistant.evals.run import DEFAULT_CASES
from assistant.evals.runner import Runner, run_all
from assistant.evals.schema import EvalCase, load_cases

CASES = load_cases(DEFAULT_CASES)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_eval_case(case: EvalCase) -> None:
    """Every sample conversation must pass: routing, slots, tool calls, refusals, languages."""
    result = await Runner().run_case(case)
    problems = [f"turn {t.index} ({t.said!r}): {t.problems}" for t in result.turns if t.problems]
    assert not problems, "\n".join(problems)


def test_sample_suite_covers_all_launch_languages_and_flows() -> None:
    langs = {c.language for c in CASES}
    assert {"en", "hi", "te"} <= langs
    per_lang = {lang: sum(c.language == lang for c in CASES) for lang in ("en", "hi", "te")}
    assert all(n >= 10 for n in per_lang.values()), per_lang
    tags = {t for c in CASES for t in c.tags}
    assert {"booking", "cancel", "reschedule", "faq", "medical", "handoff", "race", "code-mixed",
            "romanized", "orders", "returns"} <= tags
    assert {c.industry for c in CASES} == {"clinic", "support"}


async def test_metrics_report_accuracy_tokens_and_cost() -> None:
    results = await run_all(CASES)
    m = compute_metrics(results)
    assert m.turn_accuracy == 100.0 and m.case_accuracy == 100.0 and not m.failures
    assert m.cases == len(CASES) and set(m.by_language) >= {"en", "hi", "te"}
    # the headline cost claim: most conversations need no LLM at all
    assert m.llm_free_conversation_pct >= 75 and m.llm_free_turn_pct >= 90
    assert 0 < m.avg_tokens_per_conversation < 400
    assert 0 < m.avg_cost_per_conversation_usd < 0.001
    text = render_text(m)
    assert "turns:" in text and "avg tokens / conversation" in text and "te" in text


async def test_harness_detects_wrong_expectations() -> None:
    bad = EvalCase.model_validate({
        "id": "bad", "language": "en",
        "turns": [{"say": "hello", "expect": {
            "intent": "book_appointment", "route": "extractor", "language": "te",
            "replies_contain": ["no such text"], "options_include": ["nope"], "llm_calls": 3}}],
    })
    result = await Runner().run_case(bad)
    assert not result.passed
    joined = " ".join(result.turns[0].problems)
    for needle in ("intent", "route", "language", "no such text", "nope", "llm_calls"):
        assert needle in joined
    m = compute_metrics([result])
    assert m.turn_accuracy == 0.0 and m.failures[0].case == "bad" and "FAILURES" in render_text(m)


async def test_harness_flags_unscripted_and_unused_llm_use() -> None:
    # (1) the LLM is reached but the author scripted nothing
    unscripted = EvalCase.model_validate({"id": "u", "language": "en", "turns": [
        {"say": "book appointment"}, {"say": "blah blah"}]})
    r = await Runner().run_case(unscripted)
    assert "unexpected LLM call" in " ".join(r.turns[1].problems)
    # (2) a scripted extraction that the engine never needed
    unused = EvalCase.model_validate({"id": "v", "language": "en", "turns": [
        {"say": "hello", "extraction": {"intent": "faq"}}]})
    r2 = await Runner().run_case(unused)
    assert "never used" in " ".join(r2.turns[0].problems)


def test_schema_rejects_typos_and_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate({"id": "x", "language": "en", "turns": [
            {"say": "hi", "expect": {"replys_contain": ["x"]}}]})  # misspelt key
    f = tmp_path / "dup.yaml"
    f.write_text('- {id: a, language: en, turns: [{say: hi}]}\n- {id: a, language: en, turns: [{say: hi}]}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(f)


async def test_conversation_closed_then_new_conversation_starts_fresh() -> None:
    case = EvalCase.model_validate({"id": "two", "language": "en", "turns": [
        {"say": "book appointment"}, {"payload": "general"}, {"payload": "2026-11-13"},
        {"payload": "10:00"}, {"say": "Ravi Kumar"},
        {"say": "yes", "expect": {"status": "closed", "bookings": 1}},
        {"say": "hello", "expect": {"intent": "menu", "status": "active", "bookings": 1}}]})
    assert (await Runner().run_case(case)).passed

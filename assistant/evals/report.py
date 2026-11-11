"""Aggregate eval metrics: accuracy, tokens, cost, LLM usage, per language."""
from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from assistant.evals.runner import CaseResult


class LangStats(BaseModel):
    cases: int
    cases_passed: int
    turns: int
    turns_passed: int
    turn_accuracy: float
    avg_tokens_per_conversation: float
    avg_cost_per_conversation_usd: float


class Failure(BaseModel):
    case: str
    turn: int
    said: str
    problems: list[str]


class Metrics(BaseModel):
    mode: str
    cases: int
    cases_passed: int
    case_accuracy: float
    turns: int
    turns_passed: int
    turn_accuracy: float
    avg_tokens_per_conversation: float
    avg_cost_per_conversation_usd: float
    avg_llm_calls_per_conversation: float
    llm_free_conversation_pct: float
    llm_free_turn_pct: float
    by_language: dict[str, LangStats]
    failures: list[Failure]


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def _avg(total: float, n: int, digits: int = 1) -> float:
    return round(total / n, digits) if n else 0.0


def compute_metrics(results: list[CaseResult], mode: str = "stub") -> Metrics:
    turns = [t for r in results for t in r.turns]
    by_lang: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_lang[r.case.language].append(r)
    langs = {}
    for lang, rs in sorted(by_lang.items()):
        ts = [t for r in rs for t in r.turns]
        langs[lang] = LangStats(
            cases=len(rs), cases_passed=sum(r.passed for r in rs), turns=len(ts),
            turns_passed=sum(t.passed for t in ts),
            turn_accuracy=_pct(sum(t.passed for t in ts), len(ts)),
            avg_tokens_per_conversation=_avg(sum(r.tokens for r in rs), len(rs)),
            avg_cost_per_conversation_usd=_avg(sum(r.cost_usd for r in rs), len(rs), 6),
        )
    return Metrics(
        mode=mode, cases=len(results), cases_passed=sum(r.passed for r in results),
        case_accuracy=_pct(sum(r.passed for r in results), len(results)),
        turns=len(turns), turns_passed=sum(t.passed for t in turns),
        turn_accuracy=_pct(sum(t.passed for t in turns), len(turns)),
        avg_tokens_per_conversation=_avg(sum(r.tokens for r in results), len(results)),
        avg_cost_per_conversation_usd=_avg(sum(r.cost_usd for r in results), len(results), 6),
        avg_llm_calls_per_conversation=_avg(sum(r.llm_calls for r in results), len(results), 2),
        llm_free_conversation_pct=_pct(sum(r.llm_calls == 0 for r in results), len(results)),
        llm_free_turn_pct=_pct(sum(t.llm_calls == 0 for t in turns), len(turns)),
        by_language=langs,
        failures=[Failure(case=r.case.id, turn=t.index, said=t.said, problems=t.problems)
                  for r in results for t in r.turns if not t.passed],
    )


def render_text(m: Metrics) -> str:
    lines = [
        f"Evaluation ({m.mode} mode)",
        f"  cases:  {m.cases_passed}/{m.cases} passed  ({m.case_accuracy}%)",
        f"  turns:  {m.turns_passed}/{m.turns} correct  ({m.turn_accuracy}%)",
        f"  avg tokens / conversation:   {m.avg_tokens_per_conversation}",
        f"  avg cost / conversation:     ${m.avg_cost_per_conversation_usd:.6f}",
        f"  avg LLM calls / conversation: {m.avg_llm_calls_per_conversation}",
        (f"  conversations with ZERO LLM calls: {m.llm_free_conversation_pct}%   "
         f"(turns: {m.llm_free_turn_pct}%)"),
        "",
        f"  {'lang':<6}{'cases':>7}{'pass':>6}{'turn acc':>10}{'tokens/conv':>13}{'$/conv':>12}",
    ]
    for lang, s in m.by_language.items():
        lines.append(f"  {lang:<6}{s.cases:>7}{s.cases_passed:>6}{s.turn_accuracy:>9}%"
                     f"{s.avg_tokens_per_conversation:>13}{s.avg_cost_per_conversation_usd:>12.6f}")
    if m.failures:
        lines += ["", "FAILURES:"]
        for f in m.failures:
            lines.append(f"  [{f.case}] turn {f.turn} ({f.said!r})")
            lines.extend(f"      - {p}" for p in f.problems)
    return "\n".join(lines)

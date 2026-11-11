"""CLI:  python -m assistant.evals.run [--cases DIR|FILE] [--json] [--min-accuracy 100] [--live]

Stub mode (default): deterministic, offline, free; the LLM is replayed from the YAML.
--live [--fast M --strong M]: a real LLM extractor with ANY LiteLLM model ids, so you can
        compare providers side by side (provider key from the environment); stubbed
        extractions are ignored and real tokens/cost are reported. RAG answers stay scripted
        (they need a populated knowledge base); expectations on `route` may differ in live mode.
Exit code 1 if turn accuracy is below --min-accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from assistant.evals.report import compute_metrics, render_text
from assistant.evals.runner import run_all
from assistant.evals.schema import load_cases

DEFAULT_CASES = Path(__file__).parent / "cases"


async def _main(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    if args.language:
        cases = [c for c in cases if c.language == args.language]
    extractor = None
    if args.live:
        from assistant.core.config import get_settings
        from assistant.llm.factory import build_llm
        from assistant.llm.profiles import StaticModels
        from assistant.router.extractor import LLMExtractor

        s = get_settings()
        fast = args.fast or s.llm_fast_model
        if not fast:
            print("--live needs --fast MODEL (or LLM_FAST_MODEL) and the provider API key in "
                  "the environment", file=sys.stderr)
            return 2
        # Compare providers without touching the running platform: pass any model ids.
        extractor = LLMExtractor(
            build_llm(s), max_tokens=s.llm_max_output_tokens,
            resolver=StaticModels(fast, args.strong or s.llm_strong_model,
                                  args.fallback or s.llm_fallback_model))
    results = await run_all(cases, live_extractor=extractor)
    m = compute_metrics(results, mode="live" if args.live else "stub")
    print(m.model_dump_json(indent=2) if args.json else render_text(m))
    return 0 if m.turn_accuracy >= args.min_accuracy else 1


def main() -> None:
    ap = argparse.ArgumentParser(prog="assistant.evals.run")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--language", help="only cases reported under this language code")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-accuracy", type=float, default=100.0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--fast", help="live mode: fast model id (default LLM_FAST_MODEL)")
    ap.add_argument("--strong", help="live mode: strong model id")
    ap.add_argument("--fallback", help="live mode: fallback model id")
    sys.exit(asyncio.run(_main(ap.parse_args())))


if __name__ == "__main__":
    main()

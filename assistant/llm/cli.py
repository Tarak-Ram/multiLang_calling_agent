"""Operator CLI for switching models whenever you want (no redeploy, no admin key needed).

  python -m assistant.llm.cli list
  python -m assistant.llm.cli set NAME --fast P/M [--strong P/M] [--fallback P/M]
                                       [--embedding P/M] [--notes "..."]
  python -m assistant.llm.cli test NAME          # probe the models (a few tokens of real usage)
  python -m assistant.llm.cli activate NAME      # verify, switch, and re-embed if needed
  python -m assistant.llm.cli reembed            # embed any chunk missing the current model
  python -m assistant.llm.cli delete NAME

Model ids are LiteLLM ids ("provider/model"). API keys are read from the environment
(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY); they are never stored.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from assistant.core.config import get_settings
from assistant.core.db import make_engine, make_session_factory
from assistant.core.logging import setup_logging
from assistant.core.redis import make_redis
from assistant.llm.embeddings import EmbedderFactory
from assistant.llm.profiles import ProfileSpec, ProfileStore, env_profile
from assistant.llm.switch import (
    ActivationError,
    activate_profile,
    probe_profile,
    reembed_current,
    run_embedding_migration,
)
from assistant.router.cache import SemanticCache


async def _run(args: argparse.Namespace) -> int:
    s = get_settings()
    setup_logging("WARNING", False)
    engine, redis = make_engine(s), make_redis(s)
    try:
        factory = make_session_factory(engine)
        store = ProfileStore(factory, env_profile(s))
        await store.ensure_seeded()
        embedders = EmbedderFactory()
        if args.cmd == "list":
            snap = await store.snapshot(fresh=True)
            for p in sorted(snap.profiles.values(), key=lambda x: x.name):
                mark = "*" if p.name == snap.active else " "
                print(f"{mark} {p.name:<16} fast={p.fast or '-'}  strong={p.strong or '-'}  "
                      f"fallback={p.fallback or '-'}  embedding={p.embedding_model or '-'}")
            print(f"\nembedding in use: {snap.embedding_model or '-'}   "
                  f"migration: {json.dumps(snap.migration)}")
            return 0
        if args.cmd == "set":
            await store.upsert_profile(ProfileSpec(
                name=args.name, fast=args.fast or "", strong=args.strong or "",
                fallback=args.fallback or "", embedding_model=args.embedding or "",
                notes=args.notes or ""))
            print(f"saved profile {args.name!r} (not active yet: run `activate {args.name}`)")
            return 0
        if args.cmd == "delete":
            print("deleted" if await store.delete_profile(args.name) else "not found")
            return 0
        if args.cmd in ("test", "activate"):
            from assistant.llm.litellm_client import LiteLLMClient

            llm = LiteLLMClient(timeout_s=s.llm_timeout_s)
            snap = await store.snapshot(fresh=True)
            if args.name not in snap.profiles:
                print(f"unknown profile {args.name!r}", file=sys.stderr)
                return 2
            if args.cmd == "test":
                probes = await probe_profile(snap.profiles[args.name], llm, embedders)
                for pr in probes:
                    print(f"{'ok  ' if pr.ok else 'FAIL'} {pr.role:<9} {pr.model}  "
                          f"{pr.error or str(pr.latency_ms) + ' ms'}")
                return 0 if all(pr.ok for pr in probes) else 1
            try:
                res = await activate_profile(store, args.name, llm=llm, embedders=embedders,
                                             verify=not args.no_verify)
            except ActivationError as exc:
                print(f"NOT switched: {exc}", file=sys.stderr)
                return 1
            print(f"active profile: {res.previous} -> {res.active}  (embedding: {res.embedding})")
            if res.embedding == "migration_needed":
                print("re-embedding the knowledge base with the new embedding model ...")
                out = await run_embedding_migration(store, factory, embedders, redis,
                                                    SemanticCache(redis))
                print(f"migration: {out.get('state')} {json.dumps(out.get('last', out))}")
                return 0 if out.get("state") == "idle" else 1
            return 0
        if args.cmd == "reembed":
            print(f"embedded {await reembed_current(store, factory, embedders)} chunks")
            return 0
        return 2
    finally:
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(prog="assistant.llm.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("reembed")
    st = sub.add_parser("set")
    st.add_argument("name")
    for flag in ("fast", "strong", "fallback", "embedding", "notes"):
        st.add_argument(f"--{flag}")
    for name in ("test", "activate", "delete"):
        sp = sub.add_parser(name)
        sp.add_argument("name")
        if name == "activate":
            sp.add_argument("--no-verify", action="store_true")
    sys.exit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()

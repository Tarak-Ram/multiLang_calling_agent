"""Operator CLI.  python -m assistant.tenants.cli import path/to/tenant.yaml [--no-embed]

Imports (idempotently) a tenant YAML, then embeds new knowledge chunks and clears that
tenant's semantic cache so no stale answer survives a KB change.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from assistant.core.config import get_settings
from assistant.core.db import make_engine, make_session_factory
from assistant.core.logging import setup_logging
from assistant.core.redis import make_redis
from assistant.llm.embeddings import EmbedderFactory
from assistant.llm.profiles import ProfileStore, env_profile
from assistant.rag.kb import refresh_tenant_knowledge
from assistant.router.cache import SemanticCache
from assistant.tenants.loader import import_tenant, parse_tenant_yaml

log = logging.getLogger("assistant.cli")


async def import_file(path: Path, embed: bool = True) -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    data = parse_tenant_yaml(path)  # validates before touching the DB
    engine = make_engine(s)
    redis = make_redis(s)
    try:
        factory = make_session_factory(engine)
        store = ProfileStore(factory, env_profile(s))
        await store.ensure_seeded()
        snap = await store.snapshot(fresh=True)  # the ACTIVE embedding model (runtime-switchable)
        embedder = EmbedderFactory().get(snap.embedding_model, snap.embedding_dim) if embed else None
        cache = SemanticCache(redis, embedder)
        async with factory() as session, session.begin():
            tenant = await import_tenant(session, data)
            n = await refresh_tenant_knowledge(
                session, tenant.id, [lang.code for lang in data.config.languages], embedder, cache
            )
        log.info("tenant_imported", extra={"slug": data.slug, "faqs": len(data.faqs),
                                           "documents": len(data.documents), "embedded": n})
    finally:
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(prog="assistant.tenants.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import a tenant YAML file")
    imp.add_argument("path", type=Path)
    imp.add_argument("--no-embed", action="store_true", help="skip embedding (FAQ-only matching)")
    args = ap.parse_args()
    asyncio.run(import_file(args.path, embed=not args.no_embed))


if __name__ == "__main__":
    main()

"""Switching the active model profile safely.

LLM models:       verified (tiny probe call per model) then switched immediately.
Embedding model:  two-phase, because vectors from different models cannot be compared.
   1. activate -> the LLM part takes effect at once; `embedding_effective` still names the OLD
      model, so retrieval and the semantic cache keep working on the old vectors.
   2. run_embedding_migration -> re-embeds every chunk with the NEW model (resumable, batched).
   3. When nothing is left to embed, `embedding_effective` flips atomically to the new model and
      the cache is cleared. Retrieval never sees a mix of vector spaces.
If a migration fails midway, nothing user-visible changes: the old model stays effective and the
migration can simply be run again.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.locks import LockNotAcquiredError, redis_lock
from assistant.core.models import EMBEDDING_DIM, KbChunk, Tenant
from assistant.llm.base import LLMClient, LLMError, LLMRequest, LLMUnavailableError
from assistant.llm.embeddings import EmbedderFactory
from assistant.llm.profiles import ProfileSpec, ProfileStore
from assistant.rag.kb import embed_pending, swap_in_staged_embeddings
from assistant.router.cache import SemanticCache
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.llm.switch")
MIGRATION_LOCK = "lock:embedding-migration"


class ProbeResult(BaseModel):
    role: str  # fast | strong | fallback | embedding
    model: str
    ok: bool
    error: str | None = None
    latency_ms: int = 0


class ActivationError(Exception):
    def __init__(self, message: str, probes: list[ProbeResult] | None = None,
                 code: Literal["unknown", "verify", "migration"] = "verify") -> None:
        super().__init__(message)
        self.probes = probes or []
        self.code = code


class ActivationResult(BaseModel):
    active: str
    previous: str
    embedding: Literal["unchanged", "migration_needed", "switched_off"]
    probes: list[ProbeResult] = []


async def probe_profile(spec: ProfileSpec, llm: LLMClient, embedders: EmbedderFactory
                        ) -> list[ProbeResult]:
    """One tiny real call per distinct model. Costs a few tokens; proves the model id is valid
    and that the provider key in the environment works BEFORE customers are switched onto it."""
    results: list[ProbeResult] = []
    seen: set[str] = set()
    for role in ("fast", "strong", "fallback"):
        model = getattr(spec, role)
        if not model or model in seen:
            continue
        seen.add(model)
        start = time.perf_counter()
        try:
            await llm.complete(LLMRequest(model=model, system="Reply with the single word: ok",
                                          user="ping", max_tokens=5, json_mode=False))
            results.append(ProbeResult(role=role, model=model, ok=True,
                                       latency_ms=int((time.perf_counter() - start) * 1000)))
        except (LLMError, LLMUnavailableError) as exc:
            results.append(ProbeResult(role=role, model=model, ok=False, error=str(exc)[:200]))
    if spec.embedding_model:
        start = time.perf_counter()
        try:
            emb = embedders.get(spec.embedding_model, spec.embedding_dim)
            assert emb is not None
            vec = (await emb.embed(["ping"]))[0]
            ok = len(vec) == EMBEDDING_DIM
            results.append(ProbeResult(
                role="embedding", model=spec.embedding_model, ok=ok,
                error=None if ok else f"returns {len(vec)} dimensions; the database needs "
                                      f"{EMBEDDING_DIM}",
                latency_ms=int((time.perf_counter() - start) * 1000)))
        except (LLMError, LLMUnavailableError) as exc:
            results.append(ProbeResult(role="embedding", model=spec.embedding_model, ok=False,
                                       error=str(exc)[:200]))
    return results


async def activate_profile(
    store: ProfileStore, name: str, *, llm: LLMClient, embedders: EmbedderFactory,
    verify: bool = True,
) -> ActivationResult:
    snap = await store.snapshot(fresh=True)
    spec = snap.profiles.get(name)
    if spec is None:
        raise ActivationError(f"unknown profile {name!r}", code="unknown")
    mig = snap.migration
    if mig.get("state") == "running" and spec.embedding_model != mig.get("target_model"):
        raise ActivationError("an embedding migration is in progress; wait for it to finish "
                              "(or switch to a profile with the same embedding model)",
                              code="migration")
    probes: list[ProbeResult] = []
    if verify:
        probes = await probe_profile(spec, llm, embedders)
        if any(not p.ok for p in probes):
            bad = ", ".join(f"{p.role}={p.model} ({p.error})" for p in probes if not p.ok)
            raise ActivationError(f"verification failed, nothing was changed: {bad}", probes)
    previous = snap.active
    await store.set_active(name)  # LLM models switch here, immediately

    embedding: Literal["unchanged", "migration_needed", "switched_off"] = "unchanged"
    if spec.embedding_model != snap.embedding_model:
        if not spec.embedding_model:
            await store.set_embedding_effective("", spec.embedding_dim)
            await store.set_migration({"state": "idle"})
            embedding = "switched_off"
        else:
            await store.set_migration({"state": "pending", "target_model": spec.embedding_model,
                                       "target_dim": spec.embedding_dim})
            embedding = "migration_needed"
    log.info("profile_activated", extra={"profile": name, "previous": previous,
                                         "embedding": embedding})
    return ActivationResult(active=name, previous=previous, embedding=embedding, probes=probes)


async def _progress(factory: async_sessionmaker[AsyncSession], target: str) -> tuple[int, int]:
    async with factory() as s:
        total = (await s.execute(select(func.count()).select_from(KbChunk)
                                 .where(KbChunk.approved.is_(True)))).scalar_one()
        done = (await s.execute(select(func.count()).select_from(KbChunk).where(
            KbChunk.approved.is_(True),
            or_(KbChunk.embedding_next_model == target,
                KbChunk.embedding_model == target)))).scalar_one()
    return int(done), int(total)


async def run_embedding_migration(
    store: ProfileStore, factory: async_sessionmaker[AsyncSession], embedders: EmbedderFactory,
    redis: Redis, cache: SemanticCache | None = None, batch: int = 32,
) -> dict[str, Any]:
    """Phase 2/3. Safe to call any time, and to re-run after a failure (it resumes). Returns the
    final state. Until the final swap, retrieval is completely unaffected."""
    try:
        async with redis_lock(redis, MIGRATION_LOCK, ttl_ms=3_600_000, wait_s=0):
            return await _migrate(store, factory, embedders, cache, batch)
    except LockNotAcquiredError:
        return {"state": "already_running"}


async def _migrate(
    store: ProfileStore, factory: async_sessionmaker[AsyncSession], embedders: EmbedderFactory,
    cache: SemanticCache | None, batch: int,
) -> dict[str, Any]:
    snap = await store.snapshot(fresh=True)
    mig = snap.migration
    target, dim = str(mig.get("target_model", "")), int(mig.get("target_dim", EMBEDDING_DIM))
    if mig.get("state") not in ("pending", "running", "error") or not target:
        return {"state": "idle", "note": "nothing to migrate"}
    embedder = embedders.get(target, dim)
    assert embedder is not None
    started = datetime.now(UTC).isoformat()
    try:
        for _round in range(5):  # chunks added while we run are picked up by the next round
            async with factory() as s:
                tenants = list((await s.execute(
                    select(KbChunk.tenant_id).distinct().where(
                        KbChunk.embedding_next_model.is_distinct_from(target),
                        KbChunk.embedding_model.is_distinct_from(target)))).scalars().all())
            if not tenants:
                break
            for tid in tenants:
                async with factory() as s, s.begin():  # staging: live vectors keep serving
                    await embed_pending(s, embedder, tid, batch, staging=True)
                done, total = await _progress(factory, target)
                await store.set_migration({"state": "running", "target_model": target,
                                           "target_dim": dim, "done": done, "total": total,
                                           "started_at": started})
        else:
            raise RuntimeError("knowledge base keeps changing; re-run the migration")
    except Exception as exc:
        log.exception("embedding_migration_failed")
        done, total = await _progress(factory, target)
        state = {"state": "error", "target_model": target, "target_dim": dim, "done": done,
                 "total": total, "error": str(exc)[:300], "started_at": started}
        await store.set_migration(state)
        return state

    # Everything is staged on the new model: swap vectors AND flip the effective model in ONE
    # transaction, so no query ever sees a half-switched state. Then drop cached answers.
    async with factory() as s, s.begin():
        await swap_in_staged_embeddings(s, target)
        await store.set_embedding_effective(target, dim, session=s)
    done, total = await _progress(factory, target)
    final = {"state": "idle", "last": {"model": target, "chunks": done, "finished_at":
                                       datetime.now(UTC).isoformat(), "started_at": started}}
    await store.set_migration(final)
    if cache is not None:
        await _purge_caches(factory, cache)
    log.info("embedding_migration_done", extra={"model": target, "chunks": done})
    return final


async def _purge_caches(factory: async_sessionmaker[AsyncSession], cache: SemanticCache) -> None:
    async with factory() as s:
        tenants = (await s.execute(select(Tenant))).scalars().all()
    for t in tenants:
        langs = [lang.code for lang in TenantConfig.model_validate(t.config).languages]
        await cache.invalidate(t.id, languages=langs)


async def reembed_current(
    store: ProfileStore, factory: async_sessionmaker[AsyncSession], embedders: EmbedderFactory,
) -> int:
    """Embed any chunk that lacks a vector from the CURRENT effective model (e.g. knowledge
    imported while another model was effective). Idempotent."""
    snap = await store.snapshot(fresh=True)
    emb = embedders.get(snap.embedding_model, snap.embedding_dim)
    if emb is None:
        return 0
    n = 0
    async with factory() as s:
        tenants = list((await s.execute(select(KbChunk.tenant_id).distinct())).scalars().all())
    for tid in tenants:
        async with factory() as s, s.begin():
            n += await embed_pending(s, emb, tid)
    return n

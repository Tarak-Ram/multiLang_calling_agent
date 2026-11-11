"""Runtime model switching: profiles, hot resolution, verification, safe embedding migration."""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import EMBEDDING_DIM, KbChunk, KbKind, Tenant
from assistant.llm.base import LLMError, LLMRequest, LLMResponse
from assistant.llm.embeddings import HashEmbedder
from assistant.llm.profiles import (
    DEFAULT,
    RULES_ONLY,
    ProfileSpec,
    ProfileStore,
    StaticModels,
)
from assistant.llm.switch import (
    ActivationError,
    activate_profile,
    probe_profile,
    reembed_current,
    run_embedding_migration,
)
from assistant.orchestrator.types import ConvState
from assistant.rag.answerer import KnowledgeAnswerer
from assistant.rag.kb import embed_pending
from assistant.rag.retriever import search
from assistant.router.cache import SemanticCache
from assistant.router.extractor import LLMExtractor
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from assistant.tenants.schema import TenantConfig
from tests.engine_helpers import CFG, SAMPLE
from tests.test_extractor import HINTS, state

ENV = ProfileSpec(name=DEFAULT, fast="prov/a-fast", strong="prov/a-strong",
                  fallback="prov/a-backup", embedding_model="emb-a")


class Emb(HashEmbedder):
    """HashEmbedder with a configurable model name (a distinct 'vector space' per name)."""

    def __init__(self, model: str, dim: int = EMBEDDING_DIM, fail_on_call: int | None = None
                 ) -> None:
        super().__init__(dim)
        self.model, self.calls, self.fail_on_call = model, 0, fail_on_call

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise LLMError("provider exploded", transient=True)
        return await super().embed(texts)


class Factory:
    def __init__(self, **special: Emb) -> None:
        self.special = special
        self.made: list[str] = []

    def get(self, model: str, dim: int) -> Emb | None:
        if not model:
            return None
        self.made.append(model)
        return self.special.get(model) or Emb(model, dim)


class ProbeLLM:
    """Provider stand-in: models listed in `bad` fail like a wrong id / missing API key."""

    def __init__(self, bad: set[str] | None = None) -> None:
        self.bad, self.reqs = bad or set(), []

    async def complete(self, r: LLMRequest) -> LLMResponse:
        self.reqs.append(r)
        if r.model in self.bad:
            raise LLMError("AuthenticationError: no API key for this provider", transient=False)
        return LLMResponse(text=json.dumps({"intent": "book_appointment", "confidence": 0.9}),
                           model=r.model)


def sf(session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]


async def store_with(session: AsyncSession, **kw: Any) -> ProfileStore:
    st = ProfileStore(sf(session), ENV, **kw)
    await st.ensure_seeded()
    await st.upsert_profile(ProfileSpec(name="b", fast="prov/b-fast", strong="prov/b-strong",
                                        fallback="prov/b-backup", embedding_model="emb-a"))
    await st.upsert_profile(ProfileSpec(name="cheap", fast="prov/cheap", embedding_model="emb-a"))
    return st


# --------------------------------------------------------------------------- profiles
async def test_first_boot_seeds_builtin_profiles_from_env_once(session: AsyncSession) -> None:
    st = ProfileStore(sf(session), ENV)
    await st.ensure_seeded()
    snap = await st.snapshot(fresh=True)
    assert set(snap.profiles) == {RULES_ONLY, DEFAULT} and snap.active == DEFAULT
    assert snap.embedding_model == "emb-a" and snap.profiles[RULES_ONLY].fast == ""
    m = await st.resolve(CFG)
    assert (m.fast, m.strong, m.fallback, m.embedding_model) == (
        "prov/a-fast", "prov/a-strong", "prov/a-backup", "emb-a")
    # a later boot with DIFFERENT env must not overwrite what operators switched to
    other = ProfileStore(sf(session), ProfileSpec(name=DEFAULT, fast="prov/other"))
    await other.ensure_seeded()
    assert (await other.snapshot(fresh=True)).profiles[DEFAULT].fast == "prov/a-fast"


async def test_change_propagates_within_ttl_and_not_before(session: AsyncSession) -> None:
    now = [0.0]
    a = await store_with(session, clock=lambda: now[0])
    b = ProfileStore(sf(session), ENV, clock=lambda: now[0])  # "another process"
    assert (await b.resolve(CFG)).fast == "prov/a-fast"  # b caches this
    await a.set_active("b")
    assert (await a.resolve(CFG)).fast == "prov/b-fast"  # the writer sees it immediately
    assert (await b.resolve(CFG)).fast == "prov/a-fast"  # b still inside its 5 s cache
    now[0] = 6.0
    assert (await b.resolve(CFG)).fast == "prov/b-fast"  # ... and catches up on its own


async def test_tenant_can_pick_a_profile_or_pin_models(session: AsyncSession) -> None:
    st = await store_with(session)
    cfg = CFG.model_copy(deep=True)
    cfg.models.profile = "cheap"
    m = await st.resolve(cfg)
    assert (m.fast, m.strong, m.profile) == ("prov/cheap", "", "cheap")
    assert m.embedding_model == "emb-a"  # embedding is platform-wide, never per tenant
    cfg.models.strong = "prov/tenant-pinned"  # an explicit pin beats the profile
    assert (await st.resolve(cfg)).strong == "prov/tenant-pinned"
    cfg.models.profile = "does-not-exist"  # unknown profile -> falls back to the active one
    assert (await st.resolve(cfg)).fast == "prov/a-fast"


async def test_profile_rules_builtins_active_and_dimension(session: AsyncSession) -> None:
    st = await store_with(session)
    with pytest.raises(ValueError, match="built-in"):
        await st.delete_profile(RULES_ONLY)
    with pytest.raises(ValueError, match="built-in"):
        await st.delete_profile(DEFAULT)
    await st.set_active("b")
    with pytest.raises(ValueError, match="active"):
        await st.delete_profile("b")  # an in-use profile cannot be removed from under the platform
    assert await st.delete_profile("cheap") is True and await st.delete_profile("cheap") is False
    with pytest.raises(ValueError, match=str(EMBEDDING_DIM)):
        await st.upsert_profile(ProfileSpec(name="x", embedding_model="m", embedding_dim=768))
    with pytest.raises(ValueError):
        ProfileSpec(name="Bad Name!")  # type: ignore[call-arg]


async def test_static_models_still_honour_tenant_pins() -> None:
    m = StaticModels("f", "s", "b")
    cfg = CFG.model_copy(deep=True)
    cfg.models.fast = "pinned"
    r = await m.resolve(cfg)
    assert (r.fast, r.strong, r.fallback) == ("pinned", "s", "b")


# -------------------------------------------------------------- hot switch (no restart)
async def test_one_extractor_follows_the_active_profile_live(session: AsyncSession) -> None:
    st = await store_with(session, ttl_s=0.0)
    llm = ProbeLLM()
    ex = LLMExtractor(llm, resolver=st)  # built ONCE, like the worker does
    assert await ex.extract(CFG, state(), "hi", HINTS)
    assert llm.reqs[-1].model == "prov/a-fast" and llm.reqs[-1].fallback_model == "prov/a-backup"

    await activate_profile(st, "b", llm=llm, embedders=Factory(), verify=False)  # type: ignore[arg-type]
    assert await ex.extract(CFG, state(), "hi", HINTS)
    assert llm.reqs[-1].model == "prov/b-fast" and llm.reqs[-1].fallback_model == "prov/b-backup"

    calls = len(llm.reqs)
    await activate_profile(st, RULES_ONLY, llm=llm, embedders=Factory(), verify=False)  # type: ignore[arg-type]
    assert await ex.extract(CFG, state(), "hi", HINTS) is None
    assert len(llm.reqs) == calls  # rules-only: zero provider calls, zero cost


# ------------------------------------------------------------------------ verification
async def test_failed_probe_blocks_the_switch_and_changes_nothing(session: AsyncSession) -> None:
    st = await store_with(session)
    llm = ProbeLLM(bad={"prov/b-strong"})
    with pytest.raises(ActivationError) as e:
        await activate_profile(st, "b", llm=llm, embedders=Factory(), verify=True)  # type: ignore[arg-type]
    assert e.value.code == "verify" and "prov/b-strong" in str(e.value) and "nothing was changed" in str(e.value)
    assert [p.ok for p in e.value.probes] == [True, False, True, True]  # fast, strong, fallback, embedding
    assert (await st.snapshot(fresh=True)).active == DEFAULT  # untouched
    ok = await activate_profile(st, "b", llm=llm, embedders=Factory(), verify=False)  # type: ignore[arg-type]
    assert ok.active == "b" and ok.previous == DEFAULT and ok.probes == []  # explicit override


async def test_probe_dedupes_models_and_checks_embedding_dimension(session: AsyncSession) -> None:
    st = await store_with(session)
    llm = ProbeLLM()
    spec = ProfileSpec(name="x", fast="prov/same", strong="prov/same", fallback="prov/other",
                       embedding_model="emb-small")
    wrong = Factory(**{"emb-small": Emb("emb-small", dim=256)})  # returns 256-dim vectors
    probes = await probe_profile(spec, llm, wrong)  # type: ignore[arg-type]
    assert [p.model for p in probes] == ["prov/same", "prov/other", "emb-small"]
    assert len(llm.reqs) == 2  # the shared model was probed once
    emb = probes[-1]
    assert not emb.ok and "256" in (emb.error or "") and str(EMBEDDING_DIM) in (emb.error or "")
    assert llm.reqs[0].max_tokens == 5 and llm.reqs[0].json_mode is False  # tiny + cheap
    _ = st


async def test_unknown_profile_and_blocked_while_embedding_migration_runs(
    session: AsyncSession,
) -> None:
    st = await store_with(session)
    llm, f = ProbeLLM(), Factory()
    with pytest.raises(ActivationError) as e:
        await activate_profile(st, "nope", llm=llm, embedders=f, verify=False)  # type: ignore[arg-type]
    assert e.value.code == "unknown"
    await st.upsert_profile(ProfileSpec(name="emb-b", fast="prov/x", embedding_model="emb-b"))
    await st.set_migration({"state": "running", "target_model": "emb-b"})
    await st.upsert_profile(ProfileSpec(name="emb-c", fast="prov/x", embedding_model="emb-c"))
    with pytest.raises(ActivationError) as e2:
        await activate_profile(st, "emb-c", llm=llm, embedders=f, verify=False)  # type: ignore[arg-type]
    assert e2.value.code == "migration"
    with pytest.raises(ActivationError):  # a profile on the OLD embedding would silently abort it
        await activate_profile(st, "b", llm=llm, embedders=f, verify=False)  # type: ignore[arg-type]
    ok = await activate_profile(st, "emb-b", llm=llm, embedders=f, verify=False)  # type: ignore[arg-type]
    assert ok.active == "emb-b"  # same TARGET embedding model: allowed (LLM models just change)


async def test_switching_embeddings_off_is_immediate(session: AsyncSession) -> None:
    st = await store_with(session)
    res = await activate_profile(st, RULES_ONLY, llm=ProbeLLM(), embedders=Factory(),  # type: ignore[arg-type]
                                 verify=False)
    snap = await st.snapshot(fresh=True)
    assert res.embedding == "switched_off" and snap.embedding_model == "" and snap.migration == {"state": "idle"}


# ---------------------------------------------------------- the embedding migration
async def tenant_with_knowledge(session: AsyncSession) -> Tenant:
    data = parse_tenant_yaml(SAMPLE)
    data.slug, data.whatsapp_phone_number_id = "mig", "pn-mig"
    t = await import_tenant(session, data)
    await embed_pending(session, Emb("emb-a"), t.id)
    await session.commit()
    return t


async def models_in_use(session: AsyncSession) -> dict[str | None, int]:
    rows = (await session.execute(select(KbChunk.embedding_model, func.count())
                                  .group_by(KbChunk.embedding_model))).all()
    return {m: int(n) for m, n in rows}


def answerer(session: AsyncSession, st: ProfileStore, factory: Factory,
             cache: SemanticCache | None = None) -> KnowledgeAnswerer:
    return KnowledgeAnswerer(sf(session), None, cache, None, resolver=st,
                             embedders=factory)  # type: ignore[arg-type]


def conv_state(t: Tenant) -> ConvState:
    return ConvState(tenant_id=t.id, conversation_id=uuid.uuid4(), customer_phone="+919876543210")


async def test_embedding_switch_survives_a_failure_then_swaps_atomically(
    session: AsyncSession,
) -> None:
    t = await tenant_with_knowledge(session)
    cfg = TenantConfig.model_validate(t.config)
    cfg.thresholds.faq_similarity = 0.80
    st = await store_with(session, ttl_s=0.0)
    await st.upsert_profile(ProfileSpec(name="emb-b", fast="prov/b-fast", embedding_model="emb-b"))
    redis = FakeAsyncRedis(decode_responses=True)
    cache = SemanticCache(redis)
    n_chunks = sum((await models_in_use(session)).values())
    assert n_chunks >= 7 and await models_in_use(session) == {"emb-a": n_chunks}

    old_vec = (await Emb("emb-a").embed(["what are your timings?"]))[0]
    await cache.put(t.id, "en", "timings?", "stale answer", old_vec, scope="emb-a")

    # 1) switch: LLM part is live at once, embedding part is only *scheduled*
    res = await activate_profile(st, "emb-b", llm=ProbeLLM(), embedders=Factory(), verify=False)  # type: ignore[arg-type]
    assert res.embedding == "migration_needed"
    snap = await st.snapshot(fresh=True)
    assert snap.active == "emb-b" and snap.embedding_model == "emb-a"  # retrieval still on emb-a
    assert snap.migration["state"] == "pending"
    ans = answerer(session, st, Factory())
    hit = await ans.lookup(cfg, conv_state(t), "what are your timing", "en")
    assert hit and "9am-6pm" in hit.text  # semantic FAQ works during the switch

    # 2) the new provider fails midway: nothing user-visible may change
    boom = Factory(**{"emb-b": Emb("emb-b", fail_on_call=2)})
    out = await run_embedding_migration(st, sf(session), boom, redis, cache, batch=2)  # type: ignore[arg-type]
    assert out["state"] == "error" and "exploded" in out["error"]
    snap = await st.snapshot(fresh=True)
    assert snap.embedding_model == "emb-a" and snap.migration["state"] == "error"
    assert await models_in_use(session) == {"emb-a": n_chunks}
    again = await answerer(session, st, Factory()).lookup(cfg, conv_state(t), "what are your timing", "en")
    assert again and "9am-6pm" in again.text  # still served by the OLD vectors

    # 3) provider recovers: re-run resumes and swaps atomically
    ok = await run_embedding_migration(st, sf(session), Factory(), redis, cache, batch=2)  # type: ignore[arg-type]
    assert ok["state"] == "idle" and ok["last"]["model"] == "emb-b" and ok["last"]["chunks"] == n_chunks
    snap = await st.snapshot(fresh=True)
    assert snap.embedding_model == "emb-b" and snap.migration["state"] == "idle"
    assert await models_in_use(session) == {"emb-b": n_chunks}
    staged = (await session.execute(select(func.count()).select_from(KbChunk).where(
        KbChunk.embedding_next.is_not(None)))).scalar_one()
    assert staged == 0  # staging area cleaned up

    # 4) after the swap: new model is used, old-model vectors can never be compared
    new = await answerer(session, st, Factory()).lookup(cfg, conv_state(t), "what are your timing", "en")
    assert new and "9am-6pm" in new.text
    q = (await Emb("emb-a").embed(["what are your timing"]))[0]
    assert await search(session, t.id, q, kinds=[KbKind.FAQ], embedding_model="emb-a") == []
    assert await cache.get(t.id, "en", "timings?", 0.5, old_vec, scope="emb-a") is None  # cache purged


async def test_only_one_migration_runs_at_a_time(session: AsyncSession) -> None:
    from assistant.core.locks import redis_lock
    from assistant.llm.switch import MIGRATION_LOCK

    st = await store_with(session)
    redis = FakeAsyncRedis(decode_responses=True)
    async with redis_lock(redis, MIGRATION_LOCK):
        out = await run_embedding_migration(st, sf(session), Factory(), redis)  # type: ignore[arg-type]
    assert out == {"state": "already_running"}
    idle = await run_embedding_migration(st, sf(session), Factory(), redis)  # type: ignore[arg-type]
    assert idle["state"] == "idle"  # nothing pending: a harmless no-op


async def test_reembed_current_fixes_chunks_from_the_wrong_model(session: AsyncSession) -> None:
    t = await tenant_with_knowledge(session)
    st = await store_with(session)
    await session.execute(KbChunk.__table__.update().where(KbChunk.tenant_id == t.id).values(
        embedding_model="something-old"))
    await session.commit()
    n = await reembed_current(st, sf(session), Factory())  # type: ignore[arg-type]
    assert n >= 7 and set(await models_in_use(session)) == {"emb-a"}
    assert await reembed_current(st, sf(session), Factory()) == 0  # idempotent
    _ = t

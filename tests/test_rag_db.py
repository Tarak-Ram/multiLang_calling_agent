"""RAG pipeline against real Postgres+pgvector. HashEmbedder is non-semantic (character
trigrams), so thresholds are lowered here: these tests verify PIPELINE LOGIC, isolation and
guards, not embedding quality."""
import uuid

import pytest
from fakeredis import FakeAsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import EMBEDDING_DIM, KbChunk, KbKind, Tenant
from assistant.llm.base import LLMRequest, LLMResponse, LLMUnavailableError
from assistant.llm.embeddings import HashEmbedder
from assistant.llm.usage import collect_usage
from assistant.orchestrator.types import ConvState
from assistant.rag.answerer import SYSTEM_PROMPT, KnowledgeAnswerer
from assistant.rag.kb import embed_pending
from assistant.rag.retriever import search
from assistant.router.cache import SemanticCache
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from assistant.tenants.schema import TenantConfig
from tests.engine_helpers import SAMPLE

EMB = HashEmbedder(EMBEDDING_DIM)


class ScriptLLM:
    def __init__(self, by_model: dict[str, str | Exception]) -> None:
        self.by_model, self.reqs = by_model, []

    async def complete(self, r: LLMRequest) -> LLMResponse:
        self.reqs.append(r)
        out = self.by_model[r.model]  # KeyError == "unexpected call to this model"
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=r.model, input_tokens=300, output_tokens=25,
                           cost_usd=0.002)


def ok(answer: str | None, conf: float = 0.9) -> str:
    import json
    return json.dumps({"answer": answer, "confidence": conf})


async def setup(session: AsyncSession, slug: str = "demo", embed: bool = True
                ) -> tuple[Tenant, TenantConfig, ConvState]:
    data = parse_tenant_yaml(SAMPLE)
    data.slug = slug
    data.whatsapp_phone_number_id = f"pn-{slug}"
    tenant = await import_tenant(session, data)
    if embed:
        await embed_pending(session, EMB, tenant.id)
    await session.commit()
    cfg = TenantConfig.model_validate(tenant.config)
    cfg.thresholds.rag_min_score = 0.10  # HashEmbedder similarities are low by nature
    cfg.thresholds.faq_similarity = 0.80
    cfg.thresholds.cache_similarity = 0.80
    state = ConvState(tenant_id=tenant.id, conversation_id=uuid.uuid4(),
                      customer_phone="+919876543210")
    return tenant, cfg, state


def make(session: AsyncSession, llm: ScriptLLM | None = None, *, embedder: bool = True,
         cache: bool = True) -> KnowledgeAnswerer:
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    sc = SemanticCache(FakeAsyncRedis(decode_responses=True), EMB) if cache and embedder else None
    return KnowledgeAnswerer(factory, EMB if embedder else None, sc, llm, "fast", "strong")


# ------------------------------------------------------------------ kb + retriever
async def test_import_creates_chunks_without_vectors_then_embed_pending(
    session: AsyncSession,
) -> None:
    tenant, _, _ = await setup(session, embed=False)
    from sqlalchemy import select
    rows = (await session.execute(select(KbChunk).where(KbChunk.tenant_id == tenant.id))
            ).scalars().all()
    docs = [r for r in rows if r.kind == KbKind.DOC]
    assert len([r for r in rows if r.kind == KbKind.FAQ]) == 3 and len(docs) >= 3
    assert all(r.embedding is None for r in rows) and all(r.token_count > 0 for r in docs)
    assert await embed_pending(session, EMB, tenant.id) == len(rows)
    assert await embed_pending(session, EMB, tenant.id) == 0  # idempotent


async def test_reimport_replaces_yaml_documents_not_duplicates(session: AsyncSession) -> None:
    from sqlalchemy import func, select
    tenant, _, _ = await setup(session, embed=False)
    n1 = (await session.execute(select(func.count()).select_from(KbChunk))).scalar_one()
    again = parse_tenant_yaml(SAMPLE)
    again.slug = "demo"  # same tenant as setup()
    await import_tenant(session, again)
    n2 = (await session.execute(select(func.count()).select_from(KbChunk))).scalar_one()
    assert n1 == n2 and tenant.id


async def test_retriever_is_tenant_scoped_and_filters(session: AsyncSession) -> None:
    t1, _, _ = await setup(session, "one")
    t2, _, _ = await setup(session, "two")
    vec = (await EMB.embed(["consultation fee general physician"]))[0]
    hits = await search(session, t1.id, vec, limit=10)
    assert hits and {h.id for h in hits}.isdisjoint({h.id for h in await search(session, t2.id, vec, limit=10)})
    assert hits == sorted(hits, key=lambda h: -h.score)
    en_docs = await search(session, t1.id, vec, languages=["en"], kinds=[KbKind.DOC], limit=10)
    assert en_docs and all(h.language == "en" and h.kind == KbKind.DOC for h in en_docs)
    assert "500" in en_docs[0].content  # fee chunk ranks first
    assert await search(session, t1.id, vec, min_score=0.999) == []


async def test_unapproved_chunks_are_never_retrieved(session: AsyncSession) -> None:
    from sqlalchemy import update
    t, _, _ = await setup(session)
    await session.execute(update(KbChunk).where(KbChunk.tenant_id == t.id).values(approved=False))
    vec = (await EMB.embed(["consultation fee"]))[0]
    assert await search(session, t.id, vec) == []


# ------------------------------------------------------------------ lookup (no generation)
async def test_exact_faq_needs_no_embedder_or_llm(session: AsyncSession) -> None:
    _, cfg, st = await setup(session, embed=False)
    a = make(session, embedder=False)
    ans = await a.lookup(cfg, st, "What are your timings?", "en")
    assert ans and ans.source == "faq" and "9am-6pm" in ans.text
    hi = await a.lookup(cfg, st, "आपका समय क्या है", "hi")  # punctuation-insensitive, Hindi
    assert hi and "सोम-शुक्र" in hi.text


async def test_lexical_fallback_when_no_embedder(session: AsyncSession) -> None:
    _, cfg, st = await setup(session, embed=False)
    a = make(session, embedder=False)
    assert (await a.lookup(cfg, st, "your timings", "en")) is not None
    assert (await a.lookup(cfg, st, "do you have parking", "en")) is None


async def test_semantic_faq_match_and_language_isolation(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    a = make(session)
    ans = await a.lookup(cfg, st, "what are your timing", "en")
    assert ans and ans.source == "faq" and "9am-6pm" in ans.text
    te = await a.lookup(cfg, st, "what are your timing", "te")  # Telugu FAQ set only
    assert te is None or "మేము" in te.text  # never returns the English answer for Telugu


async def test_faq_is_tenant_isolated(session: AsyncSession) -> None:
    _, cfg, st = await setup(session, "one")
    _, _, st2 = await setup(session, "two")
    a = make(session, embedder=False)
    assert await a.lookup(cfg, st, "What are your timings?", "en")
    from sqlalchemy import delete
    await session.execute(delete(KbChunk).where(KbChunk.tenant_id == st2.tenant_id))
    await session.commit()
    assert await a.lookup(cfg, st2, "What are your timings?", "en") is None  # t2 has none


# ------------------------------------------------------------------ generate (RAG)
Q = "how much does the general physician consultation cost"


async def test_rag_answer_grounded_cached_and_replayed_free(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    llm = ScriptLLM({"fast": ok("General physician consultation costs 500 rupees.")})
    a = make(session, llm)
    with collect_usage() as usage:
        ans = await a.generate(cfg, st, Q, "en")
    assert ans and ans.source == "rag" and "500" in ans.text
    assert [u.step for u in usage] == ["answer"] and len(llm.reqs) == 1
    req = llm.reqs[0]
    assert req.system == SYSTEM_PROMPT and "CONTEXT:" in req.user and req.user.count("\n[") <= 3
    assert len(req.user) < 1800  # <=3 short chunks, never the whole KB
    again = await a.lookup(cfg, st, Q, "en")  # same question: served from semantic cache
    assert again and again.source == "cache" and again.text == ans.text
    assert len(llm.reqs) == 1  # zero extra LLM calls


async def test_invented_price_is_rejected(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    llm = ScriptLLM({"fast": ok("The consultation costs 650 rupees.")})  # not in KB
    a = make(session, llm)
    assert await a.generate(cfg, st, Q, "en") is None
    assert await a.lookup(cfg, st, Q, "en") is None  # and it was NOT cached


async def test_model_declines_or_low_confidence_gives_none(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    for out in (ok(None, 0.9), ok("Costs 500 rupees.", 0.2), "garbage"):
        llm = ScriptLLM({"fast": out, "strong": ok(None, 0.9)})
        assert await make(session, llm).generate(cfg, st, Q, "en") is None


async def test_clean_decline_or_ungrounded_never_escalates(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    for out in (ok(None, 0.95), ok("It costs 650 rupees.", 0.95)):  # declined / invented number
        llm = ScriptLLM({"fast": out})  # a call to "strong" would raise KeyError
        assert await make(session, llm).generate(cfg, st, Q, "en") is None
        assert [r.model for r in llm.reqs] == ["fast"]


async def test_nothing_relevant_means_no_llm_call(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    cfg.thresholds.rag_min_score = 0.95  # nothing is that similar
    llm = ScriptLLM({"fast": ok("x")})
    assert await make(session, llm).generate(cfg, st, "zzzz qqqq", "en") is None
    assert llm.reqs == []  # irrelevant question never costs tokens


async def test_escalates_to_strong_only_when_fast_unsure(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    llm = ScriptLLM({"fast": ok("Costs 500 rupees.", 0.3),
                     "strong": ok("General consultation costs 500 rupees.", 0.9)})
    with collect_usage() as usage:
        ans = await make(session, llm).generate(cfg, st, Q, "en")
    assert ans and [u.step for u in usage] == ["answer", "escalate"]


async def test_llm_down_or_unconfigured_degrades_to_none(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    down = ScriptLLM({"fast": LLMUnavailableError("down")})  # no escalation when unavailable
    assert await make(session, down).generate(cfg, st, Q, "en") is None
    assert [r.model for r in down.reqs] == ["fast"]
    assert await make(session, None).generate(cfg, st, Q, "en") is None  # FAQ-only mode
    assert await make(session, ScriptLLM({"fast": ok("x")}), embedder=False).generate(
        cfg, st, Q, "en") is None  # retrieval needs embeddings


async def test_cross_language_context_is_allowed_for_generation(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)
    llm = ScriptLLM({"fast": ok("జనరల్ ఫిజీషియన్ ఫీజు 500 రూపాయలు.")})
    ans = await make(session, llm).generate(cfg, st, Q, "te")
    assert ans and "500" in ans.text
    assert "language=Telugu" in llm.reqs[0].user  # LLM told which language to answer in


async def test_embedding_is_computed_once_for_lookup_then_generate(session: AsyncSession) -> None:
    _, cfg, st = await setup(session)

    class Counting(HashEmbedder):
        calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            Counting.calls += 1
            return await super().embed(texts)

    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    llm = ScriptLLM({"fast": ok("General physician consultation costs 500 rupees.")})
    a = KnowledgeAnswerer(factory, Counting(EMBEDDING_DIM), None, llm, "fast")
    assert await a.lookup(cfg, st, Q, "en") is None
    assert await a.generate(cfg, st, Q, "en")
    assert Counting.calls == 1  # memoised between the two stages


@pytest.mark.parametrize("faq_lang", ["en", "hi"])
async def test_exact_faq_wins_without_touching_llm(session: AsyncSession, faq_lang: str) -> None:
    _, cfg, st = await setup(session)
    llm = ScriptLLM({"fast": ok("should not be called")})
    q = {"en": "What are your timings?", "hi": "आपका समय क्या है?"}[faq_lang]
    ans = await make(session, llm).lookup(cfg, st, q, faq_lang)
    assert ans and ans.source == "faq" and llm.reqs == []

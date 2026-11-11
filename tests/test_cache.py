import uuid

from fakeredis import FakeAsyncRedis

from assistant.llm.embeddings import HashEmbedder
from assistant.router.cache import SemanticCache

T1, T2 = uuid.uuid4(), uuid.uuid4()


def make(r: FakeAsyncRedis | None = None, **kw: object) -> SemanticCache:
    return SemanticCache(r or FakeAsyncRedis(decode_responses=True), HashEmbedder(), **kw)  # type: ignore[arg-type]


async def test_hit_above_threshold_miss_below() -> None:
    c = make()
    await c.put(T1, "en", "What are your clinic timings?", "Mon-Fri 9-6.")
    hit = await c.get(T1, "en", "what are your clinic timings", 0.9)
    assert hit and hit.answer == "Mon-Fri 9-6." and hit.score > 0.9
    assert await c.get(T1, "en", "How much is a dental cleaning?", 0.9) is None


async def test_telugu_entries_work() -> None:
    c = make()
    await c.put(T1, "te", "మీ టైమింగ్స్ ఏమిటి?", "ఉదయం 9 నుండి సాయంత్రం 6 వరకు.")
    hit = await c.get(T1, "te", "మీ టైమింగ్స్ ఏమిటి", 0.9)
    assert hit and "ఉదయం" in hit.answer


async def test_tenant_and_language_isolation() -> None:
    c = make()
    await c.put(T1, "en", "What are your timings?", "Tenant1 answer")
    assert await c.get(T2, "en", "What are your timings?", 0.5) is None  # other tenant
    assert await c.get(T1, "hi", "What are your timings?", 0.5) is None  # other language
    assert await c.get(T1, "en", "What are your timings?", 0.5) is not None


async def test_invalidate_clears_cache_for_tenant() -> None:
    c = make()
    await c.put(T1, "en", "timings?", "old answer")
    await c.invalidate(T1, languages=["en"])
    assert await c.get(T1, "en", "timings?", 0.5) is None


async def test_ttl_expiry() -> None:
    now = [1000.0]
    c = make(ttl_s=60, clock=lambda: now[0])
    await c.put(T1, "en", "timings?", "answer")
    assert await c.get(T1, "en", "timings?", 0.9)
    now[0] += 61
    assert await c.get(T1, "en", "timings?", 0.9) is None


async def test_eviction_keeps_newest_max_entries() -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 1
        return now[0]

    c = make(max_entries=2, clock=clock)
    for q in ("alpha question one", "bravo question two", "charlie question three"):
        await c.put(T1, "en", q, f"a-{q}")
    assert await c.get(T1, "en", "alpha question one", 0.95) is None  # oldest evicted
    assert await c.get(T1, "en", "charlie question three", 0.95)


async def test_second_process_sees_writes_via_version_counter() -> None:
    r = FakeAsyncRedis(decode_responses=True)
    a, b = make(r), make(r)  # two worker processes, shared Redis
    assert await b.get(T1, "en", "timings?", 0.5) is None  # b caches the empty state
    await a.put(T1, "en", "timings?", "answer")
    hit = await b.get(T1, "en", "timings?", 0.9)  # version bumped -> b reloads
    assert hit and hit.answer == "answer"
    await a.invalidate(T1, languages=["en"])
    assert await b.get(T1, "en", "timings?", 0.5) is None


async def test_precomputed_vector_skips_embedding_call() -> None:
    class Counting(HashEmbedder):
        calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            Counting.calls += 1
            return await super().embed(texts)

    c = SemanticCache(FakeAsyncRedis(decode_responses=True), Counting())  # type: ignore[arg-type]
    vec = (await HashEmbedder().embed(["timings?"]))[0]
    await c.put(T1, "en", "timings?", "answer", vector=vec)
    assert await c.get(T1, "en", "timings?", 0.9, vector=vec)
    assert Counting.calls == 0

"""Router layer 3: per-tenant, per-language semantic cache for repeated FAQ answers.

Storage (Redis, no RediSearch needed):
  scache:{tenant}:{lang}      HASH  entry_id -> JSON {q, a, v(base64 float32), t(created), exp}
  scache:{tenant}:{lang}:ver  INT   bumped on every write/invalidate
Lookups cost one GET: each process keeps a numpy matrix of the vectors and reloads it only when
the version changed. Brute-force cosine is exact and fast for <= a few hundred entries per
tenant+language (the cap). If a tenant outgrows that, swap this class for pgvector/Redis-Stack.

Only cache answers derived from tenant knowledge (never personal data or booking state):
the caller decides. Tenant id AND language are in every key, so there is no cross-tenant or
cross-language leakage. Call invalidate() whenever a tenant's KB changes (Step 6).
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis

from assistant.llm.embeddings import Embedder
from assistant.router.rules import normalize


@dataclass(slots=True)
class CacheHit:
    answer: str
    question: str
    score: float


@dataclass(slots=True)
class _Index:
    version: int
    ids: list[str]
    matrix: np.ndarray  # (n, dim) L2-normalised float32
    answers: list[str]
    questions: list[str]
    expires: list[float]


def _key(tenant_id: uuid.UUID, lang: str, scope: str = "") -> str:
    """`scope` = the embedding model: entries made with another model can never be hit."""
    return f"scache:{tenant_id}:{lang}:{scope}"


def _ver(key: str) -> str:
    return key.replace("scache:", "scachever:", 1)


def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class SemanticCache:
    def __init__(
        self, redis: Redis, embedder: Embedder | None = None, *, max_entries: int = 300,
        ttl_s: int = 7 * 86_400, clock: Callable[[], float] = time.time,
    ) -> None:
        self._r, self._emb = redis, embedder
        self._max, self._ttl, self._clock = max_entries, ttl_s, clock
        self._local: dict[str, _Index] = {}

    async def _index(self, key: str) -> _Index | None:
        ver = int(await self._r.get(_ver(key)) or 0)
        cached = self._local.get(key)
        if cached and cached.version == ver:
            return cached
        raw = await self._r.hgetall(key)  # type: ignore[misc]
        entries = []
        for eid, blob in raw.items():
            d = json.loads(blob)
            entries.append((eid if isinstance(eid, str) else eid.decode(), d))
        if not entries:
            self._local.pop(key, None)
            return None
        idx = _Index(
            version=ver,
            ids=[e[0] for e in entries],
            matrix=np.stack([np.frombuffer(base64.b64decode(e[1]["v"]), dtype=np.float32)
                             for e in entries]),
            answers=[e[1]["a"] for e in entries],
            questions=[e[1]["q"] for e in entries],
            expires=[e[1]["exp"] for e in entries],
        )
        self._local[key] = idx
        return idx

    async def get(
        self, tenant_id: uuid.UUID, lang: str, question: str, threshold: float,
        vector: list[float] | None = None, scope: str = "",
    ) -> CacheHit | None:
        """Best non-expired entry with cosine >= threshold, else None.

        Text is normalised (case, punctuation, joiners) before embedding, so 'Timings?' and
        'timings' are the same question. Pass `vector` if already embedded this turn."""
        idx = await self._index(_key(tenant_id, lang, scope))
        if idx is None:
            return None
        q = _unit(await self._vector(question, vector))
        scores = idx.matrix @ q
        now = self._clock()
        for i in np.argsort(-scores):
            if scores[i] < threshold:
                return None
            if idx.expires[i] > now:
                return CacheHit(idx.answers[i], idx.questions[i], float(scores[i]))
        return None

    async def put(
        self, tenant_id: uuid.UUID, lang: str, question: str, answer: str,
        vector: list[float] | None = None, scope: str = "",
    ) -> None:
        key = _key(tenant_id, lang, scope)
        vec = _unit(await self._vector(question, vector))
        now = self._clock()
        entry = {"q": question, "a": answer, "t": now, "exp": now + self._ttl,
                 "v": base64.b64encode(vec.astype(np.float32).tobytes()).decode()}
        await self._r.hset(key, uuid.uuid4().hex, json.dumps(entry, ensure_ascii=False))  # type: ignore[misc]
        await self._evict(key)
        await self._r.incr(_ver(key))
        await self._r.expire(key, self._ttl)

    async def _evict(self, key: str) -> None:
        """Keep at most max_entries: drop expired first, then oldest."""
        raw = await self._r.hgetall(key)  # type: ignore[misc]
        if len(raw) <= self._max:
            return
        now = self._clock()
        meta = sorted(((json.loads(b), (e if isinstance(e, str) else e.decode()))
                       for e, b in raw.items()),
                      key=lambda m: (m[0]["exp"] > now, m[0]["t"]))
        for _, eid in meta[: len(raw) - self._max]:
            await self._r.hdel(key, eid)  # type: ignore[misc]

    async def _vector(self, question: str, vector: list[float] | None) -> list[float]:
        if vector is not None:
            return vector
        if self._emb is None:
            raise ValueError("no embedder configured: pass a precomputed vector")
        return (await self._emb.embed([normalize(question)]))[0]

    async def invalidate(self, tenant_id: uuid.UUID, lang: str | None = None,
                         languages: list[str] | None = None) -> None:
        """Drop cached answers for every embedding model (call after a KB/FAQ change)."""
        for code in ([lang] if lang else (languages or [])):
            async for raw in self._r.scan_iter(match=f"scache:{tenant_id}:{code}:*"):
                key = raw if isinstance(raw, str) else raw.decode()
                await self._r.delete(key)
                await self._r.incr(_ver(key))
                self._local.pop(key, None)

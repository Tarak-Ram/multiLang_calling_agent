"""Embedding interface. Provider is a config choice (LiteLLM model id); none is hard-coded."""
from __future__ import annotations

import hashlib
import math
import time
from typing import Protocol

from assistant.llm.base import LLMError
from assistant.llm.usage import UsageRecord, record


class Embedder(Protocol):
    model: str  # tags every stored vector: vectors from different models are never compared
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteLLMEmbedder:
    """Any embedding model LiteLLM supports. `dim` must equal models.EMBEDDING_DIM (the
    pgvector column size), which is verified at startup by the caller."""

    def __init__(self, model: str, dim: int, timeout_s: float = 10.0) -> None:
        self.model, self.dim, self._timeout = model, dim, timeout_s

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm  # lazy: HashEmbedder/tests/evals never need the SDK

        start = time.perf_counter()
        try:
            resp = await litellm.aembedding(model=self.model, input=texts, timeout=self._timeout)
        except Exception as exc:
            raise LLMError(f"embed:{type(exc).__name__}", transient=True) from exc
        vectors = [list(map(float, d["embedding"])) for d in resp.data]
        if vectors and len(vectors[0]) != self.dim:
            raise LLMError(f"embedding dim {len(vectors[0])} != configured {self.dim}",
                           transient=False)
        usage = getattr(resp, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 - pricing lookup must never fail a reply
            cost = 0.0
        record(UsageRecord("embed", self.model,
                           input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                           latency_ms=int((time.perf_counter() - start) * 1000), cost_usd=cost))
        return vectors


class HashEmbedder:
    """DETERMINISTIC, NON-SEMANTIC embedder for tests/dev only: hashed character 3-grams.
    Similar strings score high, but it knows nothing about meaning or translation. Never use
    it in production; startup refuses it unless APP_ENV != prod."""

    model = "hash-3gram"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            s = f"  {t.casefold()}  "
            for i in range(len(s) - 2):
                h = int.from_bytes(hashlib.md5(s[i:i + 3].encode()).digest()[:4], "little")
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class EmbedderFactory:
    """One LiteLLMEmbedder per (model, dim), created on first use. Lets the active profile
    change the embedding model at runtime without rebuilding anything else."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], Embedder] = {}

    def get(self, model: str, dim: int) -> Embedder | None:
        if not model:
            return None
        key = (model, dim)
        if key not in self._cache:
            self._cache[key] = LiteLLMEmbedder(model, dim)
        return self._cache[key]

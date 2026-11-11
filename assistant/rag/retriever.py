"""Vector search over a tenant's knowledge base (pgvector, cosine). Always tenant-scoped."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import KbChunk, KbKind, scoped


@dataclass(slots=True)
class Retrieved:
    id: uuid.UUID
    kind: KbKind
    language: str
    question: str | None
    content: str
    score: float  # cosine similarity, 1.0 = identical

    @property
    def context_text(self) -> str:
        """How a chunk is shown to the LLM (FAQ rows keep their question for context)."""
        return f"Q: {self.question} A: {self.content}" if self.question else self.content


async def search(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    vector: list[float],
    *,
    limit: int = 3,
    min_score: float = 0.0,
    languages: list[str] | None = None,
    kinds: list[KbKind] | None = None,
    embedding_model: str | None = None,
) -> list[Retrieved]:
    """Top `limit` approved chunks with similarity >= min_score, best first.

    Note: with an HNSW index, filtered queries return the best matches *within the index scan*;
    that is exact enough at per-tenant KB sizes (hundreds of chunks). For very large shared
    tables, raise hnsw.ef_search or partition by tenant.
    """
    dist = KbChunk.embedding.cosine_distance(vector)
    stmt = (
        scoped(KbChunk, tenant_id)
        .where(KbChunk.approved.is_(True), KbChunk.embedding.is_not(None))
        .add_columns(dist.label("dist"))
        .order_by(dist)
        .limit(limit)
    )
    if languages:
        stmt = stmt.where(KbChunk.language.in_(languages))
    if kinds:
        stmt = stmt.where(KbChunk.kind.in_(kinds))
    if embedding_model is not None:  # never compare vectors from different models
        stmt = stmt.where(KbChunk.embedding_model == embedding_model)
    out: list[Retrieved] = []
    for chunk, d in (await session.execute(stmt)).all():
        score = 1.0 - float(d)
        if score >= min_score:
            out.append(Retrieved(chunk.id, chunk.kind, chunk.language, chunk.question,
                                 chunk.content, score))
    return out

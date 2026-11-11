"""Knowledge-base maintenance: chunk documents, embed pending chunks, keep the cache honest."""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import KbChunk, KbKind, scoped
from assistant.llm.embeddings import Embedder
from assistant.rag.chunker import chunk_text, estimate_tokens
from assistant.router.cache import SemanticCache
from assistant.tenants.schema import DocEntry

log = logging.getLogger("assistant.rag.kb")

YAML_DOC_PREFIX = "yaml:"


async def replace_yaml_documents(
    session: AsyncSession, tenant_id: uuid.UUID, docs: list[DocEntry]
) -> int:
    """Replace the YAML-managed DOC chunks (source 'yaml:<title>'); returns chunks created.
    Embeddings are NOT computed here; call embed_pending() afterwards."""
    await session.execute(
        delete(KbChunk).where(
            KbChunk.tenant_id == tenant_id, KbChunk.kind == KbKind.DOC,
            KbChunk.source.like(f"{YAML_DOC_PREFIX}%"),
        )
    )
    n = 0
    for doc in docs:
        for piece in chunk_text(doc.text):
            session.add(KbChunk(
                tenant_id=tenant_id, kind=KbKind.DOC, language=doc.language,
                source=f"{YAML_DOC_PREFIX}{doc.title}", content=piece,
                token_count=estimate_tokens(piece),
            ))
            n += 1
    await session.flush()
    return n


def embedding_text(chunk: KbChunk) -> str:
    """What gets embedded: FAQ -> the question (that is what customers phrase); DOC -> title +
    chunk so a chunk like 'It costs 500' still carries its subject."""
    if chunk.kind == KbKind.FAQ and chunk.question:
        return chunk.question
    title = (chunk.source or "").removeprefix(YAML_DOC_PREFIX)
    return f"{title}\n{chunk.content}" if title else chunk.content


async def embed_pending(
    session: AsyncSession, embedder: Embedder, tenant_id: uuid.UUID, batch: int = 32,
    *, staging: bool = False,
) -> int:
    """Embed the tenant's chunks that lack a vector from `embedder`'s model. Returns how many.

    staging=False: fill `embedding` (the live vectors) for chunks with none or from another model.
    staging=True:  fill `embedding_next` instead, used while switching embedding models, so the
                   live vectors keep serving retrieval until the whole set is ready to swap in.
    """
    model_col = KbChunk.embedding_next_model if staging else KbChunk.embedding_model
    vec_col = KbChunk.embedding_next if staging else KbChunk.embedding
    rows = list(await session.scalars(
        scoped(KbChunk, tenant_id).where(
            or_(vec_col.is_(None), model_col.is_distinct_from(embedder.model)))))
    for i in range(0, len(rows), batch):
        group = rows[i:i + batch]
        vectors = await embedder.embed([embedding_text(c) for c in group])
        for chunk, vec in zip(group, vectors, strict=True):
            if staging:
                chunk.embedding_next, chunk.embedding_next_model = vec, embedder.model
            else:
                chunk.embedding, chunk.embedding_model = vec, embedder.model
    await session.flush()
    log.info("kb_embedded", extra={"chunks": len(rows), "staging": staging})
    return len(rows)


async def swap_in_staged_embeddings(session: AsyncSession, model: str) -> int:
    """One UPDATE: staged vectors of `model` become the live vectors. Chunks that have no staged
    vector for `model` (added a moment ago) keep their old vector but stay tagged with their own
    model, so they are simply not retrieved until embedded (never compared across spaces)."""
    res = await session.execute(
        update(KbChunk).where(KbChunk.embedding_next_model == model).values(
            embedding=KbChunk.embedding_next, embedding_model=KbChunk.embedding_next_model,
            embedding_next=None, embedding_next_model=None))
    return int(res.rowcount or 0)  # type: ignore[attr-defined]


async def refresh_tenant_knowledge(
    session: AsyncSession, tenant_id: uuid.UUID, languages: list[str],
    embedder: Embedder | None, cache: SemanticCache | None,
) -> int:
    """After any KB change: embed new chunks and drop cached answers (they may be stale)."""
    embedded = await embed_pending(session, embedder, tenant_id) if embedder else 0
    if cache:
        await cache.invalidate(tenant_id, languages=languages)
    return embedded


async def faq_pairs(
    session: AsyncSession, tenant_id: uuid.UUID, language: str
) -> list[tuple[str, str]]:
    """(question, approved answer) for lexical FAQ matching (needs no embeddings)."""
    rows = (await session.execute(
        select(KbChunk.question, KbChunk.content).where(
            KbChunk.tenant_id == tenant_id, KbChunk.kind == KbKind.FAQ,
            KbChunk.language == language, KbChunk.approved.is_(True),
            KbChunk.question.is_not(None),
        )
    )).all()
    return [(q, a) for q, a in rows if q]

"""Import a tenant YAML file into the DB (idempotent upsert by slug)."""
from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import KbChunk, KbKind, Tenant
from assistant.rag.kb import replace_yaml_documents
from assistant.tenants.schema import TenantFile


def parse_tenant_yaml(path: Path) -> TenantFile:
    """Read and validate a tenant YAML file (raises pydantic.ValidationError if invalid)."""
    return TenantFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


async def import_tenant(session: AsyncSession, data: TenantFile) -> Tenant:
    """Upsert the tenant and replace its YAML-managed FAQ rows and documents.

    Chunks are inserted WITHOUT embeddings. Run rag.kb.refresh_tenant_knowledge() (the CLI does)
    to embed them and clear the tenant's semantic cache. Until then FAQ answers still work via
    exact/lexical matching; only semantic matching and RAG need the vectors.
    """
    tenant = (
        await session.execute(select(Tenant).where(Tenant.slug == data.slug))
    ).scalar_one_or_none()
    cfg = data.config.model_dump(mode="json")
    if tenant is None:
        tenant = Tenant(slug=data.slug, name=data.name)
        session.add(tenant)
    tenant.name = data.name
    tenant.industry = data.industry
    tenant.whatsapp_phone_number_id = data.whatsapp_phone_number_id
    tenant.config = cfg
    await session.flush()

    await session.execute(
        delete(KbChunk).where(
            KbChunk.tenant_id == tenant.id, KbChunk.kind == KbKind.FAQ, KbChunk.source == "yaml"
        )
    )
    for faq in data.faqs:
        session.add(
            KbChunk(
                tenant_id=tenant.id,
                kind=KbKind.FAQ,
                language=faq.language,
                source="yaml",
                question=faq.question,
                content=faq.answer,
                token_count=len(faq.answer.split()),
            )
        )
    await session.flush()
    await replace_yaml_documents(session, tenant.id, data.documents)
    return tenant

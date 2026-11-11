"""Route an inbound channel account (WhatsApp phone_number_id) to a tenant."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import Tenant


async def resolve_tenant_id(
    factory: async_sessionmaker[AsyncSession], phone_number_id: str
) -> uuid.UUID | None:
    """Return the active tenant owning this number, or None (unknown/disabled -> drop)."""
    async with factory() as s:
        return (
            await s.execute(
                select(Tenant.id).where(
                    Tenant.whatsapp_phone_number_id == phone_number_id, Tenant.is_active.is_(True)
                )
            )
        ).scalar_one_or_none()

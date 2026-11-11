"""Async SQLAlchemy engine/session factory. Models arrive in Step 2."""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from assistant.core.config import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    """Create the pooled async engine (pre_ping drops dead connections)."""
    return create_async_engine(settings.database_url, pool_pre_ping=True, pool_size=10)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Yield a session; commit on success, rollback on error."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping_db(engine: AsyncEngine) -> bool:
    """Return True if a trivial query succeeds (raises otherwise)."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True

"""Shared fixtures. DB tests run against a real Postgres+pgvector (TEST_DATABASE_URL) or skip."""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from assistant.core.models import Base

TEST_DB = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://assistant:assistant@localhost:5432/assistant_test"
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Fresh schema per test; skips cleanly when no Postgres is reachable."""
    engine = create_async_engine(TEST_DB)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # noqa: BLE001 - any connect failure means "no DB"
        await engine.dispose()
        pytest.skip(f"Postgres not available: {exc!r}")
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
    await engine.dispose()

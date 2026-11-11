"""Thin lazy wrapper over the Arq pool so the API can start without Redis being reachable and
tests can substitute a fake."""
from __future__ import annotations

from typing import Any, Protocol

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings


class JobQueue(Protocol):
    async def enqueue(self, name: str, *args: Any, job_id: str | None = None) -> None: ...
    async def close(self) -> None: ...


class ArqJobQueue:
    """Creates the pool on first use."""

    def __init__(self, redis_dsn: str) -> None:
        self._settings = RedisSettings.from_dsn(redis_dsn)
        self._pool: ArqRedis | None = None

    async def enqueue(self, name: str, *args: Any, job_id: str | None = None) -> None:
        if self._pool is None:
            self._pool = await create_pool(self._settings)
        await self._pool.enqueue_job(name, *args, _job_id=job_id)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()

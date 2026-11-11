"""Redis primitives: per-conversation lock, fixed-window rate limiter, dedupe keys.

Every key embeds tenant_id (strict isolation).
"""
from __future__ import annotations

import asyncio
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis

# Compare-and-delete so a lock that expired and was re-taken by someone else is never released
# by the previous holder.
_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end
"""


class LockNotAcquiredError(Exception):
    """Another worker is processing this conversation."""


def conversation_lock_key(tenant_id: uuid.UUID, phone: str) -> str:
    return f"lock:conv:{tenant_id}:{phone}"


@asynccontextmanager
async def redis_lock(
    redis: Redis, key: str, *, ttl_ms: int = 30_000, wait_s: float = 5.0, poll_s: float = 0.05
) -> AsyncIterator[None]:
    """Mutual exclusion across processes. TTL bounds the damage if a holder crashes."""
    token = secrets.token_hex(8)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    while not await redis.set(key, token, nx=True, px=ttl_ms):
        if loop.time() >= deadline:
            raise LockNotAcquiredError(key)
        await asyncio.sleep(poll_s)
    try:
        yield
    finally:
        await redis.eval(_RELEASE, 1, key, token)  # type: ignore[misc]


async def allow(redis: Redis, key: str, limit: int, window_s: int = 60) -> bool:
    """Fixed-window rate limit: True while the count in the current window is <= limit."""
    async with redis.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, window_s, nx=True)
        count, _ = await pipe.execute()
    return int(count) <= limit


def dedupe_key(tenant_id: uuid.UUID, wa_message_id: str) -> str:
    return f"dedupe:{tenant_id}:{wa_message_id}"


async def first_time(redis: Redis, key: str, ttl_s: int = 86_400) -> bool:
    """True exactly once per key within the TTL (SET NX)."""
    return bool(await redis.set(key, "1", nx=True, ex=ttl_s))

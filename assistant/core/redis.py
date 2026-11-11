"""Redis client factory (sessions, semantic cache, rate limits, locks)."""
from __future__ import annotations

from redis.asyncio import Redis

from assistant.core.config import Settings


def make_redis(settings: Settings) -> Redis:
    client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return client

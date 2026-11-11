import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from assistant.core.locks import LockNotAcquiredError, allow, first_time, redis_lock


async def test_lock_is_exclusive_and_released() -> None:
    r = FakeAsyncRedis()
    async with redis_lock(r, "k"):
        with pytest.raises(LockNotAcquiredError):
            async with redis_lock(r, "k", wait_s=0.1):
                pass
    async with redis_lock(r, "k", wait_s=0.1):  # free again after release
        pass


async def test_lock_serializes_concurrent_work() -> None:
    r = FakeAsyncRedis()
    order: list[str] = []

    async def worker(name: str) -> None:
        async with redis_lock(r, "k", wait_s=2):
            order.append(f"{name}-in")
            await asyncio.sleep(0.05)
            order.append(f"{name}-out")

    await asyncio.gather(worker("a"), worker("b"))
    assert order[0][0] == order[1][0] and order[2][0] == order[3][0]  # never interleaved


async def test_stale_holder_cannot_release_new_lock() -> None:
    r = FakeAsyncRedis()
    async with redis_lock(r, "k", ttl_ms=50):
        await asyncio.sleep(0.1)  # TTL expires
        await r.set("k", "someone-else")
    assert await r.get("k") == b"someone-else"


async def test_rate_limit_and_first_time() -> None:
    r = FakeAsyncRedis()
    assert [await allow(r, "rl", 2) for _ in range(3)] == [True, True, False]
    assert await first_time(r, "d") and not await first_time(r, "d")

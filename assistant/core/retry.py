"""Async retry with exponential backoff + full jitter."""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

log = logging.getLogger("assistant.retry")


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn` up to `attempts` times, sleeping random(0, min(max, base*2^n)) between tries.

    Only exceptions in `retry_on` are retried; anything else propagates immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            log.warning("retrying", extra={"attempt": attempt, "delay_s": round(delay, 2),
                                           "error": type(exc).__name__})
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover

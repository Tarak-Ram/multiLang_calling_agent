"""Circuit breaker + retry + fallback model around any LLMClient."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal

from assistant.llm.base import LLMClient, LLMError, LLMRequest, LLMResponse, LLMUnavailableError

log = logging.getLogger("assistant.llm.resilience")

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Opens after `threshold` consecutive transient failures; after `recovery_s` lets ONE
    probe through (half-open). Success closes it; failure re-opens it."""

    def __init__(
        self, threshold: int = 5, recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold, self.recovery_s, self._clock = threshold, recovery_s, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self._opened_at >= self.recovery_s else "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def success(self) -> None:
        self._failures, self._opened_at, self._probing = 0, None, False

    def failure(self) -> None:
        self._failures += 1
        self._probing = False
        if self._failures >= self.threshold or self._opened_at is not None:
            self._opened_at = self._clock()


class ResilientLLM:
    """Per-model breakers. Order: requested model (with retries) -> fallback model.

    Permanent errors (bad request, auth) are raised without retry and don't trip the breaker.
    """

    def __init__(
        self, inner: LLMClient, fallback_model: str | None = None, *, attempts: int = 2,
        threshold: int = 5, recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner, self._fallback = inner, fallback_model or None
        self._attempts = attempts
        self._breaker_args = (threshold, recovery_s, clock)
        self.breakers: dict[str, CircuitBreaker] = {}

    def _breaker(self, model: str) -> CircuitBreaker:
        return self.breakers.setdefault(model, CircuitBreaker(*self._breaker_args))

    async def _try(self, req: LLMRequest) -> LLMResponse:
        breaker = self._breaker(req.model)
        if not breaker.allow():
            raise LLMError("circuit_open", transient=True)
        for attempt in range(1, self._attempts + 1):
            try:
                resp = await self._inner.complete(req)
            except LLMError as exc:
                if not exc.transient:
                    raise
                breaker.failure()
                if attempt < self._attempts and breaker.state == "closed":
                    continue  # one quick retry (no sleep: user is waiting on a chat reply)
                raise
            breaker.success()
            return resp
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(self, req: LLMRequest) -> LLMResponse:
        try:
            return await self._try(req)
        except LLMError as primary_exc:
            fallback = req.fallback_model or self._fallback
            if not fallback or fallback == req.model:
                raise LLMUnavailableError(str(primary_exc)) from primary_exc
            log.warning("llm_fallback", extra={"from": req.model, "to": fallback,
                                               "error": str(primary_exc)})
            try:
                return await self._try(req.model_copy(update={"model": fallback,
                                                              "fallback_model": None}))
            except LLMError as exc:
                raise LLMUnavailableError(str(exc)) from exc

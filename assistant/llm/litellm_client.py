"""LiteLLM adapter: one code path for any provider/model.

Cost rules enforced here: reasoning/thinking is never requested, output is capped, JSON mode is
on, temperature 0. Provider API keys are read by LiteLLM from the environment.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import litellm
from litellm import exceptions

from assistant.llm.base import LLMError, LLMRequest, LLMResponse

log = logging.getLogger("assistant.llm")

litellm.drop_params = True  # silently drop params a given provider doesn't support
litellm.suppress_debug_info = True

_TRANSIENT = (
    exceptions.RateLimitError,
    exceptions.Timeout,
    exceptions.APIConnectionError,
    exceptions.ServiceUnavailableError,
    exceptions.InternalServerError,
)


def _supports_explicit_cache_control(model: str) -> bool:
    """Anthropic-family models need an explicit cache marker on the stable prefix; other
    providers cache prefixes automatically and may reject the extra field."""
    m = model.lower()
    return "claude" in m or m.startswith("anthropic/")


class LiteLLMClient:
    def __init__(
        self,
        timeout_s: float = 15.0,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._timeout = timeout_s
        self._completion = completion_fn or litellm.acompletion  # injectable for tests

    async def complete(self, req: LLMRequest) -> LLMResponse:
        system: Any = req.system
        if _supports_explicit_cache_control(req.model):
            system = [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": req.user}],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "timeout": self._timeout,
            "num_retries": 0,  # retries/fallback are ours (ResilientLLM), so they are visible
        }
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = await self._completion(**kwargs)
        except _TRANSIENT as exc:
            raise LLMError(f"{type(exc).__name__}", transient=True) from exc
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {str(exc)[:200]}", transient=False) from exc
        latency = int((time.perf_counter() - start) * 1000)

        usage = getattr(resp, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or getattr(usage, "cache_read_input_tokens", 0)
                  or 0)
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001 - unpriced model: cost 0, tokens still recorded
            log.warning("cost_unknown", extra={"model": req.model})
            cost = 0.0
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            model=req.model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            cached_input_tokens=int(cached),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency,
            cost_usd=cost,
        )

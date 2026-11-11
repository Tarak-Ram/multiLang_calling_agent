"""Provider-agnostic LLM interface. Nothing outside llm/ imports a provider SDK."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """A single short completion. Stable text goes in `system` (provider prompt caching works
    on prefixes); per-turn data goes in `user`."""

    model: str  # LiteLLM model id, e.g. "provider/model-name" (chosen in config)
    system: str
    user: str
    max_tokens: int = 150  # replies are 1-3 short sentences; JSON is compact
    json_mode: bool = True
    temperature: float = 0.0
    fallback_model: str | None = None  # per-request fallback (from the active profile)


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


class LLMError(Exception):
    """Provider failure. `transient` = worth retrying / counts toward the circuit breaker."""

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class LLMUnavailableError(Exception):
    """Primary and fallback models both failed or are circuit-broken."""


class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResponse: ...

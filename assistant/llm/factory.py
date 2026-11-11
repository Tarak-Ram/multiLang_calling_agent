"""Build the LLM stack from settings. Which MODELS are used is decided per request by the
active model profile (llm/profiles.py), so this only builds provider-agnostic plumbing."""
from __future__ import annotations

from assistant.core.config import Settings
from assistant.llm.litellm_client import LiteLLMClient
from assistant.llm.profiles import ModelResolver
from assistant.llm.resilience import ResilientLLM
from assistant.router.extractor import LLMExtractor


def build_llm(s: Settings) -> ResilientLLM:
    """Shared resilient client (circuit breaker per model). The fallback model comes from the
    active profile on each request; LLM_FALLBACK_MODEL is only the last-resort default."""
    return ResilientLLM(LiteLLMClient(timeout_s=s.llm_timeout_s),
                        fallback_model=s.llm_fallback_model)


def build_extractor(s: Settings, llm: ResilientLLM, resolver: ModelResolver) -> LLMExtractor:
    """With a profile whose `fast` model is empty (e.g. `rules-only`) it simply never calls out."""
    return LLMExtractor(llm, max_tokens=s.llm_max_output_tokens, resolver=resolver)

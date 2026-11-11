"""Application settings, loaded only from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed process-level config. Per-tenant config lives in the DB (Step 2)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "postgresql+asyncpg://assistant:assistant@localhost:5432/assistant"
    redis_url: str = "redis://localhost:6379/0"

    whatsapp_verify_token: SecretStr = SecretStr("")
    whatsapp_app_secret: SecretStr = SecretStr("")
    whatsapp_access_token: SecretStr = SecretStr("")
    whatsapp_api_version: str = "v21.0"

    llm_fast_model: str = ""
    llm_strong_model: str = ""
    llm_fallback_model: str = ""
    llm_max_output_tokens: int = 150
    llm_timeout_s: float = 15.0
    # LiteLLM embedding model id (empty = semantic cache / semantic FAQ matching disabled).
    embedding_model: str = ""
    embedding_dim: int = 1024

    # Admin API (handoff inbox, cost reports). Empty = admin endpoints disabled (503).
    admin_api_key: SecretStr = SecretStr("")
    # Optional Google Calendar mirror: path to a service-account JSON (see tools/calendar.py).
    google_service_account_file: str = ""

    # Optional Langfuse export of turn traces (empty = log lines only, nothing leaves the box).
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

    rate_limit_per_minute: int = 60
    shutdown_grace_seconds: int = 20

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (override in tests)."""
    return Settings()

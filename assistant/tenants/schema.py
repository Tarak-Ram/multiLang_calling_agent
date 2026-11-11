"""Typed, validated shape of Tenant.config (stored as JSONB, imported from YAML).

New industry or language = new YAML, no code. Flow definitions themselves arrive in Step 4.
"""
from __future__ import annotations

from datetime import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class DayHours(BaseModel):
    open: time
    close: time


class LanguageConfig(BaseModel):
    code: str = Field(pattern=r"^[a-z]{2,3}$")  # ISO 639-1/2: te, hi, en
    name: str
    # Phrases for the zero-LLM rules layer (greetings, yes/no). Config, not code.
    greetings: list[str] = []
    yes: list[str] = []
    no: list[str] = []
    # intent -> trigger phrases (deterministic, zero-LLM). Reserved intents: human, menu,
    # medical_advice; any other key names a flow (e.g. book_appointment).
    keywords: dict[str, list[str]] = {}


class EscalationConfig(BaseModel):
    phone: str | None = None  # staff WhatsApp number to notify
    email: str | None = None
    message: dict[str, str] = {}  # per-language "connecting you to our staff"


class ModelConfig(BaseModel):
    """LiteLLM model ids; empty = fall back to platform defaults from env."""

    fast: str | None = None
    strong: str | None = None
    profile: str | None = None  # use another model profile's LLM models (embedding stays global)


class ThresholdConfig(BaseModel):
    intent_confidence: float = Field(0.75, ge=0, le=1)
    cache_similarity: float = Field(0.92, ge=0, le=1)
    rag_min_score: float = Field(0.55, ge=0, le=1)
    faq_similarity: float = Field(0.90, ge=0, le=1)  # approved FAQ answered verbatim
    answer_confidence: float = Field(0.70, ge=0, le=1)  # RAG answer self-confidence


class TimingConfig(BaseModel):
    reminder_after_hours: int = Field(4, ge=1)
    expire_after_hours: int = Field(24, ge=1)


class ServiceDef(BaseModel):
    """A bookable service (clinic: department/doctor). `resource` is what can be double-booked."""

    id: str
    names: dict[str, str]  # {lang: display name}
    resource: str
    duration_minutes: int = Field(30, ge=5, le=480)

    def name(self, lang: str, default: str = "en") -> str:
        return self.names.get(lang) or self.names.get(default) or next(iter(self.names.values()))


class BookingConfig(BaseModel):
    slot_minutes: int = Field(30, ge=5, le=240)  # start-time grid
    horizon_days: int = Field(7, ge=1, le=60)  # how far ahead customers may book
    min_lead_minutes: int = Field(60, ge=0)  # no bookings starting sooner than this
    services: list[ServiceDef] = []
    calendar_id: str | None = None  # Google Calendar to mirror confirmed bookings into


class IntegrationConfig(BaseModel):
    """Tenant-owned HTTPS endpoint (order lookup, CRM, report status, returns)."""

    url: str
    secret_env: str | None = None  # NAME of the env var holding the bearer token (never the token)
    timeout_s: float = Field(5.0, gt=0, le=30)


class TenantConfig(BaseModel):
    """Everything per-tenant that is not a column on `tenants`."""

    timezone: str = "Asia/Kolkata"
    default_language: str = "en"
    languages: list[LanguageConfig]
    business_hours: dict[Weekday, DayHours] = {}
    flows_enabled: list[str] = []
    tools_enabled: list[str] = []
    escalation: EscalationConfig = EscalationConfig()
    models: ModelConfig = ModelConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    timing: TimingConfig = TimingConfig()
    # WhatsApp template names by purpose; must be pre-approved in Meta (24h-window rule).
    templates: dict[str, str] = {}
    # Per-tenant overrides of the industry's reply texts: {message_key: {lang: text}}.
    messages: dict[str, dict[str, str]] = {}
    booking: BookingConfig = BookingConfig()
    integrations: dict[str, IntegrationConfig] = {}

    @field_validator("languages")
    @classmethod
    def _unique_languages(cls, v: list[LanguageConfig]) -> list[LanguageConfig]:
        codes = [x.code for x in v]
        if not codes or len(set(codes)) != len(codes):
            raise ValueError("languages must be non-empty with unique codes")
        return v

    def model_post_init(self, _ctx: object, /) -> None:
        if self.default_language not in {x.code for x in self.languages}:
            raise ValueError("default_language must be one of languages")


class FaqEntry(BaseModel):
    """Pre-approved answer. Preferred over generation; embedded at import time (Step 5/6)."""

    question: str
    answer: str
    language: str


class DocEntry(BaseModel):
    """Free-text knowledge (policies, prices, doctor bios...). Chunked + embedded on import."""

    title: str
    language: str
    text: str


class TenantFile(BaseModel):
    """Top-level YAML document."""

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    name: str
    industry: str = "clinic"
    whatsapp_phone_number_id: str | None = None
    config: TenantConfig
    faqs: list[FaqEntry] = []
    documents: list[DocEntry] = []

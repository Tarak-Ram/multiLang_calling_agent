"""Value types passed through the orchestrator (pure data, no IO)."""
from __future__ import annotations

import uuid
from typing import Any, Protocol

from pydantic import BaseModel

from assistant.channels.base import Option
from assistant.core.models import ConversationStatus
from assistant.tenants.schema import TenantConfig


class Reply(BaseModel):
    text: str
    options: list[Option] = []
    list_button: str = ""  # label when options render as a list (>3 options)


class ConvState(BaseModel):
    """Everything persisted between webhooks. Mirrors the `conversations` row."""

    tenant_id: uuid.UUID
    conversation_id: uuid.UUID
    customer_phone: str
    customer_name: str | None = None
    language: str | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    flow: str | None = None
    state: str = "greet"  # greet | identify_intent | collect_slots | confirm | close | handoff
    slots: dict[str, Any] = {}  # user-facing slot values; private keys start with "_"
    summary: str = ""
    # Last 2-4 turns as (role, text), role in {"customer","bot"}. Loaded per message, never
    # persisted on the conversation row; this is all the history an LLM call ever sees.
    recent: list[tuple[str, str]] = []


class TurnInput(BaseModel):
    text: str | None = None
    payload_id: str | None = None
    msg_type: str = "text"


class TurnResult(BaseModel):
    replies: list[Reply]
    state: ConvState
    route: str  # rule | menu | extractor | human_owned | ...  (cost reporting in Step 10)
    needs_human: bool = False
    handoff_reason: str | None = None
    tool_calls: list[dict[str, Any]] = []


class Extraction(BaseModel):
    """Output contract of the LLM extractor (Step 5)."""

    intent: str | None = None
    slots: dict[str, str] = {}
    confidence: float = 0.0
    language: str | None = None
    needs_human: bool = False


class ExtractHints(BaseModel):
    """What the engine already knows, so the LLM only fills the gap (and costs fewer tokens)."""

    intents: list[str]  # allowed intent labels for this tenant
    today: str  # "2026-11-12 Thu" in the tenant's timezone (resolves "tomorrow")
    slot: str | None = None  # slot currently being asked
    options: list[Option] = []  # valid choices for that slot; LLM must return an id or null
    slot_names: list[str] = []  # slots still missing in the active flow


class Extractor(Protocol):
    """Fills gaps the deterministic rules can't. Never controls the flow."""

    async def extract(
        self, cfg: TenantConfig, state: ConvState, text: str, hints: ExtractHints
    ) -> Extraction | None: ...


class Answer(BaseModel):
    text: str
    source: str  # faq | cache | rag


class Answerer(Protocol):
    """Answers customer questions ONLY from tenant knowledge. None = 'we don't know'."""

    async def lookup(
        self, cfg: TenantConfig, state: ConvState, text: str, lang: str, *,
        exact_only: bool = False,
    ) -> Answer | None:
        """Cheap path, no generation: approved FAQ (exact/semantic) then semantic cache.
        `exact_only` restricts it to the high-precision exact-FAQ tier (used to let an exact
        FAQ question beat keyword intents, e.g. 'What is your return policy?')."""
        ...

    async def generate(
        self, cfg: TenantConfig, state: ConvState, text: str, lang: str
    ) -> Answer | None:
        """RAG with a small LLM over retrieved chunks, grounding-checked. Costs tokens."""
        ...

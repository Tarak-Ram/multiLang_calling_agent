"""Channel-neutral message model and adapter interface.

Every channel (WhatsApp now; voice, web chat later) produces `InboundMessage` and consumes
`OutboundMessage`, so the orchestrator never knows which channel it is talking to.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel

MsgType = Literal["text", "interactive", "audio", "image", "document", "location", "unsupported"]


class InboundMessage(BaseModel):
    """Normalized customer message."""

    channel: str = "whatsapp"
    channel_account_id: str  # WhatsApp phone_number_id -> resolves the tenant
    message_id: str  # channel-unique id (wamid) used for dedupe
    from_phone: str  # E.164 with leading '+'
    timestamp: datetime
    msg_type: MsgType = "text"
    text: str | None = None  # text body, or the title of a tapped button/list row
    payload_id: str | None = None  # button/list id for deterministic menu routing
    media_id: str | None = None  # voice notes / images; transcribed by channels/voice/voicenote.py once an STT is configured
    contact_name: str | None = None


class Option(BaseModel):
    """A tappable choice (WhatsApp button / list row). `id` comes back as payload_id."""

    id: str
    title: str


class OutboundMessage(BaseModel):
    """Reply to send. Exactly one of text / template should be set."""

    to_phone: str
    text: str | None = None
    template_name: str | None = None
    template_language: str | None = None
    template_params: list[str] = []


class ChannelAdapter(Protocol):
    """Outbound side of a channel. Inbound parsing is channel-specific and lives in the adapter
    package (e.g. channels/whatsapp/parser.py)."""

    async def send_text(self, account_id: str, to_phone: str, body: str) -> str: ...

    async def send_options(
        self, account_id: str, to_phone: str, body: str, options: list[Option], list_button: str
    ) -> str: ...

    async def send_template(
        self, account_id: str, to_phone: str, name: str, language: str, params: list[str]
    ) -> str: ...

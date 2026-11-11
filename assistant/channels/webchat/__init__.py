"""Web-chat channel (stub): same InboundMessage in, same ChannelAdapter out.

TODO(phase 2): a WebSocket/SSE endpoint that authenticates the embedded widget (signed session
token carrying tenant + a verified visitor id), calls `to_inbound`, and pushes replies through
`WebChatAdapter.send_*` to the visitor's open socket. Visitors are keyed by a verified id in the
`phone` field's slot (prefixed `web:`) so conversations stay in the one (tenant, customer) model.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from assistant.channels.base import InboundMessage, Option


class WebChatMessage(BaseModel):
    tenant_id: str
    visitor_id: str  # already verified by the session layer
    message_id: str
    text: str | None = None
    option_id: str | None = None  # a tapped quick-reply


def to_inbound(m: WebChatMessage) -> InboundMessage:
    return InboundMessage(
        channel="webchat", channel_account_id=m.tenant_id, message_id=m.message_id,
        from_phone=f"web:{m.visitor_id}", timestamp=datetime.now(UTC),
        msg_type="interactive" if m.option_id else "text", text=m.text, payload_id=m.option_id,
    )


class WebChatAdapter:
    """Outbound side: pushes to whatever `push` the socket layer provides."""

    def __init__(self, push: Callable[[str, dict[str, object]], Awaitable[None]]) -> None:
        self._push = push

    async def send_text(self, account_id: str, to_phone: str, body: str) -> str:
        await self._push(to_phone, {"type": "text", "text": body})
        return f"web:{datetime.now(UTC).timestamp()}"

    async def send_options(self, account_id: str, to_phone: str, body: str,
                           options: list[Option], list_button: str) -> str:
        await self._push(to_phone, {"type": "options", "text": body,
                                    "options": [o.model_dump() for o in options]})
        return f"web:{datetime.now(UTC).timestamp()}"

    async def send_template(self, account_id: str, to_phone: str, name: str, language: str,
                            params: list[str]) -> str:
        raise NotImplementedError("templates are a WhatsApp concept; web chat has no 24h rule")

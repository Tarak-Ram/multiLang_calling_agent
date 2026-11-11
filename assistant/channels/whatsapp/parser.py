"""Parse a WhatsApp Cloud API webhook payload into normalized inbound messages."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from assistant.channels.base import InboundMessage, MsgType

log = logging.getLogger("assistant.whatsapp.parser")

_MEDIA_TYPES: dict[str, MsgType] = {"audio": "audio", "image": "image", "document": "document"}


def _parse_one(
    msg: dict[str, Any], phone_number_id: str, names: dict[str, str]
) -> InboundMessage | None:
    mtype = msg.get("type")
    text: str | None = None
    payload_id: str | None = None
    media_id: str | None = None
    norm: MsgType

    if mtype == "text":
        norm, text = "text", msg.get("text", {}).get("body")
    elif mtype == "interactive":
        inter = msg.get("interactive", {})
        reply = inter.get("button_reply") or inter.get("list_reply") or {}
        norm, text, payload_id = "interactive", reply.get("title"), reply.get("id")
    elif mtype == "button":  # quick-reply on a template message
        btn = msg.get("button", {})
        norm, text, payload_id = "interactive", btn.get("text"), btn.get("payload")
    elif mtype in _MEDIA_TYPES:
        norm = _MEDIA_TYPES[mtype]
        body = msg.get(mtype, {})
        media_id, text = body.get("id"), body.get("caption")
    elif mtype == "location":
        norm = "location"
    else:
        norm = "unsupported"

    wa_id, mid = msg.get("from"), msg.get("id")
    if not wa_id or not mid:
        log.warning("message_missing_fields")
        return None
    try:
        ts = datetime.fromtimestamp(int(msg["timestamp"]), UTC)
    except (KeyError, ValueError):
        ts = datetime.now(UTC)
    return InboundMessage(
        channel_account_id=phone_number_id,
        message_id=mid,
        from_phone=f"+{wa_id.lstrip('+')}",
        timestamp=ts,
        msg_type=norm,
        text=text,
        payload_id=payload_id,
        media_id=media_id,
        contact_name=names.get(wa_id),
    )


def parse_webhook(payload: dict[str, Any]) -> list[InboundMessage]:
    """Extract customer messages. Delivery/read `statuses` are ignored here.

    TODO(future): use `statuses` (failed/delivered) to detect undeliverable template sends.
    """
    out: list[InboundMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            value = change.get("value", {})
            pnid = value.get("metadata", {}).get("phone_number_id")
            if not pnid:
                continue
            names = {
                c.get("wa_id", ""): c.get("profile", {}).get("name", "")
                for c in value.get("contacts", [])
            }
            for raw in value.get("messages", []):
                parsed = _parse_one(raw, pnid, names)
                if parsed:
                    out.append(parsed)
    return out

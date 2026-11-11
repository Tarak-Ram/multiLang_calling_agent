"""Realistic WhatsApp Cloud API webhook payload builders for tests."""
from __future__ import annotations

from typing import Any

PNID = "111222333"


def envelope(messages: list[dict[str, Any]], pnid: str = PNID, name: str = "Ravi") -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "WABA", "changes": [{"field": "messages", "value": {
            "messaging_product": "whatsapp",
            "metadata": {"display_phone_number": "15550001111", "phone_number_id": pnid},
            "contacts": [{"profile": {"name": name}, "wa_id": "919876543210"}],
            "messages": messages,
        }}]}],
    }


def text_msg(mid: str = "wamid.A", body: str = "నమస్కారం", ts: str = "1700000000") -> dict[str, Any]:
    return {"from": "919876543210", "id": mid, "timestamp": ts, "type": "text",
            "text": {"body": body}}


STATUS_ONLY = {"object": "whatsapp_business_account", "entry": [{"changes": [{"field": "messages",
    "value": {"metadata": {"phone_number_id": PNID}, "statuses": [{"id": "x", "status": "read"}]}}]}]}

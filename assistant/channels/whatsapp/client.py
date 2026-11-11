"""WhatsApp Cloud API send client with retry/backoff.

This layer is deliberately dumb: it does NOT enforce the 24-hour window. The orchestrator
(Step 8) decides free-form vs template; calling send_text outside the window is rejected by
Meta (error 131047) and surfaces here as WhatsAppError.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from assistant.channels.base import Option
from assistant.core.retry import retry_async

log = logging.getLogger("assistant.whatsapp.client")


class WhatsAppError(Exception):
    """Non-retryable API error (4xx)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"WhatsApp API {status}: {body[:300]}")
        self.status = status
        self.body = body


class _RetryableError(Exception):
    """429 / 5xx from the API."""


class WhatsAppClient:
    """Implements ChannelAdapter for WhatsApp.

    Retry caveat: Cloud API has no idempotency key, so a retry after a read-timeout can in rare
    cases deliver a duplicate. We accept that over silently dropping replies.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        access_token: str,
        api_version: str = "v21.0",
        base_url: str = "https://graph.facebook.com",
        attempts: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self._http = http
        self._token = access_token  # TODO(future): per-tenant tokens, encrypted at rest.
        self._base = f"{base_url}/{api_version}"
        self._attempts = attempts
        self._base_delay = base_delay

    async def _post(self, phone_number_id: str, body: dict[str, Any]) -> str:
        url = f"{self._base}/{phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self._token}"}

        async def attempt() -> str:
            resp = await self._http.post(url, json=body, headers=headers, timeout=10.0)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _RetryableError(str(resp.status_code))
            if resp.status_code >= 400:
                raise WhatsAppError(resp.status_code, resp.text)
            return str(resp.json()["messages"][0]["id"])

        return await retry_async(
            attempt,
            retry_on=(_RetryableError, httpx.TransportError),
            attempts=self._attempts,
            base_delay=self._base_delay,
        )

    async def send_text(self, account_id: str, to_phone: str, body: str) -> str:
        """Free-form text (only valid inside the 24h window). Returns the wamid."""
        return await self._post(
            account_id,
            {
                "messaging_product": "whatsapp",
                "to": to_phone.lstrip("+"),
                "type": "text",
                "text": {"body": body, "preview_url": False},
            },
        )

    async def send_options(
        self, account_id: str, to_phone: str, body: str, options: list[Option], list_button: str
    ) -> str:
        """Interactive reply buttons (<=3 options) or a list (<=10). Titles are truncated to
        WhatsApp's limits (button 20, list row 24 chars); ids are what we get back."""
        if not options:
            return await self.send_text(account_id, to_phone, body)
        if len(options) <= 3:
            action: dict[str, Any] = {
                "buttons": [
                    {"type": "reply", "reply": {"id": o.id, "title": o.title[:20]}}
                    for o in options
                ]
            }
            interactive: dict[str, Any] = {"type": "button", "body": {"text": body[:1024]},
                                           "action": action}
        else:
            rows = [{"id": o.id, "title": o.title[:24]} for o in options[:10]]
            interactive = {
                "type": "list", "body": {"text": body[:1024]},
                "action": {"button": list_button[:20] or "Choose",
                           "sections": [{"title": (list_button[:24] or "Choose"), "rows": rows}]},
            }
        return await self._post(
            account_id,
            {"messaging_product": "whatsapp", "to": to_phone.lstrip("+"),
             "type": "interactive", "interactive": interactive},
        )

    async def download_media(self, media_id: str, max_bytes: int = 5_000_000) -> bytes:
        """Fetch a media file (voice note, image) the customer sent. Two hops: media id -> a
        short-lived https URL -> bytes. The URL comes from Meta, but we still require https and
        a size cap before following it."""
        headers = {"Authorization": f"Bearer {self._token}"}

        async def meta() -> dict[str, Any]:
            r = await self._http.get(f"{self._base}/{media_id}", headers=headers, timeout=10.0)
            if r.status_code == 429 or r.status_code >= 500:
                raise _RetryableError(str(r.status_code))
            if r.status_code >= 400:
                raise WhatsAppError(r.status_code, r.text)
            return dict(r.json())

        info = await retry_async(meta, retry_on=(_RetryableError, httpx.TransportError),
                                 attempts=self._attempts, base_delay=self._base_delay)
        url = str(info.get("url", ""))
        if not url.startswith("https://"):
            raise WhatsAppError(502, "media URL is not https")
        r = await self._http.get(url, headers=headers, timeout=20.0)
        if r.status_code >= 400:
            raise WhatsAppError(r.status_code, "media download failed")
        if len(r.content) > max_bytes:
            raise WhatsAppError(413, "media too large")
        return r.content

    async def send_template(
        self, account_id: str, to_phone: str, name: str, language: str, params: list[str]
    ) -> str:
        """Pre-approved template (allowed outside the window). Returns the wamid."""
        template: dict[str, Any] = {"name": name, "language": {"code": language}}
        if params:
            template["components"] = [
                {"type": "body", "parameters": [{"type": "text", "text": p} for p in params]}
            ]
        return await self._post(
            account_id,
            {
                "messaging_product": "whatsapp",
                "to": to_phone.lstrip("+"),
                "type": "template",
                "template": template,
            },
        )

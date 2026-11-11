"""WhatsApp webhook: verification handshake + signed message delivery.

Design: verify -> parse -> (tenant, rate limit, dedupe) -> enqueue -> 200 fast.
All real work happens in the Arq job so Meta's short timeout is never at risk and nothing waits
in memory.
"""
from __future__ import annotations

import hmac
import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from assistant.channels.whatsapp.parser import parse_webhook
from assistant.channels.whatsapp.signature import verify_signature
from assistant.core.locks import allow, dedupe_key, first_time
from assistant.core.logging import bind_context
from assistant.core.pii import mask_phone

router = APIRouter(prefix="/webhooks/whatsapp", tags=["whatsapp"])
log = logging.getLogger("assistant.whatsapp.webhook")


@router.get("")
async def verify(
    request: Request,
    mode: str | None = Query(None, alias="hub.mode"),
    token: str | None = Query(None, alias="hub.verify_token"),
    challenge: str | None = Query(None, alias="hub.challenge"),
) -> PlainTextResponse:
    """Meta's one-time subscription handshake."""
    expected = request.app.state.settings.whatsapp_verify_token.get_secret_value()
    if mode == "subscribe" and expected and token and hmac.compare_digest(token, expected):
        return PlainTextResponse(challenge or "")
    raise HTTPException(403, "verification failed")


@router.post("")
async def receive(request: Request) -> dict[str, object]:
    """Receive message events. Returns 2xx for anything validly signed so Meta doesn't retry
    messages we deliberately drop (unknown tenant, rate-limited, duplicate)."""
    st = request.app.state
    raw = await request.body()
    secret = st.settings.whatsapp_app_secret.get_secret_value()
    if not verify_signature(secret, raw, request.headers.get("x-hub-signature-256")):
        log.warning("bad_signature")
        raise HTTPException(401, "invalid signature")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "invalid json") from None

    accepted = 0
    for msg in parse_webhook(payload):
        tenant_id = await st.tenant_resolver(msg.channel_account_id)
        if tenant_id is None:
            log.warning("unknown_tenant", extra={"account": msg.channel_account_id})
            continue
        bind_context(tenant_id=str(tenant_id))
        phone = mask_phone(msg.from_phone)

        limit = st.settings.rate_limit_per_minute
        if not await allow(st.redis, f"rl:{tenant_id}:{msg.from_phone}", limit):
            log.warning("rate_limited", extra={"customer": phone})
            continue

        key = dedupe_key(tenant_id, msg.message_id)
        if not await first_time(st.redis, key):
            log.info("duplicate_webhook", extra={"wamid": msg.message_id})
            continue
        try:
            await st.queue.enqueue(
                "process_inbound",
                str(tenant_id),
                msg.model_dump_json(),
                job_id=f"in:{tenant_id}:{msg.message_id}",
            )
        except Exception:
            await st.redis.delete(key)  # let Meta's retry through
            log.exception("enqueue_failed")
            raise HTTPException(503, "queue unavailable") from None
        accepted += 1
    return {"status": "ok", "accepted": accepted}

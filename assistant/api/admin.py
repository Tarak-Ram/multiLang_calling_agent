"""Operator/staff admin API: handoff inbox, staff replies, resolve/resume, cost report.

Auth: X-Admin-Key header compared in constant time with ADMIN_API_KEY. If the key is unset the
whole router answers 503 (fail closed). This is a PLATFORM-operator key; per-tenant staff
credentials (so a clinic sees only its own data) are a documented follow-up.
"""
from __future__ import annotations

import hmac
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.costs import CostReport, tenant_cost_report
from assistant.core.models import (
    EMBEDDING_DIM,
    Conversation,
    ConversationStatus,
    Direction,
    Message,
    Sender,
    Tenant,
)
from assistant.llm.base import LLMClient
from assistant.llm.profiles import ProfileSpec, ProfileStore
from assistant.llm.switch import ActivationError, activate_profile, probe_profile

log = logging.getLogger("assistant.admin")


async def require_admin(request: Request) -> None:
    key = request.app.state.settings.admin_api_key.get_secret_value()
    if not key:
        raise HTTPException(503, "admin API disabled (ADMIN_API_KEY not set)")
    supplied = request.headers.get("x-admin-key", "")
    if not hmac.compare_digest(supplied.encode(), key.encode()):
        raise HTTPException(401, "invalid admin key")


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class HandoffItem(BaseModel):
    conversation_id: uuid.UUID
    tenant: str
    customer_phone: str
    language: str | None
    summary: str | None
    notified: bool
    window_open: bool  # staff may send free-form only while True
    since: datetime


class MessageItem(BaseModel):
    at: datetime
    direction: str
    sender: str
    text: str | None


class ReplyBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class ResolveBody(BaseModel):
    action: Literal["close", "resume_bot"] = "close"


@router.get("/handoffs", response_model=list[HandoffItem])
async def list_handoffs(request: Request, tenant: str | None = Query(None)) -> list[HandoffItem]:
    """Open conversations waiting for a human, oldest first."""
    now = datetime.now(UTC)
    async with request.app.state.session_factory() as s:
        q = (select(Conversation, Tenant.slug).join(Tenant, Tenant.id == Conversation.tenant_id)
             .where(Conversation.status == ConversationStatus.HANDOFF)
             .order_by(Conversation.updated_at).limit(200))
        if tenant:
            q = q.where(Tenant.slug == tenant)
        rows = (await s.execute(q)).all()
    return [HandoffItem(
        conversation_id=c.id, tenant=slug, customer_phone=c.customer_phone, language=c.language,
        summary=c.handoff_summary, notified=c.handoff_notified_at is not None,
        window_open=bool(c.window_expires_at and c.window_expires_at > now), since=c.updated_at,
    ) for c, slug in rows]


async def _get_conversation(s: AsyncSession, conversation_id: uuid.UUID) -> Conversation:
    conv = await s.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return conv


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageItem])
async def conversation_messages(
    request: Request, conversation_id: uuid.UUID, limit: int = Query(30, ge=1, le=200)
) -> list[MessageItem]:
    async with request.app.state.session_factory() as s:
        await _get_conversation(s, conversation_id)
        rows = (await s.execute(
            select(Message).where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc()).limit(limit))).scalars().all()
    return [MessageItem(at=m.created_at, direction=m.direction.value, sender=m.sender.value,
                        text=m.text) for m in reversed(rows)]


@router.post("/conversations/{conversation_id}/reply")
async def staff_reply(request: Request, conversation_id: uuid.UUID, body: ReplyBody
                      ) -> dict[str, str]:
    """Staff message to the customer. Free-form only inside the 24h window."""
    async with request.app.state.session_factory() as s, s.begin():
        conv = await _get_conversation(s, conversation_id)
        tenant = await s.get(Tenant, conv.tenant_id)
        assert tenant is not None
        if conv.status != ConversationStatus.HANDOFF:
            raise HTTPException(409, "conversation is not in handoff")
        if not (conv.window_expires_at and conv.window_expires_at > datetime.now(UTC)):
            raise HTTPException(409, "outside the 24h window: only approved templates may be sent")
        try:
            wamid = await request.app.state.channel.send_text(
                tenant.whatsapp_phone_number_id or "", conv.customer_phone, body.text)
        except Exception:
            log.exception("staff_reply_failed")
            raise HTTPException(502, "could not send message") from None
        s.add(Message(tenant_id=tenant.id, conversation_id=conv.id, direction=Direction.OUTBOUND,
                      sender=Sender.HUMAN, wa_message_id=wamid, text=body.text,
                      language=conv.language, meta={"send_status": "sent", "kind": "staff"}))
    return {"status": "sent"}


@router.post("/conversations/{conversation_id}/resolve")
async def resolve(request: Request, conversation_id: uuid.UUID, body: ResolveBody
                  ) -> dict[str, str]:
    """Staff is done. `close` ends it; `resume_bot` hands the customer back to the assistant."""
    async with request.app.state.session_factory() as s, s.begin():
        conv = await _get_conversation(s, conversation_id)
        if conv.status != ConversationStatus.HANDOFF:
            raise HTTPException(409, "conversation is not in handoff")
        if body.action == "close":
            conv.status = ConversationStatus.CLOSED
        else:
            tenant = await s.get(Tenant, conv.tenant_id)
            hours = int((tenant.config or {}).get("timing", {}).get("expire_after_hours", 24))                 if tenant else 24
            conv.status = ConversationStatus.ACTIVE
            conv.flow, conv.current_state, conv.collected_slots = None, "identify_intent", {}
            conv.expires_at = datetime.now(UTC) + timedelta(hours=hours)
        return {"status": conv.status.value}


@router.get("/tenants/{slug}/cost-report", response_model=CostReport)
async def cost_report(request: Request, slug: str, days: int = Query(30, ge=1, le=366)
                      ) -> CostReport:
    until = datetime.now(UTC)
    async with request.app.state.session_factory() as s:
        tenant = (await s.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
        if tenant is None:
            raise HTTPException(404, "tenant not found")
        return await tenant_cost_report(s, tenant.id, slug, until - timedelta(days=days), until)


# ------------------------------------------------------------------ model switching (runtime)
class ProfileBody(BaseModel):
    """Model ids only. API KEYS are never accepted here (unknown fields are rejected, so an
    `api_key` field fails loudly instead of being silently dropped): set keys in the environment."""

    model_config = ConfigDict(extra="forbid")

    fast: str = ""
    strong: str = ""
    fallback: str = ""
    embedding_model: str = ""
    embedding_dim: int = EMBEDDING_DIM
    notes: str = ""


class ActivateBody(BaseModel):
    profile: str
    verify: bool = True  # probe every model first; nothing changes if any probe fails


class TestBody(BaseModel):
    profile: str


async def _store(request: Request) -> ProfileStore:
    store: ProfileStore = request.app.state.profiles
    await store.ensure_seeded()
    return store


def _probe_client(request: Request) -> LLMClient:
    """Raw provider client (no fallback) so a probe tests exactly the model asked about."""
    client: LLMClient | None = getattr(request.app.state, "probe_llm", None)
    if client is None:
        from assistant.llm.litellm_client import LiteLLMClient

        client = LiteLLMClient(timeout_s=request.app.state.settings.llm_timeout_s)
        request.app.state.probe_llm = client
    return client


@router.get("/models")
async def models_overview(request: Request) -> dict[str, Any]:
    """Profiles, which one is active, the effective embedding model and migration progress."""
    snap = await (await _store(request)).snapshot(fresh=True)
    return {
        "active": snap.active,
        "embedding": {"effective_model": snap.embedding_model, "dim": snap.embedding_dim},
        "migration": snap.migration,
        "profiles": [p.model_dump() | {"active": p.name == snap.active}
                     for p in sorted(snap.profiles.values(), key=lambda p: p.name)],
    }


@router.put("/models/profiles/{name}")
async def upsert_model_profile(request: Request, name: str, body: ProfileBody) -> dict[str, str]:
    store = await _store(request)
    try:
        await store.upsert_profile(ProfileSpec(name=name, **body.model_dump()))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from None
    return {"status": "saved", "profile": name}


@router.delete("/models/profiles/{name}")
async def delete_model_profile(request: Request, name: str) -> dict[str, str]:
    try:
        removed = await (await _store(request)).delete_profile(name)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    if not removed:
        raise HTTPException(404, "profile not found")
    return {"status": "deleted"}


@router.post("/models/test")
async def test_model_profile(request: Request, body: TestBody) -> dict[str, Any]:
    """Probe a profile's models WITHOUT activating it (a few tokens of real usage)."""
    snap = await (await _store(request)).snapshot(fresh=True)
    spec = snap.profiles.get(body.profile)
    if spec is None:
        raise HTTPException(404, "profile not found")
    probes = await probe_profile(spec, _probe_client(request), request.app.state.embedders)
    return {"profile": body.profile, "ok": all(p.ok for p in probes),
            "probes": [p.model_dump() for p in probes]}


@router.post("/models/activate")
async def activate_model_profile(request: Request, body: ActivateBody) -> dict[str, Any]:
    """Switch the platform to another profile. LLM models change within ~5 s everywhere; a
    different embedding model triggers a safe background re-embed (retrieval keeps using the
    old vectors until the new ones are complete)."""
    store = await _store(request)
    try:
        result = await activate_profile(
            store, body.profile, llm=_probe_client(request),
            embedders=request.app.state.embedders, verify=body.verify)
    except ActivationError as exc:
        status = {"unknown": 404, "migration": 409}.get(exc.code, 422)
        raise HTTPException(status, {"error": str(exc),
                                     "probes": [p.model_dump() for p in exc.probes]}) from None
    if result.embedding == "migration_needed":
        await request.app.state.queue.enqueue("job_embedding_migration",
                                              job_id=f"embmig:{uuid.uuid4().hex[:8]}")
    return result.model_dump()

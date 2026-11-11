"""Arq worker. Chosen over Celery: async-native, fits the all-async stack, Redis-only.

Jobs: process_inbound (Step 3+), reminders/expiry/outbound (Step 8).
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from arq import Retry, cron, func
from arq.connections import RedisSettings

from assistant.channels.base import InboundMessage
from assistant.channels.voice.voicenote import transcribe_voice_note
from assistant.channels.whatsapp.client import WhatsAppClient
from assistant.core.config import get_settings
from assistant.core.db import make_engine, make_session_factory
from assistant.core.locks import LockNotAcquiredError, conversation_lock_key, redis_lock
from assistant.core.logging import bind_context, clear_context, setup_logging
from assistant.core.models import ConversationStatus, Tenant
from assistant.core.pii import mask_phone
from assistant.core.redis import make_redis
from assistant.jobs import maintenance
from assistant.jobs.handoff import notify_handoff
from assistant.jobs.queue import ArqJobQueue
from assistant.llm.embeddings import EmbedderFactory
from assistant.llm.factory import build_extractor, build_llm
from assistant.llm.profiles import ProfileStore, env_profile
from assistant.llm.switch import run_embedding_migration
from assistant.llm.trace import LangfuseExporter
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.flows import load_industries
from assistant.orchestrator.ingest import ingest_inbound
from assistant.orchestrator.service import ConversationService
from assistant.rag.answerer import KnowledgeAnswerer
from assistant.router.cache import SemanticCache
from assistant.tools.calendar import (
    GoogleCalendarAdapter,
    service_account_token_provider,
    sync_bookings,
)
from assistant.tools.registry import build_registry

log = logging.getLogger("assistant.worker")


async def startup(ctx: dict[str, Any]) -> None:
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    ctx["engine"] = make_engine(s)
    ctx["session_factory"] = make_session_factory(ctx["engine"])
    ctx["redis"] = make_redis(s)
    ctx["http"] = httpx.AsyncClient()
    llm = build_llm(s)
    # Runtime model switching: models are resolved per request from the active profile (DB, 5 s
    # cache), so switching provider / embedding model needs no restart. Seeds from env once.
    profiles = ProfileStore(ctx["session_factory"], env_profile(s))
    await profiles.ensure_seeded()
    embedders = EmbedderFactory()
    cache = SemanticCache(ctx["redis"])  # vectors are passed in, scoped by embedding model
    ctx.update(profiles=profiles, embedders=embedders, cache=cache)
    answerer = KnowledgeAnswerer(  # FAQ answers work even with no LLM/embedder configured
        ctx["session_factory"], None, cache, llm, max_tokens=s.llm_max_output_tokens,
        resolver=profiles, embedders=embedders,
    )
    channel = WhatsAppClient(ctx["http"], s.whatsapp_access_token.get_secret_value(),
                             s.whatsapp_api_version)
    tools = build_registry(ctx["http"], channel, allow_private_integrations=not s.is_prod)
    ctx["service"] = ConversationService(
        Orchestrator(load_industries(), tools, extractor=build_extractor(s, llm, profiles),
                     answerer=answerer),
        channel,
        trace_exporter=(LangfuseExporter(ctx["http"], s.langfuse_host, s.langfuse_public_key,
                                         s.langfuse_secret_key.get_secret_value())
                        if s.langfuse_host and s.langfuse_public_key else None),
    )
    ctx["stt"] = None  # TODO(phase 2): a SpeechToText provider (paid: needs owner's choice)
    ctx["llm"] = llm
    ctx["fast_model"] = s.llm_fast_model
    ctx["channel"] = channel
    ctx["queue"] = ArqJobQueue(s.redis_url)
    ctx["calendar"] = None
    if s.google_service_account_file:  # optional Google Calendar mirror
        ctx["calendar"] = GoogleCalendarAdapter(
            ctx["http"], service_account_token_provider(s.google_service_account_file))
    log.info("worker_startup")


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["queue"].close()
    await ctx["http"].aclose()
    await ctx["redis"].aclose()
    await ctx["engine"].dispose()
    log.info("worker_shutdown")


async def ping(ctx: dict[str, Any]) -> str:
    """Smoke-test job proving the queue works end to end."""
    return "pong"


async def process_inbound(ctx: dict[str, Any], tenant_id: str, message_json: str) -> str:
    """Handle one inbound message under the per-conversation lock.

    Load state -> persist message -> (Step 4: orchestrator decides + replies) -> save.
    Nothing is held in memory between messages; a duplicate returns 'duplicate'.
    """
    clear_context()
    tid = uuid.UUID(tenant_id)
    msg = InboundMessage.model_validate_json(message_json)
    bind_context(tenant_id=tenant_id, wamid=msg.message_id)
    if msg.msg_type == "audio":  # voice note: transcribe first when an STT is configured
        msg = await transcribe_voice_note(msg, ctx["channel"], ctx["stt"])
    service: ConversationService = ctx["service"]
    try:
        async with redis_lock(ctx["redis"], conversation_lock_key(tid, msg.from_phone)):
            async with ctx["session_factory"]() as session, session.begin():
                result = await ingest_inbound(session, tid, msg)
                if result is None:
                    log.info("duplicate_message")
                    return "duplicate"
                bind_context(conversation_id=str(result.conversation.id))
                log.info(
                    "inbound_stored",
                    extra={"customer": mask_phone(msg.from_phone), "type": msg.msg_type,
                           "new_conversation": result.is_new_conversation},
                )
                tenant = await session.get(Tenant, tid)
                if tenant is None:
                    return "unknown_tenant"
                pendings = await service.process(session, tenant, result, msg)
                needs_notify = (result.conversation.status == ConversationStatus.HANDOFF
                                and result.conversation.handoff_notified_at is None)
                conversation_id = result.conversation.id
            # committed: state is safe; now send (still under the lock to keep reply order)
            await service.deliver(ctx["session_factory"], pendings)
            if needs_notify:  # staff alert + summary polish run offline, not on the reply path
                await ctx["queue"].enqueue("job_notify_handoff", str(conversation_id),
                                           job_id=f"handoff:{conversation_id}")
            return "replied" if pendings else "no_reply"
    except LockNotAcquiredError:
        raise Retry(defer=1) from None  # another worker owns this conversation; try again


async def job_embedding_migration(ctx: dict[str, Any]) -> dict[str, Any]:
    """Phase 2/3 of an embedding-model switch (see llm/switch.py). Long-running, one at a time."""
    return await run_embedding_migration(ctx["profiles"], ctx["session_factory"],
                                         ctx["embedders"], ctx["redis"], ctx["cache"])


async def job_embedding_migration_watch(ctx: dict[str, Any]) -> dict[str, Any]:
    """Safety net: if a switch left a migration 'pending' (e.g. the API could not enqueue it),
    start it. Never auto-retries an 'error' state, so a broken provider can't burn money."""
    snap = await ctx["profiles"].snapshot(fresh=True)
    if snap.migration.get("state") == "pending":
        return await job_embedding_migration(ctx)
    return {}


async def job_notify_handoff(ctx: dict[str, Any], conversation_id: str) -> str:
    outcome = await notify_handoff(ctx["session_factory"], ctx["channel"], ctx["llm"],
                                   ctx["fast_model"], uuid.UUID(conversation_id))
    if outcome == "send_failed":
        raise Retry(defer=60)  # Arq retries (max_tries); the alert must not be lost
    return outcome


# ------------------------------------------------------------------ scheduled jobs (Arq cron)
def _now() -> datetime:
    return datetime.now(UTC)


async def job_expire(ctx: dict[str, Any]) -> int:
    return await maintenance.expire_conversations(ctx["session_factory"], _now())


async def job_reminders(ctx: dict[str, Any]) -> dict[str, int]:
    return await maintenance.send_reminders(ctx["session_factory"], ctx["redis"], ctx["service"],
                                            _now())


async def job_redeliver(ctx: dict[str, Any]) -> dict[str, int]:
    return await maintenance.redeliver_pending(ctx["session_factory"], ctx["redis"],
                                               ctx["service"], _now())


async def job_sync_calendar(ctx: dict[str, Any]) -> dict[str, int]:
    if ctx.get("calendar") is None:
        return {}
    return await sync_bookings(ctx["session_factory"], ctx["calendar"])


class WorkerSettings:
    """Arq discovers this class via `arq assistant.jobs.worker.WorkerSettings`."""

    functions = [ping, process_inbound, job_notify_handoff,
                 func(job_embedding_migration, timeout=3600, max_tries=1)]
    cron_jobs = [  # unique=True: only one worker runs a given tick
        cron(job_expire, minute={0, 10, 20, 30, 40, 50}, second=0, unique=True),
        cron(job_reminders, minute={5, 20, 35, 50}, second=0, unique=True),
        cron(job_redeliver, minute=set(range(60)), second=30, unique=True),
        cron(job_sync_calendar, minute=set(range(60)), second=0, unique=True),
        cron(job_embedding_migration_watch, minute={1, 11, 21, 31, 41, 51}, second=0, unique=True,
             timeout=3600),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    job_timeout = 60
    max_tries = 5

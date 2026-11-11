"""FastAPI application: lifespan (graceful shutdown), request-id middleware, health routes."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial

import httpx
from fastapi import FastAPI, Request, Response

from assistant.api.admin import router as admin_router
from assistant.api.health import router as health_router
from assistant.api.whatsapp import router as whatsapp_router
from assistant.channels.whatsapp.client import WhatsAppClient
from assistant.core.config import get_settings
from assistant.core.db import make_engine, make_session_factory
from assistant.core.logging import bind_context, clear_context, setup_logging
from assistant.core.redis import make_redis
from assistant.jobs.queue import ArqJobQueue
from assistant.llm.embeddings import EmbedderFactory
from assistant.llm.profiles import ProfileStore, env_profile
from assistant.tenants.resolver import resolve_tenant_id

log = logging.getLogger("assistant.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources on start; close them on SIGTERM so in-flight work drains."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    app.state.settings = settings
    app.state.engine = make_engine(settings)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.redis = make_redis(settings)
    app.state.queue = ArqJobQueue(settings.redis_url)
    app.state.profiles = ProfileStore(app.state.session_factory, env_profile(settings))
    app.state.embedders = EmbedderFactory()
    app.state.http = httpx.AsyncClient()
    app.state.channel = WhatsAppClient(
        app.state.http, settings.whatsapp_access_token.get_secret_value(),
        settings.whatsapp_api_version)  # used by the admin API for staff replies
    app.state.tenant_resolver = partial(resolve_tenant_id, app.state.session_factory)
    log.info("startup", extra={"env": settings.app_env})
    try:
        yield
    finally:
        log.info("shutdown")
        await app.state.queue.close()
        await app.state.http.aclose()
        await app.state.redis.aclose()
        await app.state.engine.dispose()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(title="Multi-language AI Assistant", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        clear_context()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        bind_context(request_id=rid)
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(
            "request",
            extra={
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    app.include_router(health_router)
    app.include_router(whatsapp_router)
    app.include_router(admin_router)
    # Rate limiting is per (tenant, customer) inside the webhook, where both are known.
    return app


app = create_app()

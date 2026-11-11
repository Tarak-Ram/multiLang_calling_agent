"""Liveness (process up) and readiness (dependencies reachable) probes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from assistant.core.db import ping_db

router = APIRouter(prefix="/health", tags=["health"])
log = logging.getLogger("assistant.health")


@router.get("/live")
async def live() -> dict[str, str]:
    """Cheap check with no dependencies; used by the container healthcheck."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, object]:
    """Checks Postgres and Redis; returns 503 if either is down."""
    checks: dict[str, bool] = {}
    try:
        checks["postgres"] = await ping_db(request.app.state.engine)
    except Exception:
        log.warning("postgres_unready", exc_info=True)
        checks["postgres"] = False
    try:
        checks["redis"] = bool(await request.app.state.redis.ping())
    except Exception:
        log.warning("redis_unready", exc_info=True)
        checks["redis"] = False
    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return {"status": "ok" if ok else "degraded", "checks": checks}

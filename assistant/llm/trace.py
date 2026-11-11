"""Per-turn trace in a Langfuse-compatible shape.

Every turn emits ONE structured log line (`turn_trace`) and, if configured, the same data can be
pushed to a Langfuse server (cloud or self-hosted) through its public ingestion API. Nothing is
sent anywhere by default: no Langfuse keys = log line only.

Privacy: traces carry ids, routes, models, token counts, latency and cost, never message text,
names or phone numbers.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from assistant.llm.usage import UsageRecord

log = logging.getLogger("assistant.trace")


def build_trace(
    *, tenant_id: uuid.UUID, conversation_id: uuid.UUID, message_id: uuid.UUID, route: str,
    language: str | None, latency_ms: int, usage: list[UsageRecord], needs_human: bool,
    trace_id: str | None = None, started_at: datetime | None = None,
) -> dict[str, Any]:
    """Plain dict, JSON-serialisable, also logged as-is."""
    tid = trace_id or uuid.uuid4().hex
    total_in = sum(u.input_tokens for u in usage)
    total_out = sum(u.output_tokens for u in usage)
    return {
        "trace_id": tid, "tenant_id": str(tenant_id), "conversation_id": str(conversation_id),
        "message_id": str(message_id), "route": route, "language": language,
        "needs_human": needs_human, "latency_ms": latency_ms,
        "started_at": (started_at or datetime.now(UTC)).isoformat(),
        "llm_calls": sum(1 for u in usage if u.step != "embed"),
        "input_tokens": total_in, "output_tokens": total_out,
        "cached_input_tokens": sum(u.cached_input_tokens for u in usage),
        "cost_usd": round(sum(u.cost_usd for u in usage), 6),
        "spans": [{"step": u.step, "model": u.model, "input_tokens": u.input_tokens,
                   "cached_input_tokens": u.cached_input_tokens, "output_tokens": u.output_tokens,
                   "latency_ms": u.latency_ms, "cost_usd": round(u.cost_usd, 6)} for u in usage],
    }


def emit_trace(trace: dict[str, Any]) -> None:
    log.info("turn_trace", extra=trace)


def to_langfuse_batch(trace: dict[str, Any]) -> dict[str, Any]:
    """Langfuse `POST /api/public/ingestion` body: one trace + one generation per span."""
    start = datetime.fromisoformat(trace["started_at"])
    batch: list[dict[str, Any]] = [{
        "id": uuid.uuid4().hex, "type": "trace-create", "timestamp": trace["started_at"],
        "body": {
            "id": trace["trace_id"], "name": f"turn:{trace['route']}",
            "sessionId": trace["conversation_id"], "timestamp": trace["started_at"],
            "metadata": {"tenant_id": trace["tenant_id"], "route": trace["route"],
                         "language": trace["language"], "needs_human": trace["needs_human"],
                         "latency_ms": trace["latency_ms"]},
        },
    }]
    cursor = start
    for sp in trace["spans"]:
        end = cursor + timedelta(milliseconds=sp["latency_ms"])
        batch.append({
            "id": uuid.uuid4().hex, "type": "generation-create", "timestamp": cursor.isoformat(),
            "body": {
                "id": uuid.uuid4().hex, "traceId": trace["trace_id"], "name": sp["step"],
                "model": sp["model"], "startTime": cursor.isoformat(), "endTime": end.isoformat(),
                "usage": {"input": sp["input_tokens"], "output": sp["output_tokens"],
                          "total": sp["input_tokens"] + sp["output_tokens"], "unit": "TOKENS"},
                "costDetails": {"total": sp["cost_usd"]},
                "metadata": {"cached_input_tokens": sp["cached_input_tokens"]},
            },
        })
        cursor = end
    return {"batch": batch}


class LangfuseExporter:
    """Optional. Active only when host + both keys are configured. Failures never affect a
    customer turn: they are logged and dropped."""

    def __init__(self, http: httpx.AsyncClient, host: str, public_key: str, secret_key: str) -> None:
        self._http, self._url = http, host.rstrip("/") + "/api/public/ingestion"
        self._auth = (public_key, secret_key)

    async def export(self, trace: dict[str, Any]) -> None:
        try:
            resp = await self._http.post(self._url, json=to_langfuse_batch(trace), auth=self._auth,
                                         timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            log.warning("langfuse_export_failed")

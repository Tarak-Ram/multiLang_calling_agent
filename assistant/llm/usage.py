"""Per-turn usage collection (tokens, latency, cost) via a ContextVar.

LLM/embedding callers `record()`; the service drains the list after the turn and writes
`usage_costs` rows. Using a ContextVar keeps the orchestrator signature free of plumbing and
is safe across concurrent asyncio tasks.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from assistant.llm.base import LLMResponse


@dataclass(slots=True)
class UsageRecord:
    step: str  # extract | escalate | embed | answer | summary ...
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


_current: ContextVar[list[UsageRecord] | None] = ContextVar("usage_records", default=None)


@contextmanager
def collect_usage() -> Iterator[list[UsageRecord]]:
    """Collect everything recorded inside the block."""
    records: list[UsageRecord] = []
    token = _current.set(records)
    try:
        yield records
    finally:
        _current.reset(token)


def record(rec: UsageRecord) -> None:
    """No-op outside a collect_usage() block (e.g. offline scripts)."""
    records = _current.get()
    if records is not None:
        records.append(rec)


def record_response(step: str, resp: LLMResponse) -> None:
    record(UsageRecord(step, resp.model, resp.input_tokens, resp.cached_input_tokens,
                       resp.output_tokens, resp.latency_ms, resp.cost_usd))


async def persist_usage(
    session: AsyncSession, tenant_id: uuid.UUID, conversation_id: uuid.UUID | None,
    message_id: uuid.UUID | None, route: str, records: Iterable[UsageRecord],
    trace_id: str | None = None,
) -> None:
    """Write one `usage_costs` row per recorded call (feeds the per-tenant cost report)."""
    from assistant.core.models import UsageCost  # local: keep llm.usage importable without models

    for u in records:
        session.add(UsageCost(
            tenant_id=tenant_id, conversation_id=conversation_id, message_id=message_id,
            trace_id=trace_id, step=u.step, route=route, model=u.model, input_tokens=u.input_tokens,
            cached_input_tokens=u.cached_input_tokens, output_tokens=u.output_tokens,
            latency_ms=u.latency_ms, cost_usd=Decimal(str(round(u.cost_usd, 6))),
        ))

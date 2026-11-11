"""Per-tenant cost report built from `usage_costs` (+ message routes). Pure SQL aggregates."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import (
    Conversation,
    ConversationStatus,
    Direction,
    Message,
    UsageCost,
)

COMPLETION_STEPS_EXCLUDED = ("embed",)  # embeddings are API calls but not LLM completions


class Breakdown(BaseModel):
    key: str
    calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float
    avg_latency_ms: float


class CostReport(BaseModel):
    tenant: str
    since: datetime
    until: datetime
    conversations: int  # distinct conversations with customer activity
    turns: int  # inbound customer messages
    api_calls: int  # every billed call (completion or embedding)
    total_cost_usd: float
    cost_per_conversation_usd: float
    cost_per_turn_usd: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    avg_tokens_per_turn: float
    llm_free_turn_pct: float  # % of turns answered with NO LLM completion (rules/FAQ/cache)
    route_mix: dict[str, int]  # turns by route: rule | faq | cache | rag | extractor | ...
    by_step: list[Breakdown]
    by_model: list[Breakdown]
    handoffs: int
    handoff_rate_pct: float


def _breakdown(rows: list[tuple[str, int, int, int, int, float, float]]) -> list[Breakdown]:
    return [Breakdown(key=k or "-", calls=c, input_tokens=i, cached_input_tokens=ci,
                      output_tokens=o, cost_usd=round(float(cost), 6),
                      avg_latency_ms=round(float(lat or 0), 1))
            for k, c, i, ci, o, cost, lat in rows]


async def tenant_cost_report(
    session: AsyncSession, tenant_id: uuid.UUID, slug: str, since: datetime, until: datetime
) -> CostReport:
    inbound = [Message.tenant_id == tenant_id, Message.direction == Direction.INBOUND,
               Message.created_at >= since, Message.created_at < until]
    usage = [UsageCost.tenant_id == tenant_id, UsageCost.created_at >= since,
             UsageCost.created_at < until]

    turns = (await session.execute(select(func.count()).select_from(Message).where(*inbound))
             ).scalar_one()
    convs = (await session.execute(select(func.count(distinct(Message.conversation_id)))
                                   .where(*inbound))).scalar_one()
    totals = (await session.execute(select(
        func.count(), func.coalesce(func.sum(UsageCost.cost_usd), 0),
        func.coalesce(func.sum(UsageCost.input_tokens), 0),
        func.coalesce(func.sum(UsageCost.cached_input_tokens), 0),
        func.coalesce(func.sum(UsageCost.output_tokens), 0)).where(*usage))).one()
    api_calls, cost_raw, tin, tcached, tout = totals
    cost = float(cost_raw)

    def grouped(col: Any) -> Any:
        return (select(col, func.count(), func.coalesce(func.sum(UsageCost.input_tokens), 0),
                       func.coalesce(func.sum(UsageCost.cached_input_tokens), 0),
                       func.coalesce(func.sum(UsageCost.output_tokens), 0),
                       func.coalesce(func.sum(UsageCost.cost_usd), 0),
                       func.avg(UsageCost.latency_ms))
                .where(*usage).group_by(col).order_by(func.sum(UsageCost.cost_usd).desc()))

    by_step = _breakdown([tuple(r) for r in (await session.execute(grouped(UsageCost.step))).all()])
    by_model = _breakdown([tuple(r) for r in (await session.execute(grouped(UsageCost.model))).all()])

    # group by the LABEL: repeating the ->> expression would bind a second parameter and
    # Postgres would then treat SELECT and GROUP BY as different expressions.
    routes = (await session.execute(
        select(Message.meta["route"].astext.label("route"), func.count()).where(*inbound)
        .group_by("route"))).all()
    route_mix = {(r or "unknown"): int(n) for r, n in routes}

    # turns that triggered at least one LLM completion (not just an embedding)
    llm_turns = (await session.execute(
        select(func.count(distinct(UsageCost.message_id))).where(
            *usage, UsageCost.message_id.is_not(None),
            UsageCost.step.not_in(COMPLETION_STEPS_EXCLUDED)))).scalar_one()
    llm_free_pct = round(100 * (turns - llm_turns) / turns, 1) if turns else 0.0

    handoffs = (await session.execute(
        select(func.count()).select_from(Conversation).where(
            Conversation.tenant_id == tenant_id, Conversation.status == ConversationStatus.HANDOFF,
            Conversation.created_at >= since, Conversation.created_at < until))).scalar_one()

    return CostReport(
        tenant=slug, since=since, until=until, conversations=convs, turns=turns,
        api_calls=api_calls, total_cost_usd=round(cost, 6),
        cost_per_conversation_usd=round(cost / convs, 6) if convs else 0.0,
        cost_per_turn_usd=round(cost / turns, 6) if turns else 0.0,
        input_tokens=int(tin), cached_input_tokens=int(tcached), output_tokens=int(tout),
        avg_tokens_per_turn=round((int(tin) + int(tout)) / turns, 1) if turns else 0.0,
        llm_free_turn_pct=llm_free_pct, route_mix=route_mix, by_step=by_step, by_model=by_model,
        handoffs=handoffs, handoff_rate_pct=round(100 * handoffs / convs, 1) if convs else 0.0,
    )

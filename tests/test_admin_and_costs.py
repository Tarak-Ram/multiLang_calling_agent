"""Admin API (real DB through the real app) and the per-tenant cost report."""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.api.main import create_app
from assistant.core.config import get_settings
from assistant.core.costs import tenant_cost_report
from assistant.core.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Direction,
    Message,
    Sender,
    Tenant,
    UsageCost,
)
from tests.conftest import TEST_DB
from tests.test_jobs import NOW, Env

KEY = "admin-secret"


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    async def send_text(self, account: str, to: str, body: str) -> str:
        self.sent.append((account, to, body))
        return "wamid.staff1"


@pytest.fixture
def admin_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def client() -> TestClient:
    return TestClient(create_app(), headers={"x-admin-key": KEY})


async def handoff(e: Env, window_open: bool = True, phone: str = "+919876543210") -> Conversation:
    e.now = datetime.now(UTC)  # the admin API uses the real clock, so build windows from it
    c = await e.conv(status=ConversationStatus.HANDOFF, phone=phone,
                     window_h_left=5 if window_open else -1)
    c.handoff_summary = "Reason: test"
    await e.session.commit()
    return c


# ---------------------------------------------------------------------------------- auth
def test_admin_requires_key_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("ADMIN_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/admin/handoffs").status_code == 503  # no key configured: disabled
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/admin/handoffs").status_code == 401
        assert c.get("/admin/handoffs", headers={"x-admin-key": "wrong"}).status_code == 401
    get_settings.cache_clear()


# ------------------------------------------------------------------------------ handoffs
async def test_handoff_inbox_reply_and_resolve(session: AsyncSession, admin_env: None) -> None:
    e = await Env(session).start()
    c = await handoff(e)
    other = await handoff(e, window_open=False, phone="+919000000031")
    session.add(Message(tenant_id=e.tenant.id, conversation_id=c.id, direction=Direction.INBOUND,
                        sender=Sender.CUSTOMER, text="please call me", meta={}))
    await session.commit()

    with client() as http:
        ch = Channel()
        http.app.state.channel = ch  # type: ignore[attr-defined]
        items = http.get("/admin/handoffs", params={"tenant": "jobs"}).json()
        assert {i["conversation_id"] for i in items} == {str(c.id), str(other.id)}
        mine = next(i for i in items if i["conversation_id"] == str(c.id))
        assert mine["summary"] == "Reason: test" and mine["window_open"] is True
        assert http.get("/admin/handoffs", params={"tenant": "nope"}).json() == []

        msgs = http.get(f"/admin/conversations/{c.id}/messages").json()
        assert msgs[-1]["text"] == "please call me" and msgs[-1]["sender"] == "customer"

        ok = http.post(f"/admin/conversations/{c.id}/reply", json={"text": "Hello, this is Dr. Rao's desk"})
        assert ok.status_code == 200 and ch.sent == [("pn-jobs", "+919876543210",
                                                      "Hello, this is Dr. Rao's desk")]
        late = http.post(f"/admin/conversations/{other.id}/reply", json={"text": "hi"})
        assert late.status_code == 409 and "24h" in late.text  # outside window: template only
        assert http.post(f"/admin/conversations/{c.id}/reply", json={"text": ""}).status_code == 422
        assert http.post(f"/admin/conversations/{uuid.uuid4()}/reply",
                         json={"text": "x"}).status_code == 404

        assert http.post(f"/admin/conversations/{c.id}/resolve",
                         json={"action": "resume_bot"}).json() == {"status": "active"}
        assert http.post(f"/admin/conversations/{c.id}/resolve", json={}).status_code == 409
        assert http.post(f"/admin/conversations/{other.id}/resolve", json={}).json() == {
            "status": "closed"}
        assert http.post(f"/admin/conversations/{other.id}/reply", json={"text": "x"}
                         ).status_code == 409  # not in handoff any more

    await session.refresh(c)
    assert c.status == ConversationStatus.ACTIVE and c.flow is None and c.collected_slots == {}
    staff_msg = (await session.execute(select(Message).where(Message.sender == Sender.HUMAN))
                 ).scalar_one()
    assert staff_msg.wa_message_id == "wamid.staff1" and staff_msg.direction == Direction.OUTBOUND


# -------------------------------------------------------------------------------- costs
async def seed_usage(e: Env, tenant: Tenant, phone: str = "+919876543210") -> dict[str, Any]:
    s = e.session
    cust = Customer(tenant_id=tenant.id, phone=phone)
    s.add(cust)
    await s.flush()
    conv = Conversation(tenant_id=tenant.id, customer_id=cust.id, customer_phone=phone)
    s.add(conv)
    await s.flush()
    msgs = []
    for route in ("rule", "rule", "faq", "extractor", "rag"):
        m = Message(tenant_id=tenant.id, conversation_id=conv.id, direction=Direction.INBOUND,
                    sender=Sender.CUSTOMER, text="x", meta={"route": route})
        s.add(m)
        msgs.append(m)
    await s.flush()
    rows = [  # (message index, step, model, in, cached, out, cost, latency)
        (3, "extract", "fast", 200, 100, 20, "0.000300", 400),
        (4, "embed", "emb", 30, 0, 0, "0.000010", 90),
        (4, "answer", "fast", 400, 0, 40, "0.000700", 700),
        (4, "escalate", "strong", 500, 0, 60, "0.004000", 1500),
        (2, "embed", "emb", 20, 0, 0, "0.000010", 80),  # FAQ turn: only an embedding
    ]
    for i, step, model, tin, tc, tout, cost, lat in rows:
        s.add(UsageCost(tenant_id=tenant.id, conversation_id=conv.id, message_id=msgs[i].id,
                        step=step, model=model, input_tokens=tin, cached_input_tokens=tc,
                        output_tokens=tout, cost_usd=Decimal(cost), latency_ms=lat))
    await s.commit()
    return {"conv": conv}


async def test_cost_report_aggregates_and_isolates_tenants(session: AsyncSession) -> None:
    e = await Env(session).start()
    await seed_usage(e, e.tenant)
    # another tenant with big spend must never leak into this report
    other = Tenant(slug="other", name="o")
    session.add(other)
    await session.flush()
    await seed_usage(e, other, phone="+919000000500")

    since, until = datetime.now(UTC) - timedelta(days=1), datetime.now(UTC) + timedelta(days=1)
    r = await tenant_cost_report(session, e.tenant.id, "jobs", since, until)
    assert (r.turns, r.conversations, r.api_calls) == (5, 1, 5)
    assert r.total_cost_usd == pytest.approx(0.00502)
    assert r.cost_per_conversation_usd == pytest.approx(0.00502)
    assert r.cost_per_turn_usd == pytest.approx(0.001004)
    assert (r.input_tokens, r.cached_input_tokens, r.output_tokens) == (1150, 100, 120)
    assert r.avg_tokens_per_turn == pytest.approx((1150 + 120) / 5)
    assert r.route_mix == {"rule": 2, "faq": 1, "extractor": 1, "rag": 1}
    # 3 of 5 turns used no LLM completion (2 rule + the FAQ turn that only embedded)
    assert r.llm_free_turn_pct == 60.0
    steps = {b.key: b for b in r.by_step}
    assert steps["escalate"].cost_usd == pytest.approx(0.004) and steps["embed"].calls == 2
    assert next(b.key for b in r.by_step) == "escalate"  # ordered by cost
    assert {b.key for b in r.by_model} == {"fast", "strong", "emb"}
    assert r.handoffs == 0 and r.handoff_rate_pct == 0.0

    empty = await tenant_cost_report(session, e.tenant.id, "jobs", since - timedelta(days=30),
                                     since - timedelta(days=10))
    assert (empty.turns, empty.total_cost_usd, empty.llm_free_turn_pct) == (0, 0.0, 0.0)  # no /0


async def test_cost_report_endpoint(session: AsyncSession, admin_env: None) -> None:
    e = await Env(session).start()
    await seed_usage(e, e.tenant)
    with client() as http:
        r = http.get("/admin/tenants/jobs/cost-report", params={"days": 7})
        assert r.status_code == 200
        body = r.json()
        assert body["tenant"] == "jobs" and body["turns"] == 5 and body["llm_free_turn_pct"] == 60.0
        assert http.get("/admin/tenants/ghost/cost-report").status_code == 404
        assert http.get("/admin/tenants/jobs/cost-report", params={"days": 0}).status_code == 422
    _ = NOW

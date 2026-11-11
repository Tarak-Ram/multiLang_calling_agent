"""Integration tests for the constraints that protect correctness (need Postgres)."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import (
    Booking,
    Conversation,
    ConversationStatus,
    Customer,
    Direction,
    Message,
    Sender,
    Tenant,
    scoped,
)
from assistant.tenants.loader import import_tenant, parse_tenant_yaml

SAMPLE = Path(__file__).parents[1] / "assistant/tenants/templates/clinic_sample.yaml"


async def _tenant_customer(s: AsyncSession, slug: str = "t1") -> tuple[Tenant, Customer]:
    t = Tenant(slug=slug, name=slug)
    s.add(t)
    await s.flush()
    c = Customer(tenant_id=t.id, phone="+919876543210")
    s.add(c)
    await s.flush()
    return t, c


async def test_one_open_conversation_per_customer(session: AsyncSession) -> None:
    t, c = await _tenant_customer(session)
    session.add(Conversation(tenant_id=t.id, customer_id=c.id, customer_phone=c.phone))
    await session.flush()
    session.add(Conversation(tenant_id=t.id, customer_id=c.id, customer_phone=c.phone))
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_expired_then_new_conversation_allowed(session: AsyncSession) -> None:
    t, c = await _tenant_customer(session)
    old = Conversation(tenant_id=t.id, customer_id=c.id, customer_phone=c.phone)
    session.add(old)
    await session.flush()
    old.status = ConversationStatus.EXPIRED
    await session.flush()
    session.add(Conversation(tenant_id=t.id, customer_id=c.id, customer_phone=c.phone))
    await session.flush()


async def test_webhook_message_id_deduped_per_tenant(session: AsyncSession) -> None:
    t, c = await _tenant_customer(session)
    conv = Conversation(tenant_id=t.id, customer_id=c.id, customer_phone=c.phone)
    session.add(conv)
    await session.flush()

    def msg() -> Message:
        return Message(
            tenant_id=t.id,
            conversation_id=conv.id,
            direction=Direction.INBOUND,
            sender=Sender.CUSTOMER,
            wa_message_id="wamid.1",
            text="hi",
        )

    session.add(msg())
    await session.flush()
    session.add(msg())
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_double_booking_blocked(session: AsyncSession) -> None:
    t, c = await _tenant_customer(session)
    start = datetime.now(UTC) + timedelta(days=1)

    def b(key: str) -> Booking:
        return Booking(
            tenant_id=t.id, customer_id=c.id, resource="dr-rao", patient_name="A",
            starts_at=start, ends_at=start + timedelta(minutes=30), idempotency_key=key,
        )

    session.add(b("k1"))
    await session.flush()
    session.add(b("k2"))
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_scoped_query_isolates_tenants(session: AsyncSession) -> None:
    t1, _ = await _tenant_customer(session, "a")
    t2, _ = await _tenant_customer(session, "b")
    rows = (await session.execute(scoped(Customer, t1.id))).scalars().all()
    assert len(rows) == 1 and rows[0].tenant_id == t1.id and t2.id != t1.id


async def test_yaml_import_is_idempotent(session: AsyncSession) -> None:
    data = parse_tenant_yaml(SAMPLE)
    t = await import_tenant(session, data)
    t2 = await import_tenant(session, data)
    assert t.id == t2.id and t.config["default_language"] == "te"

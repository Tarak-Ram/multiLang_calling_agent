from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.channels.whatsapp.parser import parse_webhook
from assistant.core.models import ConversationStatus, Customer, Message, Tenant
from assistant.orchestrator.ingest import ingest_inbound
from tests.whatsapp_fixtures import envelope, text_msg


async def _tenant(s: AsyncSession) -> Tenant:
    t = Tenant(slug="t", name="t")
    s.add(t)
    await s.flush()
    return t


async def test_ingest_creates_customer_conversation_and_window(session: AsyncSession) -> None:
    t = await _tenant(session)
    [m] = parse_webhook(envelope([text_msg("w1")]))
    r = await ingest_inbound(session, t.id, m)
    assert r is not None and r.is_new_conversation
    assert r.conversation.status == ConversationStatus.ACTIVE
    assert r.conversation.window_expires_at == m.timestamp + timedelta(hours=24)
    cust = (await session.execute(select(Customer))).scalar_one()
    assert cust.name == "Ravi" and cust.phone == "+919876543210"


async def test_duplicate_returns_none_and_keeps_transaction_usable(session: AsyncSession) -> None:
    t = await _tenant(session)
    [m] = parse_webhook(envelope([text_msg("w1")]))
    assert await ingest_inbound(session, t.id, m) is not None
    assert await ingest_inbound(session, t.id, m) is None
    count = (await session.execute(select(func.count()).select_from(Message))).scalar_one()
    assert count == 1


async def test_second_message_reuses_open_conversation_and_resets_reminder(
    session: AsyncSession,
) -> None:
    t = await _tenant(session)
    [m1] = parse_webhook(envelope([text_msg("w1", ts="1700000000")]))
    [m2] = parse_webhook(envelope([text_msg("w2", ts="1700000100")]))
    r1 = await ingest_inbound(session, t.id, m1)
    assert r1 is not None
    r1.conversation.status = ConversationStatus.AWAITING_CUSTOMER
    r2 = await ingest_inbound(session, t.id, m2)
    assert r2 is not None and not r2.is_new_conversation
    assert r2.conversation.id == r1.conversation.id
    assert r2.conversation.status == ConversationStatus.ACTIVE
    assert r2.conversation.last_customer_msg_at == m2.timestamp

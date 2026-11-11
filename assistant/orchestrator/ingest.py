"""Persist an inbound message and advance conversation bookkeeping (no LLM, no reply yet).

Caller must hold the per-conversation Redis lock. The DB unique constraints are the final
source of truth for dedupe, so a Redis outage cannot cause double-processing.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.channels.base import InboundMessage
from assistant.core.models import (
    OPEN_STATUSES,
    Conversation,
    ConversationStatus,
    Customer,
    Direction,
    Message,
    Sender,
)

WINDOW = timedelta(hours=24)  # WhatsApp customer-service window
RECAP_MAX_AGE = timedelta(hours=72)  # offer a recap if the expired chat is this recent


@dataclass(slots=True)
class IngestResult:
    conversation: Conversation
    message: Message
    is_new_conversation: bool


async def _expired_flow(
    session: AsyncSession, tenant_id: uuid.UUID, msg: InboundMessage
) -> str | None:
    """Flow of this customer's most recent EXPIRED conversation, if it was in progress and
    recent enough to be worth a recap."""
    row = (await session.execute(
        select(Conversation.flow, Conversation.expires_at, Conversation.updated_at).where(
            Conversation.tenant_id == tenant_id,
            Conversation.customer_phone == msg.from_phone,
            Conversation.status == ConversationStatus.EXPIRED,
            Conversation.flow.is_not(None),
        ).order_by(Conversation.updated_at.desc()).limit(1)
    )).first()
    if row is None:
        return None
    # Age = time since the conversation's own deadline (not DB row timestamps).
    expired_at = row.expires_at or row.updated_at
    if msg.timestamp - expired_at > RECAP_MAX_AGE:
        return None
    return str(row.flow)


async def ingest_inbound(
    session: AsyncSession, tenant_id: uuid.UUID, msg: InboundMessage
) -> IngestResult | None:
    """Store the message. Returns None if it was a duplicate (already processed)."""
    stmt = pg_insert(Customer).values(
        tenant_id=tenant_id, phone=msg.from_phone, name=msg.contact_name
    )
    customer_id = (
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=["tenant_id", "phone"],
                # keep an existing name; only fill it if we have none yet
                set_={"name": func.coalesce(Customer.name, stmt.excluded.name)},
            ).returning(Customer.id)
        )
    ).scalar_one()

    conv = (
        await session.execute(
            select(Conversation).where(
                Conversation.tenant_id == tenant_id,
                Conversation.customer_phone == msg.from_phone,
                Conversation.status.in_(OPEN_STATUSES),
            )
        )
    ).scalar_one_or_none()
    if conv is not None and conv.expires_at is not None and conv.expires_at < msg.timestamp:
        # The expiry job hasn't run yet, but the conversation is past its deadline: a late
        # reply must start fresh, exactly as if the job had already expired it.
        conv.status = ConversationStatus.EXPIRED
        await session.flush()
        conv = None
    is_new = conv is None
    if conv is None:
        recap_flow = await _expired_flow(session, tenant_id, msg)
        conv = Conversation(
            tenant_id=tenant_id,
            customer_id=customer_id,
            customer_phone=msg.from_phone,
            status=ConversationStatus.ACTIVE,
            # The engine turns this private marker into a ONE-line recap in the customer's
            # language, then removes it. We keep only the flow name, never old slot values.
            collected_slots={"_recap": {"flow": recap_flow}} if recap_flow else {},
        )
        session.add(conv)
        await session.flush()

    row = Message(
        tenant_id=tenant_id,
        conversation_id=conv.id,
        direction=Direction.INBOUND,
        sender=Sender.CUSTOMER,
        wa_message_id=msg.message_id,
        channel=msg.channel,
        msg_type=msg.msg_type,
        text=msg.text,
        meta={"payload_id": msg.payload_id, "media_id": msg.media_id},
    )
    try:
        async with session.begin_nested():  # savepoint: duplicate must not poison the txn
            session.add(row)
            await session.flush()
    except IntegrityError:
        return None

    conv.last_customer_msg_at = msg.timestamp
    conv.window_expires_at = msg.timestamp + WINDOW
    if conv.status == ConversationStatus.AWAITING_CUSTOMER:
        conv.status = ConversationStatus.ACTIVE  # customer replied
    conv.reminder_sent_at = None
    await session.flush()
    return IngestResult(conv, row, is_new)

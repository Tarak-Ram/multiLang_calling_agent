"""SQLAlchemy 2 models.

Tenant isolation rule: every table except `tenants` carries `tenant_id` (FK, indexed, part of
every unique key) and every query MUST filter on it. Use `scoped()` below to make that the
default path instead of an afterthought.
"""
from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    Select,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimension of kb_chunks.embedding. Fixed in the migration; changing the embedding model to
# one with a different size needs a new migration + re-embed. The embedding model chosen in EMBEDDING_MODEL must output exactly this size.
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    """Declarative base with deterministic constraint names (needed for Alembic diffs)."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


def _enum(e: type[enum.Enum]) -> Enum:
    """Store enums as VARCHAR + CHECK (not native PG enums: far easier to migrate)."""
    return Enum(e, native_enum=False, length=24, values_callable=lambda x: [m.value for m in x])


class ConversationStatus(enum.StrEnum):
    ACTIVE = "active"  # bot is mid-flow, waiting on us or the customer
    AWAITING_CUSTOMER = "awaiting_customer"  # we asked something, customer silent
    HANDOFF = "handoff"  # a human owns it
    CLOSED = "closed"  # flow completed
    EXPIRED = "expired"  # timed out; a late reply starts a fresh conversation


OPEN_STATUSES = (
    ConversationStatus.ACTIVE,
    ConversationStatus.AWAITING_CUSTOMER,
    ConversationStatus.HANDOFF,
)


class Direction(enum.StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class Sender(enum.StrEnum):
    CUSTOMER = "customer"
    BOT = "bot"
    HUMAN = "human"
    SYSTEM = "system"  # reminders, templates


class KbKind(enum.StrEnum):
    FAQ = "faq"  # pre-approved answer, preferred over generation
    DOC = "doc"  # free-text chunk for RAG


class BookingStatus(enum.StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class _Timestamps:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _tenant_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )


class Tenant(_Timestamps, Base):
    """A business using the platform. `config` is validated by tenants.schema.TenantConfig."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = _pk()
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    industry: Mapped[str] = mapped_column(String(32), default="clinic")
    # Meta's phone_number_id: how an inbound webhook is routed to its tenant.
    whatsapp_phone_number_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))


class Customer(_Timestamps, Base):
    __tablename__ = "customers"
    __table_args__ = (UniqueConstraint("tenant_id", "phone"),)

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    phone: Mapped[str] = mapped_column(String(20))  # E.164, digits with leading '+'
    name: Mapped[str | None] = mapped_column(String(200))
    language: Mapped[str | None] = mapped_column(String(8))  # detected once, then stored


class Conversation(_Timestamps, Base):
    """Durable state machine record, keyed logically by (tenant_id, customer_phone).

    At most ONE open conversation per (tenant, phone) (partial unique index); expired/closed
    ones stay as history, so a late reply creates a new row with a recap.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        Index(
            "uq_conversations_open_per_customer",
            "tenant_id",
            "customer_phone",
            unique=True,
            postgresql_where=text("status IN ('active','awaiting_customer','handoff')"),
        ),
        Index("ix_conversations_tenant_status_last_msg", "tenant_id", "status", "last_customer_msg_at"),
    )

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customers.id", ondelete="CASCADE")
    )
    customer_phone: Mapped[str] = mapped_column(String(20))
    status: Mapped[ConversationStatus] = mapped_column(
        _enum(ConversationStatus), default=ConversationStatus.ACTIVE
    )
    flow: Mapped[str | None] = mapped_column(String(48))  # e.g. "book_appointment"
    current_state: Mapped[str] = mapped_column(String(48), default="greet")
    collected_slots: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    short_summary: Mapped[str] = mapped_column(Text, default="")  # running summary, token-capped
    language: Mapped[str | None] = mapped_column(String(8))
    last_customer_msg_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # WhatsApp 24h customer-service window = last_customer_msg_at + 24h, stored for cheap queries.
    window_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reminder_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    handoff_summary: Mapped[str | None] = mapped_column(Text)
    handoff_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list[Message]] = relationship(back_populates="conversation")


class Message(Base):
    """Immutable log of every inbound/outbound message. Webhook dedupe lives here."""

    __tablename__ = "messages"
    __table_args__ = (
        # Idempotency: the same WhatsApp message id can never be stored twice per tenant.
        UniqueConstraint("tenant_id", "wa_message_id"),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    direction: Mapped[Direction] = mapped_column(_enum(Direction))
    sender: Mapped[Sender] = mapped_column(_enum(Sender))
    wa_message_id: Mapped[str | None] = mapped_column(String(128))  # NULL allowed (unique skips)
    channel: Mapped[str] = mapped_column(String(16), default="whatsapp")
    msg_type: Mapped[str] = mapped_column(String(16), default="text")
    text: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8))
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)  # intent, route, template...
    # Stamped in Python, not by Postgres now(): now() is the TRANSACTION start, so an inbound
    # message and its reply written in one transaction would tie and lose their order.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
        server_default=func.now(), nullable=False,
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class KbChunk(_Timestamps, Base):
    """Tenant knowledge: approved FAQ answers and RAG chunks, per language."""

    __tablename__ = "kb_chunks"
    __table_args__ = (
        Index("ix_kb_chunks_tenant_lang_kind", "tenant_id", "language", "kind"),
        Index(
            "ix_kb_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    kind: Mapped[KbKind] = mapped_column(_enum(KbKind), default=KbKind.DOC)
    language: Mapped[str] = mapped_column(String(8))
    source: Mapped[str | None] = mapped_column(String(200))  # doc name / url
    question: Mapped[str | None] = mapped_column(Text)  # set for FAQ entries
    content: Mapped[str] = mapped_column(Text)  # approved answer or chunk text (short!)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    approved: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # Vectors from different models live in different spaces and must never be compared.
    # Retrieval only uses chunks whose embedding_model equals the platform's effective model.
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    # Staging area for an embedding-model switch: the NEW model's vectors are written here while
    # `embedding` keeps serving retrieval, then swapped in atomically (llm/switch.py).
    embedding_next: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_next_model: Mapped[str | None] = mapped_column(String(200))


class ModelProfile(Base):
    """A named set of models (platform-level, not tenant data). Holds model IDS only, never API
    keys: keys stay in the environment, where LiteLLM reads them."""

    __tablename__ = "model_profiles"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    fast: Mapped[str] = mapped_column(String(200), default="", server_default="")
    strong: Mapped[str] = mapped_column(String(200), default="", server_default="")
    fallback: Mapped[str] = mapped_column(String(200), default="", server_default="")
    embedding_model: Mapped[str] = mapped_column(String(200), default="", server_default="")
    embedding_dim: Mapped[int] = mapped_column(Integer, default=EMBEDDING_DIM,
                                               server_default=str(EMBEDDING_DIM))
    notes: Mapped[str] = mapped_column(Text, default="", server_default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PlatformSetting(Base):
    """Small key/value store for runtime switches: active_profile, embedding_effective,
    embedding_migration."""

    __tablename__ = "platform_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Booking(_Timestamps, Base):
    __tablename__ = "bookings"
    __table_args__ = (
        # Hard guarantee against double booking: one live booking per resource per start time.
        Index(
            "uq_bookings_live_slot",
            "tenant_id",
            "resource",
            "starts_at",
            unique=True,
            postgresql_where=text("status IN ('pending','confirmed')"),
        ),
        UniqueConstraint("tenant_id", "idempotency_key"),
        Index("ix_bookings_tenant_customer", "tenant_id", "customer_id"),
    )

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customers.id", ondelete="CASCADE")
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    resource: Mapped[str] = mapped_column(String(64))  # doctor / room / service provider id
    service: Mapped[str | None] = mapped_column(String(100))
    patient_name: Mapped[str] = mapped_column(String(200))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[BookingStatus] = mapped_column(
        _enum(BookingStatus), default=BookingStatus.PENDING
    )
    calendar_event_id: Mapped[str | None] = mapped_column(String(128))
    calendar_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    idempotency_key: Mapped[str] = mapped_column(String(128))  # prevents duplicate execute


class UsageCost(Base):
    """One row per billable step (LLM call, embedding...) for traces and the cost report."""

    __tablename__ = "usage_costs"
    __table_args__ = (
        Index("ix_usage_costs_tenant_created", "tenant_id", "created_at"),
        Index("ix_usage_costs_conversation", "conversation_id"),
    )

    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant_fk()
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL")
    )
    trace_id: Mapped[str | None] = mapped_column(String(64))
    step: Mapped[str] = mapped_column(String(32))  # intent|rag|escalate|embed|summary|...
    route: Mapped[str | None] = mapped_column(String(24))  # rule|cache|faq|small|strong|human
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal(0))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )




def scoped(model: type[Base], tenant_id: uuid.UUID) -> Select[Any]:
    """Start every tenant-owned query here so the tenant filter cannot be forgotten."""
    if not hasattr(model, "tenant_id"):
        raise TypeError(f"{model.__name__} is not tenant-scoped")
    stmt = select(model).where(model.tenant_id == tenant_id)
    return cast("Select[Any]", stmt)

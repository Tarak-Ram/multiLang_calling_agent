"""Appointment tools backed by Postgres (the source of truth for availability).

All tools use ctx.session: the same transaction that saves the conversation state, so a booking
and its conversation transition commit or roll back together. Double-booking is prevented by
(1) an advisory lock per (tenant, resource) that serialises overlapping bookings, (2) an overlap
check inside that lock, and (3) a partial unique index as the final backstop.
Google Calendar is only a one-way mirror, pushed later by jobs.maintenance.sync_calendar.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import ClassVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.core.models import Booking, BookingStatus, Customer
from assistant.tenants.schema import ServiceDef, TenantConfig
from assistant.tools.base import ToolContext, ToolResult
from assistant.tools.labels import date_label, time_label

log = logging.getLogger("assistant.tools.booking")
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
LIVE = (BookingStatus.PENDING, BookingStatus.CONFIRMED)


# --------------------------------------------------------------------- pure slot maths
def service_by_id(cfg: TenantConfig, service_id: str | None) -> ServiceDef | None:
    return next((s for s in cfg.booking.services if s.id == service_id), None)


def candidate_starts(cfg: TenantConfig, svc: ServiceDef, day: date, now: datetime
                     ) -> list[datetime]:
    """Grid of possible start times on `day` (tenant tz): inside business hours, the service
    must END before closing, and not sooner than min_lead from `now`."""
    tz = ZoneInfo(cfg.timezone)
    hours = cfg.business_hours.get(_DAYS[day.weekday()])  # type: ignore[call-overload]
    if hours is None:
        return []
    step = timedelta(minutes=cfg.booking.slot_minutes)
    dur = timedelta(minutes=svc.duration_minutes)
    cursor = datetime.combine(day, hours.open, tz)
    close = datetime.combine(day, hours.close, tz)
    earliest = now.astimezone(tz) + timedelta(minutes=cfg.booking.min_lead_minutes)
    out = []
    while cursor + dur <= close:
        if cursor >= earliest:
            out.append(cursor)
        cursor += step
    return out


def free_starts(cands: Sequence[datetime], duration: timedelta,
                busy: Sequence[tuple[datetime, datetime]]) -> list[datetime]:
    """Candidates whose [start, start+duration) overlaps no busy interval."""
    return [c for c in cands if all(c + duration <= b0 or c >= b1 for b0, b1 in busy)]


async def busy_intervals(
    session: AsyncSession, tenant_id: uuid.UUID, resource: str, start: datetime, end: datetime,
    exclude_id: uuid.UUID | None = None,
) -> list[tuple[datetime, datetime]]:
    q = select(Booking.starts_at, Booking.ends_at).where(
        Booking.tenant_id == tenant_id, Booking.resource == resource,
        Booking.status.in_(LIVE), Booking.starts_at < end, Booking.ends_at > start,
    )
    if exclude_id:
        q = q.where(Booking.id != exclude_id)
    return [(a, b) for a, b in (await session.execute(q)).all()]


async def _service_for(ctx: ToolContext, slots: dict[str, str]) -> ServiceDef | None:
    """Service from the slots: either chosen directly, or via the booking being rescheduled."""
    cfg = ctx.config
    if slots.get("service"):
        return service_by_id(cfg, slots["service"])
    if slots.get("booking"):
        b = await _owned_booking(ctx, slots["booking"])
        return service_by_id(cfg, b.service) if b else None
    return None


async def _owned_booking(ctx: ToolContext, booking_id: str) -> Booking | None:
    """The customer's own future live booking. Ownership is checked on EVERY access so one
    customer can never touch another's appointment by guessing an id."""
    try:
        bid = uuid.UUID(booking_id)
    except ValueError:
        return None
    session: AsyncSession = ctx.session
    row = (await session.execute(
        select(Booking).join(Customer, Customer.id == Booking.customer_id).where(
            Booking.id == bid, Booking.tenant_id == ctx.tenant_id,
            Customer.phone == ctx.customer_phone, Booking.status.in_(LIVE),
        )
    )).scalar_one_or_none()
    return row


async def _slot_options(ctx: ToolContext, svc: ServiceDef, day: date,
                        exclude_id: uuid.UUID | None = None) -> list[datetime]:
    cfg, now = ctx.config, datetime.now(UTC)
    cands = candidate_starts(cfg, svc, day, now)
    if not cands:
        return []
    busy = await busy_intervals(ctx.session, ctx.tenant_id, svc.resource, cands[0],
                                cands[-1] + timedelta(minutes=svc.duration_minutes), exclude_id)
    return free_starts(cands, timedelta(minutes=svc.duration_minutes), busy)


# --------------------------------------------------------------------------- option tools
class OptIn(BaseModel):
    slots: dict[str, str] = {}


class ListServices:
    name: ClassVar[str] = "list_services"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        opts = [{"id": s.id, "title": s.name(ctx.language, ctx.config.default_language)}
                for s in ctx.config.booking.services]
        return ToolResult(ok=True, data={"options": opts})


class ListDates:
    name: ClassVar[str] = "list_dates"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        svc = await _service_for(ctx, args.slots)
        if svc is None:
            return ToolResult(ok=True, data={"options": []})
        exclude = _uuid_or_none(args.slots.get("booking"))
        today = datetime.now(ZoneInfo(ctx.config.timezone)).date()
        opts = []
        for i in range(ctx.config.booking.horizon_days + 1):
            day = today + timedelta(days=i)
            if await _slot_options(ctx, svc, day, exclude):
                opts.append({"id": day.isoformat(), "title": date_label(day, ctx.language)})
        return ToolResult(ok=True, data={"options": opts[:10]})


class ListSlots:
    name: ClassVar[str] = "list_slots"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        svc = await _service_for(ctx, args.slots)
        day = _parse_date(args.slots.get("date"))
        if svc is None or day is None:
            return ToolResult(ok=True, data={"options": []})
        starts = await _slot_options(ctx, svc, day, _uuid_or_none(args.slots.get("booking")))
        return ToolResult(ok=True, data={"options": [
            {"id": s.strftime("%H:%M"), "title": time_label(s)} for s in starts[:10]]})


class ListMyBookings:
    name: ClassVar[str] = "list_my_bookings"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = OptIn

    async def run(self, ctx: ToolContext, args: OptIn) -> ToolResult:
        tz = ZoneInfo(ctx.config.timezone)
        rows = (await ctx.session.execute(
            select(Booking).join(Customer, Customer.id == Booking.customer_id).where(
                Booking.tenant_id == ctx.tenant_id, Customer.phone == ctx.customer_phone,
                Booking.status.in_(LIVE), Booking.starts_at > datetime.now(UTC),
            ).order_by(Booking.starts_at).limit(5)
        )).scalars().all()
        opts = []
        for b in rows:
            local = b.starts_at.astimezone(tz)
            opts.append({"id": str(b.id),
                         "title": f"{date_label(local.date(), ctx.language)} {time_label(local)}"})
        return ToolResult(ok=True, data={"options": opts})


# ------------------------------------------------------------------------- action tools
class BookIn(BaseModel):
    service: str
    date: str
    time: str
    patient_name: str
    phone: str
    idempotency_key: str


class RescheduleIn(BaseModel):
    booking: str
    date: str
    time: str
    phone: str
    idempotency_key: str


class CancelIn(BaseModel):
    booking: str
    phone: str
    idempotency_key: str = ""


def _parse_date(v: str | None) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


def _uuid_or_none(v: str | None) -> uuid.UUID | None:
    try:
        return uuid.UUID(v) if v else None
    except ValueError:
        return None


async def _lock_resource(session: AsyncSession, tenant_id: uuid.UUID, resource: str) -> None:
    """Serialise concurrent bookings of the same resource until this transaction ends."""
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                          {"k": f"book:{tenant_id}:{resource}"})


def _resolve_start(ctx: ToolContext, svc: ServiceDef, date_s: str, time_s: str
                   ) -> datetime | None:
    """The requested start, ONLY if it is on the bookable grid (never trust slot strings)."""
    day = _parse_date(date_s)
    try:
        hh, mm = time_s.split(":")
        t = time(int(hh), int(mm))
    except ValueError:
        return None
    if day is None:
        return None
    today = datetime.now(ZoneInfo(ctx.config.timezone)).date()
    if day > today + timedelta(days=ctx.config.booking.horizon_days):
        return None  # the tool enforces the horizon itself; option lists are not a security gate
    for c in candidate_starts(ctx.config, svc, day, datetime.now(UTC)):
        if c.timetz().replace(tzinfo=None) == t:
            return c
    return None


class BookAppointment:
    name: ClassVar[str] = "book_appointment"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = BookIn

    async def run(self, ctx: ToolContext, args: BookIn) -> ToolResult:
        svc = service_by_id(ctx.config, args.service)
        start = _resolve_start(ctx, svc, args.date, args.time) if svc else None
        if svc is None or start is None:
            return ToolResult(ok=False, error_code="invalid_slot")
        session: AsyncSession = ctx.session
        end = start + timedelta(minutes=svc.duration_minutes)
        customer = (await session.execute(select(Customer).where(
            Customer.tenant_id == ctx.tenant_id, Customer.phone == ctx.customer_phone,
        ))).scalar_one()
        try:
            async with session.begin_nested():
                await _lock_resource(session, ctx.tenant_id, svc.resource)
                existing = (await session.execute(select(Booking).where(
                    Booking.tenant_id == ctx.tenant_id,
                    Booking.idempotency_key == args.idempotency_key))).scalar_one_or_none()
                if existing:  # retried execute: same booking, no duplicate
                    return ToolResult(ok=True, data={"ref": _ref(existing)})
                if await busy_intervals(session, ctx.tenant_id, svc.resource, start, end):
                    return ToolResult(ok=False, error_code="slot_taken")
                booking = Booking(
                    tenant_id=ctx.tenant_id, customer_id=customer.id,
                    conversation_id=ctx.conversation_id, resource=svc.resource, service=svc.id,
                    patient_name=args.patient_name, starts_at=start, ends_at=end,
                    status=BookingStatus.CONFIRMED, idempotency_key=args.idempotency_key,
                )
                session.add(booking)
                await session.flush()
        except IntegrityError:  # lost a race the lock didn't cover: unique index says taken
            return ToolResult(ok=False, error_code="slot_taken")
        return ToolResult(ok=True, data={"ref": _ref(booking)})


class CancelBooking:
    name: ClassVar[str] = "cancel_booking"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = CancelIn

    async def run(self, ctx: ToolContext, args: CancelIn) -> ToolResult:
        b = await _owned_booking(ctx, args.booking)
        if b is None:
            return ToolResult(ok=False, error_code="not_found")
        b.status = BookingStatus.CANCELLED  # calendar event (if any) is removed by the sync job
        await ctx.session.flush()
        return ToolResult(ok=True, data={"ref": _ref(b)})


class RescheduleBooking:
    name: ClassVar[str] = "reschedule_booking"
    group: ClassVar[str] = "booking"
    input_model: ClassVar[type[BaseModel]] = RescheduleIn

    async def run(self, ctx: ToolContext, args: RescheduleIn) -> ToolResult:
        b = await _owned_booking(ctx, args.booking)
        if b is None:
            return ToolResult(ok=False, error_code="not_found")
        svc = service_by_id(ctx.config, b.service)
        start = _resolve_start(ctx, svc, args.date, args.time) if svc else None
        if svc is None or start is None:
            return ToolResult(ok=False, error_code="invalid_slot")
        end = start + timedelta(minutes=svc.duration_minutes)
        session: AsyncSession = ctx.session
        try:
            async with session.begin_nested():
                await _lock_resource(session, ctx.tenant_id, svc.resource)
                if await busy_intervals(session, ctx.tenant_id, svc.resource, start, end,
                                        exclude_id=b.id):
                    return ToolResult(ok=False, error_code="slot_taken")
                b.starts_at, b.ends_at = start, end
                await session.flush()
        except IntegrityError:
            return ToolResult(ok=False, error_code="slot_taken")
        return ToolResult(ok=True, data={"ref": _ref(b)})


def _ref(b: Booking) -> str:
    return b.id.hex[:8].upper()

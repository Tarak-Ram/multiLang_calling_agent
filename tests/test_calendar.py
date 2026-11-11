import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import Booking, BookingStatus
from assistant.tools.booking import (
    BookAppointment,
    CancelBooking,
    CancelIn,
    ListMyBookings,
    OptIn,
    RescheduleBooking,
    RescheduleIn,
)
from assistant.tools.calendar import GoogleCalendarAdapter, sync_bookings
from tests.test_booking_tools import book_args, first_slot, make_ctx, setup


async def token() -> str:
    return "ya29.test"


async def test_google_adapter_requests() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.method == "DELETE":
            return httpx.Response(410)  # already gone: must be treated as success
        return httpx.Response(200, json={"id": "evt1"})

    cal = GoogleCalendarAdapter(httpx.AsyncClient(transport=httpx.MockTransport(handler)), token)
    start = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
    eid = await cal.create_event("cal@x", "General: Ravi", "Booking AB12", start,
                                 start + timedelta(minutes=30), "Asia/Kolkata")
    await cal.update_event("cal@x", eid, start, start + timedelta(minutes=30), "Asia/Kolkata")
    await cal.delete_event("cal@x", eid)
    create, patch, delete = seen
    assert eid == "evt1" and create.method == "POST" and create.url.path.endswith("/calendars/cal@x/events")
    assert create.headers["authorization"] == "Bearer ya29.test"
    body = json.loads(create.content)
    assert body["start"]["timeZone"] == "Asia/Kolkata" and body["summary"] == "General: Ravi"
    assert "phone" not in json.dumps(body).lower()  # no phone numbers pushed to the calendar
    assert patch.method == "PATCH" and delete.method == "DELETE"


class FakeCal:
    def __init__(self) -> None:
        self.events: dict[str, tuple[datetime, datetime]] = {}
        self.fail_for: set[str] = set()
        self.n = 0

    async def create_event(self, cal: str, summary: str, desc: str, start: datetime,
                           end: datetime, tz: str) -> str:
        if any(k in desc for k in self.fail_for):
            raise RuntimeError("google down")
        self.n += 1
        self.events[f"e{self.n}"] = (start, end)
        return f"e{self.n}"

    async def update_event(self, cal: str, eid: str, start: datetime, end: datetime,
                           tz: str) -> None:
        self.events[eid] = (start, end)

    async def delete_event(self, cal: str, eid: str) -> None:
        self.events.pop(eid)


async def test_sync_creates_updates_deletes_and_is_idempotent(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    tenant.config = {**tenant.config, "booking": {**tenant.config["booking"], "calendar_id": "c@x"}}
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    await BookAppointment().run(ctx, book_args(d, t, "k1"))
    await session.commit()
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    cal = FakeCal()

    assert await sync_bookings(factory, cal) == {"created": 1, "updated": 0, "deleted": 0, "failed": 0}
    assert await sync_bookings(factory, cal) == {"created": 0, "updated": 0, "deleted": 0, "failed": 0}
    assert len(cal.events) == 1  # second run is a no-op (not dirty any more)

    [opt] = (await ListMyBookings().run(ctx, OptIn())).data["options"]
    slots_after = (await first_slot(ctx))[1]
    await RescheduleBooking().run(ctx, RescheduleIn(
        booking=opt["id"], date=d, time=slots_after, phone=ctx.customer_phone, idempotency_key="r"))
    await session.commit()
    assert (await sync_bookings(factory, cal))["updated"] == 1

    await CancelBooking().run(ctx, CancelIn(booking=opt["id"], phone=ctx.customer_phone))
    await session.commit()
    assert (await sync_bookings(factory, cal))["deleted"] == 1 and cal.events == {}
    assert (await sync_bookings(factory, cal))["deleted"] == 0  # stable, no resync loop


async def test_one_failing_booking_does_not_block_the_batch(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    tenant.config = {**tenant.config, "booking": {**tenant.config["booking"], "calendar_id": "c@x"}}
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    r1 = await BookAppointment().run(ctx, book_args(d, t, "k1"))
    slots = (await first_slot(ctx))
    ctx2 = await make_ctx(session, tenant, cfg, phone="+919000000002")
    t2 = (await __import__("tests.test_booking_tools", fromlist=["ListSlots"]).ListSlots().run(
        ctx2, OptIn(slots={"service": "general", "date": d}))).data["options"][1]["id"]
    await BookAppointment().run(ctx2, book_args(d, t2, "k2", phone="+919000000002"))
    await session.commit()
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    cal = FakeCal()
    cal.fail_for = {r1.data["ref"]}  # the first booking's description contains its ref
    stats = await sync_bookings(factory, cal)
    assert stats["created"] == 1 and stats["failed"] == 1 and slots
    cal.fail_for = set()
    assert (await sync_bookings(factory, cal))["created"] == 1  # failed one retried next tick
    rows: Any = (await session.execute(select(Booking).where(
        Booking.status == BookingStatus.CONFIRMED))).scalars().all()
    for r in rows:
        await session.refresh(r)
    assert all(r.calendar_event_id for r in rows)


async def test_tenant_without_calendar_is_skipped(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)  # sample has calendar_id: null
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    await BookAppointment().run(ctx, book_args(d, t, "k1"))
    await session.commit()
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    cal = FakeCal()
    assert await sync_bookings(factory, cal) == {"created": 0, "updated": 0, "deleted": 0, "failed": 0}

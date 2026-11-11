"""Booking tools against real Postgres. Dates come from the tools' own option lists, so these
tests do not depend on today's date."""
import asyncio
import uuid
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import Booking, BookingStatus, Conversation, Customer, Tenant
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from assistant.tenants.schema import DayHours, TenantConfig
from assistant.tools.base import ToolContext
from assistant.tools.booking import (
    BookAppointment,
    BookIn,
    CancelBooking,
    CancelIn,
    ListDates,
    ListMyBookings,
    ListServices,
    ListSlots,
    OptIn,
    RescheduleBooking,
    RescheduleIn,
    candidate_starts,
    free_starts,
)
from tests.engine_helpers import SAMPLE

PHONE = "+919876543210"
OTHER = "+919000000001"


async def make_ctx(session: AsyncSession, tenant: Tenant, cfg: TenantConfig,
                   phone: str = PHONE, lang: str = "en") -> ToolContext:
    cust = (await session.execute(select(Customer).where(
        Customer.tenant_id == tenant.id, Customer.phone == phone))).scalar_one_or_none()
    if cust is None:
        cust = Customer(tenant_id=tenant.id, phone=phone, name="T")
        session.add(cust)
        await session.flush()
    conv = Conversation(tenant_id=tenant.id, customer_id=cust.id, customer_phone=phone)
    session.add(conv)
    await session.flush()
    return ToolContext(tenant_id=tenant.id, customer_phone=phone, conversation_id=conv.id,
                       config=cfg, language=lang, session=session)


async def setup(session: AsyncSession, slug: str = "demo") -> tuple[Tenant, TenantConfig]:
    data = parse_tenant_yaml(SAMPLE)
    data.slug = slug
    data.whatsapp_phone_number_id = f"pn-{slug}"
    # open all week so tests never hit a closed day
    for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        data.config.business_hours[d] = DayHours(open=time(9), close=time(18))  # type: ignore[index]
    tenant = await import_tenant(session, data)
    cfg = TenantConfig.model_validate(tenant.config)
    cfg.booking.min_lead_minutes = 0
    return tenant, cfg


async def first_slot(ctx: ToolContext, service: str = "general") -> tuple[str, str]:
    dates = (await ListDates().run(ctx, OptIn(slots={"service": service}))).data["options"]
    d = dates[0]["id"]
    slots = (await ListSlots().run(ctx, OptIn(slots={"service": service, "date": d}))).data[
        "options"]
    return d, slots[0]["id"]


def book_args(d: str, t: str, key: str, service: str = "general", phone: str = PHONE) -> BookIn:
    return BookIn(service=service, date=d, time=t, patient_name="Ravi Kumar", phone=phone,
                  idempotency_key=key)


# ------------------------------------------------------------------ pure slot maths
def test_candidate_starts_respect_hours_duration_and_lead() -> None:
    cfg = TenantConfig.model_validate(parse_tenant_yaml(SAMPLE).config.model_dump())
    svc = cfg.booking.services[0]
    tz = ZoneInfo(cfg.timezone)
    thursday = date(2026, 11, 12)
    now = datetime(2026, 11, 12, 10, 0, tzinfo=tz)  # 10:00 local, lead 60 min -> >= 11:00
    starts = candidate_starts(cfg, svc, thursday, now)
    assert starts[0].astimezone(tz).strftime("%H:%M") == "11:00"
    assert starts[-1].astimezone(tz).strftime("%H:%M") == "17:30"  # 30-min visit ends by 18:00
    assert candidate_starts(cfg, svc, date(2026, 11, 15), now) == []  # Sunday closed


def test_free_starts_excludes_overlaps_only() -> None:
    tz = UTC
    c = [datetime(2026, 1, 1, h, m, tzinfo=tz) for h, m in ((9, 0), (9, 30), (10, 0), (10, 30))]
    busy = [(datetime(2026, 1, 1, 9, 30, tzinfo=tz), datetime(2026, 1, 1, 10, 0, tzinfo=tz))]
    out = free_starts(c, timedelta(minutes=30), busy)
    assert [x.strftime("%H:%M") for x in out] == ["09:00", "10:00", "10:30"]
    out60 = free_starts(c, timedelta(minutes=60), busy)  # a 60-min visit at 9:00 would overlap
    assert [x.strftime("%H:%M") for x in out60] == ["10:00"][:1] or "09:00" not in [
        x.strftime("%H:%M") for x in out60]


# ------------------------------------------------------------------ option tools
async def test_services_are_localized(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    ctx = await make_ctx(session, tenant, cfg, lang="te")
    opts = (await ListServices().run(ctx, OptIn())).data["options"]
    assert opts[0] == {"id": "general", "title": "జనరల్ ఫిజీషియన్"}


async def test_dates_and_slots_listing(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    ctx = await make_ctx(session, tenant, cfg, lang="hi")
    dates = (await ListDates().run(ctx, OptIn(slots={"service": "general"}))).data["options"]
    assert 1 <= len(dates) <= cfg.booking.horizon_days + 1
    date.fromisoformat(dates[0]["id"])
    assert any(ch in dates[0]["title"] for ch in "सोमंगलबुधगुरुशुक्रशनिरवि")  # Hindi weekday
    slots = (await ListSlots().run(ctx, OptIn(slots={"service": "general",
                                                     "date": dates[-1]["id"]}))).data["options"]
    assert slots and all(s["title"].endswith(("AM", "PM")) for s in slots)
    assert (await ListSlots().run(ctx, OptIn(slots={"service": "nope", "date": "x"}))
            ).data["options"] == []


# ------------------------------------------------------------------ booking
async def test_book_then_slot_disappears_and_other_customer_blocked(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    a = await make_ctx(session, tenant, cfg)
    b = await make_ctx(session, tenant, cfg, phone=OTHER)
    d, t = await first_slot(a)
    res = await BookAppointment().run(a, book_args(d, t, "k1"))
    assert res.ok and len(res.data["ref"]) == 8
    left = (await ListSlots().run(a, OptIn(slots={"service": "general", "date": d}))
            ).data["options"]
    assert t not in [s["id"] for s in left]
    clash = await BookAppointment().run(b, book_args(d, t, "k2", phone=OTHER))
    assert not clash.ok and clash.error_code == "slot_taken"
    other_doctor = await BookAppointment().run(b, book_args(d, t, "k3", "derm", OTHER))
    assert other_doctor.ok  # different resource, same time: fine


async def test_retried_execute_is_idempotent(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    r1 = await BookAppointment().run(ctx, book_args(d, t, "same-key"))
    r2 = await BookAppointment().run(ctx, book_args(d, t, "same-key"))
    assert r1.ok and r2.ok and r1.data["ref"] == r2.data["ref"]
    n = (await session.execute(select(func.count()).select_from(Booking))).scalar_one()
    assert n == 1


async def test_overlapping_longer_service_is_blocked(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    cfg.booking.services[0].duration_minutes = 60  # general visits now 60 minutes
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    assert (await BookAppointment().run(ctx, book_args(d, t, "k1"))).ok
    hh, mm = map(int, t.split(":"))
    later = (datetime.combine(date.today(), time(hh, mm)) + timedelta(minutes=30)).strftime("%H:%M")
    res = await BookAppointment().run(ctx, book_args(d, later, "k2"))  # starts mid-visit
    assert not res.ok and res.error_code in ("slot_taken", "invalid_slot")


async def test_invalid_slots_are_rejected_not_trusted(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    ctx = await make_ctx(session, tenant, cfg)
    d, t = await first_slot(ctx)
    tomorrow = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
    cases = [
        book_args(d, "10:15", "a"),  # off the 30-minute grid
        book_args(d, "03:00", "b"),  # before opening
        book_args(d, "17:45", "c"),  # would end after closing
        book_args("2020-01-01", t, "d"),  # in the past
        book_args("not-a-date", t, "e"),
        book_args(d, "xx", "f"),
        book_args(d, t, "g", service="unknown"),
        book_args((date.fromisoformat(d) + timedelta(days=30)).isoformat(), t, "h"),  # > horizon
    ]
    for args in cases:
        res = await BookAppointment().run(ctx, args)
        assert not res.ok and res.error_code == "invalid_slot", args
    assert tomorrow
    n = (await session.execute(select(func.count()).select_from(Booking))).scalar_one()
    assert n == 0


async def test_concurrent_bookings_exactly_one_wins(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    await session.commit()
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]

    async def attempt(phone: str, key: str) -> bool:
        async with factory() as s, s.begin():
            ctx = await make_ctx(s, tenant, cfg, phone=phone)
            d, t = day_slot
            return (await BookAppointment().run(ctx, book_args(d, t, key, phone=phone))).ok

    async with factory() as s:
        day_slot = await first_slot(await make_ctx(s, tenant, cfg, phone="+911111111111"))
    results = await asyncio.gather(*(attempt(f"+9122222222{i:02d}", f"k{i}") for i in range(6)))
    assert sorted(results) == [False] * 5 + [True]
    async with factory() as s:
        n = (await s.execute(select(func.count()).select_from(Booking).where(
            Booking.status == BookingStatus.CONFIRMED))).scalar_one()
    assert n == 1


# ------------------------------------------------------------------ cancel / reschedule
async def test_list_cancel_and_ownership(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    mine = await make_ctx(session, tenant, cfg)
    theirs = await make_ctx(session, tenant, cfg, phone=OTHER)
    d, t = await first_slot(mine)
    await BookAppointment().run(mine, book_args(d, t, "k1"))
    [opt] = (await ListMyBookings().run(mine, OptIn())).data["options"]
    assert (await ListMyBookings().run(theirs, OptIn())).data["options"] == []
    steal = await CancelBooking().run(theirs, CancelIn(booking=opt["id"], phone=OTHER))
    assert not steal.ok and steal.error_code == "not_found"  # cannot cancel someone else's
    junk = await CancelBooking().run(mine, CancelIn(booking="not-a-uuid", phone=PHONE))
    assert junk.error_code == "not_found"
    ok = await CancelBooking().run(mine, CancelIn(booking=opt["id"], phone=PHONE))
    assert ok.ok
    assert (await ListMyBookings().run(mine, OptIn())).data["options"] == []
    again = await CancelBooking().run(mine, CancelIn(booking=opt["id"], phone=PHONE))
    assert again.error_code == "not_found"  # already cancelled
    freed = (await ListSlots().run(mine, OptIn(slots={"service": "general", "date": d}))
             ).data["options"]
    assert t in [s["id"] for s in freed]  # slot is bookable again


async def test_reschedule_moves_booking_and_checks_conflicts(session: AsyncSession) -> None:
    tenant, cfg = await setup(session)
    a = await make_ctx(session, tenant, cfg)
    b = await make_ctx(session, tenant, cfg, phone=OTHER)
    d, t = await first_slot(a)
    slots = (await ListSlots().run(a, OptIn(slots={"service": "general", "date": d}))
             ).data["options"]
    t2, t3 = slots[1]["id"], slots[2]["id"]
    await BookAppointment().run(a, book_args(d, t, "k1"))
    await BookAppointment().run(b, book_args(d, t2, "k2", phone=OTHER))
    [opt] = (await ListMyBookings().run(a, OptIn())).data["options"]
    # offering slots for a reschedule must not hide the booking's own slot, but hides t2
    offered = (await ListSlots().run(a, OptIn(slots={"booking": opt["id"], "date": d}))
               ).data["options"]
    assert t in [s["id"] for s in offered] and t2 not in [s["id"] for s in offered]
    clash = await RescheduleBooking().run(a, RescheduleIn(booking=opt["id"], date=d, time=t2,
                                                          phone=PHONE, idempotency_key="r1"))
    assert clash.error_code == "slot_taken"
    ok = await RescheduleBooking().run(a, RescheduleIn(booking=opt["id"], date=d, time=t3,
                                                       phone=PHONE, idempotency_key="r2"))
    assert ok.ok
    row = (await session.execute(select(Booking).where(Booking.id == uuid.UUID(opt["id"])))
           ).scalar_one()
    assert row.starts_at.astimezone(ZoneInfo(cfg.timezone)).strftime("%H:%M") == t3
    stolen = await RescheduleBooking().run(b, RescheduleIn(booking=opt["id"], date=d, time=t,
                                                           phone=OTHER, idempotency_key="r3"))
    assert stolen.error_code == "not_found"

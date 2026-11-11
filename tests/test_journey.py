"""End-to-end through the REAL stack: webhook-shaped messages -> ingest -> orchestrator ->
real YAML flows -> real booking tools -> Postgres -> channel. Only the channel is faked."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import InboundMessage, Option
from assistant.channels.whatsapp.parser import parse_webhook
from assistant.core.models import Booking, BookingStatus, Conversation, ConversationStatus, Tenant
from assistant.orchestrator.engine import Orchestrator
from assistant.orchestrator.flows import load_industries
from assistant.orchestrator.ingest import ingest_inbound
from assistant.orchestrator.service import ConversationService, PendingSend
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from assistant.tenants.schema import DayHours
from assistant.tools.registry import build_registry
from tests.engine_helpers import SAMPLE
from tests.whatsapp_fixtures import envelope, text_msg

PHONE_PN = "pn-journey"


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[Option]]] = []

    async def send_text(self, account_id: str, to: str, body: str) -> str:
        self.sent.append((body, []))
        return f"out{len(self.sent)}"

    async def send_options(self, account_id: str, to: str, body: str, options: list[Option],
                           list_button: str) -> str:
        self.sent.append((body, options))
        return f"out{len(self.sent)}"

    async def send_template(self, *a: Any) -> str:
        return "wt"


class Journey:
    def __init__(self, session: AsyncSession) -> None:
        self.session, self.n, self.channel = session, 0, Channel()
        self.tenant: Tenant
        self.svc: ConversationService
        self.last: list[PendingSend] = []

    async def start(self) -> Journey:
        data = parse_tenant_yaml(SAMPLE)
        data.slug = "journey"
        data.whatsapp_phone_number_id = PHONE_PN
        for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
            data.config.business_hours[d] = DayHours.model_validate(  # type: ignore[index]
                {"open": "09:00", "close": "18:00"})
        data.config.booking.min_lead_minutes = 0
        self.tenant = await import_tenant(self.session, data)
        await self.session.commit()
        self.factory = async_sessionmaker(self.session.bind, expire_on_commit=False)  # type: ignore[arg-type]
        orch = Orchestrator(load_industries(),
                            build_registry(httpx.AsyncClient(), self.channel))  # type: ignore[arg-type]
        self.svc = ConversationService(orch, self.channel)  # real clock
        return self

    async def send(self, text: str | None = None, payload: str | None = None) -> str:
        self.n += 1
        ts = str(int(datetime.now(UTC).timestamp()))
        msg: InboundMessage = parse_webhook(envelope([text_msg(f"w{self.n}", text or "x", ts)]))[0]
        msg = msg.model_copy(update={"text": text, "payload_id": payload})
        ing = await ingest_inbound(self.session, self.tenant.id, msg)
        assert ing is not None
        self.last = await self.svc.process(self.session, self.tenant, ing, msg)
        await self.session.commit()
        await self.svc.deliver(self.factory, self.last)
        return " | ".join(p.text for p in self.last)

    def first_option(self) -> str:
        return next(p.options[0].id for p in reversed(self.last) if p.options)

    async def bookings(self, status: BookingStatus | None = None) -> list[Booking]:
        q = select(Booking)
        if status:
            q = q.where(Booking.status == status)
        return list((await self.session.execute(q)).scalars().all())


async def book(j: Journey, name: str, yes: str) -> None:
    await j.send(payload="menu:book_appointment")
    await j.send(payload=j.first_option())  # service
    await j.send(payload=j.first_option())  # date
    await j.send(payload=j.first_option())  # time
    await j.send(name)
    await j.send(yes)


async def test_telugu_booking_then_cancel_in_telugu(session: AsyncSession) -> None:
    j = await Journey(session).start()
    greeting = await j.send("నమస్కారం")
    assert "నమస్కారం" in greeting and any(o.id == "menu:book_appointment" for o in j.last[0].options)
    await book(j, "రవి కుమార్", "అవును")  # Telugu patient name + Telugu "yes"
    [b] = await j.bookings(BookingStatus.CONFIRMED)
    assert b.patient_name == "రవి కుమార్" and b.service == "general" and b.resource == "dr-rao"
    assert "అపాయింట్‌మెంట్ నిర్ధారించబడింది" in j.last[0].text and len(b.id.hex[:8]) == 8

    cancel_prompt = await j.send("రద్దు")  # Telugu 'cancel' keyword -> cancel flow, no LLM
    assert "అపాయింట్‌మెంట్" in cancel_prompt
    await j.send(payload=j.first_option())
    done = await j.send("అవును")
    assert "రద్దు చేయబడింది" in done
    assert await j.bookings(BookingStatus.CONFIRMED) == []
    convs = (await session.execute(select(Conversation))).scalars().all()
    assert all(c.status == ConversationStatus.CLOSED for c in convs if c.flow == "cancel_appointment")


async def test_english_booking_then_reschedule(session: AsyncSession) -> None:
    j = await Journey(session).start()
    await j.send("hi")
    await book(j, "Ravi Kumar", "yes")
    [before] = await j.bookings(BookingStatus.CONFIRMED)

    prompt = await j.send("I want to reschedule")
    assert "Which appointment" in prompt
    await j.send(payload=j.first_option())  # the booking
    await j.send(payload=j.first_option())  # new date
    slots = j.last[0].options
    assert len(slots) >= 2
    await j.send(payload=slots[1].id)  # a different time than the first
    assert "Move your appointment" in j.last[0].text
    done = await j.send("yes")
    assert "Rescheduled" in done
    await session.refresh(before)
    [after] = await j.bookings(BookingStatus.CONFIRMED)
    assert after.id == before.id  # same booking, moved (not a second one)


async def test_cancel_with_no_bookings_says_so_and_shows_menu(session: AsyncSession) -> None:
    j = await Journey(session).start()
    reply = await j.send("cancel appointment")
    assert "no upcoming appointments" in reply
    assert any(o.id == "menu:book_appointment" for p in j.last for o in p.options)


@pytest.mark.parametrize("greeting", ["hello", "नमस्ते"])
async def test_two_customers_cannot_take_the_same_slot(session: AsyncSession, greeting: str) -> None:
    j = await Journey(session).start()
    await j.send(greeting)
    await book(j, "First Person", "yes" if greeting == "hello" else "हाँ")
    [first] = await j.bookings(BookingStatus.CONFIRMED)

    # second customer (different phone) sees that slot gone and cannot book it
    j.n += 100
    ts = str(int(datetime.now(UTC).timestamp()))
    other = parse_webhook(envelope([{"from": "919000000009", "id": "o1", "timestamp": ts,
                                     "type": "text", "text": {"body": "book appointment"}}]))[0]
    ing = await ingest_inbound(session, j.tenant.id, other)
    assert ing
    pend = await j.svc.process(session, j.tenant, ing, other)
    assert pend and pend[-1].options  # service list shown
    taken = first.starts_at
    assert len([b for b in await j.bookings() if b.starts_at == taken]) == 1


async def test_resume_after_silence_revalidates_slot_against_live_data(
    session: AsyncSession,
) -> None:
    """Customer A reaches the confirmation and goes quiet; customer B books A's slot; when A
    finally says yes, A must NOT get a booking and is offered the remaining times."""
    from assistant.core.models import Customer
    from assistant.tenants.schema import TenantConfig
    from assistant.tools.base import ToolContext
    from assistant.tools.booking import BookAppointment, BookIn

    j = await Journey(session).start()
    await j.send("hi")
    await j.send(payload="menu:book_appointment")
    await j.send(payload=j.first_option())  # service
    await j.send(payload=j.first_option())  # date
    await j.send(payload=j.first_option())  # time
    await j.send("Ravi Kumar")
    assert "Please confirm" in j.last[0].text
    conv = (await session.execute(select(Conversation))).scalar_one()
    slots = dict(conv.collected_slots)

    # --- while A is silent: B books exactly A's slot (state lives only in Postgres) ---
    cfg = TenantConfig.model_validate(j.tenant.config)
    b_cust = Customer(tenant_id=j.tenant.id, phone="+919000000077", name="B")
    session.add(b_cust)
    await session.flush()
    b_conv = Conversation(tenant_id=j.tenant.id, customer_id=b_cust.id,
                          customer_phone="+919000000077")
    session.add(b_conv)
    await session.flush()
    steal = await BookAppointment().run(
        ToolContext(tenant_id=j.tenant.id, customer_phone="+919000000077",
                    conversation_id=b_conv.id, config=cfg, session=session),
        BookIn(service=slots["service"], date=slots["date"], time=slots["time"],
               patient_name="B", phone="+919000000077", idempotency_key="b1"))
    assert steal.ok
    await session.commit()

    # --- A finally replies "yes" ---
    reply = await j.send("yes")
    mine = [b for b in await j.bookings(BookingStatus.CONFIRMED) if b.patient_name == "Ravi Kumar"]
    assert mine == []  # never double-booked
    assert "no longer available" in reply
    offered = [o.id for o in j.last[-1].options]
    assert slots["time"] not in offered and offered  # remaining times are offered instead
    await session.refresh(conv)
    assert conv.collected_slots["date"] == slots["date"] and "time" not in conv.collected_slots

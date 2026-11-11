"""Service integration (real Postgres): state persistence across webhooks, send + 24h window."""
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.channels.base import InboundMessage, Option
from assistant.channels.whatsapp.parser import parse_webhook
from assistant.core.models import ConversationStatus, Customer, Direction, Message, Tenant
from assistant.orchestrator.ingest import ingest_inbound
from assistant.orchestrator.service import ConversationService
from assistant.tenants.loader import import_tenant, parse_tenant_yaml
from tests.engine_helpers import OPEN_NOW, SAMPLE, make
from tests.whatsapp_fixtures import envelope, text_msg


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, list[Option]]] = []
        self.fail = False

    async def send_text(self, account_id: str, to_phone: str, body: str) -> str:
        if self.fail:
            raise RuntimeError("meta down")
        self.sent.append((to_phone, body, []))
        return f"wamid.out{len(self.sent)}"

    async def send_options(self, account_id: str, to_phone: str, body: str,
                           options: list[Option], list_button: str) -> str:
        self.sent.append((to_phone, body, options))
        return f"wamid.out{len(self.sent)}"

    async def send_template(self, *a: Any) -> str:
        raise AssertionError("not used")


async def _setup(session: AsyncSession) -> tuple[Tenant, ConversationService, FakeChannel]:
    data = parse_tenant_yaml(SAMPLE)
    data.whatsapp_phone_number_id = "111222333"
    tenant = await import_tenant(session, data)
    orch, _, _ = make()
    ch = FakeChannel()
    return tenant, ConversationService(orch, ch, clock=lambda: OPEN_NOW), ch


def _msg(mid: str, body: str, **kw: Any) -> InboundMessage:
    m = parse_webhook(envelope([text_msg(mid, body, ts=str(int(OPEN_NOW.timestamp())))]))[0]
    return m.model_copy(update=kw)


async def test_state_persists_across_webhooks_and_replies_are_sent(session: AsyncSession) -> None:
    tenant, svc, ch = await _setup(session)
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]

    m1 = _msg("w1", "I want an appointment")
    ing = await ingest_inbound(session, tenant.id, m1)
    assert ing is not None
    pend = await svc.process(session, tenant, ing, m1)
    await session.commit()
    await svc.deliver(factory, pend)

    # a NEW webhook later: state is reloaded from Postgres, nothing in memory
    m2 = _msg("w2", "General", )
    m2 = m2.model_copy(update={"payload_id": "gen"})
    ing2 = await ingest_inbound(session, tenant.id, m2)
    assert ing2 is not None and not ing2.is_new_conversation
    await svc.process(session, tenant, ing2, m2)
    conv = ing2.conversation
    assert conv.flow == "book_appointment" and conv.collected_slots["service"] == "gen"
    assert conv.status == ConversationStatus.AWAITING_CUSTOMER and conv.language == "en"
    assert conv.expires_at is not None

    assert ch.sent[0][2] and ch.sent[0][2][0].id == "gen"  # service options sent as interactive
    outs = (await session.execute(select(Message).where(Message.direction == Direction.OUTBOUND))
            ).scalars().all()
    assert outs[0].meta["send_status"] == "sent" and outs[0].wa_message_id == "wamid.out1"


async def test_handoff_persisted_and_language_saved_on_customer(session: AsyncSession) -> None:
    tenant, svc, _ = await _setup(session)
    m = _msg("w1", "నాకు జ్వరం ఉంది")
    ing = await ingest_inbound(session, tenant.id, m)
    assert ing
    await svc.process(session, tenant, ing, m)
    assert ing.conversation.status == ConversationStatus.HANDOFF
    assert "[medical_question]" in (ing.conversation.handoff_summary or "")
    cust = (await session.execute(select(Customer))).scalar_one()
    assert cust.language == "te"


async def test_free_form_blocked_outside_24h_window(session: AsyncSession) -> None:
    tenant, svc, ch = await _setup(session)
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    m = _msg("w1", "hello")
    ing = await ingest_inbound(session, tenant.id, m)
    assert ing
    pend = await svc.process(session, tenant, ing, m)
    await session.commit()
    svc.clock = lambda: OPEN_NOW + timedelta(hours=25)  # window has closed
    await svc.deliver(factory, pend)
    assert ch.sent == []
    row = (await session.execute(select(Message).where(Message.direction == Direction.OUTBOUND))
           ).scalars().first()
    await session.refresh(row)
    assert row and row.meta["send_status"] == "blocked_outside_window"


async def test_send_failure_recorded_not_raised(session: AsyncSession) -> None:
    tenant, svc, ch = await _setup(session)
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    ch.fail = True
    m = _msg("w1", "talk to staff")
    ing = await ingest_inbound(session, tenant.id, m)
    assert ing
    pend = await svc.process(session, tenant, ing, m)
    await session.commit()
    await svc.deliver(factory, pend)  # must not raise
    row = (await session.execute(select(Message).where(Message.direction == Direction.OUTBOUND))
           ).scalars().first()
    await session.refresh(row)
    assert row and row.meta["send_status"] == "failed"


async def test_usage_rows_and_recent_turns_flow_through_service(session: AsyncSession) -> None:
    from decimal import Decimal

    from assistant.core.models import UsageCost
    from assistant.llm.base import LLMRequest, LLMResponse
    from assistant.orchestrator.engine import Orchestrator
    from assistant.router.extractor import LLMExtractor
    from assistant.tools.base import ToolRegistry

    class LLM:
        def __init__(self) -> None:
            self.reqs: list[LLMRequest] = []

        async def complete(self, r: LLMRequest) -> LLMResponse:
            self.reqs.append(r)
            return LLMResponse(text='{"intent":"faq","slots":{},"confidence":0.9}', model=r.model,
                               input_tokens=111, cached_input_tokens=80, output_tokens=9,
                               latency_ms=42, cost_usd=0.00033)

    llm = LLM()
    from assistant.orchestrator.flows import load_industries
    orch = Orchestrator(load_industries(), ToolRegistry(), LLMExtractor(llm, "fast-model"))
    tenant, _, ch = await _setup(session)
    svc = ConversationService(orch, ch, clock=lambda: OPEN_NOW)
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]

    m1 = _msg("w1", "hello")  # rule: greeting -> no LLM, no usage
    ing = await ingest_inbound(session, tenant.id, m1)
    assert ing
    await svc.deliver(factory, await svc.process(session, tenant, ing, m1))
    await session.commit()
    assert (await session.execute(select(UsageCost))).scalars().all() == []

    m2 = _msg("w2", "random unusual question")  # no rule -> extractor -> 1 LLM call
    ing2 = await ingest_inbound(session, tenant.id, m2)
    assert ing2
    await svc.process(session, tenant, ing2, m2)
    [u] = (await session.execute(select(UsageCost))).scalars().all()
    assert (u.step, u.model, u.input_tokens, u.cached_input_tokens, u.output_tokens) == (
        "extract", "fast-model", 111, 80, 9)
    assert u.cost_usd == Decimal("0.000330") and u.message_id == ing2.message.id
    # the LLM saw the earlier turns (customer 'hello' + the bot greeting) but not the current msg twice
    user_block = llm.reqs[0].user
    assert "customer: hello" in user_block and user_block.rstrip().endswith(
        "message: random unusual question")


async def test_recent_turns_keep_order_within_one_transaction(session: AsyncSession) -> None:
    """Inbound + bot reply are written in the same txn; their order must still be exact."""
    tenant, svc, _ = await _setup(session)
    m1 = _msg("w1", "hello")
    ing = await ingest_inbound(session, tenant.id, m1)
    assert ing
    await svc.process(session, tenant, ing, m1)  # writes customer msg + bot greeting
    m2 = _msg("w2", "second")
    ing2 = await ingest_inbound(session, tenant.id, m2)
    assert ing2
    turns = await svc._recent_turns(session, ing2.conversation.id, ing2.message.id)
    assert [r for r, _ in turns] == ["customer", "bot"] and turns[0][1] == "hello"


async def test_faq_question_answered_end_to_end_with_zero_llm_cost(session: AsyncSession) -> None:
    """Customer asks a FAQ in Hindi: rules miss -> approved FAQ answered verbatim -> sent.
    No LLM, no embeddings configured at all."""
    from assistant.core.models import UsageCost
    from assistant.orchestrator.engine import Orchestrator
    from assistant.orchestrator.flows import load_industries
    from assistant.rag.answerer import KnowledgeAnswerer
    from assistant.tools.base import ToolRegistry

    tenant, _, ch = await _setup(session)
    await session.commit()
    factory = async_sessionmaker(session.bind, expire_on_commit=False)  # type: ignore[arg-type]
    orch = Orchestrator(load_industries(), ToolRegistry(), None,
                        KnowledgeAnswerer(factory))  # FAQ-only answerer
    svc = ConversationService(orch, ch, clock=lambda: OPEN_NOW)

    m = _msg("w1", "आपका समय क्या है?")
    ing = await ingest_inbound(session, tenant.id, m)
    assert ing
    pend = await svc.process(session, tenant, ing, m)
    await session.commit()
    await svc.deliver(factory, pend)

    assert [body for _, body, _ in ch.sent] == [
        "हम सोम-शुक्र सुबह 9 से शाम 6 और शनिवार सुबह 9 से दोपहर 1 बजे तक खुले हैं।"]
    assert ing.message.meta["route"] == "faq" and ing.conversation.language == "hi"
    assert (await session.execute(select(UsageCost))).scalars().all() == []

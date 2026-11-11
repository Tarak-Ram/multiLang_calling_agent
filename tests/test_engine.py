import pytest

from assistant.core.models import ConversationStatus
from assistant.orchestrator.types import Extraction
from tests.engine_helpers import CLOSED_NOW, Backend, Chat, FakeAnswerer, FakeExtractor, make


async def book_with_menu(chat: Chat, greeting: str) -> None:
    await chat.say(greeting)
    assert "menu:book_appointment" in chat.option_ids
    await chat.say(payload="menu:book_appointment")
    await chat.say(payload="gen")
    await chat.say(payload="2026-11-13")
    await chat.say(payload="10:00")
    await chat.say("Ravi Kumar")


@pytest.mark.parametrize(
    ("greeting", "lang", "yes"),
    [("hello", "en", "yes"), ("నమస్కారం", "te", "అవును"), ("नमस्ते", "hi", "हाँ")],
)
async def test_full_booking_flow_in_each_language(greeting: str, lang: str, yes: str) -> None:
    orch, be, cfg = make()
    chat = Chat(orch, cfg)
    await book_with_menu(chat, greeting)
    assert chat.state.language == lang
    assert chat.state.state == "confirm"
    # confirmation shows ALL critical slots incl. phone
    body = chat.texts[0]
    for needed in ("Ravi Kumar", "General", "Fri 13 Nov", "10:00", "+919876543210"):
        assert needed in body or lang != "en", needed
    assert "Ravi Kumar" in body and "+919876543210" in body
    await chat.say(yes)
    assert chat.closed and be.booked[0].time == "10:00" and be.booked[0].patient_name == "Ravi Kumar"
    assert "BK1" in chat.texts[0]


async def test_no_booking_executes_before_confirmation() -> None:
    orch, be, cfg = make()
    chat = Chat(orch, cfg)
    await book_with_menu(chat, "hi")
    assert be.booked == []


async def test_confirm_no_restarts_collection() -> None:
    orch, be, cfg = make()
    chat = Chat(orch, cfg)
    await book_with_menu(chat, "hi")
    await chat.say("no")
    assert chat.state.state == "collect_slots" and chat.option_ids == ["gen", "derm"]
    assert be.booked == []


async def test_slot_taken_between_confirm_and_yes_is_revalidated() -> None:
    orch, be, cfg = make()
    chat = Chat(orch, cfg)
    await book_with_menu(chat, "hi")
    be.taken.add(("2026-11-13", "10:00"))  # someone else books while customer is silent
    await chat.say("yes")
    assert be.booked == []  # never double-booked
    assert chat.state.state == "collect_slots"
    assert chat.option_ids == ["11:00"] and "no longer available" in chat.texts[0]
    assert "time" not in chat.state.slots and chat.state.slots["date"] == "2026-11-13"


async def test_race_inside_tool_maps_error_to_reask() -> None:
    orch, be, cfg = make()
    chat = Chat(orch, cfg)
    await book_with_menu(chat, "hi")
    be.race = True  # lost the race between revalidation and the booking call
    await chat.say("yes")
    assert be.booked == [] and chat.state.state == "collect_slots"
    assert "time" not in chat.state.slots and "no longer available" in chat.texts[0]


async def test_typed_option_title_and_number_accepted() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    await chat.say("I want an appointment")  # keyword intent skips the menu
    assert chat.state.flow == "book_appointment" and chat.option_ids == ["gen", "derm"]
    await chat.say("dermatology")
    assert chat.state.slots["service"] == "derm"
    await chat.say("2")
    assert chat.state.slots["date"] == "2026-11-14"


async def test_unclear_answers_then_handoff() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    for _ in range(3):
        await chat.say("blah blah")
    assert chat.state.status == ConversationStatus.HANDOFF
    assert chat.last and chat.last.needs_human and chat.last.handoff_reason == "unclear:service"


async def test_medical_question_refused_and_routed_to_staff() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    r = await chat.say("what medicine should I take for fever")
    assert r.needs_human and r.handoff_reason == "medical_question"
    assert "can't give medical advice" in chat.texts[0]
    assert chat.state.status == ConversationStatus.HANDOFF


async def test_medical_question_telugu() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    r = await chat.say("నాకు జ్వరం ఉంది")
    assert r.needs_human and "వైద్య సలహా" in chat.texts[0]


async def test_human_requested_and_hours_aware_message() -> None:
    orch, _, cfg = make()
    open_chat = Chat(orch, cfg)
    await open_chat.say("talk to staff")
    assert open_chat.texts == ["Connecting you to our staff."]
    closed = Chat(orch, cfg, now=CLOSED_NOW)
    await closed.say("talk to staff")
    assert "closed" in closed.texts[0]


async def test_handoff_conversation_gets_no_bot_replies() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    await chat.say("talk to staff")
    r = await chat.say("hello?")
    assert r.replies == [] and r.route == "human_owned"


async def test_menu_keyword_resets_flow() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    await chat.say("menu")
    assert chat.state.flow is None and chat.state.state == "identify_intent"
    assert "menu:book_appointment" in chat.option_ids


async def test_tool_not_enabled_for_tenant_hands_off_safely() -> None:
    orch, be, cfg = make(tools_enabled=[])
    chat = Chat(orch, cfg)
    r = await chat.say("book appointment")
    assert r.needs_human and (r.handoff_reason or "").startswith("tool_unavailable")
    assert be.booked == []


async def test_no_slots_available_hands_off() -> None:
    be = Backend()
    be.no_slots = True
    orch, _, cfg = make(be)
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    await chat.say(payload="gen")
    r = await chat.say(payload="2026-11-13")
    assert r.handoff_reason == "no_options:time"


async def test_flow_disabled_for_tenant_goes_to_human() -> None:
    orch, _, cfg = make()
    cfg.flows_enabled = ["book_appointment"]  # cancel flow exists but this tenant disabled it
    chat = Chat(orch, cfg)
    r = await chat.say("cancel appointment")  # longest keyword beats 'appointment'
    assert r.handoff_reason == "flow_unavailable:cancel_appointment"


async def test_voice_note_gets_text_only_notice() -> None:
    orch, _, cfg = make()
    chat = Chat(orch, cfg)
    chat.state.language = "en"
    await chat.say(None)
    assert "text" in chat.texts[0]


async def test_extractor_called_only_when_rules_fail() -> None:
    ext = FakeExtractor(Extraction(intent="book_appointment", confidence=0.9,
                                   slots={"service": "gen", "date": "2026-11-13", "time": "11:00"}))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await chat.say("hello")  # rule hit: greeting -> no LLM
    assert ext.calls == 0
    await chat.say("kal subah doctor se milna hai")  # no rule -> extractor
    assert ext.calls == 1 and chat.last and chat.last.route == "extractor"
    # extracted slots were validated against live options and pre-filled; only name is missing
    assert chat.state.slots["time"] == "11:00" and chat.texts[-1].startswith("Please tell me")


async def test_extractor_hallucinated_slot_is_rejected() -> None:
    ext = FakeExtractor(Extraction(intent="book_appointment", confidence=0.9,
                                   slots={"service": "gen", "date": "2026-12-25"}))  # not offered
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await chat.say("something unusual")
    assert chat.state.slots.get("service") == "gen" and "date" not in chat.state.slots


async def test_low_confidence_extraction_ignored() -> None:
    ext = FakeExtractor(Extraction(intent="book_appointment", confidence=0.3))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await chat.say("something unusual")
    assert chat.state.flow is None and "menu:book_appointment" in chat.option_ids


async def test_all_message_keys_exist_in_every_language() -> None:
    from assistant.orchestrator.flows import load_industries

    d = load_industries()["clinic"]
    for key, variants in d.messages.items():
        assert {"en", "hi", "te"} <= set(variants), key
    flow = d.flows["book_appointment"]
    keys = {s.prompt for s in flow.slots} | {flow.confirm_prompt, flow.execute.success}
    keys |= {a.message for a in flow.execute.on_error.values()}
    keys |= {f"menu_{m.id}" for m in d.menu}
    assert keys <= set(d.messages)


# ---------------------------------------------------------------- Step 5: extractor/answerer hooks
async def test_extractor_receives_current_slot_options_and_today() -> None:
    ext = FakeExtractor(Extraction(intent=None, confidence=0.9, slots={"date": "2026-11-14"}))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    await chat.say(payload="gen")
    await chat.say("saturday please")  # no rule/option match -> extractor with choices
    h = ext.hints[-1]
    assert h.slot == "date" and [o.id for o in h.options] == ["2026-11-13", "2026-11-14"]
    assert h.today == "2026-11-12 Thu" and "faq" in h.intents and "book_appointment" in h.intents
    assert chat.state.slots["date"] == "2026-11-14"  # mapped to a live option id


async def test_faq_intent_answers_from_knowledge_without_starting_a_flow() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    ans = FakeAnswerer("We open 9am-6pm.")
    orch, _, cfg = make(extractor=ext, answerer=ans)
    chat = Chat(orch, cfg)
    r = await chat.say("what time do you open")
    assert chat.texts == ["We open 9am-6pm."] and r.route == "rag" and chat.state.flow is None


async def test_unknown_answer_says_so_and_offers_human() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    orch, _, cfg = make(extractor=ext, answerer=FakeAnswerer(None))
    chat = Chat(orch, cfg)
    r = await chat.say("do you treat dinosaurs")
    assert "don't have that information" in chat.texts[0] and "menu:human" in chat.option_ids
    assert r.route == "unknown" and not r.needs_human  # offered, not forced


async def test_faq_question_mid_flow_is_answered_then_slot_reasked() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    orch, _, cfg = make(extractor=ext, answerer=FakeAnswerer("Parking is free."))
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    await chat.say(payload="gen")
    await chat.say("is parking free")
    assert chat.texts[0] == "Parking is free." and chat.texts[-1].startswith("Which date")
    assert chat.state.state == "collect_slots" and "_retries" not in chat.state.slots or \
        chat.state.slots["_retries"] == {}


async def test_extractor_flagged_complaint_goes_to_human() -> None:
    ext = FakeExtractor(Extraction(intent=None, confidence=0.9, needs_human=True))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    r = await chat.say("this is the worst service ever")
    assert r.needs_human and r.handoff_reason == "extractor_needs_human"


async def test_extractor_detected_medical_question_refused() -> None:
    ext = FakeExtractor(Extraction(intent="medical_advice", confidence=0.9))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    r = await chat.say("is it ok to take two tablets together")  # no keyword match
    assert r.needs_human and "can't give medical advice" in chat.texts[0]


async def test_romanized_telugu_corrected_by_extractor_language() -> None:
    ext = FakeExtractor(Extraction(intent="book_appointment", confidence=0.9, language="te"))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await chat.say("nenu doctor ni kalavali")  # Latin letters -> rules say 'en', LLM says 'te'
    assert chat.state.language == "te" and "విభాగానికి" in chat.texts[-1]


async def test_hints_not_requested_when_rules_resolve_everything() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    orch, _, cfg = make(extractor=ext)
    chat = Chat(orch, cfg)
    await book_with_menu(chat, "hello")
    await chat.say("yes")
    assert ext.calls == 0  # a whole booking with button taps = zero LLM calls


async def test_approved_faq_lookup_beats_the_llm() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    ans = FakeAnswerer("generated", lookup="Mon-Sat 9 to 6.")
    orch, _, cfg = make(extractor=ext, answerer=ans)
    chat = Chat(orch, cfg)
    r = await chat.say("what are your timings")
    assert chat.texts == ["Mon-Sat 9 to 6."] and r.route == "faq"
    assert ext.calls == 0 and ans.calls == 0  # no LLM extractor, no RAG generation


async def test_faq_lookup_midflow_answers_free_then_reasks_slot() -> None:
    ext = FakeExtractor(Extraction(intent="faq", confidence=0.9))
    orch, _, cfg = make(extractor=ext, answerer=FakeAnswerer("gen", lookup="Parking is free."))
    chat = Chat(orch, cfg)
    await chat.say("book appointment")
    await chat.say(payload="gen")
    await chat.say("is parking free")
    assert chat.texts[0] == "Parking is free." and chat.texts[-1].startswith("Which date")
    assert ext.calls == 0


async def test_lookup_not_tried_when_rules_already_matched() -> None:
    ans = FakeAnswerer("gen", lookup="should not be used")
    orch, _, cfg = make(answerer=ans)
    chat = Chat(orch, cfg)
    await chat.say("hello")
    assert ans.lookups == 0 and "menu:book_appointment" in chat.option_ids


import pytest as _pytest


@_pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("Ravi Kumar", True),
        ("రవి కుమార్", True),  # Telugu: vowel signs + virama are combining marks
        ("रवि कुमार", True),  # Hindi
        ("శ్రీనివాస రావు", True),
        ("Mary-Ann O'Neil", True),
        ("Dr. Rao", True),
        ("R2D2", False),  # digits
        ("", False),
        ("x", False),
        ("a b c d e f", False),  # more than 5 words
        ("what are your timings?", False),  # punctuation => a question, not a name
        ("Ravi\nKumar; DROP", False),
    ],
)
async def test_patient_name_validation_is_script_neutral(name: str, valid: bool) -> None:
    from assistant.orchestrator.engine import _Turn

    assert (_Turn._valid_name(name) is not None) is valid

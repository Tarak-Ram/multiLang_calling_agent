import pytest

from assistant.orchestrator.language import detect_language, resolve_language
from assistant.router.hours import is_open
from assistant.router.rules import RuleHit, apply_rules, normalize
from tests.engine_helpers import CFG, CLOSED_NOW, OPEN_NOW

ENABLED = ["te", "hi", "en"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello there", "en"),
        ("నమస్కారం", "te"),
        ("नमस्ते", "hi"),
        ("నాకు appointment కావాలి", "te"),  # Telugu + English code-mix -> Telugu
        ("मुझे appointment चाहिए", "hi"),
        ("12345 👍", None),
    ],
)
def test_detect_language(text: str, expected: str | None) -> None:
    assert detect_language(text, ENABLED) == expected


def test_unsupported_script_not_detected() -> None:
    assert detect_language("வணக்கம்", ENABLED) is None  # Tamil not enabled for this tenant
    assert detect_language("வணக்கம்", ENABLED + ["ta"]) == "ta"


def test_language_is_sticky_but_switchable() -> None:
    assert resolve_language(None, "ok", ENABLED, "te") == "en"
    assert resolve_language("te", "ok", ENABLED, "te") == "te"  # Latin reply doesn't flip
    assert resolve_language("te", "नमस्ते", ENABLED, "te") == "hi"  # clear switch
    assert resolve_language(None, "👍", ENABLED, "te") == "te"  # nothing detectable -> default


def test_normalize_handles_zwnj_and_punctuation() -> None:
    assert normalize("అపాయింట్‌మెంట్!!") == normalize("అపాయింట్మెంట్".replace("్మ", "్‌మ"))
    assert normalize("  Hello,   WORLD! ") == "hello world"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", RuleHit("yes")),
        ("అవును", RuleHit("yes")),
        ("नहीं", RuleHit("no")),
        ("hi", RuleHit("greeting")),
        ("hello doctor", RuleHit("greeting")),
        ("hi I want to book", RuleHit("intent", "book_appointment")),
        ("అపాయింట్‌మెంట్ కావాలి", RuleHit("intent", "book_appointment")),
        ("अपॉइंटमेंट बुक करना है", RuleHit("intent", "book_appointment")),
        ("cancel my appointment", RuleHit("intent", "cancel_appointment")),
        ("I have symptoms", RuleHit("intent", "medical_advice")),
        ("what is the weather", None),
    ],
)
def test_rules(text: str, expected: RuleHit | None) -> None:
    assert apply_rules(CFG, text, None) == expected


def test_payload_wins_over_text() -> None:
    assert apply_rules(CFG, "no", "confirm_yes") == RuleHit("payload", "confirm_yes")


def test_business_hours() -> None:
    assert is_open(CFG, OPEN_NOW) and not is_open(CFG, CLOSED_NOW)
    always = CFG.model_copy(update={"business_hours": {}})
    assert is_open(always, CLOSED_NOW)

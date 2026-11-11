import pytest

from assistant.rag.answerer import lexical_score, parse_answer
from assistant.rag.chunker import chunk_text, estimate_tokens
from assistant.rag.guards import numbers_grounded, numbers_in, trim_answer


def test_chunks_respect_budget_and_keep_sentences_whole() -> None:
    text = " ".join(f"Sentence number {i} talks about the clinic." for i in range(30))
    chunks = chunk_text(text, max_tokens=40)
    assert len(chunks) > 3 and all(estimate_tokens(c) <= 40 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_chunk_overlap_repeats_last_sentence() -> None:
    chunks = chunk_text("Aaa bbb ccc. Ddd eee fff. Ggg hhh iii. Jjj kkk lll.", max_tokens=8)
    assert len(chunks) >= 2
    assert chunks[0].split(". ")[-1].strip(".") in chunks[1]


def test_paragraphs_never_merge_and_danda_splits_hindi() -> None:
    chunks = chunk_text("पहला वाक्य है। दूसरा वाक्य है।\n\nTelugu para. Second.", max_tokens=12)
    assert not any("है।" in c and "Telugu" in c for c in chunks)
    assert any("पहला" in c for c in chunks)


def test_oversized_sentence_is_hard_split() -> None:
    chunks = chunk_text("word " * 200, max_tokens=30)
    assert len(chunks) > 3 and all(estimate_tokens(c) <= 30 for c in chunks)


def test_empty_text() -> None:
    assert chunk_text("   \n\n  ") == []


@pytest.mark.parametrize(
    ("answer", "context", "ok"),
    [
        ("It costs 500 rupees.", "General consultation costs 500 rupees.", True),
        ("It costs 600 rupees.", "General consultation costs 500 rupees.", False),  # invented
        ("We open at 9 am.", "Open 9:00 to 18:00.", True),
        ("We open at 09:00.", "Open 9 am to 6 pm.", True),
        ("Fee is 1,500.", "The fee is 1500 rupees.", True),
        ("ఫీజు 500 రూపాయలు.", "fee 500 rupees", True),  # Telugu digits/ASCII mix is fine
        ("फीस ५०० रुपये है।", "fee is 500 rupees", True),  # Devanagari digits normalised
        ("फीस ७०० रुपये है।", "fee is 500 rupees", False),
        ("No numbers here.", "anything", True),
        ("Call 040 1234.", "Phone is 040 1234", True),
        ("Call 040 9999.", "Phone is 040 1234", False),
    ],
)
def test_numbers_grounded(answer: str, context: str, ok: bool) -> None:
    assert numbers_grounded(answer, context) is ok


def test_numbers_in_normalises_forms() -> None:
    assert numbers_in("9:00 and 09:00") == {"9.0"}


def test_trim_answer_caps_sentences_and_length() -> None:
    assert trim_answer("One. Two. Three. Four.") == "One. Two. Three."
    assert len(trim_answer("x" * 900)) == 500


def test_parse_answer_variants() -> None:
    assert parse_answer('{"answer":"Yes.","confidence":0.9}') == ("Yes.", 0.9)
    assert parse_answer('```json\n{"answer":null,"confidence":0.1}\n```') == (None, 0.1)
    assert parse_answer("nonsense") == (None, 0.0)
    assert parse_answer('{"answer":"  ","confidence":"high"}') == (None, 0.0)
    assert parse_answer('{"answer":"x","confidence":5}')[1] == 1.0


def test_lexical_score() -> None:
    assert lexical_score("What are your timings?", "what are your timings") == 1.0
    assert lexical_score("what are your clinic timings", "clinic timings today") > 0.5
    assert lexical_score("book a dentist", "parking available") == 0.0
    assert lexical_score("", "x") == 0.0

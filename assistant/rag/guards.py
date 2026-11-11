"""Deterministic post-generation checks. These do not trust the LLM.

Rule: never state a price, time, quantity or phone number that is not literally in the
retrieved context. Cheap, language-independent, and catches the most damaging hallucinations.
"""
from __future__ import annotations

import re

# Telugu and Devanagari digits -> ASCII, so "९ बजे" and "9" compare equal.
_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯०१२३४५६७८९", "01234567890123456789")
_NUMBER = re.compile(r"\d+(?:[.:,]\d+)*")


def _canon(token: str) -> str:
    """'09:00' -> '9.0', '1,500' -> '1500', '6.50' -> '6.50' compared as int parts."""
    token = token.replace(",", "")
    return ".".join(str(int(p)) if p.isdigit() else p for p in re.split(r"[.:]", token))


def numbers_in(text: str) -> set[str]:
    return {_canon(m) for m in _NUMBER.findall(text.translate(_DIGITS))}


def numbers_grounded(answer: str, context: str) -> bool:
    """True if every number in `answer` also appears in `context` ('9' matches '9:00')."""
    ctx = numbers_in(context)
    for n in numbers_in(answer):
        if n in ctx or f"{n}.0" in ctx:
            continue
        if n.endswith(".0") and n[:-2] in ctx:
            continue
        return False
    return True


def trim_answer(text: str, max_sentences: int = 3, max_chars: int = 500) -> str:
    """Replies are 1-3 short sentences."""
    parts = re.split(r"(?<=[.!?।॥])\s+", text.strip())
    out = " ".join(parts[:max_sentences])
    return out[:max_chars].rstrip()

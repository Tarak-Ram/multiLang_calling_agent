"""Split documents into SHORT chunks (a few sentences) so a RAG prompt stays ~300 tokens.

Indic-aware: sentence boundaries include the danda (। ॥) as well as . ! ?.
"""
from __future__ import annotations

import math
import re

_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")


def estimate_tokens(text: str) -> int:
    """Cheap provider-neutral estimate. Indic scripts tokenize ~3x worse than English, so
    non-ASCII characters are weighted heavier. Only used for budgeting, not billing."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return max(1, math.ceil(ascii_chars / 4 + (len(text) - ascii_chars) / 1.5))


def _hard_split(sentence: str, max_tokens: int) -> list[str]:
    """A single over-long sentence: split on words."""
    parts, cur = [], ""
    for word in sentence.split():
        if cur and estimate_tokens(f"{cur} {word}") > max_tokens:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_text(text: str, max_tokens: int = 100, overlap_sentences: int = 1) -> list[str]:
    """Greedy sentence packing per paragraph; the last sentence(s) of a chunk repeat at the
    start of the next so an answer split across a boundary is still retrievable."""
    chunks: list[str] = []
    for para in _PARAGRAPH.split(text.strip()):
        sentences: list[str] = []
        for s in _SENTENCE_END.split(" ".join(para.split())):
            if not s:
                continue
            sentences.extend(_hard_split(s, max_tokens) if estimate_tokens(s) > max_tokens else [s])
        cur: list[str] = []
        for s in sentences:
            if cur and estimate_tokens(" ".join([*cur, s])) > max_tokens:
                chunks.append(" ".join(cur))
                cur = cur[-overlap_sentences:] if overlap_sentences else []
                if cur and estimate_tokens(" ".join([*cur, s])) > max_tokens:
                    cur = []  # overlap would overflow; drop it rather than exceed the budget
            cur.append(s)
        if cur:
            chunks.append(" ".join(cur))
    return chunks

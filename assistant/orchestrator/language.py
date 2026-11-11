"""Language detection by Unicode script: free, instant, handles code-mixing.

Limitation (by design, documented): romanized Telugu/Hindi ("nenu appointment kavali") looks
Latin and is detected as English here; the LLM extractor also returns `language` and the
engine switches the reply language when it disagrees (engine._Turn._extract).
"""
from __future__ import annotations

import unicodedata

# (start, end, ISO code). Devanagari is shared by hi/mr/ne: the tenant's enabled one wins.
_SCRIPTS: tuple[tuple[int, int, str], ...] = (
    (0x0C00, 0x0C7F, "te"),
    (0x0900, 0x097F, "hi"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0980, 0x09FF, "bn"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0B00, 0x0B7F, "or"),
)
_DEVANAGARI_SIBLINGS = ("hi", "mr", "ne")


def detect_language(text: str, enabled: list[str]) -> str | None:
    """Dominant NON-Latin script among enabled languages; else 'en' if Latin letters exist;
    None when there are no letters at all (digits/emoji) so the caller keeps the old value."""
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        code = next((c for lo, hi, c in _SCRIPTS if lo <= cp <= hi), None)
        if code:
            counts[code] = counts.get(code, 0) + 1
        elif unicodedata.name(ch, "").startswith("LATIN"):
            latin += 1
    if counts:
        code = max(counts, key=lambda c: counts[c])
        if code == "hi":
            code = next((c for c in _DEVANAGARI_SIBLINGS if c in enabled), code)
        return code if code in enabled else None
    return "en" if latin and "en" in enabled else None


def resolve_language(
    stored: str | None, text: str | None, enabled: list[str], default: str
) -> str:
    """First message: detect. Later: keep the stored language unless the customer clearly
    switches to another enabled non-Latin script. Latin-only replies ('ok', 'yes') never
    flip a Telugu/Hindi conversation to English."""
    detected = detect_language(text or "", enabled)
    if stored is None:
        return detected or default
    if detected and detected != "en" and detected != stored:
        return detected
    return stored

"""Router layer 1: deterministic rules. Zero LLM calls, zero cost.

Matches (in order): interactive payload ids, exact yes/no, keyword intents, greetings.
All phrases come from tenant config, so a new language needs no code.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from assistant.tenants.schema import TenantConfig

_INVISIBLE = dict.fromkeys(map(ord, "‌‍﻿"))  # ZWNJ/ZWJ vary by keyboard (Telugu)

HitKind = Literal["payload", "yes", "no", "intent", "greeting"]


@dataclass(frozen=True, slots=True)
class RuleHit:
    kind: HitKind
    value: str = ""


def normalize(text: str) -> str:
    """NFC + casefold, drop invisible joiners and punctuation, keep letters/marks/digits."""
    t = unicodedata.normalize("NFC", text).translate(_INVISIBLE).casefold()
    t = "".join(c if unicodedata.category(c)[0] in "LMN" or c.isspace() else " " for c in t)
    return " ".join(t.split())


def _contains(text: str, phrase: str) -> bool:
    """Latin phrases match at word start (so 'symptom' hits 'symptoms'); Indic = substring."""
    if phrase.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(phrase), text) is not None
    return phrase in text


def match_intent(cfg: TenantConfig, text_norm: str) -> str | None:
    """Longest matching keyword wins. Equal-length hits for different intents are ambiguous
    -> None, so the caller can fall through to the (paid) LLM extractor instead of guessing."""
    best: dict[str, int] = {}
    for lang in cfg.languages:
        for intent, phrases in lang.keywords.items():
            for p in phrases:
                pn = normalize(p)
                if pn and _contains(text_norm, pn):
                    best[intent] = max(best.get(intent, 0), len(pn))
    if not best:
        return None
    top = max(best.values())
    winners = [i for i, n in best.items() if n == top]
    return winners[0] if len(winners) == 1 else None


def apply_rules(cfg: TenantConfig, text: str | None, payload_id: str | None) -> RuleHit | None:
    """Return the first deterministic match, or None if the LLM layer is needed."""
    if payload_id:
        return RuleHit("payload", payload_id)
    t = normalize(text or "")
    if not t:
        return None
    for lang in cfg.languages:
        if t in {normalize(p) for p in lang.yes}:
            return RuleHit("yes")
        if t in {normalize(p) for p in lang.no}:
            return RuleHit("no")
    intent = match_intent(cfg, t)
    if intent:
        return RuleHit("intent", intent)
    tokens = t.split()
    for lang in cfg.languages:
        greetings = {normalize(p) for p in lang.greetings}
        if t in greetings or (len(tokens) <= 3 and tokens[0] in greetings):
            return RuleHit("greeting")
    return None

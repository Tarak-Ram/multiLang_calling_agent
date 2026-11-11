"""Router layer 2/5: LLM intent + slot extraction, JSON only, cheapest model first.

Cost design:
  * SYSTEM prompt is constant (no tenant/turn data) so provider prefix caching can apply.
  * USER block is a few compact lines: summary + missing slots + last 2-4 turns + message.
  * max_tokens is small; reasoning is never requested.
  * Escalates to the strong model only if the fast model's confidence is below the tenant's
    threshold, and only when a strong model is configured.
Accuracy design: the output is validated against what the engine allows; unknown intents and
slots are dropped, and the engine re-validates slot values against live options anyway.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from assistant.llm.base import LLMClient, LLMRequest, LLMUnavailableError
from assistant.llm.profiles import ModelResolver, StaticModels
from assistant.llm.usage import record_response
from assistant.orchestrator.types import ConvState, ExtractHints, Extraction
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.extractor")

SYSTEM_PROMPT = (
    "You classify ONE customer message for a business chat assistant. Customers write Telugu, "
    "Hindi, English or a mix, sometimes in Latin letters.\n"
    "Reply with ONLY compact JSON: "
    '{"intent":str|null,"slots":{},"confidence":0..1,"language":"te|hi|en|..","needs_human":bool,'
    '"reply":""}\n'
    "Rules: intent MUST be one of the given intents, else null. Fill slots only with values the "
    "customer actually stated; when choices are given return the choice id, else null. Resolve "
    "relative dates using 'today'. Never invent values. Medical symptoms/medicine/dosage/diagnosis "
    "questions => intent medical_advice. needs_human=true for complaints, anger or requests for a "
    "person. Leave reply empty."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_block(state: ConvState, text: str, hints: ExtractHints) -> str:
    """Compact per-turn context. Every line is optional and omitted when empty."""
    lines = [f"today={hints.today}", f"intents={','.join(hints.intents)}"]
    if state.language:
        lines.append(f"lang={state.language}")
    if state.flow:
        lines.append(f"flow={state.flow}")
    if hints.slot_names:
        lines.append(f"missing={','.join(hints.slot_names)}")
    if hints.slot and hints.options:
        choices = ";".join(f"{o.id}={o.title}" for o in hints.options)
        lines.append(f"choices[{hints.slot}]={choices}")
    filled = {k: v for k, v in state.slots.items() if not k.startswith("_")}
    if filled:
        lines.append(f"filled={json.dumps(filled, ensure_ascii=False, separators=(',', ':'))}")
    if state.summary:
        lines.append(f"summary={state.summary}")
    for role, t in state.recent[-4:]:
        lines.append(f"{role}: {t[:200]}")
    lines.append(f"message: {text[:500]}")
    return "\n".join(lines)


def parse_extraction(raw: str, hints: ExtractHints) -> Extraction | None:
    """Tolerant JSON parse + strict validation. None if unusable."""
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data: Any = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    intent = data.get("intent")
    if intent not in hints.intents:
        intent = None
    raw_slots = data.get("slots")
    slots_raw: dict[Any, Any] = raw_slots if isinstance(raw_slots, dict) else {}
    allowed = set(hints.slot_names) | ({hints.slot} if hints.slot else set())
    slots = {str(k): str(v) for k, v in slots_raw.items()
             if k in allowed and v not in (None, "", "null")}
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    lang = data.get("language")
    return Extraction(
        intent=intent, slots=slots, confidence=conf,
        language=lang if isinstance(lang, str) and 2 <= len(lang) <= 3 else None,
        needs_human=bool(data.get("needs_human", False)),
    )


class LLMExtractor:
    """Implements orchestrator.types.Extractor."""

    def __init__(
        self, llm: LLMClient, default_fast: str = "", default_strong: str = "",
        max_tokens: int = 150, *, resolver: ModelResolver | None = None,
    ) -> None:
        """`resolver` picks the models per request (runtime switching); without one the fixed
        `default_*` models are used."""
        self._llm, self._max = llm, max_tokens
        self._resolver: ModelResolver = resolver or StaticModels(default_fast, default_strong)

    async def _call(self, model: str, step: str, user: str, hints: ExtractHints,
                    fallback: str = "") -> Extraction | None:
        resp = await self._llm.complete(LLMRequest(
            model=model, system=SYSTEM_PROMPT, user=user, max_tokens=self._max,
            fallback_model=fallback or None))
        record_response(step, resp)
        parsed = parse_extraction(resp.text, hints)
        if parsed is None:
            log.warning("extractor_bad_json", extra={"model": model})
        return parsed

    async def extract(
        self, cfg: TenantConfig, state: ConvState, text: str, hints: ExtractHints
    ) -> Extraction | None:
        m = await self._resolver.resolve(cfg)
        fast, strong = m.fast, m.strong
        if not fast:
            return None  # active profile has no LLM: rules-only mode
        user = build_user_block(state, text, hints)
        try:
            ext = await self._call(fast, "extract", user, hints, m.fallback)
            low = ext is None or ext.confidence < cfg.thresholds.intent_confidence
            if low and strong and strong != fast:
                escalated = await self._call(strong, "escalate", user, hints, m.fallback)
                ext = escalated or ext
            return ext
        except LLMUnavailableError:
            log.warning("extractor_llm_unavailable")  # degrade to rules-only for this turn
            return None

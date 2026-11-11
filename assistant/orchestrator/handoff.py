"""Handoff summary for staff: deterministic (always available, zero cost, cannot hallucinate).

An optional LLM polish runs later as an OFFLINE job (jobs/handoff.py), never on the live path.
The summary is written in English for staff and may contain the customer's phone number, so it
is only ever stored on the conversation row and sent to the tenant's own staff, never logged.
"""
from __future__ import annotations

import re

_REASONS: tuple[tuple[str, str], ...] = (
    ("user_requested", "Customer asked to talk to a person"),
    ("medical_question", "Medical question: the bot gave NO advice, clinical staff must respond"),
    ("extractor_needs_human", "Customer seems unhappy or has a complaint"),
    ("unclear", "Bot could not understand the customer after several tries"),
    ("no_options", "No availability / options were found for the request"),
    ("flow_unavailable", "Customer asked for something that is not automated"),
    ("tool_unavailable", "A required system was unavailable"),
    ("tool_crashed", "A system error occurred"),
    ("tool_failed", "A system action failed"),
    ("tool_error", "A system action was rejected"),
)
MAX_TURN_CHARS = 160


def humanize_reason(reason: str | None) -> str:
    if not reason:
        return "Handed off to staff"
    for prefix, text in _REASONS:
        if reason.startswith(prefix):
            return text
    return f"Handed off ({reason})"


def build_handoff_summary(
    *,
    reason: str | None,
    customer_name: str | None,
    phone: str,
    language: str | None,
    flow_name: str | None,
    slots: dict[str, str],
    turns: list[tuple[str, str]],
    max_turns: int = 4,
) -> str:
    """Plain-text note, ~10 lines max. `slots` should already be display labels."""
    lines = [
        f"Reason: {humanize_reason(reason)}" + (f" [{reason}]" if reason else ""),
        f"Customer: {customer_name or 'unknown'} ({phone})" + (f", language: {language}"
                                                               if language else ""),
    ]
    if flow_name:
        done = ", ".join(f"{k}={v}" for k, v in slots.items() if not k.startswith("_"))
        lines.append(f"In progress: {flow_name}" + (f" ({done})" if done else ""))
    recent = [(r, t) for r, t in turns if t.strip()][-max_turns:]
    if recent:
        lines.append("Last messages:")
        for role, text in recent:
            one_line = " ".join(text.split())
            if len(one_line) > MAX_TURN_CHARS:
                one_line = one_line[:MAX_TURN_CHARS - 1] + "…"
            lines.append(f"- {role}: {one_line}")
    return "\n".join(lines)


def one_line(summary: str, limit: int = 900) -> str:
    """WhatsApp template parameters cannot contain newlines/tabs or long runs of spaces."""
    text = re.sub(r"\s*\n\s*", " | ", summary.strip())
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text[:limit]

"""PII masking helpers used by logging and anywhere data may leak into text."""
from __future__ import annotations

import re

_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s\-()]{7,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def mask_phone(phone: str) -> str:
    """Keep the first 2 and last 2 digits: '+919876543210' -> '+91********10'."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) <= 4:
        return "*" * len(digits)
    prefix = "+" if phone.strip().startswith("+") else ""
    return f"{prefix}{digits[:2]}{'*' * (len(digits) - 4)}{digits[-2:]}"


def mask_text(text: str) -> str:
    """Mask phone numbers and emails embedded in free text."""
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub(lambda m: mask_phone(m.group(0)), text)

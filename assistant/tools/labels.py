"""Localized date/time labels for option titles (WhatsApp list rows are max 24 chars)."""
from __future__ import annotations

from datetime import date, datetime

_WEEKDAYS = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "hi": ["सोम", "मंगल", "बुध", "गुरु", "शुक्र", "शनि", "रवि"],
    "te": ["సోమ", "మంగళ", "బుధ", "గురు", "శుక్ర", "శని", "ఆది"],
}
_MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "hi": ["जन", "फ़र", "मार्च", "अप्रैल", "मई", "जून", "जुल", "अग", "सित", "अक्टू", "नव", "दिस"],
    "te": ["జన", "ఫిబ్ర", "మార్చి", "ఏప్రి", "మే", "జూన్", "జూలై", "ఆగ", "సెప్టెం", "అక్టో",
           "నవం", "డిసెం"],
}


def date_label(d: date, lang: str) -> str:
    """'Fri 13 Nov' (English) / 'शुक्र 13 नव' (Hindi). Unknown languages fall back to English."""
    code = lang if lang in _WEEKDAYS else "en"
    return f"{_WEEKDAYS[code][d.weekday()]} {d.day} {_MONTHS[code][d.month - 1]}"


def time_label(dt: datetime) -> str:
    """'10:00 AM'. 12-hour with AM/PM is what Indian customers expect and is script-neutral."""
    return dt.strftime("%I:%M %p").lstrip("0")

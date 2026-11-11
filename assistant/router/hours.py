"""Business-hours check (deterministic)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from assistant.tenants.schema import TenantConfig

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def is_open(cfg: TenantConfig, now: datetime) -> bool:
    """True if `now` falls inside the tenant's hours. No hours configured = always open."""
    if not cfg.business_hours:
        return True
    local = now.astimezone(ZoneInfo(cfg.timezone))
    hours = cfg.business_hours.get(_DAYS[local.weekday()])  # type: ignore[call-overload]
    return hours is not None and hours.open <= local.time() < hours.close

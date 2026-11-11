"""Google Calendar adapter (one-way mirror of confirmed bookings) and the sync routine.

Design: Postgres decides availability; the calendar is a convenience view for staff. A booking
is mirrored by a background job, so a Google outage can never block or fail a customer's
booking. A booking is "dirty" when confirmed with no event yet, modified since last sync, or
cancelled while still having an event.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol

import httpx
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant.core.models import Booking, BookingStatus, Tenant
from assistant.tenants.schema import TenantConfig

log = logging.getLogger("assistant.calendar")
SCOPE = "https://www.googleapis.com/auth/calendar.events"
API = "https://www.googleapis.com/calendar/v3"


class CalendarAdapter(Protocol):
    async def create_event(self, calendar_id: str, summary: str, description: str,
                           start: datetime, end: datetime, timezone: str) -> str: ...

    async def update_event(self, calendar_id: str, event_id: str, start: datetime,
                           end: datetime, timezone: str) -> None: ...

    async def delete_event(self, calendar_id: str, event_id: str) -> None: ...


TokenProvider = Callable[[], Awaitable[str]]


class GoogleCalendarAdapter:
    """Google Calendar REST v3 over httpx; auth is a pluggable async token provider."""

    def __init__(self, http: httpx.AsyncClient, token_provider: TokenProvider,
                 base_url: str = API) -> None:
        self._http, self._token, self._base = http, token_provider, base_url

    async def _request(self, method: str, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        resp = await self._http.request(
            method, f"{self._base}{path}", json=json, timeout=10.0,
            headers={"Authorization": f"Bearer {await self._token()}"})
        if resp.status_code == 410 and method == "DELETE":  # already deleted: fine
            return resp
        resp.raise_for_status()
        return resp

    @staticmethod
    def _when(dt: datetime, tz: str) -> dict[str, str]:
        return {"dateTime": dt.isoformat(), "timeZone": tz}

    async def create_event(self, calendar_id: str, summary: str, description: str,
                           start: datetime, end: datetime, timezone: str) -> str:
        resp = await self._request("POST", f"/calendars/{calendar_id}/events", {
            "summary": summary, "description": description,
            "start": self._when(start, timezone), "end": self._when(end, timezone)})
        return str(resp.json()["id"])

    async def update_event(self, calendar_id: str, event_id: str, start: datetime,
                           end: datetime, timezone: str) -> None:
        await self._request("PATCH", f"/calendars/{calendar_id}/events/{event_id}", {
            "start": self._when(start, timezone), "end": self._when(end, timezone)})

    async def delete_event(self, calendar_id: str, event_id: str) -> None:
        await self._request("DELETE", f"/calendars/{calendar_id}/events/{event_id}")


def service_account_token_provider(path: str | None = None) -> TokenProvider:
    """Token provider from a Google service-account JSON file (GOOGLE_SERVICE_ACCOUNT_FILE).

    NOT exercised by the test-suite: it needs real credentials. It uses google-auth
    (optional extra `google`), imported lazily so the dependency is only needed when used.
    Share the target calendar with the service account's email so it may write events.
    """
    import asyncio

    key_file = path or os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    if not key_file:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_FILE is not set")
    from google.auth.transport.requests import Request  # type: ignore[import-not-found]
    from google.oauth2 import service_account  # type: ignore[import-not-found]

    creds = service_account.Credentials.from_service_account_file(key_file, scopes=[SCOPE])

    async def provide() -> str:
        if not creds.valid:  # refresh() is blocking I/O: keep it off the event loop
            await asyncio.to_thread(creds.refresh, Request())
        return str(creds.token)

    return provide


def _summary(b: Booking, cfg: TenantConfig) -> tuple[str, str]:
    svc = next((s for s in cfg.booking.services if s.id == b.service), None)
    name = svc.name(cfg.default_language) if svc else (b.service or "Appointment")
    return f"{name}: {b.patient_name}", f"Booking {b.id.hex[:8].upper()}"


async def sync_bookings(
    factory: async_sessionmaker[AsyncSession], calendar: CalendarAdapter, limit: int = 50,
) -> dict[str, int]:
    """Push dirty bookings to Google Calendar. Per-booking failures are logged and retried on
    the next run; they never abort the batch."""
    stats = {"created": 0, "updated": 0, "deleted": 0, "failed": 0}
    async with factory() as s:
        dirty = (await s.execute(
            select(Booking, Tenant).join(Tenant, Tenant.id == Booking.tenant_id).where(or_(
                and_(Booking.status == BookingStatus.CONFIRMED,
                     or_(Booking.calendar_event_id.is_(None),
                         Booking.calendar_synced_at.is_(None),
                         Booking.calendar_synced_at < Booking.updated_at)),
                and_(Booking.status == BookingStatus.CANCELLED,
                     Booking.calendar_event_id.is_not(None),
                     or_(Booking.calendar_synced_at.is_(None),
                         Booking.calendar_synced_at < Booking.updated_at)),
            )).order_by(Booking.updated_at).limit(limit)
        )).all()
    for booking, tenant in dirty:
        cfg = TenantConfig.model_validate(tenant.config)
        cal_id = cfg.booking.calendar_id
        if not cal_id:
            continue
        try:
            action = await _sync_one(calendar, cal_id, cfg, booking)
        except Exception:
            log.exception("calendar_sync_failed", extra={"booking": str(booking.id)})
            stats["failed"] += 1
            continue
        stats[action] += 1
        async with factory() as s, s.begin():
            row = await s.get(Booking, booking.id)
            if row is not None:
                row.calendar_event_id = booking.calendar_event_id
                # DB time (same txn instant as updated_at) so clock skew can't cause re-syncs
                row.calendar_synced_at = func.now()
    return stats


async def _sync_one(calendar: CalendarAdapter, cal_id: str, cfg: TenantConfig, b: Booking) -> str:
    if b.status == BookingStatus.CANCELLED:
        assert b.calendar_event_id
        await calendar.delete_event(cal_id, b.calendar_event_id)
        return "deleted"
    if b.calendar_event_id:
        await calendar.update_event(cal_id, b.calendar_event_id, b.starts_at, b.ends_at,
                                    cfg.timezone)
        return "updated"
    summary, desc = _summary(b, cfg)
    b.calendar_event_id = await calendar.create_event(
        cal_id, summary, desc, b.starts_at, b.ends_at, cfg.timezone)
    log.info("calendar_event_created", extra={"booking": str(b.id)})
    return "created"


"""Tenant-integration tools: order lookup, report status, returns, CRM update, send template.

Tenants point us at their own HTTPS endpoints (config `integrations`). Because the URL is
configuration we do not fully control, every call is SSRF-guarded: https only, and the host must
not resolve to a private/loopback/link-local address (blocks cloud-metadata style attacks).
Auth is a bearer token read from the env var NAMED in config, so secrets never live in the DB.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from assistant.channels.base import ChannelAdapter
from assistant.core.retry import retry_async
from assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger("assistant.tools.integrations")


class UnsafeUrlError(ValueError):
    pass


def validate_url(url: str, *, allow_private: bool = False) -> None:
    """Raise UnsafeUrlError unless `url` is https to a public host."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise UnsafeUrlError("integration URLs must be https")
    if allow_private:
        return
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(parts.hostname, parts.port or 443)}
    except socket.gaierror as exc:
        raise UnsafeUrlError("host does not resolve") from exc
    for a in addrs:
        ip = ipaddress.ip_address(str(a).split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise UnsafeUrlError("integration host resolves to a non-public address")


class IntegrationClient:
    """Calls a tenant integration. One instance per process; stateless."""

    def __init__(self, http: httpx.AsyncClient, *, allow_private: bool = False,
                 attempts: int = 2) -> None:
        self._http, self._allow_private, self._attempts = http, allow_private, attempts

    async def call(self, ctx: ToolContext, key: str, method: str, payload: dict[str, Any]
                   ) -> dict[str, Any] | None:
        """JSON response dict, or None when the integration is missing/unsafe/failing."""
        integ = ctx.config.integrations.get(key)
        if integ is None:
            log.warning("integration_not_configured", extra={"integration": key})
            return None
        try:
            validate_url(integ.url, allow_private=self._allow_private)
        except UnsafeUrlError as exc:
            log.error("integration_url_rejected", extra={"integration": key, "reason": str(exc)})
            return None
        headers = {}
        if integ.secret_env:
            token = os.environ.get(integ.secret_env)
            if not token:
                log.error("integration_secret_missing", extra={"env": integ.secret_env})
                return None
            headers["Authorization"] = f"Bearer {token}"

        async def attempt() -> httpx.Response:
            if method == "GET":
                return await self._http.get(integ.url, params=payload, headers=headers,
                                            timeout=integ.timeout_s, follow_redirects=False)
            return await self._http.post(integ.url, json=payload, headers=headers,
                                         timeout=integ.timeout_s, follow_redirects=False)

        try:
            # Only idempotent GETs are retried; a POST that timed out may have been applied.
            resp = await (retry_async(attempt, retry_on=(httpx.TransportError,),
                                      attempts=self._attempts)
                          if method == "GET" else attempt())
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except (httpx.HTTPError, ValueError):
            log.exception("integration_call_failed", extra={"integration": key})
            return None


class _ClientTool:
    group: ClassVar[str]

    def __init__(self, client: IntegrationClient) -> None:
        self.client = client


class OrderIn(BaseModel):
    order_id: str
    phone: str


class LookupOrder(_ClientTool):
    """GET integrations.orders?order_id=&phone= -> {found, status, eta?}. The tenant's system
    must verify the phone owns the order; we send it but never trust a lookup by id alone."""

    name: ClassVar[str] = "lookup_order"
    group: ClassVar[str] = "orders"
    input_model: ClassVar[type[BaseModel]] = OrderIn

    async def run(self, ctx: ToolContext, args: OrderIn) -> ToolResult:
        data = await self.client.call(ctx, "orders", "GET",
                                      {"order_id": args.order_id, "phone": args.phone})
        if data is None:
            return ToolResult(ok=False, error_code="service_error")
        if not data.get("found"):
            return ToolResult(ok=False, error_code="not_found")
        return ToolResult(ok=True, data={"status": str(data.get("status", "")),
                                         "eta": str(data.get("eta") or "-")})


class ReportIn(BaseModel):
    report_id: str
    phone: str


class ReportStatus(_ClientTool):
    name: ClassVar[str] = "report_status"
    group: ClassVar[str] = "reports"
    input_model: ClassVar[type[BaseModel]] = ReportIn

    async def run(self, ctx: ToolContext, args: ReportIn) -> ToolResult:
        data = await self.client.call(ctx, "reports", "GET",
                                      {"report_id": args.report_id, "phone": args.phone})
        if data is None:
            return ToolResult(ok=False, error_code="service_error")
        if not data.get("found"):
            return ToolResult(ok=False, error_code="not_found")
        return ToolResult(ok=True, data={"status": str(data.get("status", ""))})


class ReturnIn(BaseModel):
    order_id: str
    reason: str
    phone: str
    idempotency_key: str


class CreateReturn(_ClientTool):
    name: ClassVar[str] = "create_return"
    group: ClassVar[str] = "orders"
    input_model: ClassVar[type[BaseModel]] = ReturnIn

    async def run(self, ctx: ToolContext, args: ReturnIn) -> ToolResult:
        data = await self.client.call(ctx, "returns", "POST", args.model_dump())
        if data is None:
            return ToolResult(ok=False, error_code="service_error")
        if not data.get("ok"):
            return ToolResult(ok=False, error_code=str(data.get("error") or "rejected"))
        return ToolResult(ok=True, data={"ref": str(data.get("ref", ""))})


class CrmIn(BaseModel):
    event: str
    phone: str
    data: dict[str, str] = {}


class CrmUpdate(_ClientTool):
    """POST integrations.crm {event, phone, data}: log a booking/complaint/etc. in the tenant CRM."""

    name: ClassVar[str] = "crm_update"
    group: ClassVar[str] = "crm"
    input_model: ClassVar[type[BaseModel]] = CrmIn

    async def run(self, ctx: ToolContext, args: CrmIn) -> ToolResult:
        data = await self.client.call(ctx, "crm", "POST", args.model_dump())
        return ToolResult(ok=data is not None, error_code=None if data is not None
                          else "service_error")


class TemplateIn(BaseModel):
    purpose: str  # key in tenant config `templates`, e.g. "reminder"
    phone: str
    params: list[str] = []


class SendTemplate:
    """Send a pre-approved WhatsApp template (the only message type allowed outside 24h)."""

    name: ClassVar[str] = "send_template"
    group: ClassVar[str] = "messaging"
    input_model: ClassVar[type[BaseModel]] = TemplateIn

    def __init__(self, channel: ChannelAdapter) -> None:
        self.channel = channel

    async def run(self, ctx: ToolContext, args: TemplateIn) -> ToolResult:
        template = ctx.config.templates.get(args.purpose)
        if template is None or not ctx.account_id:
            return ToolResult(ok=False, error_code="template_not_configured")
        try:
            wamid = await self.channel.send_template(
                ctx.account_id, args.phone, template, ctx.language, args.params)
        except Exception:
            log.exception("template_send_failed")
            return ToolResult(ok=False, error_code="send_failed")
        return ToolResult(ok=True, data={"wamid": wamid})

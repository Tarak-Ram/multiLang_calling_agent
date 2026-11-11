import json
import uuid
from typing import Any

import httpx
import pytest

from assistant.tools.base import ToolContext, ToolNotEnabledError, ToolRegistry
from assistant.tools.integrations import (
    CreateReturn,
    CrmUpdate,
    IntegrationClient,
    LookupOrder,
    ReportStatus,
    SendTemplate,
    UnsafeUrlError,
    validate_url,
)
from assistant.tools.registry import build_registry
from tests.engine_helpers import CFG

PUBLIC = "https://93.184.216.34/api"  # public IP literal: no DNS needed in tests


def ctx(integrations: dict[str, Any] | None = None, **cfg_kw: Any) -> ToolContext:
    cfg = CFG.model_copy(deep=True)
    cfg.tools_enabled = ["orders", "reports", "crm", "messaging", "booking"]
    if integrations is not None:
        from assistant.tenants.schema import IntegrationConfig
        cfg.integrations = {k: IntegrationConfig(**v) for k, v in integrations.items()}
    cfg.templates.update(cfg_kw.get("templates", {}))
    return ToolContext(tenant_id=uuid.uuid4(), customer_phone="+919876543210",
                       conversation_id=uuid.uuid4(), config=cfg, language="te",
                       account_id="PNID")


def client(handler: Any) -> IntegrationClient:
    return IntegrationClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                             attempts=2)


@pytest.mark.parametrize("url", [
    "http://93.184.216.34/x",  # not https
    "https://127.0.0.1/x",  # loopback
    "https://localhost/x",
    "https://169.254.169.254/latest/meta-data",  # cloud metadata
    "https://10.1.2.3/x",  # private
    "https://192.168.0.5/x",
    "https://[::1]/x",
    "ftp://93.184.216.34/x",
    "not a url",
])
def test_ssrf_guard_rejects(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_url(url)


def test_ssrf_guard_allows_public_and_dev_override() -> None:
    validate_url(PUBLIC)
    validate_url("https://127.0.0.1/x", allow_private=True)


async def test_lookup_order_found_not_found_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORDERS_TOKEN", "s3cret")
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.params["order_id"] == "A1":
            return httpx.Response(200, json={"found": True, "status": "Shipped", "eta": "Fri"})
        return httpx.Response(200, json={"found": False})

    c = ctx({"orders": {"url": PUBLIC, "secret_env": "ORDERS_TOKEN"}})
    tool = LookupOrder(client(handler))
    ok = await tool.run(c, tool.input_model.model_validate({"order_id": "A1", "phone": "+91"}))
    assert ok.ok and ok.data == {"status": "Shipped", "eta": "Fri"}
    assert seen[0].headers["authorization"] == "Bearer s3cret"
    assert seen[0].url.params["phone"] == "+91"  # tenant system verifies ownership
    nf = await tool.run(c, tool.input_model.model_validate({"order_id": "ZZ", "phone": "+91"}))
    assert not nf.ok and nf.error_code == "not_found"


async def test_missing_config_secret_or_unsafe_url_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(req: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call out")

    tool = LookupOrder(client(never))
    args = tool.input_model.model_validate({"order_id": "A1", "phone": "+91"})
    for c in (ctx({}),  # integration not configured
              ctx({"orders": {"url": PUBLIC, "secret_env": "NOT_SET_ANYWHERE"}}),
              ctx({"orders": {"url": "https://169.254.169.254/x"}})):
        monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
        r = await tool.run(c, args)
        assert not r.ok and r.error_code == "service_error"


async def test_get_retries_post_does_not() -> None:
    gets, posts = [0], [0]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            gets[0] += 1
            raise httpx.ConnectError("boom")
        posts[0] += 1
        raise httpx.ReadTimeout("slow")

    c = ctx({"reports": {"url": PUBLIC}, "returns": {"url": PUBLIC}})
    r1 = await ReportStatus(client(handler)).run(
        c, ReportStatus.input_model.model_validate({"report_id": "R1", "phone": "+91"}))
    r2 = await CreateReturn(client(handler)).run(c, CreateReturn.input_model.model_validate(
        {"order_id": "A1", "reason": "damaged", "phone": "+91", "idempotency_key": "k"}))
    assert r1.error_code == "service_error" and gets[0] == 2
    assert r2.error_code == "service_error" and posts[0] == 1  # never replayed


async def test_return_and_crm_payloads() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        if "reason" in bodies[-1]:
            return httpx.Response(200, json={"ok": True, "ref": "RT-9"})
        return httpx.Response(200, json={"ok": True})

    c = ctx({"returns": {"url": PUBLIC}, "crm": {"url": PUBLIC}})
    ret = await CreateReturn(client(handler)).run(c, CreateReturn.input_model.model_validate(
        {"order_id": "A1", "reason": "damaged", "phone": "+91", "idempotency_key": "k"}))
    assert ret.ok and ret.data["ref"] == "RT-9" and bodies[0]["idempotency_key"] == "k"
    crm = await CrmUpdate(client(handler)).run(c, CrmUpdate.input_model.model_validate(
        {"event": "complaint", "phone": "+91", "data": {"k": "v"}}))
    assert crm.ok

    def rejects(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "not_eligible"})

    no = await CreateReturn(client(rejects)).run(c, CreateReturn.input_model.model_validate(
        {"order_id": "A1", "reason": "x" * 3, "phone": "+91", "idempotency_key": "k"}))
    assert no.error_code == "not_eligible"


async def test_http_error_status_is_service_error() -> None:
    c = ctx({"reports": {"url": PUBLIC}})
    r = await ReportStatus(client(lambda req: httpx.Response(500))).run(
        c, ReportStatus.input_model.model_validate({"report_id": "R1", "phone": "+91"}))
    assert r.error_code == "service_error"


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[Any, ...]] = []

    async def send_template(self, *a: Any) -> str:
        self.sent.append(a)
        return "wamid.T"


async def test_send_template_uses_tenant_template_and_language() -> None:
    ch = FakeChannel()
    tool = SendTemplate(ch)  # type: ignore[arg-type]
    c = ctx(templates={"reminder": "clinic_reminder_v1"})
    r = await tool.run(c, tool.input_model.model_validate(
        {"purpose": "reminder", "phone": "+919876543210", "params": ["Ravi"]}))
    assert r.ok and ch.sent == [("PNID", "+919876543210", "clinic_reminder_v1", "te", ["Ravi"])]
    bad = await tool.run(c, tool.input_model.model_validate(
        {"purpose": "not_configured", "phone": "+91"}))
    assert bad.error_code == "template_not_configured"


async def test_registry_enforces_tenant_enablement_and_input_validation() -> None:
    reg: ToolRegistry = build_registry(httpx.AsyncClient(), FakeChannel())  # type: ignore[arg-type]
    c = ctx({"orders": {"url": PUBLIC}})
    c.config.tools_enabled = ["booking"]  # orders group NOT enabled for this tenant
    with pytest.raises(ToolNotEnabledError):
        await reg.run("lookup_order", c, {"order_id": "A1", "phone": "+91"})
    with pytest.raises(ToolNotEnabledError):
        await reg.run("does_not_exist", c, {})
    c.config.tools_enabled = ["orders"]
    with pytest.raises(Exception, match="order_id"):  # typed input: missing field -> error
        await reg.run("lookup_order", c, {"phone": "+91"})

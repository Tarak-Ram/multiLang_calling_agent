import json

import httpx
import pytest

from assistant.channels.whatsapp.client import WhatsAppClient, WhatsAppError


def make(handler: httpx.MockTransport, attempts: int = 3) -> WhatsAppClient:
    return WhatsAppClient(httpx.AsyncClient(transport=handler), "tok", attempts=attempts,
                          base_delay=0)


async def test_send_text_payload_and_auth() -> None:
    seen: dict = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], \
            json.loads(req.content)
        return httpx.Response(200, json={"messages": [{"id": "wamid.OUT"}]})

    wamid = await make(httpx.MockTransport(h)).send_text("PNID", "+919876543210", "hello")
    assert wamid == "wamid.OUT" and seen["auth"] == "Bearer tok"
    assert seen["url"].endswith("/v21.0/PNID/messages")
    assert seen["body"]["to"] == "919876543210" and seen["body"]["text"]["body"] == "hello"


async def test_template_params() -> None:
    seen: dict = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"messages": [{"id": "w"}]})

    await make(httpx.MockTransport(h)).send_template("P", "+91", "remind_v1", "te", ["Ravi", "5pm"])
    assert seen["template"]["language"]["code"] == "te"
    assert [p["text"] for p in seen["template"]["components"][0]["parameters"]] == ["Ravi", "5pm"]


async def test_retries_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503 if calls["n"] == 1 else 429)
        return httpx.Response(200, json={"messages": [{"id": "ok"}]})

    assert await make(httpx.MockTransport(h)).send_text("P", "+91", "x") == "ok"
    assert calls["n"] == 3


async def test_4xx_not_retried() -> None:
    calls = {"n": 0}

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"code": 131047}})

    with pytest.raises(WhatsAppError) as e:
        await make(httpx.MockTransport(h)).send_text("P", "+91", "x")
    assert e.value.status == 400 and calls["n"] == 1


async def test_gives_up_after_attempts() -> None:
    calls = {"n": 0}

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    with pytest.raises(httpx.ConnectError):
        await make(httpx.MockTransport(h), attempts=2).send_text("P", "+91", "x")
    assert calls["n"] == 2


async def test_interactive_buttons_vs_list_and_truncation() -> None:
    from assistant.channels.base import Option

    sent: list[dict] = []

    def h(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"messages": [{"id": "w"}]})

    c = make(httpx.MockTransport(h))
    await c.send_options("P", "+91", "pick", [Option(id="a", title="x" * 40),
                                                Option(id="b", title="B")], "Choose")
    btn = sent[0]["interactive"]
    assert btn["type"] == "button" and btn["action"]["buttons"][0]["reply"]["id"] == "a"
    assert len(btn["action"]["buttons"][0]["reply"]["title"]) == 20

    many = [Option(id=str(i), title=f"Row {i}") for i in range(12)]
    await c.send_options("P", "+91", "pick", many, "Choose")
    lst = sent[1]["interactive"]
    assert lst["type"] == "list" and len(lst["action"]["sections"][0]["rows"]) == 10

    await c.send_options("P", "+91", "just text", [], "")
    assert sent[2]["type"] == "text"

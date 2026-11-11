from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from assistant.llm.base import LLMError, LLMRequest, LLMResponse, LLMUnavailableError
from assistant.llm.litellm_client import LiteLLMClient
from assistant.llm.resilience import CircuitBreaker, ResilientLLM
from assistant.llm.usage import collect_usage, record_response


def fake_resp(text: str = '{"intent":null}', p: int = 120, c: int = 20, cached: int = 100) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"  {text}\n"))],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


def req(model: str = "provider/small") -> LLMRequest:
    return LLMRequest(model=model, system="SYS", user="USER")


async def test_client_params_and_usage_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0012)
    seen: dict[str, Any] = {}

    async def fn(**kw: Any) -> Any:
        seen.update(kw)
        return fake_resp()

    r = await LiteLLMClient(completion_fn=fn).complete(req())
    # cost rules: capped output, deterministic, JSON mode, no provider-side retries, no thinking
    assert seen["max_tokens"] == 150 and seen["temperature"] == 0
    assert seen["response_format"] == {"type": "json_object"} and seen["num_retries"] == 0
    assert not any("think" in k or "reason" in k for k in seen)
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}  # plain, stable prefix
    assert (r.text, r.input_tokens, r.cached_input_tokens, r.output_tokens) == (
        '{"intent":null}', 120, 100, 20)
    assert r.cost_usd == pytest.approx(0.0012)


async def test_cache_control_marker_only_for_claude_family() -> None:
    seen: list[dict[str, Any]] = []

    async def fn(**kw: Any) -> Any:
        seen.append(kw)
        return fake_resp()

    c = LiteLLMClient(completion_fn=fn)
    await c.complete(req("anthropic/some-claude-model"))
    await c.complete(req("openai/some-model"))
    sys0 = seen[0]["messages"][0]["content"]
    assert isinstance(sys0, list) and sys0[0]["cache_control"] == {"type": "ephemeral"}
    assert seen[1]["messages"][0]["content"] == "SYS"


async def test_unknown_cost_does_not_fail_the_call() -> None:
    async def fn(**kw: Any) -> Any:
        return fake_resp()

    r = await LiteLLMClient(completion_fn=fn).complete(req("custom/unpriced"))
    assert r.cost_usd == 0.0 and r.output_tokens == 20


async def test_error_mapping_transient_vs_permanent() -> None:
    async def rate(**kw: Any) -> Any:
        raise litellm.RateLimitError("slow down", "p", "m")

    async def bad(**kw: Any) -> Any:
        raise ValueError("bad request")

    with pytest.raises(LLMError) as t:
        await LiteLLMClient(completion_fn=rate).complete(req())
    assert t.value.transient
    with pytest.raises(LLMError) as p:
        await LiteLLMClient(completion_fn=bad).complete(req())
    assert not p.value.transient


class Scripted:
    """Inner client whose behaviour per model is a list of outcomes (Exception or text)."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script, self.calls = script, []

    async def complete(self, r: LLMRequest) -> LLMResponse:
        self.calls.append(r.model)
        out = self.script[r.model].pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=out, model=r.model)


TRANSIENT = LLMError("503", transient=True)


async def test_retry_then_success_on_same_model() -> None:
    inner = Scripted({"a": [TRANSIENT, "ok"]})
    r = await ResilientLLM(inner, attempts=2).complete(req("a"))
    assert r.text == "ok" and inner.calls == ["a", "a"]


async def test_fallback_model_used_when_primary_fails() -> None:
    inner = Scripted({"a": [TRANSIENT, TRANSIENT], "b": ["from-fallback"]})
    r = await ResilientLLM(inner, fallback_model="b", attempts=2).complete(req("a"))
    assert r.text == "from-fallback" and r.model == "b"


async def test_permanent_error_not_retried_and_does_not_trip_breaker() -> None:
    inner = Scripted({"a": [LLMError("auth", transient=False)] * 10})
    llm = ResilientLLM(inner, attempts=3, threshold=2)
    for _ in range(4):
        with pytest.raises(LLMUnavailableError):
            await llm.complete(req("a"))
    assert inner.calls == ["a"] * 4 and llm.breakers["a"].state == "closed"


async def test_breaker_opens_then_skips_provider_then_recovers() -> None:
    now = [0.0]
    inner = Scripted({"a": [TRANSIENT] * 4 + ["recovered"], "b": ["fb"] * 10})
    llm = ResilientLLM(inner, fallback_model="b", attempts=1, threshold=3, recovery_s=30,
                       clock=lambda: now[0])
    for _ in range(3):
        await llm.complete(req("a"))  # each fails on a, served by b
    assert llm.breakers["a"].state == "open"
    inner.calls.clear()
    await llm.complete(req("a"))
    assert inner.calls == ["b"]  # open breaker: primary not even attempted
    now[0] = 31.0  # half-open: exactly one probe
    assert llm.breakers["a"].state == "half_open"
    r = await llm.complete(req("a"))  # probe fails (4th TRANSIENT) -> re-open, fallback serves
    assert r.model == "b" and llm.breakers["a"].state == "open"
    now[0] = 70.0
    r = await llm.complete(req("a"))  # probe succeeds
    assert r.text == "recovered" and llm.breakers["a"].state == "closed"


async def test_everything_down_raises_unavailable() -> None:
    inner = Scripted({"a": [TRANSIENT] * 2, "b": [TRANSIENT] * 2})
    with pytest.raises(LLMUnavailableError):
        await ResilientLLM(inner, fallback_model="b", attempts=2).complete(req("a"))


def test_half_open_allows_single_probe() -> None:
    t = [0.0]
    b = CircuitBreaker(threshold=1, recovery_s=10, clock=lambda: t[0])
    b.failure()
    assert not b.allow()
    t[0] = 11
    assert b.allow() and not b.allow()


def test_usage_collection_is_scoped() -> None:
    record_response("extract", LLMResponse(text="", model="m"))  # outside: ignored, no error
    with collect_usage() as recs:
        record_response("extract", LLMResponse(text="", model="m", input_tokens=5, cost_usd=0.1))
    assert len(recs) == 1 and recs[0].input_tokens == 5


# ------------------------------------------------------------------------------------ traces
def test_trace_shape_and_langfuse_batch() -> None:
    import uuid as _uuid
    from datetime import UTC, datetime

    from assistant.llm.trace import build_trace, to_langfuse_batch
    from assistant.llm.usage import UsageRecord

    usage = [UsageRecord("embed", "emb", 20, 0, 0, 80, 0.00001),
             UsageRecord("extract", "fast", 200, 100, 20, 400, 0.0003),
             UsageRecord("escalate", "strong", 500, 0, 60, 1500, 0.004)]
    t = build_trace(tenant_id=_uuid.uuid4(), conversation_id=_uuid.uuid4(),
                    message_id=_uuid.uuid4(), route="extractor", language="te", latency_ms=2100,
                    usage=usage, needs_human=False,
                    started_at=datetime(2026, 11, 12, tzinfo=UTC))
    assert t["llm_calls"] == 2  # embedding is billed but is not an LLM completion
    assert (t["input_tokens"], t["output_tokens"], t["cached_input_tokens"]) == (720, 80, 100)
    assert t["cost_usd"] == pytest.approx(0.00431)
    assert "text" not in str(t).lower().replace("context", "")  # no message content in traces
    batch = to_langfuse_batch(t)["batch"]
    assert batch[0]["type"] == "trace-create" and batch[0]["body"]["sessionId"] == t["conversation_id"]
    gens = batch[1:]
    assert [g["body"]["name"] for g in gens] == ["embed", "extract", "escalate"]
    assert all(g["body"]["traceId"] == t["trace_id"] for g in gens)
    assert gens[2]["body"]["usage"] == {"input": 500, "output": 60, "total": 560, "unit": "TOKENS"}
    assert gens[1]["body"]["startTime"] == gens[0]["body"]["endTime"]  # spans laid end to end


async def test_langfuse_exporter_posts_with_basic_auth_and_swallows_failures() -> None:
    import httpx as _httpx

    from assistant.llm.trace import LangfuseExporter

    seen: list[_httpx.Request] = []

    def ok(req: _httpx.Request) -> _httpx.Response:
        seen.append(req)
        return _httpx.Response(207, json={})

    trace = {"trace_id": "t1", "conversation_id": "c", "tenant_id": "x", "route": "rule",
             "language": "en", "needs_human": False, "latency_ms": 5,
             "started_at": "2026-11-12T00:00:00+00:00", "spans": []}
    ex = LangfuseExporter(_httpx.AsyncClient(transport=_httpx.MockTransport(ok)),
                          "https://lf.example/", "pk", "sk")
    await ex.export(trace)
    assert str(seen[0].url) == "https://lf.example/api/public/ingestion"
    assert seen[0].headers["authorization"].startswith("Basic ")

    def boom(req: _httpx.Request) -> _httpx.Response:
        raise _httpx.ConnectError("down")

    bad = LangfuseExporter(_httpx.AsyncClient(transport=_httpx.MockTransport(boom)),
                           "https://lf.example", "pk", "sk")
    await bad.export(trace)  # must not raise: a trace failure can never affect a customer turn

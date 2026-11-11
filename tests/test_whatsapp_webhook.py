import json
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from pydantic import SecretStr

from assistant.api.main import create_app
from assistant.channels.whatsapp.signature import compute_signature
from assistant.core.config import Settings
from tests.whatsapp_fixtures import PNID, envelope, text_msg

SECRET = "app-secret"
TENANT = uuid.uuid4()


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple[Any, ...], str | None]] = []
        self.fail = False

    async def enqueue(self, name: str, *args: Any, job_id: str | None = None) -> None:
        if self.fail:
            raise RuntimeError("down")
        self.jobs.append((name, args, job_id))

    async def close(self) -> None: ...


@pytest.fixture
def client() -> Iterator[tuple[TestClient, FakeQueue]]:
    app = create_app()
    with TestClient(app) as c:
        q = FakeQueue()
        app.state.queue = q
        app.state.redis = FakeAsyncRedis(decode_responses=True)
        app.state.settings = Settings(
            _env_file=None, whatsapp_app_secret=SecretStr(SECRET),
            whatsapp_verify_token=SecretStr("vt"), rate_limit_per_minute=3,
        )

        async def resolver(pnid: str) -> uuid.UUID | None:
            return TENANT if pnid == PNID else None

        app.state.tenant_resolver = resolver
        yield c, q


def post(c: TestClient, payload: dict[str, Any], secret: str = SECRET) -> Any:
    raw = json.dumps(payload).encode()
    return c.post("/webhooks/whatsapp", content=raw,
                  headers={"x-hub-signature-256": compute_signature(secret, raw)})


def test_verify_handshake(client: tuple[TestClient, FakeQueue]) -> None:
    c, _ = client
    ok = c.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "vt",
                                             "hub.challenge": "1234"})
    assert ok.status_code == 200 and ok.text == "1234"
    bad = c.get("/webhooks/whatsapp", params={"hub.mode": "subscribe", "hub.verify_token": "x",
                                              "hub.challenge": "1"})
    assert bad.status_code == 403


def test_bad_signature_rejected_and_nothing_enqueued(client: tuple[TestClient, FakeQueue]) -> None:
    c, q = client
    assert post(c, envelope([text_msg()]), secret="wrong").status_code == 401
    assert c.post("/webhooks/whatsapp", content=b"{}").status_code == 401
    assert q.jobs == []


def test_valid_message_enqueued_once_and_duplicate_dropped(
    client: tuple[TestClient, FakeQueue],
) -> None:
    c, q = client
    assert post(c, envelope([text_msg("wamid.X")])).json()["accepted"] == 1
    assert post(c, envelope([text_msg("wamid.X")])).json()["accepted"] == 0  # Meta retry
    assert len(q.jobs) == 1
    name, (tenant, body), job_id = q.jobs[0]
    assert name == "process_inbound" and tenant == str(TENANT)
    assert job_id == f"in:{TENANT}:wamid.X" and "wamid.X" in body


def test_unknown_tenant_dropped_with_200(client: tuple[TestClient, FakeQueue]) -> None:
    c, q = client
    r = post(c, envelope([text_msg()], pnid="unknown"))
    assert r.status_code == 200 and q.jobs == []


def test_rate_limit_per_customer(client: tuple[TestClient, FakeQueue]) -> None:
    c, q = client
    for i in range(6):
        post(c, envelope([text_msg(f"wamid.{i}")]))
    assert len(q.jobs) == 3  # limit configured as 3/min


def test_enqueue_failure_returns_503_and_allows_meta_retry(
    client: tuple[TestClient, FakeQueue],
) -> None:
    c, q = client
    q.fail = True
    assert post(c, envelope([text_msg("wamid.R")])).status_code == 503
    q.fail = False
    assert post(c, envelope([text_msg("wamid.R")])).json()["accepted"] == 1

"""Admin API for switching models at runtime (real app + real DB)."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.api.main import create_app
from assistant.core.config import get_settings
from tests.conftest import TEST_DB
from tests.test_model_switch import Factory, ProbeLLM

KEY = "admin-secret"


class Queue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, str | None]] = []

    async def enqueue(self, name: str, *a: Any, job_id: str | None = None) -> None:
        self.jobs.append((name, job_id))

    async def close(self) -> None: ...


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("ADMIN_API_KEY", KEY)
    monkeypatch.setenv("LLM_FAST_MODEL", "prov/a-fast")
    monkeypatch.setenv("LLM_STRONG_MODEL", "prov/a-strong")
    monkeypatch.setenv("EMBEDDING_MODEL", "emb-a")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def client(bad: set[str] | None = None) -> TestClient:
    c = TestClient(create_app(), headers={"x-admin-key": KEY})
    return c


def wire(http: TestClient, bad: set[str] | None = None) -> Queue:
    q = Queue()
    http.app.state.queue = q  # type: ignore[attr-defined]
    http.app.state.probe_llm = ProbeLLM(bad)  # type: ignore[attr-defined]
    http.app.state.embedders = Factory()  # type: ignore[attr-defined]
    return q


async def test_overview_seeds_from_env_and_requires_key(session: AsyncSession, env: None) -> None:
    with TestClient(create_app()) as anon:
        assert anon.get("/admin/models").status_code == 401
    with client() as http:
        wire(http)
        body = http.get("/admin/models").json()
        assert body["active"] == "default"
        assert body["embedding"] == {"effective_model": "emb-a", "dim": 1024}
        names = {p["name"]: p for p in body["profiles"]}
        assert set(names) == {"default", "rules-only"} and names["default"]["active"] is True
        assert names["default"]["fast"] == "prov/a-fast"


async def test_create_switch_and_delete_profiles(session: AsyncSession, env: None) -> None:
    with client() as http:
        q = wire(http)
        saved = http.put("/admin/models/profiles/b", json={
            "fast": "prov/b-fast", "strong": "prov/b-strong", "fallback": "prov/b-backup",
            "embedding_model": "emb-a", "notes": "cheaper provider"})
        assert saved.status_code == 200
        # keys can never be stored: unknown fields are rejected, not silently ignored
        leak = http.put("/admin/models/profiles/x", json={"fast": "p/m", "api_key": "sk-secret"})
        assert leak.status_code == 422
        wrong_dim = http.put("/admin/models/profiles/y", json={"embedding_model": "m",
                                                                "embedding_dim": 768})
        assert wrong_dim.status_code == 422 and "1024" in wrong_dim.text
        assert http.put("/admin/models/profiles/Bad Name", json={}).status_code == 422

        assert http.post("/admin/models/test", json={"profile": "b"}).json()["ok"] is True
        assert http.post("/admin/models/test", json={"profile": "zzz"}).status_code == 404

        sw = http.post("/admin/models/activate", json={"profile": "b"}).json()
        assert sw["active"] == "b" and sw["previous"] == "default" and sw["embedding"] == "unchanged"
        assert {p["role"] for p in sw["probes"]} == {"fast", "strong", "fallback", "embedding"}
        assert q.jobs == []  # same embedding model: no migration needed
        assert http.get("/admin/models").json()["active"] == "b"

        assert http.delete("/admin/models/profiles/b").status_code == 409  # active
        assert http.delete("/admin/models/profiles/default").status_code == 409  # built-in
        http.post("/admin/models/activate", json={"profile": "rules-only"})
        assert http.delete("/admin/models/profiles/b").status_code == 200
        assert http.delete("/admin/models/profiles/b").status_code == 404


async def test_broken_provider_is_refused_with_details_and_nothing_changes(
    session: AsyncSession, env: None,
) -> None:
    with client() as http:
        wire(http, bad={"prov/c-fast"})
        http.put("/admin/models/profiles/c", json={"fast": "prov/c-fast", "strong": "prov/c-strong"})
        r = http.post("/admin/models/activate", json={"profile": "c"})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "nothing was changed" in detail["error"]
        failed = [p for p in detail["probes"] if not p["ok"]]
        assert [p["model"] for p in failed] == ["prov/c-fast"] and "API key" in failed[0]["error"]
        assert http.get("/admin/models").json()["active"] == "default"
        forced = http.post("/admin/models/activate", json={"profile": "c", "verify": False})
        assert forced.status_code == 200 and forced.json()["active"] == "c"  # explicit override
        assert http.post("/admin/models/activate", json={"profile": "nope"}).status_code == 404


async def test_embedding_change_schedules_a_background_migration(
    session: AsyncSession, env: None,
) -> None:
    with client() as http:
        q = wire(http)
        http.put("/admin/models/profiles/newemb", json={"fast": "prov/a-fast",
                                                         "embedding_model": "emb-new"})
        r = http.post("/admin/models/activate", json={"profile": "newemb"}).json()
        assert r["embedding"] == "migration_needed"
        assert len(q.jobs) == 1 and q.jobs[0][0] == "job_embedding_migration"
        ov = http.get("/admin/models").json()
        assert ov["active"] == "newemb"
        assert ov["embedding"]["effective_model"] == "emb-a"  # old model serves until swap
        assert ov["migration"]["state"] == "pending" and ov["migration"]["target_model"] == "emb-new"
        # still only PENDING: a newer choice simply re-targets it (a second job is queued; the
        # first one to run reads the latest target, the other finds nothing left to do)
        http.put("/admin/models/profiles/other", json={"fast": "p/m", "embedding_model": "emb-z"})
        assert http.post("/admin/models/activate", json={"profile": "other"}).status_code == 200
        assert http.get("/admin/models").json()["migration"]["target_model"] == "emb-z"
        assert len(q.jobs) == 2

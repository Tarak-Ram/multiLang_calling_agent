from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from assistant.api.main import create_app


class _Redis:
    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None: ...


def test_live_and_request_id() -> None:
    with TestClient(create_app()) as c:
        r = c.get("/health/live")
        assert r.status_code == 200 and "x-request-id" in r.headers


def test_ready_503_when_deps_down() -> None:
    with TestClient(create_app()) as c:
        c.app.state.redis = _Redis()
        # a Postgres that is definitely down (nothing listens on port 1), whatever the env says
        c.app.state.engine = create_async_engine("postgresql+asyncpg://x:x@127.0.0.1:1/x")
        r = c.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["redis"] is True

import pytest

from assistant.core.config import Settings


def test_env_overrides_and_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "s3cret")
    s = Settings(_env_file=None)
    assert s.is_prod and s.whatsapp_app_secret.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(s)

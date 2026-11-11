from pathlib import Path

import pytest
from pydantic import ValidationError

from assistant.tenants.loader import parse_tenant_yaml
from assistant.tenants.schema import TenantConfig

SAMPLE = Path(__file__).parents[1] / "assistant/tenants/templates/clinic_sample.yaml"


def test_sample_yaml_is_valid_and_has_three_launch_languages() -> None:
    t = parse_tenant_yaml(SAMPLE)
    assert {x.code for x in t.config.languages} == {"te", "hi", "en"}
    assert len(t.faqs) == 3


def test_default_language_must_be_configured() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(default_language="ta", languages=[{"code": "en", "name": "English"}])


def test_duplicate_language_rejected() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(languages=[{"code": "en", "name": "a"}, {"code": "en", "name": "b"}])


def test_adding_language_needs_only_config() -> None:
    cfg = TenantConfig(
        default_language="ta",
        languages=[{"code": "ta", "name": "Tamil", "greetings": ["வணக்கம்"]}],
    )
    assert cfg.languages[0].code == "ta"


def test_sample_yes_no_phrases_not_swallowed_by_yaml_booleans() -> None:
    # YAML 1.1 turns unquoted `yes:`/`no:` keys into booleans; they must be quoted.
    for lang in parse_tenant_yaml(SAMPLE).config.languages:
        assert lang.yes and lang.no and lang.greetings, lang.code

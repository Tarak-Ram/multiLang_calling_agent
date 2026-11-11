"""Industry/flow definitions (YAML) and message lookup. Config, not code."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from assistant.tenants.schema import TenantConfig

INDUSTRIES_DIR = Path(__file__).parent / "industries"


class SlotDef(BaseModel):
    name: str
    prompt: str  # message key
    # option: must match a tool-provided choice | name: a person's name | text: free text |
    # token: an id/reference like an order or report number
    validate_as: Literal["option", "name", "text", "token"] = "option"
    options_tool: str | None = None  # tool returning data["options"]; value must be one of them
    revalidate: bool = False  # re-check the chosen option is still offered before confirming
    on_empty: str | None = None  # message key if the tool returns no options (else: human)


class ErrorAction(BaseModel):
    message: str
    clear: list[str] = []
    # retry: re-ask the cleared slots | menu: back to the main menu | handoff: to a human
    then: Literal["retry", "menu", "handoff"] = "retry"


class ExecuteDef(BaseModel):
    tool: str
    success: str
    on_error: dict[str, ErrorAction] = {}


class FlowDef(BaseModel):
    slots: list[SlotDef]
    confirm_prompt: str | None  # None = read-only flow, execute as soon as slots are filled
    execute: ExecuteDef


class MenuItem(BaseModel):
    id: str
    flow: str | None = None
    action: str | None = None


class IndustryDef(BaseModel):
    industry: str
    menu: list[MenuItem]
    flows: dict[str, FlowDef]
    messages: dict[str, dict[str, str]]

    def text(self, cfg: TenantConfig, key: str, lang: str) -> str:
        """Tenant override -> industry text; language -> tenant default -> English."""
        for table in (cfg.messages, self.messages):
            variants = table.get(key)
            if variants:
                for code in (lang, cfg.default_language, "en"):
                    if code in variants:
                        return variants[code]
        return key  # visible in dev if a key is missing; covered by a test


def _parse_flow(raw: dict) -> FlowDef:  # type: ignore[type-arg]
    slots = [
        SlotDef(
            name=s["name"], prompt=s["prompt"], validate_as=s.get("validate", "option"),
            options_tool=s.get("options_tool"), revalidate=s.get("revalidate", False),
            on_empty=s.get("on_empty"),
        )
        for s in raw["slots"]
    ]
    return FlowDef(
        slots=slots, confirm_prompt=(raw.get("confirm") or {}).get("prompt"),
        execute=ExecuteDef.model_validate(raw["execute"]),
    )


def load_industry(path: Path, common: dict[str, dict[str, str]] | None = None) -> IndustryDef:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return IndustryDef(
        industry=raw["industry"],
        menu=[MenuItem.model_validate(m) for m in raw["menu"]],
        flows={k: _parse_flow(v) for k, v in raw["flows"].items()},
        messages={**(common or {}), **raw["messages"]},  # industry texts override shared ones
    )


def load_industries(directory: Path = INDUSTRIES_DIR) -> dict[str, IndustryDef]:
    """Every *.yaml is one industry, except `_common.yaml` (texts shared by all of them)."""
    common_file = directory / "_common.yaml"
    common = (yaml.safe_load(common_file.read_text(encoding="utf-8"))["messages"]
              if common_file.exists() else {})
    defs = [load_industry(p, common) for p in sorted(directory.glob("*.yaml"))
            if not p.name.startswith("_")]
    return {d.industry: d for d in defs}

"""Tool layer: typed, validated, tenant-gated functions the orchestrator may call.

The LLM never calls tools directly. The state machine does, with validated arguments.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from assistant.tenants.schema import TenantConfig


@dataclass(slots=True)
class ToolEnv:
    """Per-turn runtime handles the orchestrator passes to tools (not persisted state).

    `session` is the SAME transaction that saves the conversation state, so a booking and the
    state transition commit or roll back together."""

    session: Any = None  # sqlalchemy AsyncSession
    account_id: str | None = None  # channel account (WhatsApp phone_number_id)


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: uuid.UUID
    customer_phone: str
    conversation_id: uuid.UUID
    config: TenantConfig
    language: str = "en"
    session: Any = None
    account_id: str | None = None


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] = {}
    error_code: str | None = None  # machine-readable, mapped by the flow's on_error


class ToolNotEnabledError(Exception):
    """Tool missing from the registry or its group is not enabled for this tenant."""


class Tool(Protocol):
    name: ClassVar[str]
    group: ClassVar[str]  # matched against tenant.config.tools_enabled
    input_model: ClassVar[type[BaseModel]]

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult: ...


class ToolRegistry:
    """Name -> tool. Enforces per-tenant enablement and input validation on every call."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    async def run(self, name: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None or tool.group not in ctx.config.tools_enabled:
            raise ToolNotEnabledError(name)
        return await tool.run(ctx, tool.input_model.model_validate(args))

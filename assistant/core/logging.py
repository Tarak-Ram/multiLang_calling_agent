"""Structured JSON logging with automatic PII masking and request/conversation context."""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from assistant.core.pii import mask_text

# Propagated across awaits; set by middleware / orchestrator.
log_context: ContextVar[dict[str, Any]] = ContextVar("log_context")

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def bind_context(**kwargs: Any) -> None:
    """Add fields (request_id, tenant_id, conversation_id...) to every following log line."""
    log_context.set({**log_context.get({}), **kwargs})


def clear_context() -> None:
    log_context.set({})


class JsonFormatter(logging.Formatter):
    """One JSON object per line; masks phones/emails in message and string extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": mask_text(record.getMessage()),
            **log_context.get({}),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = mask_text(val) if isinstance(val, str) else val
        if record.exc_info:
            payload["exc"] = mask_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure root logging once at process start (api and worker)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

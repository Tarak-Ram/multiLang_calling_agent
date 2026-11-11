import json
import logging

from assistant.core.logging import JsonFormatter, bind_context, clear_context


def test_json_log_masks_pii_and_includes_context() -> None:
    clear_context()
    bind_context(request_id="r1")
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "from +919876543210", None, None)
    rec.phone = "+919876543210"
    out = json.loads(JsonFormatter().format(rec))
    assert out["request_id"] == "r1"
    assert "9876543210" not in json.dumps(out)

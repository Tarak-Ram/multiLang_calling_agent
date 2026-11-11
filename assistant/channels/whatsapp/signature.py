"""Webhook authenticity: X-Hub-Signature-256 = 'sha256=' + HMAC_SHA256(app_secret, raw_body)."""
from __future__ import annotations

import hashlib
import hmac


def compute_signature(app_secret: str, raw_body: bytes) -> str:
    return "sha256=" + hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()


def verify_signature(app_secret: str, raw_body: bytes, header: str | None) -> bool:
    """Constant-time check over the RAW bytes. Fails closed if the secret is unset."""
    if not app_secret or not header:
        return False
    return hmac.compare_digest(compute_signature(app_secret, raw_body), header)

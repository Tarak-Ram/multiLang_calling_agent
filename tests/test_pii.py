from assistant.core.pii import mask_phone, mask_text


def test_mask_phone_keeps_prefix_and_tail() -> None:
    assert mask_phone("+919876543210") == "+91********10"


def test_mask_text_phone_and_email() -> None:
    out = mask_text("call +91 98765 43210 or a.b@x.com")
    assert "98765" not in out and "a.b@x.com" not in out and "[email]" in out

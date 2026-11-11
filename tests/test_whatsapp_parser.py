from assistant.channels.whatsapp.parser import parse_webhook
from assistant.channels.whatsapp.signature import compute_signature, verify_signature
from tests.whatsapp_fixtures import PNID, STATUS_ONLY, envelope, text_msg


def test_text_message_normalized_incl_telugu() -> None:
    [m] = parse_webhook(envelope([text_msg(body="నమస్కారం")]))
    assert m.text == "నమస్కారం" and m.from_phone == "+919876543210"
    assert m.channel_account_id == PNID and m.contact_name == "Ravi" and m.msg_type == "text"


def test_button_and_list_replies_keep_payload_id() -> None:
    btn = {"from": "919876543210", "id": "w1", "timestamp": "1", "type": "interactive",
           "interactive": {"button_reply": {"id": "book", "title": "Book"}}}
    lst = {"from": "919876543210", "id": "w2", "timestamp": "1", "type": "interactive",
           "interactive": {"list_reply": {"id": "slot_10", "title": "10:00 AM"}}}
    a, b = parse_webhook(envelope([btn, lst]))
    assert (a.payload_id, a.text) == ("book", "Book")
    assert (b.payload_id, b.text) == ("slot_10", "10:00 AM")


def test_voice_note_kept_as_audio_with_media_id() -> None:
    audio = {"from": "919876543210", "id": "w3", "timestamp": "1", "type": "audio",
             "audio": {"id": "MEDIA1"}}
    [m] = parse_webhook(envelope([audio]))
    assert m.msg_type == "audio" and m.media_id == "MEDIA1" and m.text is None


def test_status_updates_and_garbage_ignored() -> None:
    assert parse_webhook(STATUS_ONLY) == []
    assert parse_webhook({}) == []
    assert parse_webhook(envelope([{"type": "text"}])) == []  # missing id/from


def test_signature_roundtrip_and_failures() -> None:
    body = b'{"a":1}'
    good = compute_signature("secret", body)
    assert verify_signature("secret", body, good)
    assert not verify_signature("secret", body + b" ", good)  # tampered
    assert not verify_signature("other", body, good)
    assert not verify_signature("secret", body, None)
    assert not verify_signature("", body, good)  # unset secret fails closed

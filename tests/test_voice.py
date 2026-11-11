"""Phase-2 plug-in proof: fake STT/TTS/telephony drive a real conversation through the SHARED
pipeline (ingest -> orchestrator -> tools -> DB -> channel adapter). Nothing else changes."""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant.channels.base import InboundMessage, Option
from assistant.channels.voice.adapter import (
    VoiceCallHandler,
    VoiceChannelAdapter,
    prepare_for_speech,
    spoken_options,
)
from assistant.channels.voice.interfaces import NotConfiguredSTT, NotConfiguredTTS
from assistant.channels.voice.types import (
    AudioChunk,
    CallDirection,
    CallEvent,
    CallEventType,
    CallSession,
    SynthesizedAudio,
    Transcript,
    VoiceNotConfiguredError,
)
from assistant.channels.voice.voicenote import transcribe_voice_note
from assistant.channels.webchat import WebChatAdapter, WebChatMessage, to_inbound
from assistant.channels.whatsapp.client import WhatsAppClient, WhatsAppError
from assistant.core.models import Conversation, Message
from assistant.orchestrator.ingest import ingest_inbound
from tests.test_journey import Journey


class FakeSTT:
    """Maps pretend audio bytes to a transcript."""

    def __init__(self, script: dict[bytes, Transcript]) -> None:
        self.script, self.calls = script, 0

    async def transcribe(self, audio: AudioChunk, language_hint: str | None = None) -> Transcript:
        self.calls += 1
        return self.script[audio.data]

    def transcribe_stream(self, chunks: AsyncIterator[AudioChunk], language_hint: str | None = None
                          ) -> AsyncIterator[Transcript]:
        raise NotImplementedError


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    async def synthesize(self, text: str, language: str, voice: str | None = None
                         ) -> SynthesizedAudio:
        self.spoken.append((text, language))
        return SynthesizedAudio(data=text.encode())


class FakeTelephony:
    def __init__(self) -> None:
        self.played: list[tuple[str, bytes]] = []

    async def play(self, call_id: str, audio: SynthesizedAudio) -> None:
        self.played.append((call_id, audio.data))

    async def place_call(self, *a: Any) -> CallSession: raise NotImplementedError
    async def hangup(self, call_id: str) -> None: ...
    def verify_webhook(self, headers: dict[str, str], body: bytes) -> bool: return True
    def parse_event(self, payload: dict[str, Any]) -> CallEvent | None: return None


def call(phone: str = "+919876543210", lang: str | None = "te") -> CallSession:
    return CallSession(call_id="CA1", tenant_id="t", phone=phone, direction=CallDirection.INBOUND,
                       language=lang, started_at=datetime.now(UTC))


def test_prepare_for_speech_strips_chat_formatting() -> None:
    assert prepare_for_speech("✅ *Booked*: General on Fri 13 Nov. Reference: BK1.") == (
        "Booked: General on Fri 13 Nov. Reference: BK1.")


def test_options_are_read_aloud_with_numbers() -> None:
    out = spoken_options([Option(id="a", title="General"), Option(id="b", title="Dermatology")],
                         "Say the number or the name.")
    assert out == "Say the number or the name. 1. General. 2. Dermatology."
    assert spoken_options([], "x") == ""


async def test_default_stubs_fail_loudly_not_silently() -> None:
    with pytest.raises(VoiceNotConfiguredError):
        await NotConfiguredSTT().transcribe(AudioChunk(data=b""))
    with pytest.raises(VoiceNotConfiguredError):
        await NotConfiguredTTS().synthesize("hi", "en")
    with pytest.raises(VoiceNotConfiguredError):  # no active call and no outbound dialer yet
        await VoiceChannelAdapter(FakeTTS(), FakeTelephony()).send_text("a", "+91", "hi")  # type: ignore[arg-type]


async def test_phone_call_runs_through_the_shared_pipeline(session: AsyncSession) -> None:
    """Caller speaks Telugu: STT -> same ingest/orchestrator/DB -> reply spoken via TTS."""
    j = await Journey(session).start()
    tts, tel = FakeTTS(), FakeTelephony()
    adapter = VoiceChannelAdapter(tts, tel)  # type: ignore[arg-type]
    j.svc.channel = adapter  # voice replies go out through the voice adapter

    async def process(tenant_id: str, msg: InboundMessage) -> None:  # == the worker's path
        ing = await ingest_inbound(session, j.tenant.id, msg)
        assert ing is not None
        pend = await j.svc.process(session, j.tenant, ing, msg)
        await session.commit()
        await j.svc.deliver(j.factory, pend)

    stt = FakeSTT({b"a1": Transcript(text="నాకు appointment కావాలి", language="te"),
                   b"a2": Transcript(text="నేను చెప్పలేను", confidence=0.2, is_final=False)})
    handler = VoiceCallHandler(stt, adapter, process)  # type: ignore[arg-type]
    c = call()
    await handler.handle(CallEvent(type=CallEventType.STARTED, call=c))
    await handler.handle(CallEvent(type=CallEventType.UTTERANCE, call=c, sequence=1,
                                   audio=AudioChunk(data=b"a1")))
    # replied by VOICE: Telugu prompt spoken, options read aloud, played into the call
    assert tel.played and tel.played[0][0] == "CA1"
    spoken_text, lang = tts.spoken[0]
    assert lang == "te" and "విభాగానికి" in spoken_text and "1." in spoken_text

    # unusable audio (non-final) produces no turn at all
    before = len(tts.spoken)
    await handler.handle(CallEvent(type=CallEventType.UTTERANCE, call=c, sequence=2,
                                   audio=AudioChunk(data=b"a2")))
    assert len(tts.spoken) == before

    # keypad digit 1 = first option, deterministic: zero STT calls, same engine
    stt_calls = stt.calls
    await handler.handle(CallEvent(type=CallEventType.DTMF, call=c, sequence=3, digits="1"))
    assert stt.calls == stt_calls
    conv = (await session.execute(select(Conversation))).scalar_one()
    assert conv.collected_slots["service"] == "general" and conv.flow == "book_appointment"

    msgs = (await session.execute(select(Message).where(Message.channel == "voice"))).scalars().all()
    inbound_ids = {m.wa_message_id for m in msgs if m.direction.value == "inbound"}
    outbound_ids = [m.wa_message_id for m in msgs if m.direction.value == "outbound"]
    assert inbound_ids == {"CA1:1", "CA1:3"}  # call-id + sequence: normal webhook dedupe applies
    assert len(outbound_ids) >= 2 and len(set(outbound_ids)) == len(outbound_ids)  # unique replies
    await handler.handle(CallEvent(type=CallEventType.ENDED, call=c))
    with pytest.raises(VoiceNotConfiguredError):  # call is over: nothing to speak into
        await adapter.send_text("a", c.phone, "late")


# ------------------------------------------------------------------ WhatsApp voice notes
def audio_msg(text: str | None = None) -> InboundMessage:
    return InboundMessage(channel_account_id="p", message_id="w1", from_phone="+91",
                          timestamp=datetime.now(UTC), msg_type="audio", media_id="MEDIA1",
                          text=text)


class Downloader:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.ids = fail, []

    async def download_media(self, media_id: str, max_bytes: int = 0) -> bytes:
        self.ids.append(media_id)
        if self.fail:
            raise WhatsAppError(500, "boom")
        return b"a1"


async def test_voice_note_transcribed_into_a_normal_text_message() -> None:
    stt = FakeSTT({b"a1": Transcript(text="అపాయింట్‌మెంట్ కావాలి", language="te", confidence=0.9)})
    out = await transcribe_voice_note(audio_msg(), Downloader(), stt, "te")  # type: ignore[arg-type]
    assert out.text == "అపాయింట్‌మెంట్ కావాలి" and out.msg_type == "text"
    assert out.message_id == "w1" and out.media_id == "MEDIA1"  # identity preserved for dedupe


async def test_voice_note_falls_back_when_not_possible() -> None:
    low = FakeSTT({b"a1": Transcript(text="?", confidence=0.2)})
    for stt, dl in ((None, Downloader()), (low, Downloader()),
                    (NotConfiguredSTT(), Downloader()), (low, Downloader(fail=True))):
        out = await transcribe_voice_note(audio_msg(), dl, stt, None)  # type: ignore[arg-type]
        assert out.msg_type == "audio" and out.text is None  # polite "please type" path
    already = await transcribe_voice_note(audio_msg("caption"), Downloader(), low, None)  # type: ignore[arg-type]
    assert already.text == "caption"


async def test_whatsapp_media_download_two_hops_and_guards() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "graph.facebook.com":
            url = "http://insecure.example/x" if "BAD" in req.url.path else "https://cdn.fb/x.ogg"
            return httpx.Response(200, json={"url": url})
        assert req.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, content=b"x" * 10 if "big" not in str(req.url) else b"y" * 99)

    c = WhatsAppClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok",
                       base_delay=0)
    assert await c.download_media("MEDIA1") == b"x" * 10
    with pytest.raises(WhatsAppError, match="https"):
        await c.download_media("BAD")
    with pytest.raises(WhatsAppError, match="large"):
        await c.download_media("MEDIA1", max_bytes=5)


# ----------------------------------------------------------------------------- web chat
async def test_webchat_maps_to_the_same_inbound_model_and_pushes_replies() -> None:
    m = to_inbound(WebChatMessage(tenant_id="t", visitor_id="v9", message_id="m1", option_id="gen",
                                  text="General"))
    assert (m.channel, m.from_phone, m.msg_type, m.payload_id) == ("webchat", "web:v9",
                                                                  "interactive", "gen")
    pushed: list[tuple[str, dict[str, object]]] = []

    async def push(to: str, payload: dict[str, object]) -> None:
        pushed.append((to, payload))

    ad = WebChatAdapter(push)
    await ad.send_options("t", "web:v9", "Pick", [Option(id="a", title="A")], "")
    assert pushed[0][1]["type"] == "options" and pushed[0][0] == "web:v9"
    with pytest.raises(NotImplementedError):
        await ad.send_template("t", "web:v9", "n", "en", [])

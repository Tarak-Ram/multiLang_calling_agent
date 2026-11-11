"""Voice channel: maps call events to the SAME InboundMessage the WhatsApp channel produces, and
speaks replies through TTS. Everything between (ingest, orchestrator, tools, cost tracking,
handoff) is the shared pipeline.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from assistant.channels.base import InboundMessage, Option
from assistant.channels.voice.interfaces import SpeechToText, TelephonyProvider, TextToSpeech
from assistant.channels.voice.types import (
    CallEvent,
    CallEventType,
    CallSession,
    VoiceNotConfiguredError,
)
from assistant.core.pii import mask_phone

log = logging.getLogger("assistant.voice")

_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿✅️]")
_MARKUP = re.compile(r"[*_`#>]+")


def prepare_for_speech(text: str) -> str:
    """Text written for chat -> text that sounds right aloud: no emoji/markdown, and short.
    (References like 'BK1' are read letter-by-letter by most TTS engines, which is what we want.)"""
    text = _MARKUP.sub("", _EMOJI.sub("", text))
    return " ".join(text.split())


def spoken_options(options: list[Option], prompt: str) -> str:
    """Voice has no buttons: read the choices aloud. The engine already accepts the option's
    NUMBER or its NAME as the answer, so 'two' / 'dermatology' work without extra code."""
    if not options:
        return ""
    listed = " ".join(f"{i}. {o.title}." for i, o in enumerate(options, 1))
    return f"{prompt} {listed}".strip()


# The shared pipeline entry point (same function the WhatsApp worker uses after parsing):
# store message -> run orchestrator -> deliver replies through the channel adapter.
TurnProcessor = Callable[[str, InboundMessage], Awaitable[None]]


class VoiceChannelAdapter:
    """Implements the ChannelAdapter outbound interface for phone calls."""

    def __init__(self, tts: TextToSpeech, telephony: TelephonyProvider,
                 choose_prompt: str = "Say the number or the name.") -> None:
        self._tts, self._tel, self._prompt = tts, telephony, choose_prompt
        self._calls: dict[str, CallSession] = {}  # phone -> active call (this process)
        self._out_seq: dict[str, int] = {}  # call_id -> spoken replies so far

    def attach(self, call: CallSession) -> None:
        self._calls[call.phone] = call

    def detach(self, phone: str) -> None:
        self._calls.pop(phone, None)

    async def send_text(self, account_id: str, to_phone: str, body: str) -> str:
        call = self._calls.get(to_phone)
        if call is None:
            raise VoiceNotConfiguredError("no active call for this phone (outbound calling is "
                                          "started via TelephonyProvider.place_call)")
        audio = await self._tts.synthesize(prepare_for_speech(body), call.language or "en")
        await self._tel.play(call.call_id, audio)
        # Unique per reply: (tenant, message id) is a DB unique key, so a call's 2nd reply
        # must not reuse the 1st reply's id.
        self._out_seq[call.call_id] = self._out_seq.get(call.call_id, 0) + 1
        return f"voice:{call.call_id}:out{self._out_seq[call.call_id]}"

    async def send_options(self, account_id: str, to_phone: str, body: str,
                           options: list[Option], list_button: str) -> str:
        return await self.send_text(account_id, to_phone,
                                    f"{body} {spoken_options(options, self._prompt)}")

    async def send_template(self, account_id: str, to_phone: str, name: str, language: str,
                            params: list[str]) -> str:
        # WhatsApp templates have no voice equivalent; outbound calls use place_call instead.
        raise VoiceNotConfiguredError("templates are a WhatsApp concept")


class VoiceCallHandler:
    """Turns telephony events into pipeline turns."""

    def __init__(self, stt: SpeechToText, adapter: VoiceChannelAdapter,
                 process: TurnProcessor) -> None:
        self._stt, self._adapter, self._process = stt, adapter, process

    async def handle(self, event: CallEvent) -> None:
        call = event.call
        if event.type == CallEventType.STARTED:
            self._adapter.attach(call)
            return
        if event.type == CallEventType.ENDED:
            self._adapter.detach(call.phone)
            return
        self._adapter.attach(call)
        text: str | None = None
        if event.type == CallEventType.DTMF and event.digits:
            text = event.digits  # keypad digit == "pick option N", no STT cost or latency
        elif event.type == CallEventType.UTTERANCE and event.audio is not None:
            transcript = await self._stt.transcribe(event.audio, call.language)
            if transcript.is_final and transcript.text.strip():
                text = transcript.text
        if text is None:
            return  # silence / unintelligible: the call layer re-prompts
        msg = InboundMessage(
            channel="voice", channel_account_id=call.tenant_id,
            message_id=f"{call.call_id}:{event.sequence}",  # unique -> normal dedupe applies
            from_phone=call.phone, timestamp=event.call.started_at or datetime.now(UTC), text=text,
        )
        log.info("voice_turn", extra={"customer": mask_phone(call.phone)})
        await self._process(call.tenant_id, msg)


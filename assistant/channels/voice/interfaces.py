"""Provider interfaces for phase 2 (phone calls). Implement these, nothing else changes.

PHASE-2 PLUG-IN CHECKLIST
  1. Pick vendors (each is a PAID external service: confirm with the product owner first):
       SpeechToText (must support Telugu, Hindi, Indian English + code-mixing),
       TextToSpeech (natural te/hi/en voices), TelephonyProvider (Indian numbers, SIP/PSTN).
  2. Implement the three Protocols below (one small class each).
  3. Add a webhook route that verifies the telephony vendor's signature, maps the payload to a
     CallEvent via TelephonyProvider.parse_event, and calls VoiceCallHandler.handle(event).
  4. Register VoiceChannelAdapter as the channel for voice replies.
The orchestrator, router, tools, RAG, cost tracking and handoff are reused unchanged.

Latency budget for a natural call (all stages together): aim < 1.2 s from end-of-speech to the
first audio byte. That is why rules/FAQ (no LLM) matter even more on voice, and why replies are
1-2 short sentences (see prepare_for_speech).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from assistant.channels.voice.types import (
    AudioChunk,
    CallEvent,
    CallSession,
    SynthesizedAudio,
    Transcript,
    VoiceNotConfiguredError,
)


class SpeechToText(Protocol):
    async def transcribe(self, audio: AudioChunk, language_hint: str | None = None) -> Transcript:
        """One finished utterance -> text. `language_hint` = the conversation's stored language."""
        ...

    def transcribe_stream(
        self, chunks: AsyncIterator[AudioChunk], language_hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        """Streaming variant for low latency (partials, then a final)."""
        ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str, language: str, voice: str | None = None
                         ) -> SynthesizedAudio: ...


class TelephonyProvider(Protocol):
    async def place_call(self, tenant_id: str, to_phone: str, from_number: str,
                         callback_url: str) -> CallSession:
        """Start an OUTBOUND call (reminders, follow-ups). Must respect do-not-call rules."""
        ...

    async def play(self, call_id: str, audio: SynthesizedAudio) -> None: ...

    async def hangup(self, call_id: str) -> None: ...

    def verify_webhook(self, headers: dict[str, str], body: bytes) -> bool: ...

    def parse_event(self, payload: dict[str, Any]) -> CallEvent | None: ...


class NotConfiguredSTT:
    """Default until a vendor is chosen. Fails loudly, never silently drops a caller."""

    async def transcribe(self, audio: AudioChunk, language_hint: str | None = None) -> Transcript:
        raise VoiceNotConfiguredError("no SpeechToText provider configured (phase 2)")

    def transcribe_stream(self, chunks: AsyncIterator[AudioChunk],
                          language_hint: str | None = None) -> AsyncIterator[Transcript]:
        raise VoiceNotConfiguredError("no SpeechToText provider configured (phase 2)")


class NotConfiguredTTS:
    async def synthesize(self, text: str, language: str, voice: str | None = None
                         ) -> SynthesizedAudio:
        raise VoiceNotConfiguredError("no TextToSpeech provider configured (phase 2)")

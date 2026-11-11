"""Value types for the voice channel (phase 2). Provider-neutral."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class CallDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class AudioChunk(BaseModel):
    """Raw audio from the caller (or a voice note). Telephony is usually 8 kHz mu-law/PCM."""

    data: bytes
    sample_rate: int = 16_000
    encoding: Literal["pcm16", "mulaw", "alaw", "opus", "ogg_opus", "mp3"] = "pcm16"


class Transcript(BaseModel):
    text: str
    language: str | None = None  # ISO code detected by the STT, if it reports one
    confidence: float = 1.0
    is_final: bool = True


class SynthesizedAudio(BaseModel):
    data: bytes
    sample_rate: int = 16_000
    encoding: Literal["pcm16", "mulaw", "alaw", "opus", "ogg_opus", "mp3"] = "pcm16"


class CallSession(BaseModel):
    """One phone call. `phone` is E.164 and is the SAME key the conversation uses, so a caller
    who also uses WhatsApp continues one conversation across channels."""

    call_id: str
    tenant_id: str
    phone: str
    direction: CallDirection
    language: str | None = None
    started_at: datetime | None = None


class CallEventType(StrEnum):
    STARTED = "started"
    UTTERANCE = "utterance"  # caller finished speaking (end-of-speech detected)
    DTMF = "dtmf"  # keypad digit(s), a deterministic menu choice with no STT needed
    ENDED = "ended"


class CallEvent(BaseModel):
    type: CallEventType
    call: CallSession
    audio: AudioChunk | None = None  # UTTERANCE
    digits: str | None = None  # DTMF
    sequence: int = 0  # increases per event within a call (used to build dedupe ids)


class VoiceNotConfiguredError(RuntimeError):
    """Raised by the default stubs: no speech/telephony provider has been chosen yet."""

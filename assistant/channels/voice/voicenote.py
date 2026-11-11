"""WhatsApp voice notes -> text, so the normal pipeline can handle them (phase 2 ready).

Until a SpeechToText provider is configured this does nothing and the customer gets the polite
"please type" reply. Once one is, voice notes are transcribed BEFORE ingestion, with the
conversation's language as a hint (te/hi/en/code-mixed).
"""
from __future__ import annotations

import logging
from typing import Protocol

from assistant.channels.base import InboundMessage
from assistant.channels.voice.interfaces import SpeechToText
from assistant.channels.voice.types import AudioChunk, VoiceNotConfiguredError

log = logging.getLogger("assistant.voice.voicenote")
MIN_CONFIDENCE = 0.5  # below this the transcript is too unreliable to act on


class MediaDownloader(Protocol):
    async def download_media(self, media_id: str, max_bytes: int = ...) -> bytes: ...


async def transcribe_voice_note(
    msg: InboundMessage, downloader: MediaDownloader, stt: SpeechToText | None,
    language_hint: str | None = None,
) -> InboundMessage:
    """Return `msg` with `.text` filled from the audio, or unchanged if that is impossible.
    Never raises: a failed transcription falls back to the "please type" reply."""
    if msg.msg_type != "audio" or not msg.media_id or stt is None or msg.text:
        return msg
    try:
        audio = AudioChunk(data=await downloader.download_media(msg.media_id),
                           encoding="ogg_opus", sample_rate=16_000)
        transcript = await stt.transcribe(audio, language_hint)
    except VoiceNotConfiguredError:
        return msg
    except Exception:
        log.exception("voice_note_transcription_failed")
        return msg
    if not transcript.is_final or not transcript.text.strip() or transcript.confidence < MIN_CONFIDENCE:
        log.info("voice_note_unusable", extra={"confidence": transcript.confidence})
        return msg
    return msg.model_copy(update={"text": transcript.text.strip(), "msg_type": "text"})

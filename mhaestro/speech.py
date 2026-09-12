"""Optional speech input.

Section 6.3 treats "adjustable pacing, concise on-screen text, and optional audio
read-out" as accessibility features that protect validity, not as extras. Speaking
an answer instead of typing it is the input-side equivalent: it removes a barrier
for anyone who finds typing on a phone at a busy event slow or difficult, and a
longer spoken answer is not evidence of greater engagement, so whether a turn was
spoken or typed is recorded alongside it.

Text-to-speech is deliberately absent. Reading every question aloud adds seconds
to every turn, and the session budget is five minutes.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

from .config import get_secret
from .llm import _client, has_secret

TRANSCRIBE_MODELS = {
    "openai": "gpt-4o-transcribe",
    "gemini": "gemini-2.5-flash",
}


def transcription_available() -> bool:
    if has_secret("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401

            return True
        except Exception:
            return False
    return False


def transcribe(audio_bytes: bytes, *, filename: str = "reply.wav") -> Tuple[Optional[str], str]:
    """Return (text, error). Never raises -- a failed transcription falls back to typing."""
    if not audio_bytes:
        return None, "No audio was captured."
    if not transcription_available():
        return None, "Speech input is not configured."

    try:
        client = _client("openai")
        buffer = io.BytesIO(audio_bytes)
        buffer.name = filename
        response = client.audio.transcriptions.create(
            model=get_secret("TRANSCRIBE_MODEL", TRANSCRIBE_MODELS["openai"]),
            file=buffer,
        )
        text = getattr(response, "text", "") or ""
        return (text.strip() or None), ""
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__ + ": " + str(exc)

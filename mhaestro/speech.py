"""Optional speech input.

Section 6.3 treats "adjustable pacing, concise on-screen text, and optional audio
read-out" as accessibility features that protect validity, not as extras. Speaking
an answer instead of typing it is the input-side equivalent: it removes a barrier
for anyone who finds typing on a phone at a busy event slow or difficult, and a
longer spoken answer is not evidence of greater engagement, so which backend
transcribed a turn -- and whether it was spoken at all -- is recorded alongside it.

Streamlit has no transcription of its own: `st.audio_input` is a recorder that
returns raw bytes. Something external has to turn those into text, and three
things can:

*   **ElevenLabs** (`scribe_v1`) -- a purpose-built speech-to-text model, and what
    the SPIRIT project uses. Tried first when a key is present.
*   **OpenAI** -- a dedicated transcription endpoint.
*   **Gemini** -- natively multimodal, so the audio goes to an ordinary
    `generate_content` call with an instruction to transcribe it.

**Anthropic cannot.** Claude accepts text, images and PDFs, but not audio.

Every layer degrades rather than breaks. A backend with no key or no SDK installed
is skipped; a backend that errors at call time falls through to the next one; if
all of them fail the participant is told to type, which always works because the
text box is rendered before any of this and never depends on it. Nothing here can
stop a session.

Text-to-speech is deliberately absent throughout. Reading every question aloud
adds seconds to every turn, and the session budget is five minutes.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

from .config import get_secret, has_secret
from .llm import _client as _llm_client

# Preference order. The two purpose-built speech-to-text models come before the
# general multimodal one.
TRANSCRIBE_BACKENDS = ("elevenlabs", "openai", "gemini")

BACKEND_LABELS = {
    "elevenlabs": "ElevenLabs",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
}

BACKEND_KEYS = {
    "elevenlabs": "ELEVENLABS_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

BACKEND_PACKAGES = {
    "elevenlabs": "elevenlabs",
    "openai": "openai",
    "gemini": "google-genai",
}

DEFAULT_TRANSCRIBE_MODELS = {
    "elevenlabs": "scribe_v1",
    "openai": "gpt-4o-transcribe",
    "gemini": "gemini-2.5-flash",
}

# Sent with the audio on the multimodal path. Kept blunt: anything conversational
# risks the model answering the participant instead of transcribing them.
GEMINI_TRANSCRIBE_PROMPT = (
    "Transcribe this audio verbatim. Return only the words spoken, with no "
    "commentary, no speaker labels, no timestamps and no quotation marks. "
    "If there is no intelligible speech, return nothing at all."
)


# ------------------------------------------------------------------ discovery


def _import_backend(name: str):
    try:
        if name == "elevenlabs":
            from elevenlabs.client import ElevenLabs

            return ElevenLabs
        if name == "openai":
            import openai

            return openai
        if name == "gemini":
            from google import genai

            return genai
    except Exception:
        return None
    return None


def backend_installed(name: str) -> bool:
    return _import_backend(name) is not None


def available_backends() -> List[str]:
    """Backends that have both a key and an importable SDK, in preference order."""
    return [
        name
        for name in TRANSCRIBE_BACKENDS
        if has_secret(BACKEND_KEYS[name]) and backend_installed(name)
    ]


def ordered_backends() -> List[str]:
    """Available backends, with an explicit TRANSCRIBE_PROVIDER hoisted to front."""
    backends = available_backends()
    preferred = (get_secret("TRANSCRIBE_PROVIDER", "") or "").strip().lower()
    if preferred in backends:
        return [preferred] + [b for b in backends if b != preferred]
    return backends


def active_backend() -> Optional[str]:
    """The backend that will be tried first."""
    backends = ordered_backends()
    return backends[0] if backends else None


def active_model(backend: Optional[str] = None) -> str:
    backend = backend or active_backend() or ""
    # A model override only makes sense for the backend it was written for.
    override = (get_secret("TRANSCRIBE_MODEL", "") or "").strip()
    preferred = (get_secret("TRANSCRIBE_PROVIDER", "") or "").strip().lower()
    if override and (not preferred or preferred == backend):
        return override
    return DEFAULT_TRANSCRIBE_MODELS.get(backend, "")


def transcription_available() -> bool:
    return bool(ordered_backends())


def unavailable_reason() -> str:
    """Why speech input cannot be offered, in terms the researcher can act on."""
    if transcription_available():
        return ""

    # Distinguish "no key" from "key but no SDK" -- they need different fixes.
    keyed_but_missing_sdk = [
        name
        for name in TRANSCRIBE_BACKENDS
        if has_secret(BACKEND_KEYS[name]) and not backend_installed(name)
    ]
    if keyed_but_missing_sdk:
        name = keyed_but_missing_sdk[0]
        return (
            BACKEND_LABELS[name] + " is configured but its SDK is not installed "
            "(pip install " + BACKEND_PACKAGES[name] + ")."
        )

    reason = "Speech input needs an ELEVENLABS_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY."
    if has_secret("ANTHROPIC_API_KEY"):
        reason += " Claude cannot accept audio, so an Anthropic-only deployment is text-only."
    return reason


# -------------------------------------------------------------------- clients

_CLIENTS: Dict[str, object] = {}


def _client(name: str):
    if name in _CLIENTS:
        return _CLIENTS[name]
    if name == "elevenlabs":
        ElevenLabs = _import_backend("elevenlabs")
        client = ElevenLabs(api_key=get_secret(BACKEND_KEYS["elevenlabs"]))
    else:
        # OpenAI and Gemini share the clients the rest of the app already builds.
        client = _llm_client(name)
    _CLIENTS[name] = client
    return client


# --------------------------------------------------------------- transcription


def transcribe(
    audio_bytes: bytes,
    *,
    filename: str = "reply.wav",
    mime_type: str = "audio/wav",
    timeout: int = 60,
) -> Tuple[Optional[str], str, dict]:
    """Return (text, error, provenance).

    Tries each available backend in preference order, falling through whenever one
    raises, so a failing or unreachable provider costs a few seconds rather than
    the feature. Never raises: if every backend fails the caller shows a message
    and the participant types instead.

    An *empty* result from a call that otherwise succeeded means the recording had
    no intelligible speech. That is an answer, not a failure, so it does not fall
    through -- retrying silence on two more backends would just triple the wait.

    `provenance` is flat and CSV-safe: it records which backend produced the text,
    whether that required falling back, and what each attempt did.
    """
    provenance = {"asr_provider": "", "asr_model": "", "asr_fell_back": False, "asr_attempts": ""}

    if not audio_bytes:
        return None, "No audio was captured.", provenance

    backends = ordered_backends()
    if not backends:
        return None, unavailable_reason(), provenance

    attempts: List[str] = []
    errors: List[str] = []

    for index, backend in enumerate(backends):
        model = active_model(backend)
        try:
            text = _dispatch(backend, audio_bytes, filename, mime_type, model, timeout)
        except Exception as exc:  # noqa: BLE001 -- try the next backend instead
            attempts.append(backend + ":error")
            errors.append(BACKEND_LABELS[backend] + ": " + type(exc).__name__ + ": " + str(exc))
            continue

        text = (text or "").strip()
        attempts.append(backend + (":ok" if text else ":empty"))
        provenance = {
            "asr_provider": backend,
            "asr_model": model,
            "asr_fell_back": index > 0,
            "asr_attempts": ";".join(attempts),
        }
        if text:
            return text, "", provenance
        return None, "No speech was detected.", provenance

    provenance["asr_attempts"] = ";".join(attempts)
    return None, " | ".join(errors) or "Transcription failed.", provenance


def _dispatch(backend: str, audio_bytes: bytes, filename: str, mime_type: str, model: str, timeout: int) -> str:
    if backend == "elevenlabs":
        return _transcribe_elevenlabs(audio_bytes, model, timeout)
    if backend == "openai":
        return _transcribe_openai(audio_bytes, filename, model, timeout)
    if backend == "gemini":
        return _transcribe_gemini(audio_bytes, mime_type, model)
    raise ValueError("Unknown transcription backend: " + str(backend))


def _transcribe_elevenlabs(audio_bytes: bytes, model: str, timeout: int) -> str:
    client = _client("elevenlabs")
    response = client.speech_to_text.convert(
        file=io.BytesIO(audio_bytes),
        model_id=model or DEFAULT_TRANSCRIBE_MODELS["elevenlabs"],
        # Both off: one participant per session, and audio-event tags like
        # "(laughs)" would end up counted as words by the response-length measures.
        diarize=False,
        tag_audio_events=False,
        request_options={"timeout_in_seconds": timeout},
    )
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        text = response.get("text", "")
    return text or ""


def _transcribe_openai(audio_bytes: bytes, filename: str, model: str, timeout: int) -> str:
    client = _client("openai")
    buffer = io.BytesIO(audio_bytes)
    # The SDK infers the format from the filename, so it has to carry an extension.
    buffer.name = filename
    response = client.audio.transcriptions.create(model=model, file=buffer, timeout=timeout)
    return getattr(response, "text", "") or ""


def _transcribe_gemini(audio_bytes: bytes, mime_type: str, model: str) -> str:
    from google.genai import types

    client = _client("gemini")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    types.Part.from_text(text=GEMINI_TRANSCRIBE_PROMPT),
                ],
            )
        ],
        config=types.GenerateContentConfig(max_output_tokens=1000),
    )
    return response.text or ""

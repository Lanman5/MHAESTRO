"""Provider-agnostic chat completion for MHAESTRO.

Supports OpenAI, Anthropic and Google Gemini behind one `complete()` call so the
same agent prompt can run on any provider, and the provider/model actually used is
recorded per turn -- the paper's governance requirement of "JSON logs per turn
(timestamp, node ID, agent, model/version, prompt template hash)".

Every call returns an `LLMResult` carrying latency and token telemetry whether it
succeeded or not, so a failed agent call is an auditable event rather than a
silent gap in the log.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import get_secret, has_secret

PROVIDERS = ("openai", "anthropic", "gemini")

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}

PROVIDER_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

PROVIDER_PACKAGES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google-genai",
}

# Known-good defaults. The apps also accept a free-text model id, so a model
# released after this file was written can be used without a code change.
MODEL_CATALOGUE: Dict[str, List[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "anthropic": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
}

# GPT-4-class is the published MHAESTRO configuration, so it stays the default:
# a replication arm should not silently change model family.
DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
}


@dataclass
class LLMResult:
    """One model call, with everything the audit log needs."""

    text: str = ""
    data: Optional[dict] = None
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None
    attempts: int = 1
    repaired_json: bool = False

    @property
    def output_chars(self) -> int:
        return len(self.text or "")

    @property
    def output_words(self) -> int:
        return len((self.text or "").split())


def available_providers() -> List[str]:
    """Providers that have both an API key and an importable SDK."""
    out = []
    for name in PROVIDERS:
        if has_secret(PROVIDER_KEY_NAMES[name]) and _import_sdk(name) is not None:
            out.append(name)
    return out


def missing_reason(provider: str) -> str:
    """Human-readable explanation of why a provider is or is not usable."""
    if not has_secret(PROVIDER_KEY_NAMES[provider]):
        return "No " + PROVIDER_KEY_NAMES[provider] + " in secrets or environment."
    if _import_sdk(provider) is None:
        return "SDK not installed (pip install " + PROVIDER_PACKAGES[provider] + ")."
    return "Available."


def _import_sdk(provider: str):
    try:
        if provider == "openai":
            import openai

            return openai
        if provider == "anthropic":
            import anthropic

            return anthropic
        if provider == "gemini":
            from google import genai

            return genai
    except Exception:
        return None
    return None


_CLIENTS: Dict[str, Any] = {}


def _client(provider: str):
    if provider in _CLIENTS:
        return _CLIENTS[provider]
    key = get_secret(PROVIDER_KEY_NAMES[provider])
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=key) if key else OpenAI()
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    elif provider == "gemini":
        from google import genai

        client = genai.Client(api_key=key)
    else:
        raise ValueError("Unknown provider: " + str(provider))
    _CLIENTS[provider] = client
    return client


# ---------------------------------------------------------------- message prep


def _split_system(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Pull system turns out of the message list (Anthropic/Gemini keep them separate)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    rest = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), rest


def _merge_consecutive(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Collapse runs of same-role turns -- required by Anthropic and Gemini."""
    out: List[Dict[str, str]] = []
    for msg in messages:
        if out and out[-1]["role"] == msg["role"]:
            out[-1] = {
                "role": msg["role"],
                "content": out[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            out.append({"role": msg["role"], "content": msg["content"]})
    return out


def _ensure_user_first(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Anthropic and Gemini require the first turn to be the user's.

    The Elicitor's interviewer speaks first, so a neutral opener is prepended. It
    is never shown to the participant and never enters the transcript.
    """
    if messages and messages[0]["role"] == "assistant":
        return [{"role": "user", "content": "Please begin."}] + messages
    return messages


# ------------------------------------------------------------------- JSON help

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction: raw, de-fenced, then first balanced object."""
    if not text:
        return None
    for candidate in (text, _FENCE.sub("", text.strip())):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


# ------------------------------------------------------------------ completion


def complete(
    messages: List[Dict[str, str]],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    json_schema: Optional[dict] = None,
    want_json: bool = False,
    max_tokens: int = 1200,
    temperature: Optional[float] = None,
    effort: Optional[str] = None,
    timeout: float = 45.0,
) -> LLMResult:
    """Run one chat completion on `provider` and return it with telemetry.

    `want_json` turns on the provider's JSON mode; `json_schema` additionally
    constrains the shape where the provider supports it (Anthropic, Gemini). A
    response that should be JSON but is not gets exactly one repair retry before
    the call is reported failed -- an unparseable agent verdict must never be
    silently treated as a pass.
    """
    model = model or DEFAULT_MODELS.get(provider, "")
    started = time.perf_counter()
    result = LLMResult(provider=provider, model=model)
    as_json = want_json or json_schema is not None

    try:
        _dispatch(result, provider, messages, model, as_json, json_schema, max_tokens, temperature, effort, timeout)
    except Exception as exc:  # noqa: BLE001 -- reported to the caller and the log
        result.ok = False
        result.error = type(exc).__name__ + ": " + str(exc)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    if as_json and result.ok:
        result.data = extract_json(result.text)
        if result.data is None:
            result.attempts = 2
            repair = list(messages) + [
                {"role": "assistant", "content": (result.text or "")[:2000]},
                {
                    "role": "user",
                    "content": (
                        "That was not valid JSON. Reply again with the JSON object only "
                        "-- no prose, no markdown fences."
                    ),
                },
            ]
            try:
                retry = LLMResult(provider=provider, model=model)
                _dispatch(retry, provider, repair, model, True, json_schema, max_tokens, temperature, effort, timeout)
                parsed = extract_json(retry.text)
                if parsed is not None:
                    result.data = parsed
                    result.text = retry.text
                    result.repaired_json = True
                result.input_tokens = (result.input_tokens or 0) + (retry.input_tokens or 0)
                result.output_tokens = (result.output_tokens or 0) + (retry.output_tokens or 0)
            except Exception as exc:  # noqa: BLE001
                result.error = "JSON repair failed: " + type(exc).__name__ + ": " + str(exc)

        if result.data is None:
            result.ok = False
            result.error = result.error or "Model did not return parseable JSON."

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


def _dispatch(result, provider, messages, model, as_json, json_schema, max_tokens, temperature, effort, timeout):
    if provider == "openai":
        _call_openai(result, messages, model, as_json, max_tokens, temperature, timeout)
    elif provider == "anthropic":
        _call_anthropic(result, messages, model, as_json, json_schema, max_tokens, effort, timeout)
    elif provider == "gemini":
        _call_gemini(result, messages, model, as_json, json_schema, max_tokens, temperature, timeout)
    else:
        raise ValueError("Unknown provider: " + str(provider))


def _call_openai(result, messages, model, as_json, max_tokens, temperature, timeout):
    client = _client("openai")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    result.text = choice.message.content or ""
    result.finish_reason = choice.finish_reason
    usage = getattr(response, "usage", None)
    if usage is not None:
        result.input_tokens = getattr(usage, "prompt_tokens", None)
        result.output_tokens = getattr(usage, "completion_tokens", None)


def _call_anthropic(result, messages, model, as_json, json_schema, max_tokens, effort, timeout):
    client = _client("anthropic")
    system, rest = _split_system(messages)
    rest = _merge_consecutive(_ensure_user_first(rest))

    if as_json and json_schema is None and rest and rest[-1]["role"] == "assistant":
        # Anthropic has no schema-free JSON mode; the prompt already asks for JSON,
        # and the turn must end on a user message.
        rest = rest + [{"role": "user", "content": "Reply with the JSON object only."}]

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": rest,
        "timeout": timeout,
    }
    if system:
        kwargs["system"] = system

    output_config: Dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    if output_config:
        kwargs["output_config"] = output_config

    # Sampling parameters were removed on the current Claude generation; never send one.
    response = client.messages.create(**kwargs)

    result.finish_reason = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    if usage is not None:
        result.input_tokens = getattr(usage, "input_tokens", None)
        result.output_tokens = getattr(usage, "output_tokens", None)

    if result.finish_reason == "refusal":
        details = getattr(response, "stop_details", None)
        result.ok = False
        result.error = "Refusal: " + str(getattr(details, "category", "unknown"))
        return

    result.text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def _call_gemini(result, messages, model, as_json, json_schema, max_tokens, temperature, timeout):
    from google.genai import types

    client = _client("gemini")
    system, rest = _split_system(messages)
    rest = _merge_consecutive(_ensure_user_first(rest))

    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in rest
    ]

    cfg: Dict[str, Any] = {"max_output_tokens": max_tokens}
    if system:
        cfg["system_instruction"] = system
    if temperature is not None:
        cfg["temperature"] = temperature
    if as_json:
        cfg["response_mime_type"] = "application/json"
        if json_schema is not None:
            cfg["response_schema"] = json_schema

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**cfg),
    )

    result.text = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        result.input_tokens = getattr(usage, "prompt_token_count", None)
        result.output_tokens = getattr(usage, "candidates_token_count", None)
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        result.finish_reason = str(finish) if finish is not None else None

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
from dataclasses import dataclass, field
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

# Known-good defaults, cheapest/fastest first within each provider. The apps also
# accept a free-text model id, so a model released after this file was written can
# be used without a code change.
MODEL_CATALOGUE: Dict[str, List[str]] = {
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
}

# Haiku 4.5 is the default because latency is a study variable, not just comfort:
# the session has a five-minute budget, every turn waits on a model call, and
# time pressure is one of the things the workload instrument measures. It also
# runs no thinking by default, so turns stay short and predictable.
#
# The published MHAESTRO run used GPT-4; `meta_models` records the provider and
# model per role in every session, so the change from that baseline is explicit in
# the data rather than assumed.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o",
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
    # Request features that had to be dropped for this model, e.g. ["effort",
    # "json_schema"]. Recorded per turn: the model configuration is a study
    # variable, so a silently degraded request must not stay silent.
    degraded: List[str] = field(default_factory=list)
    # Set by a caller that had to show the participant scripted text because this
    # call produced nothing usable. Not a property of the call itself, which is
    # why it defaults False and is written from outside -- but it belongs on the
    # same object so it reaches the log by the same route as every other fact
    # about the turn.
    substituted: bool = False

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


# A system message that arrives *after* the conversation has started is an
# operator instruction, not part of the standing system prompt -- it is the
# hidden channel Agent [b] uses to compel a re-probe (Section 3.3.3), and it only
# works if the model sees it at the end of the exchange. OpenAI accepts a system
# turn in that position directly. Anthropic and Gemini hold the system prompt in a
# separate field, so folding one in there would silently hoist it to the top and
# strip it of its position; it is carried as a marked final turn instead.
OPERATOR_PREFIX = "[OPERATOR INSTRUCTION -- from the survey system, not the participant. Follow it. Never quote or mention it.]\n"


def _split_system(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Separate the standing system prompt from the conversation.

    Leading system turns become the system prompt. Any system turn appearing after
    conversation has begun is preserved in place as a marked operator turn.
    """
    system_parts: List[str] = []
    conversation: List[Dict[str, str]] = []
    started = False

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            if started:
                conversation.append({"role": "user", "content": OPERATOR_PREFIX + content})
            elif content:
                system_parts.append(content)
        else:
            started = True
            conversation.append({"role": role, "content": content})

    return "\n\n".join(system_parts), conversation


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
    """Anthropic and Gemini require at least one message, starting with the user's.

    Two cases need the neutral opener. The Elicitor's interviewer speaks first, so
    the history can begin with an assistant turn. And on the *opening* turn there
    is no conversation at all -- the whole request is a system prompt plus a
    control instruction -- which would otherwise send an empty `messages` array
    and fail every session at its first question.
    """
    if not messages:
        return [{"role": "user", "content": "Please begin."}]
    if messages[0]["role"] == "assistant":
        return [{"role": "user", "content": "Please begin."}] + messages
    return messages


# Models that accept `output_config.effort`. Sending it to one that does not --
# Haiku 4.5 and the 4.5-era Sonnet -- is a hard error, so an unrecognised model id
# is treated as unsupported: omitting effort costs a little latency, sending it
# where it is not supported costs the whole session.
_EFFORT_CAPABLE = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-mythos-5",
)


def supports_effort(model: str) -> bool:
    return any((model or "").startswith(prefix) for prefix in _EFFORT_CAPABLE)


def model_belongs_to(model: str) -> Optional[str]:
    """The provider whose catalogue lists this model, if any."""
    for provider, models in MODEL_CATALOGUE.items():
        if model in models:
            return provider
    return None


def resolve_model(provider: str, model: str) -> str:
    """Return a model id that can actually be used with `provider`.

    `INTERVIEWER_MODEL` and `CONTROL_MODEL` are provider-agnostic strings, and the
    provider can be changed independently of them -- in the sidebar, or by editing
    `DEFAULT_PROVIDER`. Nothing previously stopped the two being combined into a
    pairing like `anthropic/gpt-4o`, which is not a slow or degraded
    configuration but a dead one: every call 404s and every turn shows the
    participant an apology.

    A model listed in a *different* provider's catalogue is therefore treated as a
    mistake and replaced with this provider's default. A model in no catalogue is
    passed through untouched -- that is how a newly released model id reaches the
    provider.
    """
    if not model:
        return DEFAULT_MODELS.get(provider, "")
    if model in MODEL_CATALOGUE.get(provider, []):
        return model
    if model_belongs_to(model) is not None:
        return DEFAULT_MODELS.get(provider, "")
    return model


# ------------------------------------------------------- capability negotiation
#
# The apps let any model id be configured, including ones released after this file
# was written, and providers differ on which request parameters they accept:
# OpenAI's reasoning models replaced `max_tokens` with `max_completion_tokens` and
# reject a non-default `temperature`; older Claude models predate structured
# outputs; every model has its own output ceiling.
#
# Rather than maintain a table that goes stale, each (provider, model) starts from
# a best-guess prior and is then corrected by the provider's own error messages: a
# 400 naming an unsupported parameter causes that parameter to be dropped and the
# call retried, and the finding is remembered for the rest of the process so the
# same round trip is not wasted twice. Whatever had to be dropped is recorded on
# the result and lands in the per-turn log.

# OpenAI reasoning-model families: `max_completion_tokens`, no custom temperature.
_OPENAI_REASONING = ("o1", "o3", "o4", "gpt-5", "gpt-6")

# Claude generations that predate structured outputs (`output_config.format`).
_ANTHROPIC_LEGACY = ("claude-2", "claude-3", "claude-instant")

_CAPABILITIES: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _initial_capabilities(provider: str, model: str) -> Dict[str, Any]:
    model_l = (model or "").lower()
    caps: Dict[str, Any] = {
        "max_tokens_param": "max_tokens",
        "temperature": True,
        "effort": False,
        "schema": True,
        "max_tokens_cap": None,
    }

    if provider == "openai":
        if any(model_l.startswith(p) for p in _OPENAI_REASONING):
            caps["max_tokens_param"] = "max_completion_tokens"
            caps["temperature"] = False
        # OpenAI is driven through json_object mode here, not a JSON schema.
        caps["schema"] = False
    elif provider == "anthropic":
        # Sampling parameters were removed on the current Claude generation, and
        # sending one is a 400 -- never send it to any Claude model.
        caps["temperature"] = False
        caps["effort"] = supports_effort(model)
        if any(model_l.startswith(p) for p in _ANTHROPIC_LEGACY):
            caps["schema"] = False
    elif provider == "gemini":
        caps["effort"] = False

    return caps


def capabilities(provider: str, model: str) -> Dict[str, Any]:
    key = (provider, model or "")
    if key not in _CAPABILITIES:
        _CAPABILITIES[key] = _initial_capabilities(provider, model)
    return _CAPABILITIES[key]


def _status_code(exc: Exception) -> Optional[int]:
    for attribute in ("status_code", "code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def _adapt_to_error(provider: str, model: str, exc: Exception) -> Optional[str]:
    """Drop whatever the provider just rejected. Returns what was dropped, or None.

    Only rejections that clearly name a request parameter are acted on, so a
    transient fault cannot permanently degrade a model's configuration.
    """
    status = _status_code(exc)
    if status is not None and status not in (400, 404, 422):
        return None

    text = str(exc).lower()
    caps = capabilities(provider, model)

    if "max_completion_tokens" in text and caps["max_tokens_param"] != "max_completion_tokens":
        caps["max_tokens_param"] = "max_completion_tokens"
        return "max_completion_tokens"

    if "temperature" in text and caps["temperature"]:
        caps["temperature"] = False
        return "temperature"

    # Schema is tested before effort. Both live under `output_config` on Anthropic,
    # so a complaint about the *format* mentions "output_config" too -- checking
    # effort first would drop it needlessly and still not fix the real problem.
    schema_tokens = ("json_schema", "response_schema", "structured output", "format")
    if caps["schema"] and any(token in text for token in schema_tokens):
        caps["schema"] = False
        return "json_schema"

    if caps["effort"] and ("effort" in text or "output_config" in text):
        caps["effort"] = False
        return "effort"

    # An output ceiling below what we asked for. Step down rather than guess the
    # exact limit from prose that differs per provider.
    if "max_tokens" in text and any(
        token in text for token in ("greater than", "maximum", "at most", "too large", "exceed", "<=")
    ):
        current = caps["max_tokens_cap"]
        for ceiling in (8192, 4096, 2048, 1024):
            if current is None or ceiling < current:
                caps["max_tokens_cap"] = ceiling
                return "max_tokens<=" + str(ceiling)

    return None


def _effective_max_tokens(provider: str, model: str, requested: int) -> int:
    cap = capabilities(provider, model)["max_tokens_cap"]
    return min(requested, cap) if cap else requested


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

    failure = _dispatch_with_adaptation(
        result, provider, messages, model, as_json, json_schema, max_tokens, temperature, effort, timeout
    )
    if failure is not None:
        result.ok = False
        result.error = type(failure).__name__ + ": " + str(failure)
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
                repair_failure = _dispatch_with_adaptation(
                    retry, provider, repair, model, True, json_schema, max_tokens, temperature, effort, timeout
                )
                if repair_failure is not None:
                    raise repair_failure
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


def _dispatch_with_adaptation(
    result, provider, messages, model, as_json, json_schema, max_tokens, temperature, effort, timeout
) -> Optional[Exception]:
    """Call the provider, dropping any request feature it rejects, then retrying.

    Returns the exception if the call could not be made to work, else None. Each
    dropped feature is appended to `result.degraded`, and the finding is cached for
    the process so the next call for this model gets it right first time.
    """
    for _ in range(_MAX_ADAPTATIONS + 1):
        # A retry must not inherit a half-written result from the failed attempt.
        result.text = ""
        result.finish_reason = None
        result.ok = True
        result.error = None
        try:
            _dispatch(
                result, provider, messages, model, as_json, json_schema, max_tokens, temperature, effort, timeout
            )
            return None
        except Exception as exc:  # noqa: BLE001 -- classified below
            dropped = _adapt_to_error(provider, model, exc)
            if dropped is None:
                return exc
            result.degraded.append(dropped)
    return None


# One per adaptable feature, so a model that rejects several still converges.
_MAX_ADAPTATIONS = 5


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
    caps = capabilities("openai", model)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }
    # Reasoning models renamed this parameter; the name in force is learned from
    # the provider rather than assumed.
    kwargs[caps["max_tokens_param"]] = _effective_max_tokens("openai", model, max_tokens)
    if temperature is not None and caps["temperature"]:
        kwargs["temperature"] = temperature
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}
        # OpenAI rejects json_object mode unless the word "json" appears somewhere
        # in the messages. Every agent prompt here says so already, but a
        # hand-edited prompt must not be able to break the call.
        if not any("json" in (m.get("content") or "").lower() for m in messages):
            kwargs["messages"] = list(messages) + [
                {"role": "system", "content": "Reply with a single JSON object."}
            ]

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

    caps = capabilities("anthropic", model)
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": _effective_max_tokens("anthropic", model, max_tokens),
        "messages": rest,
        "timeout": timeout,
    }
    if system:
        kwargs["system"] = system

    output_config: Dict[str, Any] = {}
    if effort and caps["effort"]:
        output_config["effort"] = effort
    if json_schema is not None and caps["schema"]:
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

    # On models where adaptive thinking is on by default, a tight `max_tokens` can
    # be spent thinking before any visible text is produced. Left unflagged that
    # arrives as an empty-but-successful response and the caller shows the
    # participant a generic apology with nothing in the log to explain it.
    if not result.text.strip():
        result.ok = False
        result.error = (
            "Empty response from %s (stop_reason=%s); max_tokens=%s may be too low for "
            "this model's thinking budget." % (model, result.finish_reason, max_tokens)
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

    caps = capabilities("gemini", model)
    cfg: Dict[str, Any] = {"max_output_tokens": _effective_max_tokens("gemini", model, max_tokens)}
    if system:
        cfg["system_instruction"] = system
    if temperature is not None and caps["temperature"]:
        cfg["temperature"] = temperature
    if as_json:
        cfg["response_mime_type"] = "application/json"
        if json_schema is not None and caps["schema"]:
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


# ----------------------------------------------------------------- preflight
#
# An exhausted or revoked API key is indistinguishable, from inside a session,
# from a model that happens to be slow: both surface as a failed call, and the
# participant is shown an apology while the reason sits in an event log nobody
# reads until the evening. At an event that is the difference between losing one
# participant and losing the whole morning.
#
# So the key is tested once, before anybody is allowed to start, with the
# cheapest call the provider offers.

_PREFLIGHT: Dict[Tuple[str, str], Tuple[bool, str]] = {}


def preflight(provider: str, model: str, *, timeout: float = 20.0, refresh: bool = False) -> Tuple[bool, str]:
    """Make one minimal real call. Returns (ok, reason).

    Cached per (provider, model) for the life of the process: this runs on page
    load, and an open day is a few hundred page loads.
    """
    key = (provider, model or "")
    if not refresh and key in _PREFLIGHT:
        return _PREFLIGHT[key]

    if not has_secret(PROVIDER_KEY_NAMES.get(provider, "")):
        outcome = (False, missing_reason(provider))
    elif _import_sdk(provider) is None:
        outcome = (False, missing_reason(provider))
    else:
        result = complete(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            provider=provider,
            model=model,
            max_tokens=1000,
            timeout=timeout,
        )
        if result.ok and (result.text or "").strip():
            outcome = (True, str(result.latency_ms) + " ms")
        else:
            outcome = (False, result.error or "The model returned an empty response.")

    _PREFLIGHT[key] = outcome
    return outcome


def explain_failure(reason: str) -> str:
    """Turn a provider error into the one action that fixes it.

    The provider's own message is accurate and unreadable at a stand with someone
    waiting. These are the failures that actually happen.
    """
    text = (reason or "").lower()
    if "credit" in text or "quota" in text or "billing" in text or "429" in text:
        return "This API key has no credit left. Top it up, or switch to a provider whose key does."
    if "api key" in text or "authentication" in text or "401" in text or "invalid_api_key" in text:
        return "This API key was rejected. Check it is current and pasted in full."
    if "not found" in text or "404" in text or "does not exist" in text:
        return "This model id does not exist for this provider. Pick one from the list."
    if "permission" in text or "403" in text:
        return "This key is not permitted to use this model."
    if "timeout" in text or "timed out" in text or "connection" in text:
        return "The provider could not be reached. Check the network."
    return "Fix the provider configuration before running the study."

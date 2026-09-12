"""Any configured model must work, including ones released after this was written.

Providers disagree about request parameters and change their minds between model
generations: OpenAI's reasoning models replaced `max_tokens` with
`max_completion_tokens` and reject a custom `temperature`; older Claude models
predate structured outputs; Claude Haiku 4.5 rejects `effort`; every model has its
own output ceiling. A hardcoded table of which model accepts what goes stale the
day a new model ships.

So the layer under test learns from the provider's own rejection: a 400 naming a
parameter causes that parameter to be dropped and the call retried, and the
finding is cached. These tests build fake models that reject each parameter (and
combinations of them) and assert the call still succeeds, that what was dropped is
recorded, and that the lesson is not re-learned on every call.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mhaestro import agents, llm, prompts, schema
from mhaestro.agents import ModelPlan
from mhaestro.telemetry import SessionLog

fails = []


def check(label, condition, extra=""):
    if not condition:
        fails.append(label)
    print(("ok   " if condition else "FAIL "), label, ("| " + str(extra)) if extra else "")


class FakeBadRequest(Exception):
    """Mimics a provider's 400 for an unsupported request parameter."""

    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400


class PickyModel:
    """A fake provider that rejects whichever parameters it was told to reject.

    `rejects` maps a request key to the error message the provider would return.
    """

    def __init__(self, provider, rejects, reply="ok", max_tokens_ceiling=None):
        self.provider = provider
        self.rejects = rejects
        self.reply = reply
        self.max_tokens_ceiling = max_tokens_ceiling
        self.calls = []
        self._install()

    # -- the shape each SDK expects -------------------------------------------

    def _check(self, kwargs):
        self.calls.append(kwargs)
        for key, message in self.rejects.items():
            if key == "effort":
                if "effort" in (kwargs.get("output_config") or {}):
                    raise FakeBadRequest(message)
            elif key == "schema":
                oc = kwargs.get("output_config") or {}
                cfg = kwargs.get("config")
                if "format" in oc or (cfg is not None and getattr(cfg, "response_schema", None)):
                    raise FakeBadRequest(message)
            elif key in kwargs:
                raise FakeBadRequest(message)
        ceiling = self.max_tokens_ceiling
        if ceiling is not None:
            asked = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0
            if asked > ceiling:
                raise FakeBadRequest(
                    "max_tokens: %d is greater than the maximum of %d for this model" % (asked, ceiling)
                )

    def _install(self):
        outer = self

        def _usage():
            class _U:
                input_tokens = output_tokens = prompt_tokens = completion_tokens = 10
                prompt_token_count = candidates_token_count = 10

            return _U()

        if self.provider == "anthropic":
            class _Messages:
                def create(self, **kwargs):
                    outer._check(kwargs)

                    class _Block:
                        type = "text"
                        text = outer.reply  # resolved per call, not at construction

                    class _R:
                        stop_reason = "end_turn"
                        stop_details = None
                        content = [_Block()]
                        usage = _usage()

                    return _R()

            class _C:
                messages = _Messages()

            llm._CLIENTS["anthropic"] = _C()

        elif self.provider == "openai":
            class _Completions:
                def create(self, **kwargs):
                    outer._check(kwargs)

                    class _Choice:
                        finish_reason = "stop"

                        class message:
                            content = outer.reply

                    class _R:
                        choices = [_Choice()]
                        usage = _usage()

                    return _R()

            class _Chat:
                completions = _Completions()

            class _C:
                chat = _Chat()

            llm._CLIENTS["openai"] = _C()

        else:
            class _Models:
                def generate_content(self, **kwargs):
                    outer._check(kwargs)

                    class _R:
                        text = outer.reply
                        usage_metadata = _usage()
                        candidates = []

                    return _R()

            class _C:
                models = _Models()

            llm._CLIENTS["gemini"] = _C()


def fresh(provider, model, rejects, reply="ok", ceiling=None):
    """Clear any learned capabilities so each scenario starts from the prior."""
    llm._CAPABILITIES.pop((provider, model), None)
    return PickyModel(provider, rejects, reply=reply, max_tokens_ceiling=ceiling)


def call(provider, model, **kw):
    return llm.complete(
        [{"role": "user", "content": "Say ok. Reply as JSON if asked."}],
        provider=provider, model=model, max_tokens=1000, **kw
    )


# =================================================== one awkward model at a time
print("=== a model that rejects a single parameter still works ===")

SCENARIOS = [
    # No prefix rule covers this one, so the prior guesses `max_tokens` and the
    # correct parameter name has to be learned from the rejection.
    ("openai", "gpt-4o-next",
     {"max_tokens": "Unsupported parameter: 'max_tokens' is not supported with this "
                    "model. Use 'max_completion_tokens' instead."},
     {"temperature": 0.6}, "max_completion_tokens"),
    ("openai", "o9-hypothetical",
     {"temperature": "Unsupported value: 'temperature' does not support 0.6 with this model."},
     {"temperature": 0.6}, "temperature"),
    # Prefix matches the effort allow-list, so effort IS attempted and must be
    # unlearned from the provider's rejection.
    ("anthropic", "claude-opus-5-preview",
     {"effort": "output_config.effort: unsupported parameter for this model"},
     {"effort": "low"}, "effort"),
    ("anthropic", "claude-sonnet-5-experimental",
     {"schema": "output_config.format: json_schema is not supported on this model"},
     {"json_schema": {"type": "object"}}, "json_schema"),
    ("gemini", "gemini-future-1",
     {"schema": "response_schema is not supported for this model"},
     {"json_schema": {"type": "object"}}, "json_schema"),
]

for provider, model, rejects, kwargs, expect_dropped in SCENARIOS:
    reply = '{"ok": true}' if "json_schema" in kwargs else "ok"
    fake = fresh(provider, model, rejects, reply=reply)
    result = call(provider, model, **kwargs)
    check("%s/%s recovers from rejected %s" % (provider, model, expect_dropped),
          result.ok, result.error)
    check("  ...and records what it dropped",
          expect_dropped in result.degraded, result.degraded)
    # Nothing beyond the actual culprit should be given up: a complaint about the
    # response format must not also cost the effort parameter.
    check("  ...and drops nothing else", result.degraded == [expect_dropped], result.degraded)


print("\n=== a correct prior costs no failed round trip at all ===")

# These models are known in advance to reject the parameter, so it is never sent
# and there is nothing to recover from. Cheaper than learning, and the point of
# keeping a prior alongside the adaptation.
PRIORS = [
    ("openai", "gpt-5-hypothetical",
     {"max_tokens": "Use 'max_completion_tokens' instead."}, {"temperature": 0.6}),
    ("anthropic", "claude-haiku-4-5", {"effort": "effort unsupported"}, {"effort": "low"}),
    ("anthropic", "claude-3-legacy", {"schema": "json_schema unsupported"},
     {"json_schema": {"type": "object"}}),
    ("anthropic", "claude-haiku-4-5-x", {"temperature": "temperature removed"}, {"temperature": 0.6}),
]
for provider, model, rejects, kwargs in PRIORS:
    reply = '{"ok": true}' if "json_schema" in kwargs else "ok"
    fake = fresh(provider, model, rejects, reply=reply)
    result = call(provider, model, **kwargs)
    check("%s/%s: prior avoids the bad request entirely" % (provider, model), result.ok, result.error)
    check("  ...in a single round trip", len(fake.calls) == 1, len(fake.calls))
    check("  ...with nothing dropped", not result.degraded, result.degraded)


# ============================================ several rejections simultaneously
print("\n=== a model that rejects several parameters at once ===")

fake = fresh("openai", "o9-strict", {
    "max_tokens": "Use 'max_completion_tokens' instead.",
    "temperature": "'temperature' does not support 0.6 with this model.",
})
result = call("openai", "o9-strict", temperature=0.6)
check("recovers from two rejections in sequence", result.ok, result.error)
check("both are recorded", set(result.degraded) == {"max_completion_tokens", "temperature"},
      result.degraded)
final = fake.calls[-1]
check("the surviving request uses max_completion_tokens",
      "max_completion_tokens" in final and "max_tokens" not in final, sorted(final.keys()))
check("the surviving request has no temperature", "temperature" not in final, sorted(final.keys()))


# ================================================================ output ceiling
print("\n=== a model with a smaller output ceiling than we asked for ===")

fake = fresh("anthropic", "claude-small-ceiling", {}, ceiling=2048)
result = llm.complete([{"role": "user", "content": "hi"}],
                      provider="anthropic", model="claude-small-ceiling", max_tokens=8000)
check("steps max_tokens down until it fits", result.ok, result.error)
check("the cap is recorded", any(d.startswith("max_tokens<=") for d in result.degraded), result.degraded)
check("the final request is within the ceiling", fake.calls[-1]["max_tokens"] <= 2048,
      fake.calls[-1]["max_tokens"])


# ==================================================== the lesson is only learnt once
print("\n=== a learned limitation is cached, not re-discovered every turn ===")

fake = fresh("anthropic", "claude-opus-5-picky", {"effort": "effort unsupported"})
first = call("anthropic", "claude-opus-5-picky", effort="low")
calls_after_first = len(fake.calls)
second = call("anthropic", "claude-opus-5-picky", effort="low")
calls_after_second = len(fake.calls)

check("first call pays one failed round trip", calls_after_first == 2, calls_after_first)
check("second call costs a single round trip",
      calls_after_second - calls_after_first == 1, calls_after_second - calls_after_first)
check("second call is clean (nothing left to drop)", second.ok and not second.degraded, second.degraded)


# ======================================================= genuine faults still fail
print("\n=== faults that are not about parameters are still reported ===")

llm._CAPABILITIES.pop(("anthropic", "claude-missing"), None)


class _Missing:
    class messages:
        @staticmethod
        def create(**kwargs):
            err = Exception("model: claude-missing not found")
            err.status_code = 404
            raise err


llm._CLIENTS["anthropic"] = _Missing()
result = llm.complete([{"role": "user", "content": "hi"}],
                      provider="anthropic", model="claude-missing", max_tokens=100)
check("an unknown model id fails loudly rather than silently degrading",
      result.ok is False and "not found" in (result.error or ""), result.error)
check("nothing was dropped chasing it", not result.degraded, result.degraded)


class _Down:
    class messages:
        @staticmethod
        def create(**kwargs):
            err = Exception("overloaded_error: the service is temporarily overloaded")
            err.status_code = 529
            raise err


llm._CAPABILITIES.pop(("anthropic", "claude-haiku-4-5"), None)
llm._CLIENTS["anthropic"] = _Down()
result = llm.complete([{"role": "user", "content": "hi"}],
                      provider="anthropic", model="claude-haiku-4-5", max_tokens=100)
check("a transient outage does not permanently degrade the model",
      result.ok is False and not result.degraded, (result.error, result.degraded))
check("  ...and its capabilities are untouched",
      llm.capabilities("anthropic", "claude-haiku-4-5")["schema"] is True)


# ============================================ the agents work on an awkward model
print("\n=== the whole agent layer on a maximally awkward model ===")

POLICY = schema.normalise_policy(
    json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
)
NODE = schema.get_node(POLICY, POLICY["root_id"])

verdict_json = json.dumps({
    "adequate": True, "on_topic": True, "wants_to_move_on": False, "missing": "",
    "suggested_probe": "", "distress_signal": False, "confidence": 0.9, "reason": "ok",
    "condition": "named_specific_activities", "justification": "named the robotics lab",
})

fake = fresh("anthropic", "claude-awkward", {
    "effort": "effort is not supported on this model",
    "schema": "output_config.format is not supported on this model",
}, reply=verdict_json, ceiling=2048)

plan = ModelPlan(interviewer_provider="anthropic", interviewer_model="claude-awkward",
                 control_provider="anthropic", control_model="claude-awkward")
log = SessionLog(tool="elicitor")

text, res = agents.ask_interviewer(
    log, plan, system_prompt=prompts.ELICITOR_INTERVIEWER_TREE,
    system_values={"context": POLICY["description"], "scope_note": POLICY["scope_note"],
                   "steering_note": POLICY["steering_note"]},
    history=[], control_instruction="CONTROL: ask the opening question", node_id=NODE["id"])
check("Agent [a] works on the awkward model", res.ok and text and "went wrong" not in text, res.error)

verdict = agents.check_adequacy(log, plan, transcript="P: the robotics lab", node=NODE, reprobe_count=0)
check("Agent [b] works without structured outputs",
      verdict["adequate"] is True and verdict["fallback_used"] is False, verdict)

next_id, detail = agents.choose_branch(
    log, plan, policy=POLICY, node_id=NODE["id"], path=[NODE["id"]], transcript="P: the robotics lab")
check("Agent [c] works and picks a declared label",
      detail["outcome"] == "classified" and next_id == "ask_overall", detail)

check("every agent call succeeded", all(e.get("ok", True) for e in log.events),
      [e["error"] for e in log.events if not e.get("ok", True)])
check("the degradation is on the audit trail",
      any("json_schema" in (e.get("llm_degraded") or "") for e in log.events),
      [e.get("llm_degraded") for e in log.events])
check("events CSV carries the llm_degraded column", "llm_degraded" in log.events_csv().splitlines()[0])


# ============================================== every catalogue model is buildable
print("\n=== every model in the catalogue produces a valid request ===")

for provider, models in llm.MODEL_CATALOGUE.items():
    for model in models:
        fresh(provider, model, {})
        result = call(provider, model, effort="low", temperature=0.5)
        check("%s/%s builds and succeeds" % (provider, model), result.ok, result.error)

print("\n" + ("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)

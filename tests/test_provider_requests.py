"""Validate the request MHAESTRO builds for each provider, without a real key.

The apps failed in the field with "Sorry -- something went wrong at our end" on
every Anthropic turn, and nothing in the UI said why. Three separate faults were
responsible, all of them invisible until the request itself was inspected:

  1. The opening turn is a system prompt plus a control instruction and nothing
     else, so `messages` went out empty -- which Anthropic and Gemini reject.
  2. `output_config.effort` was sent to every Claude model, but Haiku 4.5 does not
     accept that parameter and errors on it.
  3. A control instruction is a system message arriving mid-conversation. Folding
     it into Anthropic's separate `system` field hoisted it to the top of the
     prompt and stripped it of the end-of-conversation position it needs to
     compel a re-probe.

These tests hold the request shape to each provider's actual rules so a
regression shows up here rather than in front of a visitor.
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


# ------------------------------------------------------------------ fake clients

CAPTURED = []


def _reset():
    CAPTURED.clear()


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 500
    output_tokens = 20
    prompt_tokens = 500
    completion_tokens = 20
    prompt_token_count = 500
    candidates_token_count = 20


class FakeAnthropic:
    """Stands in for anthropic.Anthropic()."""

    def __init__(self, reply="A short question?"):
        self.reply = reply
        outer = self

        class _Messages:
            def create(self, **kwargs):
                CAPTURED.append(("anthropic", kwargs))

                class _R:
                    stop_reason = "end_turn"
                    stop_details = None
                    content = [_Block(outer.reply)]
                    usage = _Usage()

                return _R()

        self.messages = _Messages()


class FakeOpenAI:
    def __init__(self, reply="A short question?"):
        outer = self
        self.reply = reply

        class _Completions:
            def create(self, **kwargs):
                CAPTURED.append(("openai", kwargs))

                class _Choice:
                    finish_reason = "stop"

                    class message:
                        content = outer.reply

                class _R:
                    choices = [_Choice()]
                    usage = _Usage()

                return _R()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class FakeGemini:
    def __init__(self, reply="A short question?"):
        outer = self
        self.reply = reply

        class _Models:
            def generate_content(self, **kwargs):
                CAPTURED.append(("gemini", kwargs))

                class _R:
                    text = outer.reply
                    usage_metadata = _Usage()
                    candidates = []

                return _R()

        self.models = _Models()


POLICY = schema.normalise_policy(
    json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
)
ROOT_NODE = schema.get_node(POLICY, POLICY["root_id"])

SYSTEM_VALUES = {
    "context": POLICY["description"],
    "scope_note": POLICY["scope_note"],
    "steering_note": POLICY["steering_note"],
}


def interviewer_request(provider, model, history, control):
    """Run Agent [a] against a fake client and return the captured request."""
    _reset()
    llm._CLIENTS[provider] = {"anthropic": FakeAnthropic, "openai": FakeOpenAI, "gemini": FakeGemini}[provider]()
    plan = ModelPlan(interviewer_provider=provider, interviewer_model=model,
                     control_provider=provider, control_model=model)
    agents.ask_interviewer(
        SessionLog(tool="elicitor"), plan,
        system_prompt=prompts.ELICITOR_INTERVIEWER_TREE,
        system_values=SYSTEM_VALUES,
        history=history, control_instruction=control, node_id=ROOT_NODE["id"],
    )
    return CAPTURED[-1][1]


OPENING = 'CONTROL: This is the first turn. Greet them, then ask: "' + ROOT_NODE["question"] + '"'
HISTORY = [
    {"role": "assistant", "content": "Which activities did you take part in today?"},
    {"role": "user", "content": "The robotics lab."},
]
REPROBE = "CONTROL: They have not said why it stood out. Ask once for that."


# ============================================================ the opening turn
print("=== the opening turn: history is empty, everything is a system message ===")

for provider, model, key in (
    ("anthropic", "claude-haiku-4-5", "messages"),
    ("gemini", "gemini-2.5-flash", "contents"),
):
    req = interviewer_request(provider, model, [], OPENING)
    payload = req.get(key)
    check(provider + ": " + key + " is never empty", bool(payload), payload)
    if provider == "anthropic":
        check("anthropic: opening turn starts with a user message",
              payload and payload[0]["role"] == "user", payload[0] if payload else None)
    else:
        check("gemini: opening turn starts with a user role",
              payload and payload[0].role == "user", payload[0].role if payload else None)

req = interviewer_request("openai", "gpt-4o", [], OPENING)
check("openai: system-only opening is left alone (it is valid there)",
      [m["role"] for m in req["messages"]] == ["system", "system"],
      [m["role"] for m in req["messages"]])


# ==================================================== effort parameter gating
print("\n=== effort is only sent to models that accept it ===")

check("haiku 4.5 does not accept effort", not llm.supports_effort("claude-haiku-4-5"))
check("sonnet 4.5 does not accept effort", not llm.supports_effort("claude-sonnet-4-5"))
check("an unknown/future model id is treated as unsupported (omitting is safe)",
      not llm.supports_effort("claude-something-new"))
check("opus 5 accepts effort", llm.supports_effort("claude-opus-5"))
check("sonnet 5 accepts effort", llm.supports_effort("claude-sonnet-5"))

req = interviewer_request("anthropic", "claude-haiku-4-5", HISTORY, REPROBE)
check("haiku request carries no effort key",
      "effort" not in (req.get("output_config") or {}), req.get("output_config"))

req = interviewer_request("anthropic", "claude-sonnet-5", HISTORY, REPROBE)
check("sonnet-5 request does carry effort",
      (req.get("output_config") or {}).get("effort") == "low", req.get("output_config"))

check("temperature is never sent to Anthropic (removed on current models; 400s)",
      "temperature" not in req, sorted(req.keys()))


# ============================================ control instruction positioning
print("\n=== the re-probe instruction must arrive last, not at the top ===")

req = interviewer_request("anthropic", "claude-haiku-4-5", HISTORY, REPROBE)
system_text = req.get("system", "")
msgs = req["messages"]
check("anthropic: control instruction is NOT folded into the system prompt",
      REPROBE not in system_text)
check("anthropic: control instruction is in the final message",
      REPROBE in msgs[-1]["content"], msgs[-1]["content"][-90:])
check("anthropic: it is marked as an operator instruction, not the participant",
      llm.OPERATOR_PREFIX.split("\n")[0] in msgs[-1]["content"])
check("anthropic: the standing system prompt is still the system field",
      "conversational survey" in system_text)
check("anthropic: roles alternate legally", [m["role"] for m in msgs] == ["user", "assistant", "user"],
      [m["role"] for m in msgs])

req = interviewer_request("gemini", "gemini-2.5-flash", HISTORY, REPROBE)
last = req["contents"][-1]
check("gemini: control instruction is in the final turn",
      REPROBE in last.parts[0].text, last.role)
check("gemini: system_instruction holds only the standing prompt",
      REPROBE not in (req["config"].system_instruction or ""))

req = interviewer_request("openai", "gpt-4o", HISTORY, REPROBE)
check("openai: control instruction stays a trailing system message (unchanged)",
      req["messages"][-1]["role"] == "system" and req["messages"][-1]["content"] == REPROBE,
      req["messages"][-1]["role"])


# ================================================== empty response is a failure
print("\n=== an empty response is reported, not shown as a silent apology ===")

_reset()
llm._CLIENTS["anthropic"] = FakeAnthropic(reply="")
plan = ModelPlan(interviewer_provider="anthropic", interviewer_model="claude-opus-5",
                 control_provider="anthropic", control_model="claude-opus-5")
log = SessionLog(tool="elicitor")
text, result = agents.ask_interviewer(
    log, plan, system_prompt=prompts.ELICITOR_INTERVIEWER_TREE,
    system_values=SYSTEM_VALUES, history=HISTORY, control_instruction=REPROBE, node_id="n",
)
check("an empty completion is marked not-ok", result.ok is False)
check("the error names the likely cause", "max_tokens" in (result.error or ""), result.error)
check("the failure is logged, not swallowed",
      any(e["event_type"] == "agent_a_question" and e["ok"] is False for e in log.events))


# ================================================================== the defaults
print("\n=== defaults ===")
check("default provider is anthropic", llm.DEFAULT_PROVIDER == "anthropic", llm.DEFAULT_PROVIDER)
check("default anthropic model is haiku 4.5",
      llm.DEFAULT_MODELS["anthropic"] == "claude-haiku-4-5", llm.DEFAULT_MODELS["anthropic"])
check("haiku is first in the anthropic catalogue",
      llm.MODEL_CATALOGUE["anthropic"][0] == "claude-haiku-4-5", llm.MODEL_CATALOGUE["anthropic"])
check("a bare ModelPlan uses them",
      ModelPlan().interviewer_model == "claude-haiku-4-5", ModelPlan().as_dict())

# The default must itself be a configuration that actually works.
req = interviewer_request(llm.DEFAULT_PROVIDER, llm.DEFAULT_MODELS[llm.DEFAULT_PROVIDER], [], OPENING)
check("the shipped default produces a valid opening request",
      bool(req.get("messages")) and "effort" not in (req.get("output_config") or {})
      and "temperature" not in req,
      {"messages": req.get("messages"), "output_config": req.get("output_config")})


# ======================================== control agents on the default model
print("\n=== control agents on the default model ===")

for name, fn in (
    ("adequacy", lambda p: agents.check_adequacy(
        SessionLog(), p, transcript="Interviewer: hi\n\nParticipant: the robotics lab",
        node=ROOT_NODE, reprobe_count=0)),
    ("traversal", lambda p: agents.choose_branch(
        SessionLog(), p, policy=POLICY, node_id="ask_overall", path=["ask_activities"],
        transcript="Interviewer: hi\n\nParticipant: good")),
):
    _reset()
    payload = {"adequate": True, "on_topic": True, "wants_to_move_on": False, "missing": "",
               "suggested_probe": "", "distress_signal": False, "confidence": 0.9, "reason": "ok",
               "condition": "clear_view_with_reason", "justification": "said why"}
    llm._CLIENTS["anthropic"] = FakeAnthropic(reply=json.dumps(payload))
    plan = ModelPlan(control_provider="anthropic", control_model="claude-haiku-4-5")
    fn(plan)
    req = CAPTURED[-1][1]
    check(name + ": no effort on haiku", "effort" not in (req.get("output_config") or {}),
          req.get("output_config"))
    check(name + ": messages non-empty and user-first",
          req["messages"] and req["messages"][0]["role"] == "user",
          [m["role"] for m in req["messages"]])
    check(name + ": json schema is attached",
          "format" in (req.get("output_config") or {}), list((req.get("output_config") or {}).keys()))

print("\n" + ("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)

"""What happens when the model provider is not there.

Three failures were possible before these tests existed, and all three were
silent:

1.  A key with no credit left turned every turn into "Sorry -- something went
    wrong at our end. Could you say that again?" -- including the very first
    turn, where it asks the participant to repeat something they never said.
2.  Nothing checked the provider before a participant was asked to consent, so
    a dead station looked identical to a working one until somebody used it.
3.  The free-form arms ended after a fixed number of *replies*, so every
    re-probe cost arm C a topic that arm D still got. The governance factor was
    silently also a topic-exposure manipulation.
"""
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ["RESEARCHER_PIN"] = "test-pin"

from streamlit.testing.v1 import AppTest

from _helpers import effective_secret
from mhaestro import agents, llm, schema
from mhaestro.telemetry import EV_QUESTION, SessionLog

PIN = effective_secret("RESEARCHER_PIN", "test-pin")
ELICIT = str(ROOT / "knowledge-elicitation" / "app.py")
POLICY = schema.normalise_policy(
    json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
)
ROOT_QUESTION = schema.get_node(POLICY, POLICY["root_id"])["question"]
CORE = schema.core_nodes(POLICY)

APOLOGY = "went wrong at our end"

fails = []


def check(label, condition, extra=""):
    if not condition:
        fails.append(label)
    print(("ok   " if condition else "FAIL "), label, ("| " + str(extra)) if extra else "")


def ss(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


# ===================================================== 1. the interviewer agent
print("\n=== Agent [a] with a dead provider ===")


def dead(messages, **kwargs):
    return llm.LLMResult(
        provider="openai", model="gpt-4o", ok=False,
        error="RateLimitError: Error code: 429 - insufficient_quota",
    )


agents.complete = dead
log = SessionLog(tool="test")
plan = agents.ModelPlan()

text, result = agents.ask_interviewer(
    log, plan,
    system_prompt=agents.P.ELICITOR_INTERVIEWER_TREE,
    system_values={"context": "an open day", "scope_note": "", "steering_note": ""},
    history=[],
    control_instruction="CONTROL: first turn",
    node_id="ask_activities",
    fallback="Welcome. " + ROOT_QUESTION,
)

check("a failed call does not apologise to the participant", APOLOGY not in text, text[:70])
check("it asks the scripted question instead", ROOT_QUESTION in text, text[:70])
check("the substitution is flagged on the result", result.substituted is True)

detail = json.loads(log.events[-1]["detail_json"])
check("the event records the substitution", detail.get("scripted_fallback") is True)
check("the event is still marked failed", log.events[-1]["ok"] is False)
check("the real provider error survives in the log", "429" in (log.events[-1]["error"] or ""))
check("the session record counts scripted turns", log.derived()["n_scripted_questions"] == 1,
      log.derived()["n_scripted_questions"])

# With no fallback supplied at all it must still not apologise.
text2, _ = agents.ask_interviewer(
    log, plan,
    system_prompt=agents.P.ELICITOR_INTERVIEWER_TREE,
    system_values={"context": "an open day", "scope_note": "", "steering_note": ""},
    history=[], node_id="", fallback="",
)
check("even with no fallback it stays a question, not an apology",
      APOLOGY not in text2 and text2.strip().endswith("?"), text2[:70])


# ============================================== 2. the opening turn, end to end
print("\n=== pressing Begin with a dead provider ===")


def begin_session_under(arm_id, complete_fn, *, preflight_ok=True):
    """Drive the app to the first interviewer message."""
    agents.complete = complete_fn
    llm.preflight = lambda p, m, **kw: (preflight_ok, "429 insufficient_quota" if not preflight_ok else "10 ms")
    at = AppTest.from_file(ELICIT, default_timeout=120)
    at.run()
    if not at.sidebar.text_input:
        return at  # blocked before the sidebar even rendered
    at.sidebar.text_input[0].set_value(PIN)
    at.run()
    force = [s for s in at.sidebar.selectbox if "Force arm" in s.label]
    if force:
        force[0].set_value(arm_id)
        at.run()
    if not at.radio:
        return at
    at.radio[0].set_value("I am 18 years old or over").run()
    [c for c in at.checkbox if "voluntarily agree" in c.label][0].set_value(True).run()
    [b for b in at.button if b.label == "Begin"][0].click().run()
    return at


os.environ["PREFLIGHT"] = "off"
at = begin_session_under("A", dead)
check("no exception on Begin", not at.exception, [e.value for e in at.exception][:1])
messages = ss(at, "messages", [])
first = messages[0]["content"] if messages else ""
check("arm A: Begin produces a first message at all", bool(first.strip()))
check("arm A: that message is a welcome, not an error", APOLOGY not in first, first[:80])
check("arm A: it greets the participant", "thank" in first.lower() or "welcome" in first.lower(), first[:60])
check("arm A: it states how long it takes", "minute" in first.lower(), first[:80])
check("arm A: it asks the root node's question", ROOT_QUESTION in first, first[:80])
check("arm A: the session is live, not stalled", ss(at, "phase") == "interview", ss(at, "phase"))

at = begin_session_under("C", dead)
first = (ss(at, "messages", [{}])[0] or {}).get("content", "")
check("arm C: Begin also produces a welcome, not an error", first.strip() and APOLOGY not in first, first[:80])


# ================================================================ 3. preflight
print("\n=== preflight ===")

os.environ["PREFLIGHT"] = "on"
at = begin_session_under("A", dead, preflight_ok=False)
check("a dead provider blocks the study before consent", not at.radio, len(at.radio))
check("no session was created", ss(at, "log") is None)
body = " ".join(str(m.value) for m in at.info) + " ".join(str(e.value) for e in at.error)
check("the participant is told calmly, with no stack trace", "not collecting responses" in body, body[:80])
check("the researcher is told the actual fault", "429" in body or "insufficient_quota" in body, body[-120:])

os.environ["PREFLIGHT"] = "off"

check("a credit failure is explained in one line",
      "credit" in llm.explain_failure("429 insufficient_quota credit_balance_exhausted").lower())
check("a bad key is explained differently",
      "key" in llm.explain_failure("401 invalid_api_key").lower())


# ================================== 4. governance must not cost topic exposure
print("\n=== free-form arms: re-probes must not eat topics ===")

class FreeformFake:
    """Just enough model to drive a free-form arm to its end.

    `always_inadequate` makes every answer fail the check, which the soft cap then
    releases after one re-probe -- so arm C spends two replies per topic where arm
    D spends one. That is precisely the condition under which counting replies
    instead of topics would truncate arm C.
    """

    def __init__(self, adequate):
        self.adequate = adequate

    def __call__(self, messages, **kwargs):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if "steering agent auditing" in system:
            return self._json({
                "adequate": self.adequate, "on_topic": True, "wants_to_move_on": False,
                "missing": "" if self.adequate else "a reason",
                "suggested_probe": "" if self.adequate else "Ask what made them feel that way.",
                "distress_signal": False, "confidence": 0.8, "reason": "scripted",
            })
        if "summariser producing" in system:
            constructs = re.findall(r"^- (.+)$", system, re.MULTILINE)
            return self._json({"answers": [
                {"construct": c, "summary": "noted", "evidence_quote": ""} for c in constructs
            ]})
        if "coding a survey transcript" in system:
            topics = re.findall(r"^- (.+)$", system, re.MULTILINE)
            return self._json({"coded": [
                {"topic": t, "status": "covered", "evidence_quote": ""} for t in topics
            ]})
        return llm.LLMResult(text="So, tell me a bit about that?", provider="fake", model="fake-1", ok=True)

    def _json(self, payload):
        return llm.LLMResult(
            text=json.dumps(payload), data=payload, provider="fake", model="fake-1", ok=True
        )


def run_freeform(arm_id, adequate):
    agents.complete = FreeformFake(adequate)
    llm.preflight = lambda p, m, **kw: (True, "10 ms")
    at = AppTest.from_file(ELICIT, default_timeout=120)
    at.run()
    at.sidebar.text_input[0].set_value(PIN)
    at.run()
    [s for s in at.sidebar.selectbox if "Force arm" in s.label][0].set_value(arm_id)
    at.run()
    at.radio[0].set_value("I am 18 years old or over").run()
    [c for c in at.checkbox if "voluntarily agree" in c.label][0].set_value(True).run()
    [b for b in at.button if b.label == "Begin"][0].click().run()
    replies = 0
    while ss(at, "phase") == "interview" and replies < 40:
        replies += 1
        at.chat_input[0].set_value("Answer %d about the robotics lab." % replies).run()
        if at.exception:
            raise AssertionError(str(at.exception[0].value))
    return at, replies


expected_topics = max(4, len(CORE))

at_c, replies_c = run_freeform("C", False)
topics_c = ss(at_c, "topics_asked")
at_d, replies_d = run_freeform("D", True)
topics_d = ss(at_d, "topics_asked")

check("arm C reaches the full topic list despite re-probing", topics_c == expected_topics,
      "%s of %s" % (topics_c, expected_topics))
check("arm D reaches the full topic list", topics_d == expected_topics,
      "%s of %s" % (topics_d, expected_topics))
check("governance does not change topic exposure", topics_c == topics_d, (topics_c, topics_d))
check("arm C did genuinely re-probe (otherwise the test proves nothing)",
      replies_c > replies_d, "C=%d replies, D=%d replies" % (replies_c, replies_d))

print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)

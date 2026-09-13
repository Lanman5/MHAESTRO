"""Drive complete sessions through all four arms with a scripted fake provider.

Exercises the parts no screen test reaches: the adequacy gate, the re-probe loop,
the soft cap, labelled-edge traversal, traversal-error recovery, the skip control,
and coverage accounting.
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
# The study controls fail closed without a PIN, so the harness supplies one.
os.environ["RESEARCHER_PIN"] = "test-pin"
# The key above is fake, so the Elicitor's live preflight would block every
# screen. Arm behaviour is what this file tests; the preflight has its own.
os.environ["PREFLIGHT"] = "off"

from streamlit.testing.v1 import AppTest

from _helpers import effective_secret
from mhaestro import agents, arms, llm, schema

PIN = effective_secret("RESEARCHER_PIN", "test-pin")

ELICIT = str(ROOT / "knowledge-elicitation" / "app.py")
POLICY = schema.normalise_policy(
    json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
)
CORE = set(schema.core_nodes(POLICY))

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


# ------------------------------------------------------------- the fake model


class Fake:
    """Scripted stand-in for `mhaestro.llm.complete`."""

    def __init__(self, *, adequacy="one_reprobe", bad_label=False):
        self.adequacy = adequacy
        self.bad_label = bad_label
        self.seen = {}
        self.calls = []

    def __call__(self, messages, **kwargs):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        self.calls.append(system[:60])

        if "steering agent auditing" in system:
            return self._adequacy(system)
        if "traversal agent" in system:
            return self._traversal(system)
        if "summariser producing" in system:
            return self._summary(system)
        if "coding a survey transcript" in system:
            return self._coverage(system)
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if "at most 25 words" in user:
            return self._result("We ask so we can act on what visitors tell us.")
        return self._result("So, tell me a bit about that?")

    def _result(self, text, data=None):
        return llm.LLMResult(
            text=text, data=data, provider="fake", model="fake-1",
            latency_ms=120, input_tokens=900, output_tokens=40, ok=True,
        )

    def _adequacy(self, system):
        node = re.search(r"current node's question: (.+)", system)
        key = node.group(1) if node else "?"
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.adequacy == "always_inadequate":
            adequate = False
        elif self.adequacy == "always_adequate":
            adequate = True
        else:  # one re-probe per node
            adequate = self.seen[key] > 1
        payload = {
            "adequate": adequate, "on_topic": True, "wants_to_move_on": False,
            "missing": "" if adequate else "a reason for that view",
            "suggested_probe": "" if adequate else "Ask what made them feel that way.",
            "distress_signal": False, "confidence": 0.8,
            "reason": "ok" if adequate else "no supporting reason given",
        }
        return self._result(json.dumps(payload), payload)

    def _traversal(self, system):
        labels = re.findall(r"^- ([a-z_]+) :", system, re.MULTILINE)
        chosen = "not_a_real_label" if self.bad_label else (labels[0] if labels else "")
        payload = {"condition": chosen, "justification": "scripted", "confidence": 0.9}
        return self._result(json.dumps(payload), payload)

    def _summary(self, system):
        constructs = re.findall(r"^- (.+)$", system, re.MULTILINE)
        payload = {"answers": [
            {"construct": c, "summary": "You said something about this.", "evidence_quote": ""}
            for c in constructs
        ]}
        return self._result(json.dumps(payload), payload)

    def _coverage(self, system):
        topics = re.findall(r"^- (.+)$", system, re.MULTILINE)
        payload = {"coded": [{"topic": t, "status": "covered", "evidence_quote": ""} for t in topics]}
        return self._result(json.dumps(payload), payload)


def run_session(arm_id, *, fake, replies=40, skip_at=None):
    agents.complete = fake
    at = AppTest.from_file(ELICIT, default_timeout=120)
    at.run()

    assert not at.sidebar.selectbox, "study controls must stay hidden until the PIN is entered"
    at.sidebar.text_input[0].set_value(PIN)
    at.run()

    force = [s for s in at.sidebar.selectbox if "Force arm" in s.label]
    assert force, "force-arm control missing"
    force[0].set_value(arm_id)
    at.run()

    at.radio[0].set_value("I am 18 years old or over").run()
    [c for c in at.checkbox if "voluntarily agree" in c.label][0].set_value(True).run()
    [b for b in at.button if b.label == "Begin"][0].click().run()

    turns = 0
    while ss(at, "phase") == "interview" and turns < replies:
        turns += 1
        if skip_at and turns == skip_at:
            [b for b in at.button if b.label == "Skip this one"][0].click().run()
            continue
        at.chat_input[0].set_value("Answer number %d about the robotics lab." % turns).run()
        if at.exception:
            raise AssertionError("arm %s turn %d raised: %s" % (arm_id, turns, at.exception[0].value))
    return at, turns


# Collected per arm so the cross-arm invariants can be checked at the end.
ARM_RECORDS = {}

# ================================================================= arm by arm
for arm_id, expect_adequacy, expect_traversal in (
    ("A", True, True),
    ("B", False, True),
    ("C", True, False),
    ("D", False, False),
):
    print("\n=== arm %s ===" % arm_id)
    fake = Fake()
    at, turns = run_session(arm_id, fake=fake)
    log = ss(at, "log")
    types = [e["event_type"] for e in log.events]

    check("arm %s completes the session" % arm_id, ss(at, "phase") == "feedback", ss(at, "phase"))
    check("arm %s terminates in a sane number of turns" % arm_id, 3 <= turns <= 20, turns)
    check("arm %s adequacy checks %s" % (arm_id, "run" if expect_adequacy else "never run"),
          ("agent_b_adequacy" in types) == expect_adequacy, types.count("agent_b_adequacy"))
    check("arm %s traversal %s" % (arm_id, "runs" if expect_traversal else "never runs"),
          ("agent_c_traversal" in types) == expect_traversal, types.count("agent_c_traversal"))
    check("arm %s meta records the factors" % arm_id,
          log.meta["arm_structure"] == ("tree" if expect_traversal else "freeform")
          and log.meta["arm_governance"] == ("adequacy" if expect_adequacy else "none"))

    if expect_traversal:
        visited = set(ss(at, "visited"))
        check("arm %s covers every core node" % arm_id, CORE <= visited, sorted(CORE - visited))
        check("arm %s records the path" % arm_id, len(log.meta["path_taken"]) >= 6, log.meta["path_taken"])
    if expect_adequacy:
        check("arm %s re-probes then advances" % arm_id, "reprobe_issued" in types, types.count("reprobe_issued"))

    check("arm %s codes coverage regardless of structure" % arm_id,
          log.meta.get("coverage_coded_rate") == 1.0, log.meta.get("coverage_coded_rate"))
    check("arm %s produced summaries" % arm_id, len(ss(at, "summaries", [])) == 4, len(ss(at, "summaries", [])))
    check("arm %s turns carry per-turn measures" % arm_id,
          all("reply_words" in t and "t_elapsed_s" in t for t in log.turns), len(log.turns))

    ARM_RECORDS[arm_id] = log

# ================================================ invariants that make arms comparable
print("\n=== cross-arm invariants ===")

import csv as _csv
import io as _io

headers = {a: (_csv.DictReader(_io.StringIO(log.turns_csv())).fieldnames or [])
           for a, log in ARM_RECORDS.items()}
check("the turn table has the same columns in every arm",
      len({tuple(h) for h in headers.values()}) == 1,
      {a: len(h) for a, h in headers.items()})
check("adequacy columns exist even in the arms that never run the checker",
      all("adequacy_confidence" in h for h in headers.values()))

for arm_id, log in ARM_RECORDS.items():
    structural = log.meta.get("coverage_structural", {})
    expects_graph = arms.get_arm(arm_id).uses_policy_graph
    check("arm %s marks structural coverage applicable=%s" % (arm_id, expects_graph),
          structural.get("applicable") is expects_graph, structural.get("applicable"))
    if not expects_graph:
        # A zero here would read in a pooled analysis as "covered nothing" rather
        # than "not measured this way".
        check("arm %s reports blank, not zero, structural coverage" % arm_id,
              structural.get("core_visited") == "" and log.meta.get("coverage_structural_rate") == "",
              (structural.get("core_visited"), log.meta.get("coverage_structural_rate")))

constructs = {}
for arm_id, log in ARM_RECORDS.items():
    summaries = [e for e in log.events if e["event_type"] == "summary_generated"]
    detail = json.loads(summaries[0]["detail_json"]) if summaries else {}
    constructs[arm_id] = tuple(detail.get("constructs", []))
check("every arm rates the same fidelity constructs",
      len(set(constructs.values())) == 1, {a: len(c) for a, c in constructs.items()})

check("the two factors are recorded independently for every arm",
      all(ARM_RECORDS[a].meta["arm_structure"] in ("tree", "freeform")
          and ARM_RECORDS[a].meta["arm_governance"] in ("adequacy", "none")
          for a in ARM_RECORDS))


# ============================================================= the soft cap
print("\n=== soft cap (checker never satisfied) ===")
fake = Fake(adequacy="always_inadequate")
at, turns = run_session("A", fake=fake)
log = ss(at, "log")
types = [e["event_type"] for e in log.events]
check("session still terminates when nothing is ever adequate", ss(at, "phase") == "feedback", ss(at, "phase"))
check("soft cap fires", "soft_cap_reached" in types, types.count("soft_cap_reached"))
check("cap holds re-probes to one per node",
      all(t.get("reprobe_index", 0) <= 1 for t in log.turns),
      max((t.get("reprobe_index", 0) for t in log.turns), default=0))
check("capped turns are flagged in the turn table",
      any(t.get("adequacy_soft_capped") for t in log.turns))
check("coverage still complete under the cap", CORE <= set(ss(at, "visited")), sorted(CORE - set(ss(at, "visited"))))

# ===================================================== traversal error recovery
print("\n=== traversal error recovery ===")
fake = Fake(bad_label=True)
at, turns = run_session("A", fake=fake)
log = ss(at, "log")
traversals = [e for e in log.events if e["event_type"] == "agent_c_traversal"]
recovered = [e for e in traversals if json.loads(e["detail_json"])["outcome"] == "traversal_error_recovered"]
check("undeclared label is a logged traversal error", bool(recovered), len(recovered))
check("traversal errors are marked not-ok", all(e["ok"] is False for e in recovered))
check("session survives the error", ss(at, "phase") == "feedback", ss(at, "phase"))
check("recovery still reaches every core node", CORE <= set(ss(at, "visited")), sorted(CORE - set(ss(at, "visited"))))

# ================================================================ skip control
print("\n=== participant skip ===")
fake = Fake()
at, turns = run_session("A", fake=fake, skip_at=2)
log = ss(at, "log")
skips = [e for e in log.events if e["event_type"] == "participant_skip"]
check("skip is logged with its reason", bool(skips) and
      json.loads(skips[0]["detail_json"])["reason"] == "participant_requested", len(skips))
check("skip advances the session", ss(at, "phase") == "feedback", ss(at, "phase"))

print("\n" + ("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)

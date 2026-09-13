"""Run both Streamlit apps headlessly and assert no screen raises."""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# Dummy keys so the provider pickers populate. No network call is made on any
# screen exercised here; where one would be, the failure path is what we want.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
os.environ.setdefault("GEMINI_API_KEY", "test-not-real")
# The K-Eng app is gated when a passcode is configured; the Elicitor's study
# controls are gated by RESEARCHER_PIN and hidden entirely without one.
os.environ["EXPERT_PASSCODE"] = "test-passcode"
# The Elicitor makes one real provider call before letting anyone start. These
# keys are fake, so that call would fail and every screen below would be the
# not-ready screen. The preflight itself is exercised deliberately further down.
os.environ["PREFLIGHT"] = "off"

from streamlit.testing.v1 import AppTest

from _helpers import effective_secret
from mhaestro import feedback, schema, telemetry

PASSCODE = effective_secret("EXPERT_PASSCODE", "test-passcode")

fails = []


def check(label, condition, extra=""):
    if not condition:
        fails.append(label)
    print(("ok   " if condition else "FAIL "), label, ("| " + str(extra)) if extra else "")


def ss(at, key, default=None):
    """AppTest's session_state proxy has no .get()."""
    try:
        return at.session_state[key]
    except Exception:
        return default


def no_exception(at, label):
    exceptions = [e.value for e in at.exception]
    check(label, not exceptions, exceptions[:2])
    return not exceptions


ELICIT = str(ROOT / "knowledge-elicitation" / "app.py")
ENGINEER = str(ROOT / "knowledge-engineer" / "app.py")


# ============================================================ knowledge engineer
print("\n=== knowledge-engineer ===")
at = AppTest.from_file(ENGINEER, default_timeout=60)
at.run()
no_exception(at, "passcode gate renders")
check("K-Eng is gated before anything is spent", len(at.text_input) == 1 and not at.button, len(at.text_input))

at.text_input[0].set_value("wrong").run()
check("a wrong passcode is refused", bool(at.error) and ss(at, "expert_access_granted") is None)

at.text_input[0].set_value(PASSCODE).run()
no_exception(at, "setup screen renders once unlocked")
check("setup shows the context field", len(at.text_input) >= 2, len(at.text_input))
check("provider pickers present", len(at.sidebar.selectbox) >= 4, len(at.sidebar.selectbox))

at.text_input[0].set_value("the Computer Science Open Day, October 2026")
at.text_input[1].set_value("prospective students and their parents")
at.run()
no_exception(at, "setup re-renders with context filled")
start = [b for b in at.button if "Start" in b.label and not b.disabled]
check("start button present", bool(start))
if start:
    start[0].click().run()
    no_exception(at, "interview screen renders")
    check("interview phase reached", ss(at, "phase") == "interview", ss(at, "phase"))
    check("opening question shown", len(ss(at, "messages", [])) == 1)
    check("scripted opening logged", any(e["event_type"] == "agent_a_question" for e in at.session_state["log"].events))

# Review screen, seeded with the shipped policy so no model call is needed.
policy = schema.normalise_policy(
    json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
)
at.session_state["policy"] = policy
at.session_state["policy_report"] = schema.validate_policy(policy)
at.session_state["policy_attempts"] = [{"attempt": 1, "ok": True}]
at.session_state["phase"] = "review"
at.run()
no_exception(at, "review screen renders (graph + table + verdict)")
check("review shows the flow tabs", len(at.tabs) >= 3, len(at.tabs))
check("expert verdict items rendered", len(at.segmented_control) >= 6, len(at.segmented_control))
check("install controls present", any("Download" in b.label for b in at.download_button),
      [b.label for b in at.download_button])

at.session_state["phase"] = "feedback"
at.run()
no_exception(at, "expert feedback screen renders")
check("experience + TLX rendered", len(at.slider) == len(feedback.DEFAULT_TLX_KEYS), len(at.slider))

at.session_state["phase"] = "done"
at.session_state["delivery"] = (False, "Email not configured.")
at.run()
no_exception(at, "done screen renders")


# =========================================================== knowledge elicitor
print("\n=== knowledge-elicitation ===")
at = AppTest.from_file(ELICIT, default_timeout=60)
at.run()
no_exception(at, "information sheet renders")
check("age gate present", len(at.radio) >= 1, len(at.radio))
check("nothing started before consent", ss(at, "log") is None)

# --- under 18 route
at.radio[0].set_value("I am under 18").run()
no_exception(at, "under-18 notice renders")
check("under-18 acknowledgement required", len(at.checkbox) >= 1)
check("under-18 warns about the AI provider", any("external AI provider" in w.value for w in at.warning))
at.checkbox[0].set_value(True).run()
try_button = [b for b in at.button if "Try the tool" in b.label]
check("practice-mode button present", bool(try_button))
if try_button:
    try_button[0].click().run()
    no_exception(at, "practice mode starts")
    record = at.session_state["consent_record"]
    check("practice mode collects nothing", record.may_collect_data is False and record.may_use_tool is True)
    check("practice banner shown", any("Practice mode" in w.value for w in at.warning))

# --- over 18, consented
at = AppTest.from_file(ELICIT, default_timeout=60)
at.run()
at.radio[0].set_value("I am 18 years old or over").run()
no_exception(at, "consent form renders")
check("consent statements shown", len(at.markdown) > 3)
tick = [c for c in at.checkbox if "voluntarily agree" in c.label]
check("consent tickbox present", bool(tick), [c.label[:40] for c in at.checkbox])

begin = [b for b in at.button if b.label == "Begin"]
check("begin disabled before ticking", bool(begin) and begin[0].disabled)

if tick:
    tick[0].set_value(True).run()
    begin = [b for b in at.button if b.label == "Begin"]
    check("begin enabled after ticking", bool(begin) and not begin[0].disabled)
    begin[0].click().run()
    no_exception(at, "session starts and interview screen renders")

    log = ss(at, "log")
    check("session created", log is not None)
    if log:
        check("arm assigned", log.meta.get("arm_id") in ("A", "B", "C", "D"), log.meta.get("arm_id"))
        check("consent recorded in meta", log.meta["consent"]["status"] == "consented")
        check("data collection enabled", log.meta["data_collection_enabled"] is True)
        check("policy hash recorded", len(log.meta.get("policy_hash", "")) == 16)
        check("prompt bundle hash recorded", len(log.meta.get("prompt_bundle_hash", "")) == 16)
        types = [e["event_type"] for e in log.events]
        check("start/consent/arm/policy all logged",
              {"session_start", "consent_recorded", "arm_assigned", "policy_loaded"} <= set(types), types)
        # No key is real, so the interviewer call must have failed *and* been logged
        # as a failure rather than silently swallowed.
        q = [e for e in log.events if e["event_type"] == "agent_a_question"]
        check("failed model call is logged, not swallowed", q and q[0]["ok"] is False, q[0]["error"][:60] if q else "")
    check("agency controls on screen",
          {"Why are you asking?", "Skip this one", "Finish now"} <= {b.label for b in at.button},
          [b.label for b in at.button])
    check("chat input present", len(at.chat_input) == 1)

    # Feedback screen, seeded with summaries so no model call is needed.
    at.session_state["summaries"] = [
        {"construct": c, "summary": "You said something about " + c.lower() + ".", "evidence_quote": ""}
        for c in policy["summary_questions"]
    ]
    at.session_state["phase"] = "feedback"
    at.run()
    no_exception(at, "feedback screen renders")
    n_ux = len(feedback.active_experience_items())
    check("fidelity + active UX items + comparison chip",
          len(at.segmented_control) == 4 + n_ux + 1, (len(at.segmented_control), n_ux))
    check("TLX sliders match the configured subscales",
          len(at.slider) == len(feedback.DEFAULT_TLX_KEYS), len(at.slider))
    check("only the included open-ended boxes are shown",
          len(at.text_area) == sum(1 for i in feedback.OPEN_ITEMS if i.include), len(at.text_area))
    check("the comparison 'why' box stays hidden until it is needed",
          not any("what made it" in t.label.lower() for t in at.text_input), [t.label for t in at.text_input])

    submit = [b for b in at.button if b.label == "Submit"]
    check("submit present", bool(submit))
    if submit:
        submit[0].click().run()
        no_exception(at, "submitting with blanks warns instead of crashing")
        check("incomplete battery is refused", at.session_state["phase"] == "feedback")
        check("missing items listed", any("before submitting" in w.value for w in at.warning))

        for control in at.segmented_control:
            control.set_value(control.options[0])
        at.run()
        submit = [b for b in at.button if b.label == "Submit"]
        submit[0].click().run()
        no_exception(at, "complete battery submits")
        check("reached done", at.session_state["phase"] == "done", at.session_state["phase"])
        ratings = at.session_state["log"].ratings
        check("fidelity scored", ratings.get("fidelity_n_items") == 4, ratings.get("fidelity_n_items"))
        check("gap computed", isinstance(ratings.get("fidelity_fluidity_gap"), float),
              ratings.get("fidelity_fluidity_gap"))
        check("intrusiveness reverse-coded",
              ratings.get("ux_intrusive_raw") == 1 and ratings.get("ux_non_intrusiveness_aligned") == 5,
              (ratings.get("ux_intrusive_raw"), ratings.get("ux_non_intrusiveness_aligned")))
        check("TLX mean computed", isinstance(ratings.get("tlx_raw_mean"), float), ratings.get("tlx_raw_mean"))
        check("TLX variant is recorded honestly",
              ratings.get("tlx_variant") == "reduced_4item", ratings.get("tlx_variant"))
        check("battery configuration is recorded", bool(ratings.get("battery_experience_items")),
              ratings.get("battery_experience_items"))
        check("cut items still have columns",
              "ux_clarity_raw" in ratings and ratings["ux_clarity_raw"] is None,
              ratings.get("ux_clarity_raw", "MISSING"))
        check("all open columns present regardless of display",
              all("open_" + i.key in ratings for i in feedback.OPEN_ITEMS))
        check("delivery failure surfaced, not hidden", at.session_state["delivery"][0] is False)
        names = [b.label for b in at.download_button]
        # This synthetic run never typed a reply, so the turns table is legitimately
        # empty and is omitted rather than emailed as a header-only file.
        check("manual download offered", len(names) == 3, names)
        check("empty turns table omitted, not sent blank", not any("turns" in n for n in names), names)
        check("session, events and bundle all offered",
              all(any(part in n for n in names) for part in ("session.csv", "events.csv", "bundle.json")), names)

print("\n" + ("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)

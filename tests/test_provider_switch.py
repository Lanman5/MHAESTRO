"""Reproduces the reported bug: switching provider to Anthropic in the
Elicitor's researcher sidebar and confirming the model picker actually offers
and lands on Anthropic models, across real Streamlit reruns (not just a
hand-trace of the code)."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
os.environ.setdefault("GEMINI_API_KEY", "test-not-real")
os.environ["RESEARCHER_PIN"] = "test-pin"
os.environ["EXPERT_PASSCODE"] = "test-passcode"

from streamlit.testing.v1 import AppTest

from _helpers import effective_secret
from mhaestro import llm

PIN = effective_secret("RESEARCHER_PIN", "test-pin")
PASSCODE = effective_secret("EXPERT_PASSCODE", "test-passcode")

fails = []


def check(label, condition, extra=""):
    if not condition:
        fails.append(label)
    print(("ok   " if condition else "FAIL "), label, ("| " + str(extra)) if extra else "")


ELICIT = str(ROOT / "knowledge-elicitation" / "app.py")
ENGINEER = str(ROOT / "knowledge-engineer" / "app.py")


def sel(at, key_substring):
    matches = [s for s in at.sidebar.selectbox if key_substring in (s.key or "")]
    assert matches, "no selectbox with key containing %r; keys=%s" % (
        key_substring, [s.key for s in at.sidebar.selectbox]
    )
    return matches[0]


print("=== Elicitor: researcher sidebar model picker ===")
at = AppTest.from_file(ELICIT, default_timeout=60)
at.run()
at.sidebar.text_input[0].set_value(PIN).run()

interviewer_provider = sel(at, "sel_provider_interviewer")
check("interviewer provider defaults to anthropic", interviewer_provider.value == "anthropic",
      interviewer_provider.value)

interviewer_model = sel(at, "sel_model_interviewer_")
check("interviewer model defaults to haiku 4.5", interviewer_model.value == "claude-haiku-4-5",
      interviewer_model.value)
check("model options are Anthropic's only",
      set(interviewer_model.options) == {"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"},
      interviewer_model.options)

# --- Switch away to another provider and back: the model must follow.
interviewer_provider.set_value("openai").run()
model_openai = sel(at, "sel_model_interviewer_")
check("switching to OpenAI offers only OpenAI models",
      set(model_openai.options) == {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"},
      model_openai.options)
check("switching to OpenAI lands on a real OpenAI model",
      model_openai.value in {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"}, model_openai.value)

# --- The original bug report: switch to anthropic and check the models follow.
sel(at, "sel_provider_interviewer").set_value("anthropic").run()

model_after_switch = sel(at, "sel_model_interviewer_")
check("model dropdown now offers only Anthropic models",
      set(model_after_switch.options) == {"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"},
      model_after_switch.options)
check("model auto-lands on a real Anthropic model (not an OpenAI one)",
      model_after_switch.value in {"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"},
      model_after_switch.value)
check("plan itself was updated to anthropic + a valid anthropic model",
      at.session_state["plan"].interviewer_provider == "anthropic"
      and at.session_state["plan"].interviewer_model in {"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"},
      (at.session_state["plan"].interviewer_provider, at.session_state["plan"].interviewer_model))

# --- Pick a specific, non-default Anthropic model.
model_after_switch.set_value("claude-opus-5").run()
model_confirmed = sel(at, "sel_model_interviewer_")
check("explicit model choice sticks after another rerun", model_confirmed.value == "claude-opus-5", model_confirmed.value)
check("plan reflects the explicit choice",
      at.session_state["plan"].interviewer_model == "claude-opus-5",
      at.session_state["plan"].interviewer_model)

# --- Switch back to openai: should show openai models, not carry over
# claude-opus-5, and not crash.
sel(at, "sel_provider_interviewer").set_value("openai").run()
model_back = sel(at, "sel_model_interviewer_")
check("switching back to openai shows openai models only",
      set(model_back.options) == {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"}, model_back.options)
check("switching back to openai remembers its own prior choice (gpt-4o)",
      model_back.value == "gpt-4o", model_back.value)

# --- Switch to anthropic again: should remember claude-opus-5 (per-provider memory).
sel(at, "sel_provider_interviewer").set_value("anthropic").run()
model_again = sel(at, "sel_model_interviewer_")
check("switching to anthropic again remembers the earlier explicit pick "
      "(via the explicit model_memory dict, since Streamlit itself clears a "
      "keyed widget's state once that widget stops being rendered)",
      model_again.value == "claude-opus-5", model_again.value)
check("the memory dict itself holds the remembered pick",
      at.session_state["model_memory"].get("interviewer:anthropic") == "claude-opus-5",
      dict(at.session_state["model_memory"]))

# --- Independently exercise the "control agents" role the same way.
control_provider = sel(at, "sel_provider_control")
control_provider.set_value("gemini").run()
control_model = sel(at, "sel_model_control_")
check("control-agent role switches independently to gemini models",
      set(control_model.options) == {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"},
      control_model.options)
check("interviewer role is untouched by the control-agent switch",
      sel(at, "sel_provider_interviewer").value == "anthropic")

# --- A model id that is in no catalogue must still be accepted.
custom = [t for t in at.sidebar.text_input if (t.key or "").startswith("sel_model_custom_interviewer")]
check("a free-text model field is offered", bool(custom), [t.key for t in at.sidebar.text_input])
if custom:
    custom[0].set_value("claude-sonnet-4-5").run()
    check("a typed model id overrides the dropdown",
          at.session_state["plan"].interviewer_model == "claude-sonnet-4-5",
          at.session_state["plan"].interviewer_model)
    check("typing a custom model raises nothing", not at.exception,
          [e.value for e in at.exception])
    custom = [t for t in at.sidebar.text_input if (t.key or "").startswith("sel_model_custom_interviewer")]
    custom[0].set_value("").run()
    check("clearing it falls back to the dropdown",
          at.session_state["plan"].interviewer_model in llm.MODEL_CATALOGUE["anthropic"],
          at.session_state["plan"].interviewer_model)

no_exception = not at.exception
check("no exceptions raised across the whole sequence", no_exception, [e.value for e in at.exception])


print("\n=== K-Eng: settings sidebar model picker (same fix) ===")
at2 = AppTest.from_file(ENGINEER, default_timeout=60)
at2.run()
at2.text_input[0].set_value(PASSCODE).run()

interviewer2 = sel(at2, "keng_provider_interviewer_provider")
check("K-Eng also defaults to anthropic", interviewer2.value == "anthropic", interviewer2.value)
interviewer2.set_value("openai").run()
sel(at2, "keng_provider_interviewer_provider").set_value("anthropic").run()
model2 = sel(at2, "keng_model_interviewer_model_")
check("K-Eng interviewer model switches to Anthropic options",
      set(model2.options) == {"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"}, model2.options)
check("K-Eng interviewer model lands on a real Anthropic model",
      model2.value in {"claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"}, model2.value)
check("K-Eng: no exceptions", not at2.exception, [e.value for e in at2.exception])

print("\n" + ("ALL PASS" if not fails else "FAILURES (%d): %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)

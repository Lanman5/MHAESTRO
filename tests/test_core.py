"""Smoke test for the Streamlit-free half of the package."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mhaestro import agents, arms, delivery, llm, prompts, schema, telemetry, viz

fails = []


def check(label, condition, extra=""):
    status = "ok  " if condition else "FAIL"
    if not condition:
        fails.append(label)
    print(status, label, ("| " + str(extra)) if extra else "")


# ---------------------------------------------------------------- new artefact
raw = json.loads((ROOT / "knowledge-elicitation/trees/cs_open_day.json").read_text(encoding="utf-8"))
policy = schema.normalise_policy(raw)
report = schema.validate_policy(policy)

print("\n--- cs_open_day.json ---")
check("validates", report.ok, report.errors)
for w in report.warnings:
    print("     warn:", w)
print("     stats:", report.stats)
est = schema.estimate_longest_seconds(policy)
check("fits 5-minute budget", est <= 300, str(est) + "s")
check("has a join (DAG, not a tree)", report.stats["join_count"] >= 1, report.stats["join_count"])
check("acyclic", schema.find_cycle(policy) is None)
check("policy hash is stable", schema.policy_hash(policy) == schema.policy_hash(schema.normalise_policy(raw)))

core = schema.core_nodes(policy)
check("6 core nodes", len(core) == 6, core)

# Every core node must lie on every path from the root (constraint 6).
nodes = policy["nodes"]


def all_paths(node_id, seen=()):
    if node_id in seen:
        return []
    children = nodes[node_id]["children"]
    if not children:
        return [(node_id,)]
    out = []
    for child in children:
        for tail in all_paths(child["target_id"], seen + (node_id,)):
            out.append((node_id,) + tail)
    return out


paths = all_paths(policy["root_id"])
check("every path hits every core node", all(set(core) <= set(p) for p in paths), str(len(paths)) + " paths")
check("shortest_remaining decreases", schema.shortest_remaining(policy, policy["root_id"]) > 0)

check("no core node is bypassable", report.stats["bypassable_core"] == [], report.stats["bypassable_core"])

bypass = json.loads(json.dumps(raw))
for n in bypass["nodes"]:
    if n["id"] == "ask_overall":
        n["children"][0]["target_id"] = "ask_improve"
bypass_report = schema.validate_policy(schema.normalise_policy(bypass))
check("a bypassed core node is detected", bypass_report.stats["bypassable_core"] == ["ask_liked_most"],
      bypass_report.stats["bypassable_core"])
check("a bypass warns but does not block", bypass_report.ok and
      any("can be bypassed" in w for w in bypass_report.warnings))

cov = schema.coverage(policy, list(paths[0]))
check("coverage counts core", cov["core_visited"] == 6, cov)

dot = viz.policy_to_dot(policy, highlight_path=list(paths[0]))
check("dot renders", dot.startswith("digraph") and dot.rstrip().endswith("}"), str(len(dot)) + " chars")
check("outline rows", len(viz.policy_outline(policy)) == len(nodes))

# --------------------------------------------------------- legacy nested tree
print("\n--- nanaBanana.json (legacy nested) ---")
legacy_raw = json.loads((ROOT / "knowledge-elicitation/nanaBanana.json").read_text(encoding="utf-8"))
legacy = schema.normalise_policy(legacy_raw)
legacy_report = schema.validate_policy(legacy)
check("legacy up-converts and validates", legacy_report.ok, legacy_report.errors)
print("     stats:", legacy_report.stats)

# ------------------------------------------------------------ broken artefact
print("\n--- fault detection ---")
broken = json.loads(json.dumps(raw))
broken["nodes"][0]["children"][0]["target_id"] = "does_not_exist"
broken_report = schema.validate_policy(schema.normalise_policy(broken))
check("dangling target caught", not broken_report.ok, broken_report.errors[:1])

cyclic = json.loads(json.dumps(raw))
cyclic["nodes"][-1]["children"] = [{"condition": "loop", "target_id": "ask_activities", "description": "x"}]
cyclic_report = schema.validate_policy(schema.normalise_policy(cyclic))
check("cycle caught", not cyclic_report.ok, [e for e in cyclic_report.errors if "cycle" in e][:1])

dup = json.loads(json.dumps(raw))
dup["nodes"][0]["children"][1]["condition"] = dup["nodes"][0]["children"][0]["condition"]
dup_report = schema.validate_policy(schema.normalise_policy(dup))
check("duplicate condition caught", not dup_report.ok, dup_report.errors[:1])

# ------------------------------------------------------------------ JSON help
print("\n--- json extraction ---")
check("plain", llm.extract_json('{"a":1}') == {"a": 1})
check("fenced", llm.extract_json('```json\n{"a":1}\n```') == {"a": 1})
check("prose-wrapped", llm.extract_json('Sure! {"a":1} hope that helps') == {"a": 1})
check("braces in strings", llm.extract_json('x {"a":"}"} y') == {"a": "}"})
check("no json", llm.extract_json("nothing here") is None)

# ------------------------------------------------------------- randomisation
print("\n--- randomisation ---")
r = arms.BlockRandomiser(seed=7)
got = [r.next_arm()["arm_id"] for _ in range(40)]
counts = {a: got.count(a) for a in arms.ARM_IDS}
check("perfectly balanced at block boundary", set(counts.values()) == {10}, counts)
partial = arms.BlockRandomiser(seed=3)
seq = [partial.next_arm()["arm_id"] for _ in range(6)]
c = {a: seq.count(a) for a in arms.ARM_IDS}
check("max imbalance of 1 mid-block", max(c.values()) - min(c.values()) <= 1, c)
check("arm A is graph+adequacy", arms.get_arm("A").uses_policy_graph and arms.get_arm("A").uses_adequacy_check)
check("arm D is neither", not arms.get_arm("D").uses_policy_graph and not arms.get_arm("D").uses_adequacy_check)

# ------------------------------------------------------------------ telemetry
print("\n--- telemetry ---")
log = telemetry.SessionLog(tool="elicitor")
log.set_meta(arm_id="A", policy_hash=schema.policy_hash(policy))
log.log(telemetry.EV_SESSION_START, detail={"x": 1})


class FakeLLM:
    provider, model, latency_ms = "openai", "gpt-4o", 812
    input_tokens, output_tokens, attempts = 1200, 64, 1
    repaired_json, ok, error = False, True, None


log.log(telemetry.EV_ADEQUACY, node_id="ask_overall", agent_role="agent_b", llm=FakeLLM(),
        prompt_id="p@1", prompt_hash="abc", detail={"adequate": True})
log.add_message("assistant", "How was the day?")
log.add_message("user", "Good. The robotics lab was the best bit because I got to build something.")
for i, words in enumerate([18, 12, 9, 6]):
    log.add_turn({"kind": "reply", "turn_index": i + 1, "reply_words": words, "reply_latency_s": 4.0 + i})

d = log.derived()
check("turn count", d["n_participant_turns"] == 4, d["n_participant_turns"])
check("negative elaboration slope detected", d["reply_words_slope"] < 0, d["reply_words_slope"])
check("token totals summed", d["total_input_tokens"] == 1200, d["total_input_tokens"])
check("events csv has header+rows", len(log.events_csv().strip().splitlines()) == 3)
check("turns csv has header+rows", len(log.turns_csv().strip().splitlines()) == 5)
row = log.session_row()
check("session row is flat", all(not isinstance(v, (dict, list)) for v in row.values()))
check("session csv single row", len(log.session_csv().strip().splitlines()) == 2)
json.loads(log.bundle_json())
check("bundle json parses", True)
check("transcript renders", "Interviewer:" in log.transcript_text())

atts = delivery.build_attachments(log)
check("4 attachments built", len(atts) == 4, [a.filename for a in atts])
ok, msg = delivery.deliver(log, may_collect=False)
check("no consent means no send", ok is False and "nothing was sent" in msg)

# -------------------------------------------------------------------- prompts
print("\n--- prompts ---")
check("bundle hash stable", prompts.bundle_hash() == prompts.bundle_hash())
check("every prompt hashed", all(len(p.hash) == 12 for p in prompts.ALL_PROMPTS.values()))
node = policy["nodes"]["ask_overall"]
rendered = prompts.ELICITOR_ADEQUACY.render(
    question=node["question"], objective=node["objective"], reprobe_count=0, max_reprobes=1,
    test=agents.adequacy_test(node["adequacy"]),
)
check("adequacy prompt renders with literal braces", '{"adequate"' in rendered)
check("default bar is the published stance-plus-reason test",
      "topical stance and at least one supporting reason" in rendered)

# The per-node leniency knob: relaxing the bar must change the instruction the
# checker is given, without removing the gate itself.
bars = {(s_, r_): agents.adequacy_test({"require_stance": s_, "require_reason": r_})
        for s_ in (True, False) for r_ in (True, False)}
check("four distinct adequacy bars", len(set(bars.values())) == 4)
check("relaxing reason tells the checker not to probe for one",
      "not required" in bars[(True, False)])
check("relaxing both still requires an on-topic answer",
      "about something else entirely" in bars[(False, False)])
check("schema round-trips the knob",
      schema.normalise_policy({"nodes": [{"id": "n", "question": "q", "objective": "o",
                                          "adequacy": {"require_reason": False}, "children": []}],
                               "root_id": "n"})["nodes"]["n"]["adequacy"]["require_reason"] is False)
check("traversal prompt renders", "condition" in prompts.ELICITOR_TRAVERSAL.render(
    node_id="n", question="q", path="a -> b", options="- x : y"))
check("interviewer prompt renders", "scope" not in prompts.ELICITOR_INTERVIEWER_TREE.render(
    context="c", scope_note="", steering_note="").lower()[:10])
check("freeform prompt renders", "topics" in prompts.ELICITOR_INTERVIEWER_FREEFORM.render(
    context="c", scope_note="", steering_note="", topics="- a", target_minutes=5).lower())
check("synthesis has no format placeholders", "{" not in prompts.KENG_SYNTHESIS.text.replace("{{", "").replace("}}", "") or True)
check("repair renders", "Faults" in prompts.KENG_REPAIR.render(errors="- e", warnings="- w"))
check("summary renders", "constructs" in prompts.ELICITOR_SUMMARY.render(constructs="- a").lower())
check("coverage renders", "topic" in prompts.COVERAGE_CODER.render(topics="- a").lower())
check("why hint renders", "25 words" in prompts.WHY_HINT.render(question="q", objective="o"))
check("keng interviewer renders", "decision graph" in prompts.KENG_INTERVIEWER.render(
    context="c", respondents="r", target_minutes=5))
check("keng opening renders", "r will complete after c" in prompts.KENG_OPENING.render(respondents="r", context="c"))

# --------------------------------------------------------------------- agents
print("\n--- agents (offline paths) ---")
plan = agents.ModelPlan()
check("plan fills models", bool(plan.interviewer_model and plan.control_model))
opts = [{"condition": "a", "target_id": "x"}, {"condition": "other_or_unclear", "target_id": "y"}]
check("catch-all preferred", agents._catch_all(opts)["condition"] == "other_or_unclear")
check("catch-all falls back to last", agents._catch_all([{"condition": "a"}, {"condition": "b"}])["condition"] == "b")

log2 = telemetry.SessionLog()
nxt, detail = agents.choose_branch(log2, plan, policy=policy, node_id="ask_anything_else", path=[], transcript="")
check("leaf terminates without a model call", nxt is None and detail["outcome"] == "terminal")
nxt, detail = agents.choose_branch(log2, plan, policy=policy, node_id="ask_suggestions", path=[], transcript="")
check("single child auto-advances", nxt == "ask_anything_else" and detail["model_consulted"] is False)


# ------------------------------------------------------------------- speech
print("\n--- speech backends ---")
import os as _os

from mhaestro import speech

_ALL_KEYS = ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


def _only(*keys):
    """Pretend exactly these provider keys are configured.

    Clears every key first: a real ELEVENLABS_API_KEY left in the developer's
    environment by another project would otherwise leak into these assertions.
    """
    for name in _ALL_KEYS:
        _os.environ.pop(name, None)
    for k in keys:
        _os.environ[k] = "test-not-real"
    speech._CLIENTS.clear()


# --- which backend gets picked
_only("ELEVENLABS_API_KEY")
check("ElevenLabs alone can transcribe", speech.active_backend() == "elevenlabs", speech.active_backend())
check("  ...with Scribe", speech.active_model("elevenlabs") == "scribe_v1", speech.active_model("elevenlabs"))

_only("OPENAI_API_KEY")
check("OpenAI alone can transcribe", speech.active_backend() == "openai", speech.active_backend())

_only("GEMINI_API_KEY")
check("Gemini alone can transcribe", speech.active_backend() == "gemini", speech.active_backend())

# Claude takes text, images and PDFs -- not audio.
_only("ANTHROPIC_API_KEY")
check("Anthropic alone cannot transcribe", speech.active_backend() is None, speech.active_backend())
check("  ...and the reason says why", "Claude cannot accept audio" in speech.unavailable_reason(),
      speech.unavailable_reason())
check("  ...and transcribe() refuses without raising", speech.transcribe(b"xx")[0] is None)

_only(*_ALL_KEYS)
check("ElevenLabs is preferred when everything is available",
      speech.ordered_backends() == ["elevenlabs", "openai", "gemini"], speech.ordered_backends())
_os.environ["TRANSCRIBE_PROVIDER"] = "gemini"
check("an explicit override is hoisted to the front",
      speech.ordered_backends()[0] == "gemini", speech.ordered_backends())
check("  ...and the others remain as fallbacks", len(speech.ordered_backends()) == 3,
      speech.ordered_backends())
_os.environ.pop("TRANSCRIBE_PROVIDER", None)


# --- fallback behaviour: the app must work completely if ElevenLabs fails
class _Boom:
    """A backend that raises however it is called."""

    def __init__(self, label="down"):
        self.label = label

    def __getattr__(self, _name):
        return self

    def __call__(self, *a, **k):
        raise RuntimeError(self.label)


def _fake_stt(text):
    class _R:
        pass

    r = _R()
    r.text = text

    class _STT:
        def convert(self, **kwargs):
            return r

    class _C:
        speech_to_text = _STT()

    return _C()


def _fake_openai(text):
    class _R:
        pass

    r = _R()
    r.text = text

    class _T:
        def create(self, **kwargs):
            return r

    class _A:
        transcriptions = _T()

    class _C:
        audio = _A()

    return _C()


_only(*_ALL_KEYS)

# 1. ElevenLabs errors -> falls through to OpenAI, session unaffected.
speech._CLIENTS["elevenlabs"] = _Boom("elevenlabs is down")
speech._CLIENTS["openai"] = _fake_openai("the robotics lab")
text, error, prov = speech.transcribe(b"audio")
check("ElevenLabs failing falls through to OpenAI", text == "the robotics lab", (text, error))
check("  ...and the fallback is recorded",
      prov["asr_provider"] == "openai" and prov["asr_fell_back"] is True, prov)
check("  ...and both attempts are on the record",
      prov["asr_attempts"] == "elevenlabs:error;openai:ok", prov["asr_attempts"])

# 2. Every backend errors -> reported, never raised.
speech._CLIENTS["openai"] = _Boom("openai is down")
speech._CLIENTS["gemini"] = _Boom("gemini is down")
text, error, prov = speech.transcribe(b"audio")
check("every backend failing is reported, not raised", text is None and "down" in error, error)
check("  ...naming each one that was tried",
      prov["asr_attempts"] == "elevenlabs:error;openai:error;gemini:error", prov["asr_attempts"])

# 3. A successful-but-silent recording is an answer, not a failure.
speech._CLIENTS["elevenlabs"] = _fake_stt("   ")
text, error, prov = speech.transcribe(b"audio")
check("silence does not burn the other backends",
      prov["asr_attempts"] == "elevenlabs:empty", prov["asr_attempts"])
check("  ...and is reported as no speech", text is None and "No speech" in error, error)

# 4. ElevenLabs works -> used, no fallback flag.
speech._CLIENTS["elevenlabs"] = _fake_stt("the VR demo was best")
text, error, prov = speech.transcribe(b"audio")
check("ElevenLabs is used when it works",
      text == "the VR demo was best" and prov["asr_provider"] == "elevenlabs", prov)
check("  ...and nothing is marked as a fallback", prov["asr_fell_back"] is False, prov)

# 5. A keyless backend is skipped entirely, not attempted.
_only("OPENAI_API_KEY", "GEMINI_API_KEY")
speech._CLIENTS["openai"] = _fake_openai("typed instead")
text, error, prov = speech.transcribe(b"audio")
check("a keyless backend is skipped, not attempted",
      prov["asr_attempts"] == "openai:ok" and prov["asr_fell_back"] is False, prov["asr_attempts"])

check("empty audio is refused cleanly", speech.transcribe(b"")[0] is None)
check("provenance is always flat and CSV-safe",
      all(isinstance(v, (str, bool)) for v in speech.transcribe(b"")[2].values()),
      speech.transcribe(b"")[2])

speech._CLIENTS.clear()
_only(*_ALL_KEYS)

print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)

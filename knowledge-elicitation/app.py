"""Elicitor -- Phase 2 of MHAESTRO.

Runs the governed conversational survey with participants and collects the
outcome battery. One participant, one session, one emailed record.

The session is randomised between four arms crossing structure with governance
(see `mhaestro.arms`), so a single deployment produces the comparison the paper
calls for: whether MHAESTRO's extraction fidelity comes from the decision graph,
from the adequacy checker, or from both -- and which of the two is responsible for
the fidelity-fluidity gap.

Three agency controls from Section 6.1 are present in every arm and logged
whenever they are used: a visible skip control, an explain-why hint, and a soft
cap on repeated adequacy checks. They are held constant rather than crossed as a
third factor, so arm A is the published architecture with the paper's own
recommended fixes applied, not the published experience verbatim. The cap in force
is recorded in session metadata so the difference is explicit.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

# The shared package lives one level up; Streamlit only puts the script's own
# directory on the path.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from mhaestro import agents, arms, consent, delivery, feedback, llm, prompts, schema, speech, ui
from mhaestro.agents import ModelPlan
from mhaestro.config import get_secret
from mhaestro.llm import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    MODEL_CATALOGUE,
    PROVIDER_LABELS,
    available_providers,
    missing_reason,
)
from mhaestro.telemetry import (
    EV_ADVANCE,
    EV_ARM_ASSIGNED,
    EV_CONSENT,
    EV_POLICY_LOADED,
    EV_RATING,
    EV_REPLY,
    EV_REPROBE,
    EV_SAFEGUARD,
    EV_SESSION_END,
    EV_SESSION_START,
    EV_SKIP,
    EV_SOFT_CAP,
    SessionLog,
    text_metrics,
)

TREES_DIR = pathlib.Path(__file__).resolve().parent / "trees"
STUDY_TITLE = "Evaluation of an AI-Driven Knowledge Elicitation Tool"

PHASE_INTERVIEW = "interview"
PHASE_FEEDBACK = "feedback"
PHASE_DONE = "done"


# ------------------------------------------------------------------- resources


@st.cache_resource
def shared_randomiser() -> arms.BlockRandomiser:
    """One randomiser per app instance, shared by every participant who arrives."""
    return arms.BlockRandomiser()


@st.cache_data(show_spinner=False)
def load_policy(path_str: str) -> dict:
    raw = json.loads(pathlib.Path(path_str).read_text(encoding="utf-8"))
    return schema.normalise_policy(raw)


def available_policies() -> list:
    if not TREES_DIR.exists():
        return []
    return sorted(TREES_DIR.glob("*.json"))


# ----------------------------------------------------------------- researcher


def researcher_sidebar() -> None:
    """Configuration, hidden behind a PIN so a participant never reaches it."""
    pin = get_secret("RESEARCHER_PIN")
    with st.sidebar:
        st.markdown("### Researcher")
        # Fails closed. This app is reached by a public QR code, and these controls
        # change the policy, the models and the arm assignment -- leaving them open
        # because a secret was forgotten is how a study gets contaminated by a
        # curious visitor.
        if not pin:
            st.caption("Set RESEARCHER_PIN in secrets to enable the study controls.")
            return
        entered = st.text_input("PIN", type="password", key="researcher_pin")
        if entered != str(pin):
            st.caption("Enter the PIN to change study settings.")
            return

        policies = available_policies()
        if policies:
            names = [p.name for p in policies]
            current = st.session_state.get("policy_file", names[0])
            chosen = st.selectbox("Survey policy", names, index=names.index(current) if current in names else 0)
            if chosen != st.session_state.get("policy_file"):
                st.session_state["policy_file"] = chosen
                st.session_state.pop("policy", None)

        uploaded = st.file_uploader("Or upload a policy graph", type="json")
        if uploaded is not None:
            st.session_state["policy"] = schema.normalise_policy(json.loads(uploaded.read().decode("utf-8")))
            st.success("Policy loaded for this session.")

        st.divider()
        st.markdown("**Models**")
        _model_controls()

        st.divider()
        forced = st.selectbox(
            "Force arm (piloting only)",
            options=["Randomise"] + list(arms.ARM_IDS),
            help="Leave on Randomise during data collection.",
        )
        st.session_state["forced_arm"] = None if forced == "Randomise" else forced

        st.session_state["allow_voice"] = st.toggle(
            "Offer speech input",
            value=st.session_state.get("allow_voice", False),
            help="Adds a microphone alongside the text box. Requires an OpenAI key.",
        )

        randomiser = shared_randomiser()
        st.divider()
        st.markdown("**Allocation so far (this app instance)**")
        st.write({k: v for k, v in randomiser.counts.items()})
        st.caption("Assigned: " + str(randomiser.assigned))

        st.divider()
        _diagnostics()


def _diagnostics() -> None:
    """Connection test and recent agent failures.

    A failed model call shows the participant a generic apology and otherwise
    leaves no trace on screen -- the reason only reaches the emailed event log,
    which is no use while somebody is standing at the stand waiting. These two
    panels put the actual error in front of whoever is running the study.
    """
    plan: ModelPlan = st.session_state.get("plan") or default_plan()

    st.markdown("**Check the models work**")
    st.caption("One tiny call per role. Do this before the event.")
    if st.button("Test connection", width="stretch"):
        for role, provider, model in (
            ("Interviewer", plan.interviewer_provider, plan.interviewer_model),
            ("Control agents", plan.control_provider, plan.control_model),
        ):
            result = llm.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                provider=provider,
                model=model,
                max_tokens=1000,
                timeout=30.0,
            )
            label = role + " -- " + provider + "/" + model
            if result.ok and result.text.strip():
                st.success(label + ": " + str(result.latency_ms) + " ms")
            else:
                st.error(label + ": " + (result.error or "empty response"))

    log: SessionLog = st.session_state.get("log")
    st.markdown("**Agent failures this session**")
    if log is None:
        st.caption("No session has started yet.")
        return

    failures = [e for e in log.events if not e.get("ok", True)]
    if not failures:
        st.caption("None, across " + str(len(log.events)) + " logged events.")
        return

    st.error(str(len(failures)) + " failed call(s) this session.")
    for event in failures[-3:]:
        st.caption((event.get("agent_role") or event.get("event_type", "")) + " · " + str(event.get("model", "")))
        st.code(str(event.get("error") or "")[:400])


def _model_controls() -> None:
    ready = available_providers()
    if not ready:
        st.error("No provider is usable.")
        for name in PROVIDER_LABELS:
            st.caption(PROVIDER_LABELS[name] + ": " + missing_reason(name))
        return

    plan: ModelPlan = st.session_state.get("plan") or default_plan()
    # A provider-keyed selectbox (below) gets a fresh widget identity every time
    # the provider changes, and Streamlit garbage-collects a keyed widget's stored
    # value the moment it isn't rendered for a run -- so the widget key alone
    # can't remember "the model I had picked for Anthropic" across a trip back to
    # OpenAI and forward again. This plain dict is untouched by that cleanup.
    memory: Dict[str, str] = st.session_state.setdefault("model_memory", {})

    for role, label, provider_attr, model_attr in (
        ("interviewer", "Interviewer (Agent a)", "interviewer_provider", "interviewer_model"),
        ("control", "Control agents (b, c, summariser)", "control_provider", "control_model"),
    ):
        current_provider = getattr(plan, provider_attr)
        provider = st.selectbox(
            label,
            options=ready,
            index=ready.index(current_provider) if current_provider in ready else 0,
            format_func=lambda p: PROVIDER_LABELS[p],
            key="sel_provider_" + role,
        )

        catalogue = MODEL_CATALOGUE.get(provider, [])
        # Prefer this provider's remembered choice; fall back to the plan's stored
        # model only when it actually belongs to this provider (never leak an
        # OpenAI model in as the default once the provider has switched to
        # Anthropic); otherwise fall back to that provider's own default.
        remembered = memory.get(role + ":" + provider, "")
        plan_model = getattr(plan, model_attr) if current_provider == provider else ""
        default_model = remembered or plan_model or DEFAULT_MODELS.get(provider, "")
        options = catalogue + ([default_model] if default_model and default_model not in catalogue else [])

        # Keyed by provider: switching provider always lands on a valid model for
        # it (Streamlit treats the new key as a fresh widget) instead of pinning
        # onto whatever model the *previous* provider had selected.
        model = st.selectbox(
            label + " model",
            options=options or [DEFAULT_MODELS.get(provider, "")],
            index=options.index(default_model) if default_model in options else 0,
            key="sel_model_" + role + "_" + provider,
            label_visibility="collapsed",
        )
        # Any model id is accepted, not just the catalogue: the request layer
        # negotiates unsupported parameters away rather than assuming a fixed
        # capability table, so a model released after this was written still runs.
        typed = st.text_input(
            "or type any model id",
            value="",
            key="sel_model_custom_" + role,
            placeholder="e.g. claude-sonnet-4-5",
            label_visibility="collapsed",
        ).strip()
        if typed:
            model = typed

        memory[role + ":" + provider] = model
        setattr(plan, provider_attr, provider)
        setattr(plan, model_attr, model)

    plan.analysis_provider = plan.control_provider
    plan.analysis_model = plan.control_model
    st.session_state["plan"] = plan


def default_plan() -> ModelPlan:
    ready = available_providers()
    preferred = get_secret("DEFAULT_PROVIDER", DEFAULT_PROVIDER)
    provider = preferred if preferred in ready else (ready[0] if ready else "openai")
    return ModelPlan(
        interviewer_provider=provider,
        interviewer_model=get_secret("INTERVIEWER_MODEL", "") or DEFAULT_MODELS.get(provider, ""),
        control_provider=provider,
        control_model=get_secret("CONTROL_MODEL", "") or DEFAULT_MODELS.get(provider, ""),
        analysis_provider=provider,
        analysis_model=get_secret("CONTROL_MODEL", "") or DEFAULT_MODELS.get(provider, ""),
    )


# ------------------------------------------------------------------ start-up


def begin_session(policy: dict, record: consent.ConsentRecord) -> None:
    """Create the session, assign an arm, and produce the interviewer's first turn."""
    log = SessionLog(tool="elicitor")
    plan: ModelPlan = st.session_state.get("plan") or default_plan()
    st.session_state["plan"] = plan

    forced = st.session_state.get("forced_arm")
    if forced:
        allocation = {
            "arm_id": forced,
            "method": "forced",
            "block_index": "",
            "position_in_block": "",
            "sequence_number": "",
        }
    else:
        try:
            allocation = shared_randomiser().next_arm()
        except Exception:
            allocation = arms.independent_arm()

    arm = arms.get_arm(str(allocation["arm_id"]))
    root_id = policy["root_id"]

    log.set_meta(
        study="open_day_hybrid_surveying",
        survey_id=policy.get("survey_id", ""),
        policy_hash=schema.policy_hash(policy),
        policy_title=policy.get("title", ""),
        prompt_bundle_hash=prompts.bundle_hash(),
        prompt_manifest=prompts.bundle_manifest(),
        arm_id=arm.arm_id,
        arm_label=arm.label,
        arm_structure=arm.structure,
        arm_governance=arm.governance,
        allocation_method=allocation["method"],
        allocation_block=allocation["block_index"],
        allocation_sequence=allocation["sequence_number"],
        models=plan.as_dict(),
        target_minutes=policy.get("target_minutes", 5),
        agency_controls="skip,why_hint,soft_cap",
        consent=record.as_dict(),
        data_collection_enabled=record.may_collect_data,
    )

    log.log(EV_SESSION_START, detail={"arm": arm.arm_id})
    log.log(EV_CONSENT, detail=record.as_dict())
    log.log(EV_ARM_ASSIGNED, detail=dict(allocation))
    log.log(
        EV_POLICY_LOADED,
        detail={
            "policy_hash": schema.policy_hash(policy),
            "stats": schema.validate_policy(policy).stats,
            "root_id": root_id,
        },
    )

    st.session_state.update(
        {
            "log": log,
            "arm": arm,
            "phase": PHASE_INTERVIEW,
            "messages": [],
            "node_id": root_id if arm.uses_policy_graph else "",
            "path": [root_id] if arm.uses_policy_graph else [],
            "visited": [root_id] if arm.uses_policy_graph else [],
            "reprobe_count": 0,
            "turn_index": 0,
            "pending_control": _opening_instruction(policy, arm),
            "interview_started_at": time.perf_counter(),
            "why_hint": "",
        }
    )
    generate_interviewer_turn(policy)


def _opening_instruction(policy: dict, arm: arms.Arm) -> str:
    if arm.uses_policy_graph:
        node = schema.get_node(policy, policy["root_id"]) or {}
        return (
            "CONTROL: This is the first turn. Greet the participant in one short sentence, say "
            "roughly how long this will take, then ask this and nothing else, in your own words: "
            '"' + node.get("question", "") + '". Purpose of the question: ' + node.get("objective", "")
        )
    return (
        "CONTROL: This is the first turn. Greet the participant in one short sentence, say roughly "
        "how long this will take, then ask your first question."
    )


def freeform_topics(policy: dict) -> list:
    """The same survey content the graph arms cover, without the ordering or gating."""
    nodes = policy.get("nodes", {})
    seen = set()
    topics = []
    for node_id in schema.core_nodes(policy):
        node = nodes[node_id]
        label = node.get("topic") or node.get("question", "")
        if label in seen:
            continue
        seen.add(label)
        topics.append("- " + label + " (" + node.get("objective", "") + ")")
    return topics


# ------------------------------------------------------------ interviewer turn


def generate_interviewer_turn(policy: dict) -> None:
    log: SessionLog = st.session_state["log"]
    plan: ModelPlan = st.session_state["plan"]
    arm: arms.Arm = st.session_state["arm"]

    if arm.uses_policy_graph:
        system_prompt = prompts.ELICITOR_INTERVIEWER_TREE
        values = {
            "context": policy.get("description", policy.get("title", "the session you have just had")),
            "scope_note": policy.get("scope_note", ""),
            "steering_note": policy.get("steering_note", ""),
        }
    else:
        system_prompt = prompts.ELICITOR_INTERVIEWER_FREEFORM
        values = {
            "context": policy.get("description", policy.get("title", "the session you have just had")),
            "scope_note": policy.get("scope_note", ""),
            "steering_note": policy.get("steering_note", ""),
            "topics": "\n".join(freeform_topics(policy)),
            "target_minutes": policy.get("target_minutes", 5),
        }

    text, _ = agents.ask_interviewer(
        log,
        plan,
        system_prompt=system_prompt,
        system_values=values,
        history=list(st.session_state["messages"]),
        control_instruction=st.session_state.get("pending_control", ""),
        node_id=st.session_state.get("node_id", ""),
    )

    st.session_state["messages"].append({"role": "assistant", "content": text})
    log.add_message("assistant", text)
    st.session_state["pending_control"] = ""
    st.session_state["question_shown_at"] = time.perf_counter()
    st.session_state["why_hint"] = ""


# --------------------------------------------------------------- the main loop


def handle_reply(policy: dict, reply: str, *, via_voice: bool = False) -> None:
    """Everything that happens between the participant pressing send and the next question."""
    log: SessionLog = st.session_state["log"]
    plan: ModelPlan = st.session_state["plan"]
    arm: arms.Arm = st.session_state["arm"]

    shown_at = st.session_state.get("question_shown_at", time.perf_counter())
    latency = round(time.perf_counter() - shown_at, 3)
    st.session_state["turn_index"] += 1
    turn_index = st.session_state["turn_index"]
    node_id = st.session_state.get("node_id", "")
    reprobe_count = st.session_state.get("reprobe_count", 0)

    # Captured before the reply is appended: the free-form arms judge adequacy
    # against the question that was actually asked.
    last_question = next(
        (m["content"] for m in reversed(st.session_state["messages"]) if m["role"] == "assistant"), ""
    )

    st.session_state["messages"].append({"role": "user", "content": reply})
    log.add_message("user", reply)

    metrics = text_metrics(reply)
    log.log(
        EV_REPLY,
        node_id=node_id,
        detail={
            "turn_index": turn_index,
            "reprobe_index": reprobe_count,
            "reply_latency_s": latency,
            "input_mode": "voice" if via_voice else "text",
            **metrics,
        },
    )

    turn_row = {
        "session_id": log.session_id,
        "pseudonym": log.pseudonym,
        "arm_id": arm.arm_id,
        "kind": "reply",
        "turn_index": turn_index,
        "node_id": node_id,
        "node_topic": (schema.get_node(policy, node_id) or {}).get("topic", ""),
        "node_priority": (schema.get_node(policy, node_id) or {}).get("priority", ""),
        "reprobe_index": reprobe_count,
        "reply_latency_s": latency,
        "input_mode": "voice" if via_voice else "text",
        "t_elapsed_s": log.elapsed(),
    }
    turn_row.update({"reply_" + k: v for k, v in metrics.items()})

    transcript = log.transcript_text()

    # ---- Agent [b]: only the adequacy arms consult it at all.
    if arm.uses_adequacy_check:
        node = (
            schema.get_node(policy, node_id)
            if arm.uses_policy_graph
            else _synthetic_node(turn_index, last_question)
        )
        verdict = agents.check_adequacy(
            log, plan, transcript=transcript, node=node or {}, reprobe_count=reprobe_count
        )
        turn_row.update(
            {
                "adequacy_checked": True,
                "adequate": verdict["adequate"],
                "adequacy_on_topic": verdict["on_topic"],
                "adequacy_wants_to_move_on": verdict["wants_to_move_on"],
                "adequacy_missing": verdict["missing"],
                "adequacy_confidence": verdict["confidence"],
                "adequacy_soft_capped": verdict.get("capped", False),
                "adequacy_fallback_used": verdict.get("fallback_used", False),
                "adequacy_reason": verdict["reason"],
            }
        )

        if verdict.get("distress_signal"):
            log.log(EV_SAFEGUARD, node_id=node_id, detail={"turn_index": turn_index, "reason": verdict["reason"]})
            st.session_state["safeguard_shown"] = True

        if verdict.get("capped"):
            log.log(
                EV_SOFT_CAP,
                node_id=node_id,
                detail={"turn_index": turn_index, "reprobe_count": reprobe_count, "missing": verdict["missing"]},
            )

        if not verdict["adequate"]:
            # Re-probe: the critique goes to Agent [a] on the hidden channel.
            st.session_state["reprobe_count"] = reprobe_count + 1
            st.session_state["pending_control"] = (
                "CONTROL: The participant has not yet covered: "
                + (verdict["missing"] or "the point of the question")
                + ". Acknowledge briefly what they did say, then ask once for that one thing. "
                + (verdict["suggested_probe"] or "")
            )
            log.log(
                EV_REPROBE,
                node_id=node_id,
                detail={"turn_index": turn_index, "missing": verdict["missing"], "probe": verdict["suggested_probe"]},
            )
            turn_row["outcome"] = "reprobe"
            log.add_turn(turn_row)
            generate_interviewer_turn(policy)
            return
    else:
        turn_row.update({"adequacy_checked": False, "adequate": "", "adequacy_soft_capped": False})

    turn_row["outcome"] = "advance"
    log.add_turn(turn_row)
    advance(policy)


def _synthetic_node(turn_index: int, question: str) -> dict:
    """The unit Agent [b] judges in the free-form arms.

    Free-form has no current node, so adequacy is tested against the turn itself --
    the question the interviewer actually asked, plus the paper's own binary
    operationalisation of adequacy as its objective. That keeps the governance
    factor identical across the structure factor, which is what makes the 2x2
    interpretable: arm C differs from arm A only in where the question came from.
    """
    return {
        "id": "freeform_turn_%d" % turn_index,
        "question": question or "the question the interviewer just asked",
        "objective": (
            "The answer states a position on the question that was asked and gives at least "
            "one supporting reason."
        ),
        "adequacy": {"max_reprobes": 1},
    }


def advance(policy: dict) -> None:
    """Move to the next node (graph arms) or the next free-form turn."""
    log: SessionLog = st.session_state["log"]
    plan: ModelPlan = st.session_state["plan"]
    arm: arms.Arm = st.session_state["arm"]
    st.session_state["reprobe_count"] = 0

    if not arm.uses_policy_graph:
        _advance_freeform(policy)
        return

    node_id = st.session_state["node_id"]
    transcript = log.transcript_text()
    next_id, detail = agents.choose_branch(
        log, plan, policy=policy, node_id=node_id, path=st.session_state["path"], transcript=transcript
    )

    next_id = _skip_optional_when_over_budget(policy, next_id, log)

    if not next_id:
        finish_interview(policy)
        return

    st.session_state["node_id"] = next_id
    st.session_state["path"].append(next_id)
    st.session_state["visited"].append(next_id)
    log.log(EV_ADVANCE, node_id=next_id, detail={"from": node_id, "condition": detail.get("condition", "")})

    node = schema.get_node(policy, next_id) or {}
    st.session_state["pending_control"] = (
        "CONTROL: Ask this next, in your own words, as one short question: \""
        + node.get("question", "")
        + "\". Purpose: "
        + node.get("objective", "")
        + ". Do not ask anything else."
    )
    generate_interviewer_turn(policy)


def _skip_optional_when_over_budget(policy: dict, next_id, log: SessionLog):
    """Drop depth probes, never core topics, when the session is running long.

    Coverage of core topics is the study's fixed constraint, so it is never traded
    for time. Optional nodes exist precisely to be the thing that gives.
    """
    budget_s = float(policy.get("target_minutes", 5)) * 60.0
    elapsed = time.perf_counter() - st.session_state.get("interview_started_at", time.perf_counter())
    guard = 0
    while next_id and elapsed > budget_s and guard < 10:
        node = schema.get_node(policy, next_id) or {}
        if node.get("priority") != schema.OPTIONAL:
            break
        children = node.get("children", [])
        log.log(
            EV_SKIP,
            node_id=next_id,
            detail={"reason": "over_time_budget", "elapsed_s": round(elapsed, 1), "budget_s": budget_s},
        )
        next_id = children[0]["target_id"] if children else None
        guard += 1
    return next_id


def _advance_freeform(policy: dict) -> None:
    """Free-form arms end on turn count or time, whichever comes first."""
    log: SessionLog = st.session_state["log"]
    # Matched to the graph arms' core spine so exposure is comparable across arms.
    max_turns = max(4, len(schema.core_nodes(policy)))
    budget_s = float(policy.get("target_minutes", 5)) * 60.0
    elapsed = time.perf_counter() - st.session_state.get("interview_started_at", time.perf_counter())

    if st.session_state["turn_index"] >= max_turns or elapsed > budget_s:
        finish_interview(policy)
        return

    st.session_state["pending_control"] = (
        "CONTROL: Continue the interview. Ask your next question, covering a topic you have not "
        "yet covered."
    )
    generate_interviewer_turn(policy)


def finish_interview(policy: dict) -> None:
    log: SessionLog = st.session_state["log"]
    plan: ModelPlan = st.session_state["plan"]

    st.session_state["pending_control"] = (
        "CONTROL: The survey is complete. Thank the participant in one short sentence. "
        "Do not ask another question."
    )
    generate_interviewer_turn(policy)

    transcript = log.transcript_text()
    constructs = list(policy.get("summary_questions") or []) or [
        (schema.get_node(policy, n) or {}).get("question", "") for n in schema.core_nodes(policy)[:5]
    ]

    with st.spinner("Putting together a summary of what you said..."):
        summaries, _ = agents.summarise(log, plan, transcript=transcript, constructs=constructs)
        topics = [
            (schema.get_node(policy, n) or {}).get("topic")
            or (schema.get_node(policy, n) or {}).get("question", "")
            for n in schema.core_nodes(policy)
        ]
        coded = agents.code_coverage(log, plan, transcript=transcript, topics=topics)

    arm: arms.Arm = st.session_state["arm"]
    structural = (
        schema.coverage(policy, st.session_state.get("visited", []))
        if arm.uses_policy_graph
        else schema.coverage_not_applicable("no decision graph in this arm; use coverage_coded")
    )
    log.set_meta(
        coverage_structural=structural,
        # Flattened alongside the JSON blob so the analysis table has a plain
        # numeric column rather than something that must be parsed per row.
        coverage_structural_rate=structural.get("core_rate", ""),
        coverage_coded=coded,
        coverage_coded_rate=(
            round(sum(1 for v in coded.values() if v == "covered") / len(coded), 4) if coded else ""
        ),
        path_taken=st.session_state.get("path", []),
        interview_seconds=round(
            time.perf_counter() - st.session_state.get("interview_started_at", time.perf_counter()), 2
        ),
    )

    st.session_state["summaries"] = summaries
    st.session_state["phase"] = PHASE_FEEDBACK


# -------------------------------------------------------------------- screens


def render_interview(policy: dict) -> None:
    log: SessionLog = st.session_state["log"]
    arm: arms.Arm = st.session_state["arm"]

    ui.header("Computer Science Open Day", policy.get("title", "Your experience today"))

    # An honest progress indicator: the shortest remaining route in the graph arms,
    # turn count in the free-form arms. Never a bar that pretends to know more.
    if arm.uses_policy_graph:
        remaining = schema.shortest_remaining(policy, st.session_state["node_id"])
        done = len(st.session_state["visited"])
        fraction = done / max(1, done + remaining)
    else:
        max_turns = max(4, len(schema.core_nodes(policy)))
        fraction = min(1.0, st.session_state["turn_index"] / max_turns)
    ui.progress_bar(fraction, "About " + str(max(1, int(round((1 - fraction) * 100)))) + "% to go")

    ui.render_chat(st.session_state["messages"])

    if st.session_state.get("safeguard_shown"):
        st.info(
            "If anything today has left you feeling uncomfortable, please speak to a member of "
            "staff at the stand. You can stop at any time."
        )

    if st.session_state.get("why_hint"):
        st.caption("Why we ask: " + st.session_state["why_hint"])

    reply = st.chat_input("Type your answer...")

    controls = st.columns([1, 1, 1])
    if controls[0].button("Why are you asking?", width="stretch"):
        node = schema.get_node(policy, st.session_state.get("node_id", "")) or {
            "question": st.session_state["messages"][-1]["content"] if st.session_state["messages"] else "",
            "objective": "understanding your experience of the open day",
        }
        st.session_state["why_hint"] = agents.why_hint(log, st.session_state["plan"], node=node)
        st.rerun()

    if controls[1].button("Skip this one", width="stretch"):
        log.log(
            EV_SKIP,
            node_id=st.session_state.get("node_id", ""),
            detail={"reason": "participant_requested", "turn_index": st.session_state["turn_index"]},
        )
        st.session_state["messages"].append({"role": "user", "content": "[skipped]"})
        log.add_message("user", "[skipped]")
        advance(policy)
        st.rerun()

    if controls[2].button("Finish now", width="stretch"):
        finish_interview(policy)
        st.rerun()

    if st.session_state.get("allow_voice") and speech.transcription_available():
        audio = st.audio_input("Or speak your answer", label_visibility="visible")
        if audio is not None and not st.session_state.get("voice_consumed"):
            text, error = speech.transcribe(audio.read())
            st.session_state["voice_consumed"] = True
            if text:
                handle_reply(policy, text, via_voice=True)
                st.session_state["voice_consumed"] = False
                st.rerun()
            elif error:
                st.warning("We could not hear that clearly -- please type your answer. (" + error + ")")

    if reply:
        handle_reply(policy, reply)
        st.rerun()


def render_feedback(policy: dict, record: consent.ConsentRecord) -> None:
    log: SessionLog = st.session_state["log"]
    ui.header("Almost done", "Two quick things")
    st.caption("This takes about two minutes and is the part that helps us improve the tool.")

    fidelity = feedback.render_fidelity_block(st.session_state.get("summaries", []))
    st.divider()
    experience = feedback.render_experience_block()
    st.divider()
    workload = feedback.render_workload_block()
    st.divider()
    open_text = feedback.render_open_block()

    st.divider()
    if st.button("Submit", type="primary", width="stretch"):
        missing = feedback.unanswered_required(fidelity, experience)
        if missing:
            st.warning(
                "Please answer these before submitting:\n\n"
                + "\n".join("- " + m for m in missing[:6])
                + ("\n- ..." if len(missing) > 6 else "")
            )
            return

        scored = feedback.score_battery(fidelity, experience, workload, open_text)
        log.ratings.update(scored)
        log.log(EV_RATING, detail={"n_items": len(scored)})
        log.log(EV_SESSION_END, detail={"completed": True})

        with st.spinner("Submitting..."):
            ok, message = delivery.deliver(log, may_collect=record.may_collect_data)
        st.session_state["delivery"] = (ok, message)
        st.session_state["phase"] = PHASE_DONE
        st.rerun()


def render_done(record: consent.ConsentRecord) -> None:
    log: SessionLog = st.session_state["log"]
    ok, message = st.session_state.get("delivery", (False, ""))

    ui.header("", "Thank you")
    if record.may_collect_data:
        if ok:
            st.success(message)
        else:
            st.error(message)
            _render_manual_download(log)
    else:
        st.info("This was a practice session. Nothing was collected, stored or sent.")

    st.write(
        "Your responses will help us understand how well an AI interviewer can capture what "
        "people actually mean. Please hand the device back to a member of the team."
    )
    st.caption("Participant reference: " + log.pseudonym)


def _render_manual_download(log: SessionLog) -> None:
    st.caption("Please show this screen to a member of the research team.")
    for attachment in delivery.build_attachments(log):
        st.download_button(
            "Download " + attachment.filename,
            data=attachment.content,
            file_name=attachment.filename,
            mime="application/json" if attachment.subtype == "json" else "text/csv",
            width="stretch",
        )


# ------------------------------------------------------------------- entry


def main() -> None:
    ui.page_setup("Open Day Survey", "🎙️")
    researcher_sidebar()

    policy = st.session_state.get("policy")
    if policy is None:
        files = available_policies()
        chosen = st.session_state.get("policy_file")
        path = next((f for f in files if f.name == chosen), files[0] if files else None)
        if path is None:
            st.error(
                "No survey policy is installed. Add a policy graph JSON file to "
                "`knowledge-elicitation/trees/`, or upload one from the researcher sidebar."
            )
            st.stop()
        policy = load_policy(str(path))
        st.session_state["policy"] = policy

    report = schema.validate_policy(policy)
    if not report.ok:
        # Section 3.2.4: the Elicitor consumes only JSON that passes schema and
        # consistency checks. A broken policy stops the app rather than producing an
        # unauditable session.
        st.error("The survey policy failed validation and cannot be run.")
        for error in report.errors:
            st.write("- " + error)
        st.stop()

    minutes = int(round(float(policy.get("target_minutes", 5))))
    record = consent.consent_gate(
        study_title=STUDY_TITLE,
        use_case="the Computer Science Open Day activities you have just experienced",
        survey_minutes=minutes,
        feedback_minutes=2,
    )

    if record.status == consent.STATUS_DECLINED:
        consent.render_declined()
        st.stop()

    if record.status == consent.STATUS_EXPLORE:
        consent.explore_banner()

    if "log" not in st.session_state:
        begin_session(policy, record)
        st.rerun()

    phase = st.session_state.get("phase", PHASE_INTERVIEW)
    if phase == PHASE_INTERVIEW:
        render_interview(policy)
    elif phase == PHASE_FEEDBACK:
        if record.may_collect_data:
            render_feedback(policy, record)
        else:
            # Under-18 practice sessions end here: no battery, no submission.
            st.session_state["phase"] = PHASE_DONE
            st.rerun()
    else:
        render_done(record)


main()

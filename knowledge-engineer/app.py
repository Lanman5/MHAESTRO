"""K-Eng -- Phase 1 of MHAESTRO.

Interviews the domain expert who is running the event, then compiles what they
said into a validated decision graph the Elicitor can execute.

Two things differ materially from the published prototype.

**The artefact is validated, not merely generated.** Section 3.2.4 defines a
schema contract the Elicitor enforces at run time -- condition labels must match
exactly, targets must exist, the graph must terminate -- but the prototype
generated the tree once and discovered contract violations only when a live
session hit them. Here the graph is generated, checked against the contract, and
the concrete faults are fed back for repair until it passes or the attempts run
out. A graph that still fails is never offered for use.

**The expert sees the graph.** Section 3.2.3 rests reproducibility on the graph
being reviewable by experts and Table 1 lists human oversight as a governance
control, yet the prototype's output was a JSON download. Here the expert gets a
drawing, a table, a per-node verdict, and a structured review that is recorded as
data -- because their judgement of whether the compiled survey actually asks what
they need is itself a study outcome.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from mhaestro import agents, delivery, feedback, prompts, schema, ui, viz
from mhaestro.agents import ModelPlan
from mhaestro.config import get_secret
from mhaestro.llm import (
    DEFAULT_MODELS,
    MODEL_CATALOGUE,
    PROVIDER_LABELS,
    available_providers,
    missing_reason,
)
from mhaestro.telemetry import (
    EV_POLICY_LOADED,
    EV_QUESTION,
    EV_RATING,
    EV_REPLY,
    EV_SESSION_END,
    EV_SESSION_START,
    SessionLog,
    text_metrics,
)

TREES_DIR = _ROOT / "knowledge-elicitation" / "trees"

PHASE_SETUP = "setup"
PHASE_INTERVIEW = "interview"
PHASE_REVIEW = "review"
PHASE_FEEDBACK = "feedback"
PHASE_DONE = "done"


# ------------------------------------------------------------------- settings


def default_plan() -> ModelPlan:
    ready = available_providers()
    preferred = get_secret("DEFAULT_PROVIDER", "openai")
    provider = preferred if preferred in ready else (ready[0] if ready else "openai")
    return ModelPlan(
        interviewer_provider=provider,
        control_provider=provider,
        analysis_provider=provider,
    )


def require_access() -> None:
    """Gate the whole app behind a shared passcode when one is configured.

    On Streamlit Community Cloud this app has a public URL, and every screen past
    the setup form spends API credits. `EXPERT_PASSCODE` (falling back to
    `RESEARCHER_PIN`) keeps it to the people running the study. Leave both unset
    only when running locally.
    """
    passcode = get_secret("EXPERT_PASSCODE") or get_secret("RESEARCHER_PIN")
    if not passcode:
        return
    if st.session_state.get("expert_access_granted"):
        return

    ui.header("MHAESTRO · Phase 1", "Knowledge Engineering")
    st.caption("This tool is for the team running the event.")
    entered = st.text_input("Passcode", type="password", key="expert_passcode")
    if entered and entered != str(passcode):
        st.error("That passcode is not recognised.")
    if entered == str(passcode):
        st.session_state["expert_access_granted"] = True
        st.rerun()
    st.stop()


def settings_sidebar() -> ModelPlan:
    plan: ModelPlan = st.session_state.get("plan") or default_plan()
    ready = available_providers()

    with st.sidebar:
        st.markdown("### Settings")
        if not ready:
            st.error("No provider is usable.")
            for name in PROVIDER_LABELS:
                st.caption(PROVIDER_LABELS[name] + ": " + missing_reason(name))
            st.stop()

        for label, provider_attr, model_attr, note in (
            ("Interview agent", "interviewer_provider", "interviewer_model", "Talks to you."),
            (
                "Analysis agent",
                "analysis_provider",
                "analysis_model",
                "Compiles the decision graph. Use the strongest model you have.",
            ),
        ):
            st.markdown("**" + label + "**")
            st.caption(note)
            current = getattr(plan, provider_attr)
            provider = st.selectbox(
                label + " provider",
                options=ready,
                index=ready.index(current) if current in ready else 0,
                format_func=lambda p: PROVIDER_LABELS[p],
                key="keng_provider_" + provider_attr,
                label_visibility="collapsed",
            )
            catalogue = MODEL_CATALOGUE.get(provider, [])
            current_model = getattr(plan, model_attr)
            options = catalogue + ([current_model] if current_model and current_model not in catalogue else [])
            model = st.selectbox(
                label + " model",
                options=options or [DEFAULT_MODELS.get(provider, "")],
                index=options.index(current_model) if current_model in options else 0,
                key="keng_model_" + model_attr,
                label_visibility="collapsed",
            )
            setattr(plan, provider_attr, provider)
            setattr(plan, model_attr, model)

        plan.control_provider = plan.analysis_provider
        plan.control_model = plan.analysis_model
        st.session_state["plan"] = plan

        st.divider()
        with st.expander("Prompt provenance"):
            st.caption("Bundle hash: " + prompts.bundle_hash())
            st.json(prompts.bundle_manifest(), expanded=False)

    return plan


# ---------------------------------------------------------------------- setup


def render_setup(plan: ModelPlan) -> None:
    ui.header(
        "MHAESTRO · Phase 1",
        "Knowledge Engineering",
        "A short interview about what you need to find out. We turn it into the survey your "
        "visitors will answer.",
    )

    with st.container(border=True):
        st.markdown("**What are we building a survey about?**")
        context = st.text_input(
            "Event or activity",
            value=st.session_state.get("ctx_context", ""),
            placeholder="e.g. the Computer Science Open Day, October 2026",
        )
        respondents = st.text_input(
            "Who will answer it",
            value=st.session_state.get("ctx_respondents", ""),
            placeholder="e.g. prospective students and their parents, straight after the visit",
        )
        left, right = st.columns(2)
        minutes = left.number_input(
            "Time budget per respondent (minutes)", min_value=2.0, max_value=20.0, value=5.0, step=0.5
        )
        role = right.text_input("Your role", value="", placeholder="e.g. Open Day lead")

    st.caption(
        "The interview takes about ten minutes. Everything you say is used only to build the "
        "survey and to record how well this tool worked for you."
    )

    if st.button("Start the interview", type="primary", width="stretch", disabled=not (context and respondents)):
        st.session_state.update({"ctx_context": context, "ctx_respondents": respondents})
        begin_session(plan, context, respondents, minutes, role)
        st.rerun()


def begin_session(plan: ModelPlan, context: str, respondents: str, minutes: float, role: str) -> None:
    log = SessionLog(tool="keng")
    log.set_meta(
        study="open_day_hybrid_surveying",
        phase="knowledge_engineering",
        context=context,
        respondents=respondents,
        target_minutes=minutes,
        expert_role=role,
        models=plan.as_dict(),
        prompt_bundle_hash=prompts.bundle_hash(),
        prompt_manifest=prompts.bundle_manifest(),
        data_collection_enabled=True,
    )
    log.log(EV_SESSION_START, detail={"context": context, "respondents": respondents})

    opening = prompts.KENG_OPENING.render(respondents=respondents, context=context)
    log.log(
        EV_QUESTION,
        agent_role=agents.ROLE_INTERVIEWER,
        prompt_id=prompts.KENG_OPENING.template_id,
        prompt_hash=prompts.KENG_OPENING.hash,
        detail={"scripted": True, "output_words": len(opening.split())},
    )

    st.session_state.update(
        {
            "log": log,
            "phase": PHASE_INTERVIEW,
            "messages": [{"role": "assistant", "content": opening}],
            "turn_index": 0,
            "question_shown_at": time.perf_counter(),
            "interview_started_at": time.perf_counter(),
        }
    )
    log.add_message("assistant", opening)


# ------------------------------------------------------------------ interview


def render_interview(plan: ModelPlan) -> None:
    log: SessionLog = st.session_state["log"]
    ui.header("MHAESTRO · Phase 1", "Scoping interview")

    ui.status_row(
        {
            "Turns": str(st.session_state["turn_index"]),
            "Elapsed": _mmss(time.perf_counter() - st.session_state["interview_started_at"]),
            "Budget/respondent": str(log.meta.get("target_minutes", 5)) + " min",
        }
    )

    ui.render_chat(st.session_state["messages"], interviewer_avatar="🧭", participant_avatar="🧑‍🏫")

    reply = st.chat_input("Your answer...")
    if st.button("I've said enough — build the survey", width="stretch"):
        st.session_state["phase"] = PHASE_REVIEW
        st.session_state.pop("policy", None)
        st.rerun()

    if reply:
        latency = round(time.perf_counter() - st.session_state.get("question_shown_at", time.perf_counter()), 3)
        st.session_state["turn_index"] += 1
        metrics = text_metrics(reply)

        st.session_state["messages"].append({"role": "user", "content": reply})
        log.add_message("user", reply)
        log.log(
            EV_REPLY,
            detail={"turn_index": st.session_state["turn_index"], "reply_latency_s": latency, **metrics},
        )
        log.add_turn(
            {
                "session_id": log.session_id,
                "kind": "reply",
                "turn_index": st.session_state["turn_index"],
                "reply_latency_s": latency,
                "t_elapsed_s": log.elapsed(),
                **{"reply_" + k: v for k, v in metrics.items()},
            }
        )

        with st.spinner("Thinking..."):
            text, _ = agents.ask_interviewer(
                log,
                plan,
                system_prompt=prompts.KENG_INTERVIEWER,
                system_values={
                    "context": log.meta.get("context", ""),
                    "respondents": log.meta.get("respondents", ""),
                    "target_minutes": log.meta.get("target_minutes", 5),
                },
                history=list(st.session_state["messages"]),
            )
        st.session_state["messages"].append({"role": "assistant", "content": text})
        log.add_message("assistant", text)
        st.session_state["question_shown_at"] = time.perf_counter()
        st.rerun()


# --------------------------------------------------------------------- review


def compile_policy(plan: ModelPlan) -> None:
    log: SessionLog = st.session_state["log"]
    started = time.perf_counter()

    with st.spinner("Compiling your survey into a decision graph and checking it..."):
        policy, report, attempts = agents.synthesise_policy(
            log,
            plan,
            transcript=log.transcript_text(interviewer="Knowledge engineer", participant="Expert"),
            context=log.meta.get("context", ""),
            respondents=log.meta.get("respondents", ""),
            target_minutes=float(log.meta.get("target_minutes", 5)),
        )

    st.session_state["policy"] = policy
    st.session_state["policy_report"] = report
    st.session_state["policy_attempts"] = attempts
    log.set_meta(
        compile_seconds=round(time.perf_counter() - started, 2),
        compile_attempts=len(attempts),
        compile_repairs_needed=max(0, len(attempts) - 1),
        policy_valid=report.ok,
        policy_hash=schema.policy_hash(policy) if policy else "",
        policy_stats=report.stats,
        policy_errors=report.errors,
        policy_warnings=report.warnings,
    )
    if policy:
        log.log(
            EV_POLICY_LOADED,
            detail={"policy_hash": schema.policy_hash(policy), "valid": report.ok, "stats": report.stats},
        )


def render_review(plan: ModelPlan) -> None:
    log: SessionLog = st.session_state["log"]

    if "policy" not in st.session_state:
        compile_policy(plan)
        st.rerun()

    policy = st.session_state.get("policy")
    report: schema.ValidationReport = st.session_state["policy_report"]

    ui.header("MHAESTRO · Phase 1", "Your survey")

    if policy is None:
        st.error("The survey could not be compiled. " + "; ".join(report.errors))
        if st.button("Try again"):
            st.session_state.pop("policy", None)
            st.rerun()
        st.stop()

    if report.ok:
        st.success(
            "This survey passed every structural check and is ready to run. "
            "Please read it over -- you are the only person who can tell us whether it asks "
            "the right things."
        )
    else:
        st.error(
            "This survey has structural faults and cannot be run as it stands. "
            "Your review is still worth recording."
        )
        for error in report.errors:
            st.write("- " + error)

    stats = report.stats or {}
    longest_seconds = schema.estimate_longest_seconds(policy)
    ui.status_row(
        {
            "Questions": str(stats.get("node_count", 0)),
            "Must-ask": str(stats.get("core_count", 0)),
            "Longest path": str(stats.get("max_depth", 0)),
            "Est. longest run": _mmss(longest_seconds),
        }
    )
    if longest_seconds > float(policy.get("target_minutes", 5)) * 60:
        st.warning(
            "The longest route through this survey is likely to overrun your "
            + str(policy.get("target_minutes", 5))
            + "-minute budget. Consider marking some questions as optional."
        )
    bypassable = (stats or {}).get("bypassable_core") or []
    if bypassable:
        # Coverage completeness is the study's fixed constraint, so a must-ask
        # question that some routes skip is the single most consequential fault
        # short of a structural one.
        st.warning(
            "**These must-ask questions can be skipped by some answers**, so not every "
            "visitor would be asked them: "
            + ", ".join("`" + node_id + "`" for node_id in bypassable)
            + ". Either accept them as optional follow-ups, or say so below and we will "
            "route every route through them."
        )

    for warning in report.warnings:
        if "can be bypassed" not in warning:
            st.caption("⚠ " + warning)

    graph_tab, table_tab, json_tab = st.tabs(["Flow", "Question by question", "Raw artefact"])

    with graph_tab:
        st.caption(
            "Blue = every respondent is asked this. Grey = asked only if the conversation goes "
            "that way. Green = the survey ends here. Labels on the arrows are the answer "
            "categories that decide the route."
        )
        st.graphviz_chart(viz.policy_to_dot(policy))

    with table_tab:
        st.dataframe(
            viz.policy_outline(policy),
            width="stretch",
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("Question", width="large"),
                "objective": st.column_config.TextColumn("What a good answer contains", width="large"),
            },
        )

    with json_tab:
        st.json(policy, expanded=False)

    st.divider()
    render_expert_review(policy, report)


def render_expert_review(policy: dict, report: schema.ValidationReport) -> None:
    """The expert's structured verdict on the compiled survey. This is study data."""
    log: SessionLog = st.session_state["log"]
    st.subheader("Your verdict")
    st.caption("Be blunt. A survey you would not actually run is the most useful thing you can tell us.")

    review = {}
    for key, text in (
        ("covers_needs", "This survey would tell me what I need to know."),
        ("would_run", "I would be happy to put this in front of real visitors."),
        ("wording", "The questions are worded the way I would word them."),
        ("branching", "The branching sends people down sensible routes."),
        ("length", "The length is right for the time I have with each person."),
        ("faithful", "This reflects what I actually said in the interview."),
    ):
        st.markdown("**" + text + "**")
        review[key] = st.segmented_control(
            text,
            options=[1, 2, 3, 4, 5],
            format_func=lambda v: feedback.AGREE_LABELS[v - 1],
            key="review_" + key,
            label_visibility="collapsed",
            default=None,
        )

    st.markdown("**Anything missing that you asked for?**")
    review["missing"] = st.text_area("Missing", label_visibility="collapsed", placeholder="Optional", height=80)
    st.markdown("**Anything here you would cut?**")
    review["cut"] = st.text_area("Cut", label_visibility="collapsed", placeholder="Optional", height=80)

    with st.expander("Flag individual questions"):
        st.caption("Only the ones you would change. Leave the rest alone.")
        node_verdicts = {}
        for node_id, node in (policy.get("nodes") or {}).items():
            columns = st.columns([3, 2])
            columns[0].markdown("`" + node_id + "` — " + node.get("question", ""))
            node_verdicts[node_id] = columns[1].segmented_control(
                node_id,
                options=["Keep", "Reword", "Cut"],
                key="node_verdict_" + node_id,
                label_visibility="collapsed",
                default=None,
            )
        review["node_verdicts"] = {k: v for k, v in node_verdicts.items() if v}

    st.divider()
    left, right = st.columns(2)

    if left.button("Record my review and continue", type="primary", width="stretch"):
        log.set_meta(expert_review=review)
        log.log(EV_RATING, detail={"kind": "expert_tree_review", **{k: v for k, v in review.items()}})
        st.session_state["phase"] = PHASE_FEEDBACK
        st.rerun()

    if right.button("Redo the interview from here", width="stretch"):
        st.session_state["phase"] = PHASE_INTERVIEW
        st.session_state.pop("policy", None)
        st.rerun()

    st.divider()
    _render_policy_export(policy, report)


def _render_policy_export(policy: dict, report: schema.ValidationReport) -> None:
    st.markdown("**Install this survey**")
    filename = (policy.get("survey_id") or "survey") + ".json"
    payload = json.dumps(policy, indent=2, ensure_ascii=False)

    left, right = st.columns(2)
    left.download_button(
        "Download " + filename,
        data=payload,
        file_name=filename,
        mime="application/json",
        width="stretch",
    )

    # Writing straight into the Elicitor's policy folder only works where the two
    # apps share a filesystem. On Streamlit Community Cloud they do not, and a file
    # written there disappears on the next restart -- so the download is the real
    # install route and this is a local-development convenience.
    if right.button("Save into the Elicitor", width="stretch", disabled=not report.ok):
        try:
            TREES_DIR.mkdir(parents=True, exist_ok=True)
            (TREES_DIR / filename).write_text(payload, encoding="utf-8")
            st.success("Saved to " + str(TREES_DIR / filename))
        except Exception as exc:  # noqa: BLE001
            st.warning(
                "Could not write to the Elicitor's policy folder (" + type(exc).__name__ + "). "
                "Download the file and commit it to the repository instead."
            )


# ------------------------------------------------------------------- feedback


def render_feedback() -> None:
    log: SessionLog = st.session_state["log"]
    ui.header("MHAESTRO · Phase 1", "How was this tool to use?")
    st.caption("Two minutes. This is about the knowledge-engineering tool, not the survey it produced.")

    experience = feedback.render_experience_block(key_prefix="keng_ux")
    st.divider()
    workload = feedback.render_workload_block(key_prefix="keng_tlx")
    st.divider()
    open_text = feedback.render_open_block(key_prefix="keng_open")

    st.divider()
    if st.button("Submit", type="primary", width="stretch"):
        scored = feedback.score_battery({}, experience, workload, open_text)
        log.ratings.update(scored)
        log.log(EV_RATING, detail={"kind": "expert_tool_experience"})
        log.log(EV_SESSION_END, detail={"completed": True})

        attachments = delivery.build_attachments(log)
        policy = st.session_state.get("policy")
        if policy:
            attachments.append(
                delivery.Attachment(
                    "keng_" + log.pseudonym + "_policy.json",
                    json.dumps(policy, indent=2, ensure_ascii=False),
                    "json",
                )
            )

        with st.spinner("Submitting..."):
            ok, message = delivery.deliver(log, may_collect=True, attachments=attachments)
        st.session_state["delivery"] = (ok, message)
        st.session_state["phase"] = PHASE_DONE
        st.rerun()


def render_done() -> None:
    log: SessionLog = st.session_state["log"]
    ok, message = st.session_state.get("delivery", (False, ""))

    ui.header("", "Thank you")
    if ok:
        st.success(message)
    else:
        st.error(message)
        for attachment in delivery.build_attachments(log):
            st.download_button(
                "Download " + attachment.filename,
                data=attachment.content,
                file_name=attachment.filename,
                mime="application/json" if attachment.subtype == "json" else "text/csv",
                width="stretch",
            )

    policy = st.session_state.get("policy")
    if policy:
        st.divider()
        _render_policy_export(policy, st.session_state["policy_report"])

    st.caption("Session reference: " + log.pseudonym)


def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return "%d:%02d" % (seconds // 60, seconds % 60)


# ---------------------------------------------------------------------- entry


def main() -> None:
    ui.page_setup("Knowledge Engineering", "🧭", wide=True)
    require_access()
    plan = settings_sidebar()

    phase = st.session_state.get("phase", PHASE_SETUP)
    if phase == PHASE_SETUP:
        render_setup(plan)
    elif phase == PHASE_INTERVIEW:
        render_interview(plan)
    elif phase == PHASE_REVIEW:
        render_review(plan)
    elif phase == PHASE_FEEDBACK:
        render_feedback()
    else:
        render_done()


main()

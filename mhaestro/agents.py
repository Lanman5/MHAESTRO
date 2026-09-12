"""The agent layer: one function per role in the paper's orchestration.

Figure 2 names three runtime roles -- Agent [a] the survey agent, Agent [b] the
steering/adequacy checker, Agent [c] the tree-traversal controller -- plus the
summariser, and Section 6.6 proposes Agent [d], a lightweight safeguarding monitor
over the same per-turn stream. Each is a thin, auditable wrapper: render a
versioned prompt, call whichever provider that role is configured for, validate
the reply, and write one event row carrying the model, the latency, the tokens and
the prompt hash.

Two invariants hold everywhere in this module:

*   **A failed agent call never becomes a silent pass.** If the checker cannot be
    reached or returns unparseable JSON, the turn advances (the participant is not
    made to wait on our infrastructure) but the event is logged as a failure with
    `fallback_used` set, so those turns can be excluded or modelled.
*   **The traversal agent can only pick a declared label.** Anything else is a
    traversal error, caught here and resolved to the node's catch-all, exactly as
    Section 3.2.4 requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import prompts as P
from . import schema as S
from .llm import DEFAULT_MODELS, DEFAULT_PROVIDER, LLMResult, complete
from .telemetry import (
    EV_ADEQUACY,
    EV_ERROR,
    EV_QUESTION,
    EV_SUMMARY,
    EV_TRAVERSAL,
    EV_WHY_HINT,
    SessionLog,
)

# Role -> the kind of work it does. Interviewer turns are conversational and want a
# little warmth; the control agents are classifiers and want determinism.
ROLE_INTERVIEWER = "agent_a_interviewer"
ROLE_CHECKER = "agent_b_checker"
ROLE_TRAVERSAL = "agent_c_traversal"
ROLE_SUMMARISER = "summariser"
ROLE_CODER = "coverage_coder"
ROLE_ANALYST = "analysis_agent"


@dataclass
class ModelPlan:
    """Which provider and model runs each role.

    Roles are configured separately because they have different requirements: the
    interviewer is on the participant's critical path and wants a fast model, while
    the analysis agent compiles the policy graph once and wants the strongest one.
    The plan is recorded in session meta so a run can be reproduced.
    """

    interviewer_provider: str = DEFAULT_PROVIDER
    interviewer_model: str = ""
    control_provider: str = DEFAULT_PROVIDER
    control_model: str = ""
    analysis_provider: str = DEFAULT_PROVIDER
    analysis_model: str = ""

    def __post_init__(self) -> None:
        self.interviewer_model = self.interviewer_model or DEFAULT_MODELS.get(self.interviewer_provider, "")
        self.control_model = self.control_model or DEFAULT_MODELS.get(self.control_provider, "")
        self.analysis_model = self.analysis_model or DEFAULT_MODELS.get(self.analysis_provider, "")

    def as_dict(self) -> Dict[str, str]:
        return {
            "interviewer": self.interviewer_provider + "/" + self.interviewer_model,
            "control": self.control_provider + "/" + self.control_model,
            "analysis": self.analysis_provider + "/" + self.analysis_model,
        }


# --------------------------------------------------------- Agent [a] interviewer


def ask_interviewer(
    log: SessionLog,
    plan: ModelPlan,
    *,
    system_prompt: P.Prompt,
    system_values: Dict[str, Any],
    history: List[Dict[str, str]],
    control_instruction: str = "",
    node_id: str = "",
) -> Tuple[str, LLMResult]:
    """Produce the next thing the participant sees.

    `control_instruction` is the hidden channel of Section 3.3.3: the steering
    agent's critique is appended as a system message the participant never sees,
    which is what compels the re-probe. It is passed here rather than written into
    the transcript so it can never leak into the visible conversation.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.render(**system_values)}
    ]
    messages.extend(history)
    if control_instruction:
        messages.append({"role": "system", "content": control_instruction})

    result = complete(
        messages,
        provider=plan.interviewer_provider,
        model=plan.interviewer_model,
        max_tokens=1000,
        temperature=0.6 if plan.interviewer_provider != "anthropic" else None,
        effort="low",
    )

    text = (result.text or "").strip()
    if not result.ok or not text:
        text = "Sorry -- something went wrong at our end. Could you say that again?"

    log.log(
        EV_QUESTION,
        node_id=node_id,
        agent_role=ROLE_INTERVIEWER,
        llm=result,
        prompt_id=system_prompt.template_id,
        prompt_hash=system_prompt.hash,
        detail={
            "had_control_instruction": bool(control_instruction),
            "control_instruction": control_instruction,
            "output_words": len(text.split()),
            "output_chars": len(text),
        },
    )
    return text, result


# ------------------------------------------------------------ Agent [b] checker


def check_adequacy(
    log: SessionLog,
    plan: ModelPlan,
    *,
    transcript: str,
    node: Dict[str, Any],
    reprobe_count: int,
) -> Dict[str, Any]:
    """The steering agent. Returns a verdict dict that is always safe to act on."""
    adequacy = node.get("adequacy", {}) or {}
    max_reprobes = int(adequacy.get("max_reprobes", 1))
    system = P.ELICITOR_ADEQUACY.render(
        question=node.get("question", ""),
        objective=node.get("objective", ""),
        reprobe_count=reprobe_count,
        max_reprobes=max_reprobes,
        test=adequacy_test(adequacy),
    )

    result = complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Transcript so far:\n\n" + transcript},
        ],
        provider=plan.control_provider,
        model=plan.control_model,
        json_schema=P.ADEQUACY_SCHEMA if plan.control_provider != "openai" else None,
        want_json=True,
        max_tokens=1024,
        temperature=0.0 if plan.control_provider != "anthropic" else None,
        effort="low",
    )

    verdict = _default_adequacy()
    fallback_used = True
    if result.ok and isinstance(result.data, dict):
        data = result.data
        verdict = {
            "adequate": bool(data.get("adequate", True)),
            "on_topic": bool(data.get("on_topic", True)),
            "wants_to_move_on": bool(data.get("wants_to_move_on", False)),
            "missing": str(data.get("missing", "") or ""),
            "suggested_probe": str(data.get("suggested_probe", "") or ""),
            "distress_signal": bool(data.get("distress_signal", False)),
            "confidence": _as_float(data.get("confidence"), 0.0),
            "reason": str(data.get("reason", "") or ""),
        }
        fallback_used = False

    # The soft cap of Section 6.1. Once a node has spent its re-probes the turn
    # advances whatever the checker says, and the shortfall is recorded rather than
    # pressed. This is the single change that most directly targets the published
    # intrusiveness score.
    capped = False
    if not verdict["adequate"] and reprobe_count >= max_reprobes:
        verdict["adequate"] = True
        verdict["capped"] = True
        capped = True
    verdict.setdefault("capped", False)

    # A participant who has said they are finished is finished, regardless.
    if verdict["wants_to_move_on"]:
        verdict["adequate"] = True

    verdict["fallback_used"] = fallback_used

    log.log(
        EV_ADEQUACY,
        node_id=node.get("id", ""),
        agent_role=ROLE_CHECKER,
        llm=result,
        prompt_id=P.ELICITOR_ADEQUACY.template_id,
        prompt_hash=P.ELICITOR_ADEQUACY.hash,
        detail={
            **verdict,
            "reprobe_count": reprobe_count,
            "max_reprobes": max_reprobes,
            "soft_capped": capped,
            # The bar this turn was judged against, so a session can be re-scored
            # later without guessing which policy was in force.
            "adequacy_test": adequacy_test(adequacy),
            "require_stance": bool(adequacy.get("require_stance", True)),
            "require_reason": bool(adequacy.get("require_reason", True)),
        },
    )
    return verdict


def adequacy_test(adequacy: Dict[str, Any]) -> str:
    """Turn a node's adequacy settings into the sentence Agent [b] is judged against.

    The published operationalisation (Section 3.3.3) is "a topical stance and one
    supporting reason", and that is the default here. `require_stance` and
    `require_reason` relax it per node, which is the right place to soften the
    gate: it changes how demanding the bar is for a particular question without
    touching whether the gate exists at all -- so the arm contrast between
    governance on and governance off survives intact.
    """
    stance = bool(adequacy.get("require_stance", True))
    reason = bool(adequacy.get("require_reason", True))
    if stance and reason:
        return "carries a topical stance and at least one supporting reason"
    if stance:
        return (
            "states a position on the question. A reason is welcome but is not required, "
            "so do not re-probe merely for the absence of one"
        )
    if reason:
        return "offers at least one reason, example or detail, in any form"
    return (
        "is a genuine attempt to answer the question, however brief. Re-probe only if the "
        "answer is about something else entirely"
    )


def _default_adequacy() -> Dict[str, Any]:
    """What we assume when the checker is unreachable: pass, and say so in the log.

    Failing open is deliberate. Failing closed would trap a participant in a loop
    because our infrastructure broke, which is exactly the experience this revision
    exists to remove.
    """
    return {
        "adequate": True,
        "on_topic": True,
        "wants_to_move_on": False,
        "missing": "",
        "suggested_probe": "",
        "distress_signal": False,
        "confidence": 0.0,
        "reason": "Adequacy check unavailable; advanced by default.",
        "capped": False,
    }


# ---------------------------------------------------------- Agent [c] traversal


def choose_branch(
    log: SessionLog,
    plan: ModelPlan,
    *,
    policy: dict,
    node_id: str,
    path: List[str],
    transcript: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Advance one edge. Returns (next_node_id, decision detail).

    Deterministic where the graph is deterministic: a leaf ends the session and a
    single child is followed without consulting a model at all (Section 3.3.4). The
    model is asked only when the node genuinely branches.
    """
    node = S.get_node(policy, node_id) or {}
    options = list(node.get("children", []))

    if not options:
        detail = {"outcome": "terminal", "condition": "", "justification": "Leaf node reached."}
        log.log(EV_TRAVERSAL, node_id=node_id, agent_role=ROLE_TRAVERSAL, detail=detail)
        return None, detail

    if len(options) == 1:
        target = options[0].get("target_id")
        detail = {
            "outcome": "deterministic",
            "condition": options[0].get("condition", ""),
            "target_id": target,
            "justification": "Single outgoing edge; no classification required.",
            "model_consulted": False,
        }
        log.log(EV_TRAVERSAL, node_id=node_id, agent_role=ROLE_TRAVERSAL, detail=detail)
        return target, detail

    labels = [str(option.get("condition", "")) for option in options]
    options_text = "\n".join(
        "- " + str(o.get("condition", "")) + " : " + str(o.get("description", "") or "(no description)")
        for o in options
    )
    system = P.ELICITOR_TRAVERSAL.render(
        node_id=node_id,
        question=node.get("question", ""),
        path=" -> ".join(path),
        options=options_text,
    )

    result = complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Transcript so far:\n\n" + transcript},
        ],
        provider=plan.control_provider,
        model=plan.control_model,
        json_schema=P.TRAVERSAL_SCHEMA if plan.control_provider != "openai" else None,
        want_json=True,
        max_tokens=1024,
        temperature=0.0 if plan.control_provider != "anthropic" else None,
        effort="low",
    )

    chosen = ""
    justification = ""
    confidence = 0.0
    if result.ok and isinstance(result.data, dict):
        chosen = str(result.data.get("condition", "") or "").strip()
        justification = str(result.data.get("justification", "") or "")
        confidence = _as_float(result.data.get("confidence"), 0.0)

    # Section 3.2.4: "Labels used at run time must match these condition values
    # exactly; otherwise the Elicitor will raise a traversal error." We raise it,
    # record it, and resolve to the catch-all rather than stalling a live session.
    traversal_error = chosen not in labels
    if traversal_error:
        recovered = _catch_all(options)
        detail = {
            "outcome": "traversal_error_recovered",
            "requested_condition": chosen,
            "condition": recovered.get("condition", ""),
            "target_id": recovered.get("target_id"),
            "justification": justification,
            "confidence": confidence,
            "valid_labels": labels,
            "model_consulted": True,
        }
        log.log(
            EV_TRAVERSAL,
            node_id=node_id,
            agent_role=ROLE_TRAVERSAL,
            llm=result,
            prompt_id=P.ELICITOR_TRAVERSAL.template_id,
            prompt_hash=P.ELICITOR_TRAVERSAL.hash,
            detail=detail,
            ok=False,
            error="Traversal error: %r is not a declared condition label." % chosen,
        )
        return recovered.get("target_id"), detail

    selected = next(o for o in options if str(o.get("condition", "")) == chosen)
    detail = {
        "outcome": "classified",
        "condition": chosen,
        "target_id": selected.get("target_id"),
        "justification": justification,
        "confidence": confidence,
        "valid_labels": labels,
        "model_consulted": True,
    }
    log.log(
        EV_TRAVERSAL,
        node_id=node_id,
        agent_role=ROLE_TRAVERSAL,
        llm=result,
        prompt_id=P.ELICITOR_TRAVERSAL.template_id,
        prompt_hash=P.ELICITOR_TRAVERSAL.hash,
        detail=detail,
    )
    return selected.get("target_id"), detail


def _catch_all(options: List[dict]) -> dict:
    """Prefer an explicit catch-all label, else the last declared branch."""
    for option in options:
        condition = str(option.get("condition", "")).lower()
        if any(token in condition for token in ("other", "unclear", "unknown", "none", "catch")):
            return option
    return options[-1]


# ------------------------------------------------------------------ summariser


def summarise(
    log: SessionLog,
    plan: ModelPlan,
    *,
    transcript: str,
    constructs: List[str],
) -> Tuple[List[Dict[str, str]], LLMResult]:
    """Per-construct summaries the participant then rates for accuracy.

    Agreement with these is the study's extraction-fidelity measure (Section 4.3.1).
    """
    system = P.ELICITOR_SUMMARY.render(
        constructs="\n".join("- " + c for c in constructs)
    )
    result = complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Transcript:\n\n" + transcript},
        ],
        provider=plan.control_provider,
        model=plan.control_model,
        json_schema=P.SUMMARY_SCHEMA if plan.control_provider != "openai" else None,
        want_json=True,
        max_tokens=1600,
        temperature=0.2 if plan.control_provider != "anthropic" else None,
    )

    answers: List[Dict[str, str]] = []
    if result.ok and isinstance(result.data, dict):
        for item in result.data.get("answers", []) or []:
            if isinstance(item, dict):
                answers.append(
                    {
                        "construct": str(item.get("construct", "")),
                        "summary": str(item.get("summary", "")),
                        "evidence_quote": str(item.get("evidence_quote", "")),
                    }
                )

    # Never show the participant a shorter list than the constructs we promised to
    # summarise -- a missing row would silently drop a fidelity item.
    covered = {a["construct"] for a in answers}
    for construct in constructs:
        if construct not in covered:
            answers.append(
                {
                    "construct": construct,
                    "summary": "Not covered in this conversation.",
                    "evidence_quote": "",
                }
            )

    log.log(
        EV_SUMMARY,
        agent_role=ROLE_SUMMARISER,
        llm=result,
        prompt_id=P.ELICITOR_SUMMARY.template_id,
        prompt_hash=P.ELICITOR_SUMMARY.hash,
        detail={
            "constructs": constructs,
            "answers": answers,
            # Section 6.2 asks for "a short cryptographic hash of the emitted summary"
            # so the artefact the participant rated can be verified later.
            "summary_hash": _hash_summaries(answers),
        },
    )
    return answers, result


def _hash_summaries(answers: List[Dict[str, str]]) -> str:
    import hashlib
    import json

    payload = json.dumps(answers, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# -------------------------------------------------------------- coverage coder


def code_coverage(
    log: SessionLog,
    plan: ModelPlan,
    *,
    transcript: str,
    topics: List[str],
) -> Dict[str, str]:
    """Post-hoc topic coverage, run identically on every arm.

    In the graph arms coverage is structurally guaranteed; in the free-form arms it
    is not. Coding both the same way is what makes coverage completeness usable as
    the fixed constraint the comparison is held against.
    """
    if not topics:
        return {}

    system = P.COVERAGE_CODER.render(topics="\n".join("- " + t for t in topics))
    result = complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Transcript:\n\n" + transcript},
        ],
        provider=plan.control_provider,
        model=plan.control_model,
        json_schema=P.COVERAGE_SCHEMA if plan.control_provider != "openai" else None,
        want_json=True,
        max_tokens=1500,
        temperature=0.0 if plan.control_provider != "anthropic" else None,
    )

    coded: Dict[str, str] = {}
    if result.ok and isinstance(result.data, dict):
        for item in result.data.get("coded", []) or []:
            if isinstance(item, dict) and item.get("topic"):
                coded[str(item["topic"])] = str(item.get("status", "not_covered"))

    log.log(
        EV_SUMMARY,
        agent_role=ROLE_CODER,
        llm=result,
        prompt_id=P.COVERAGE_CODER.template_id,
        prompt_hash=P.COVERAGE_CODER.hash,
        detail={"topics": topics, "coded": coded},
    )
    return coded


# ------------------------------------------------------------------- why hints


def why_hint(log: SessionLog, plan: ModelPlan, *, node: Dict[str, Any]) -> str:
    """Section 6.1's "brief explain-why hint that makes the governance goal legible"."""
    result = complete(
        [
            {
                "role": "user",
                "content": P.WHY_HINT.render(
                    question=node.get("question", ""), objective=node.get("objective", "")
                ),
            }
        ],
        provider=plan.control_provider,
        model=plan.control_model,
        max_tokens=512,
        temperature=0.3 if plan.control_provider != "anthropic" else None,
        effort="low",
    )
    text = (result.text or "").strip() or "We ask this so we can act on what visitors actually tell us."
    log.log(
        EV_WHY_HINT,
        node_id=node.get("id", ""),
        agent_role=ROLE_CHECKER,
        llm=result,
        prompt_id=P.WHY_HINT.template_id,
        prompt_hash=P.WHY_HINT.hash,
        detail={"hint": text},
    )
    return text


# ------------------------------------------------- analysis agent (K-Eng only)


def synthesise_policy(
    log: SessionLog,
    plan: ModelPlan,
    *,
    transcript: str,
    context: str,
    respondents: str,
    target_minutes: float,
    max_repairs: int = 2,
) -> Tuple[Optional[dict], S.ValidationReport, List[Dict[str, Any]]]:
    """Compile the expert transcript into a validated policy graph.

    The published pipeline generated the tree once and hoped it was executable. The
    contract is checkable, so here it is checked: generate, validate, and feed the
    concrete faults back for repair, up to `max_repairs` times. A graph that still
    fails is returned with its report rather than handed to a live session.
    """
    system = P.KENG_SYNTHESIS.text
    user = (
        "Event or activity: " + context + "\n"
        "Intended respondents: " + respondents + "\n"
        "Respondent time budget (minutes): " + str(target_minutes) + "\n\n"
        "Interview transcript with the expert:\n\n" + transcript
    )

    attempts: List[Dict[str, Any]] = []
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    policy: Optional[dict] = None
    report = S.ValidationReport()

    for attempt in range(max_repairs + 1):
        prompt_used = P.KENG_SYNTHESIS if attempt == 0 else P.KENG_REPAIR
        result = complete(
            messages,
            provider=plan.analysis_provider,
            model=plan.analysis_model,
            json_schema=P.POLICY_SCHEMA if plan.analysis_provider != "openai" else None,
            want_json=True,
            max_tokens=8000,
            temperature=0.2 if plan.analysis_provider != "anthropic" else None,
            effort="high",
            timeout=180.0,
        )

        if not result.ok or not isinstance(result.data, dict):
            report = S.ValidationReport()
            report.error("The analysis agent did not return a JSON object. " + (result.error or ""))
            log.log(
                EV_ERROR,
                agent_role=ROLE_ANALYST,
                llm=result,
                prompt_id=prompt_used.template_id,
                prompt_hash=prompt_used.hash,
                detail={"attempt": attempt + 1},
                ok=False,
                error=result.error or "No JSON returned.",
            )
            attempts.append({"attempt": attempt + 1, "ok": False, "errors": report.errors})
            break

        candidate = S.normalise_policy(result.data)
        candidate.setdefault("target_minutes", target_minutes)
        report = S.validate_policy(candidate)

        log.log(
            EV_SUMMARY,
            agent_role=ROLE_ANALYST,
            llm=result,
            prompt_id=prompt_used.template_id,
            prompt_hash=prompt_used.hash,
            detail={
                "attempt": attempt + 1,
                "valid": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "stats": report.stats,
                "policy_hash": S.policy_hash(candidate),
            },
            ok=report.ok,
            error="; ".join(report.errors),
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "ok": report.ok,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "stats": dict(report.stats),
            }
        )

        policy = candidate
        if report.ok:
            break
        if attempt == max_repairs:
            break

        import json as _json

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _json.dumps(result.data, ensure_ascii=False)},
            {
                "role": "user",
                "content": P.KENG_REPAIR.render(
                    errors="\n".join("- " + e for e in report.errors) or "- (none)",
                    warnings="\n".join("- " + w for w in report.warnings) or "- (none)",
                ),
            },
        ]

    return policy, report, attempts


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

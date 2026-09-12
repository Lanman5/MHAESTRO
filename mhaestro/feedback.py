"""The outcome battery: extraction fidelity, user experience, and workload.

Item wording for the experience block is taken from Part 2 of the approved
interview-questions document. Scoring follows Section 4.3.3 of the paper: all
rating items use a 1-5 agree scale, and items phrased so that agreement is a *bad*
outcome are reverse-coded as ``x' = 6 - x`` and reported aligned, so higher always
means better.

Three things here are deliberate departures from the published instrument.

*   **Repetitiveness and intrusiveness are separate items.** The published study
    collected them as one "repetitive/intrusive" item and then found the four UX
    facets did not form a scale (Cronbach's alpha = 0.54). They are different
    constructs -- a survey can be repetitive without feeling invasive -- and the
    2.45-point fidelity-fluidity gap rests entirely on that single item.
*   **Human-equivalence is asked twice**, once as an agree item and once as the
    document's Better/Same/Worse comparison with a free-text reason. It was the
    paper's weakest outcome (M=2.75) and the one the next study is meant to move,
    so it should not hang on one item.
*   **A NASA-TLX is added.** The paper argues the interaction strain it found is an
    accessibility and equity concern likely to fall hardest on people with high
    cognitive fatigue. Perceived workload is the standard instrument for that
    claim, and without it the argument stays inferential.

Nothing is pre-selected anywhere. A default of "neutral" on an unanswered item is
fabricated data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

AGREE_LABELS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


@dataclass(frozen=True)
class LikertItem:
    key: str
    text: str
    construct: str
    reverse: bool = False
    required: bool = True
    # Items are defined here whether or not they are currently asked. Turning one
    # off hides it but keeps its column in the session CSV, so the analysis table
    # has the same shape whichever configuration a session ran under.
    include: bool = True


# Part 2, "User Interface". Order follows the approved document.
#
# Four of these are the paper's baseline UX facets -- ease, visual appeal,
# coherence and (reverse-coded) non-intrusiveness -- and are never cut, because
# the whole point is to put the new numbers next to the published ones.
# `clarity` and `comfort` are off by default: clarity largely restates coherence,
# and comfort is a secondary construct. Set `include=True` to restore either.
EXPERIENCE_ITEMS: List[LikertItem] = [
    LikertItem("ease", "The program was easy to use.", "ease"),
    LikertItem("visual_appeal", "The interface was well designed and pleasant to look at.", "visual_appeal"),
    LikertItem("coherence", "The flow of the questions was coherent.", "coherence"),
    LikertItem("repetitive", "The questions felt repetitive.", "non_repetitiveness", reverse=True),
    LikertItem("intrusive", "The questions felt intrusive.", "non_intrusiveness", reverse=True),
    LikertItem(
        "clarity",
        "The questions were clear and easy to answer.",
        "clarity",
        include=False,
    ),
    LikertItem(
        "comfort",
        "I felt comfortable sharing information with the AI interviewer.",
        "comfort",
        include=False,
    ),
    LikertItem(
        "human_equivalence",
        "This tool was as effective as being interviewed by a person.",
        "human_equivalence",
    ),
    LikertItem(
        "agency",
        "I felt in control of how much I said and when we moved on.",
        "agency",
        required=False,
    ),
]


def active_experience_items() -> List[LikertItem]:
    return [item for item in EXPERIENCE_ITEMS if item.include]

HUMAN_COMPARISON_OPTIONS = ["Better", "About the same", "Worse"]

# Raw NASA-TLX: six subscales, unweighted mean, 0-100 each. Descriptors are
# reworded for a short typed conversation while keeping each subscale's meaning.
@dataclass(frozen=True)
class TLXItem:
    key: str
    title: str
    description: str
    low: str
    high: str
    reverse: bool = False


TLX_ITEMS: List[TLXItem] = [
    TLXItem(
        "mental_demand",
        "Mental demand",
        "How much thinking, remembering or deciding did this take?",
        "Very low",
        "Very high",
    ),
    TLXItem(
        "physical_demand",
        "Physical demand",
        "How physically demanding was it (typing, reading on screen)?",
        "Very low",
        "Very high",
    ),
    TLXItem(
        "temporal_demand",
        "Time pressure",
        "How hurried or rushed did the pace feel?",
        "Very low",
        "Very high",
    ),
    TLXItem(
        "performance",
        "Your performance",
        "How well do you think you got your views across?",
        "Perfectly",
        "Not at all",
        # TLX's performance scale runs good -> poor, so it is already aligned with
        # the other five (high = worse). Kept explicit so the scoring cannot drift.
    ),
    TLXItem(
        "effort",
        "Effort",
        "How hard did you have to work to answer the questions?",
        "Very low",
        "Very high",
    ),
    TLXItem(
        "frustration",
        "Frustration",
        "How irritated, stressed or annoyed did you feel?",
        "Very low",
        "Very high",
    ),
]

# Which subscales are actually shown. Sliders are the slowest widget on a phone,
# and two of the six earn little here: physical demand floors on a typed chat, and
# performance duplicates what the extraction-fidelity items already ask. Dropping
# them means this is a *reduced* TLX, not RTLX -- `tlx_variant` records which was
# administered so a write-up cannot claim the full instrument by accident. Set
# this to every key in TLX_ITEMS to restore RTLX.
DEFAULT_TLX_KEYS = ("mental_demand", "temporal_demand", "effort", "frustration")
RTLX_KEYS = tuple(item.key for item in TLX_ITEMS)


@dataclass(frozen=True)
class OpenItem:
    key: str
    text: str
    include: bool = True


# Free text is the most expensive thing on the page -- a engaged participant spends
# 30-60 seconds per box, and a rushed one abandons the whole battery at the sight
# of three. One is kept, and it is the one that produces actionable material.
OPEN_ITEMS: List[OpenItem] = [
    OpenItem("improve", "What would make the experience better?"),
    OpenItem("use_again", "What would make you more likely to use a tool like this again?", include=False),
    OpenItem("gained", "What, if anything, did you get out of taking part?", include=False),
]


def _align(value: Optional[int], reverse: bool) -> Optional[int]:
    """Reverse-code so that higher always means a better experience."""
    if value is None:
        return None
    return (6 - value) if reverse else value


# ------------------------------------------------------------------- rendering


def render_fidelity_block(summaries: List[Dict[str, str]], key_prefix: str = "fid") -> Dict[str, Any]:
    """Show each construct summary and ask whether it reflects what they said.

    Agreement here is the extraction-fidelity measure (Section 4.3.1): it asks
    whether the system captured the input faithfully, not whether the participant
    likes the topic.
    """
    st.subheader("Did we get this right?")
    st.caption(
        "Below is what the tool understood from your answers. For each one, tell us how "
        "accurately it reflects what you actually said."
    )

    responses: Dict[str, Any] = {}
    for index, item in enumerate(summaries):
        with st.container(border=True):
            st.markdown("**" + item.get("construct", "") + "**")
            st.write(item.get("summary", ""))
            score = st.segmented_control(
                "Accuracy",
                options=[1, 2, 3, 4, 5],
                format_func=lambda v: AGREE_LABELS[v - 1],
                key=key_prefix + "_" + str(index),
                label_visibility="collapsed",
                default=None,
            )
            responses["item_%d" % (index + 1)] = {
                "construct": item.get("construct", ""),
                "summary": item.get("summary", ""),
                "score": score,
            }
    return responses


def render_experience_block(key_prefix: str = "ux") -> Dict[str, Any]:
    """Part 2 of the approved question set."""
    st.subheader("How was it to use?")
    st.caption("There are no right answers. Critical feedback is more useful to us than polite feedback.")

    responses: Dict[str, Any] = {}
    for item in active_experience_items():
        st.markdown("**" + item.text + "**")
        responses[item.key] = st.segmented_control(
            item.text,
            options=[1, 2, 3, 4, 5],
            format_func=lambda v: AGREE_LABELS[v - 1],
            key=key_prefix + "_" + item.key,
            label_visibility="collapsed",
            default=None,
        )

    st.markdown("**How did the AI interviewer compare to what you would expect from a person?**")
    comparison = st.segmented_control(
        "Comparison",
        options=HUMAN_COMPARISON_OPTIONS,
        key=key_prefix + "_human_comparison",
        label_visibility="collapsed",
        default=None,
    )
    responses["human_comparison"] = comparison

    # The free-text follow-up appears only when there is something to explain.
    # "About the same" needs no reason, and an always-visible box is one more thing
    # to scroll past for everyone who was never going to fill it in.
    if comparison in ("Better", "Worse"):
        responses["human_comparison_why"] = st.text_input(
            "Briefly, what made it " + comparison.lower() + "?",
            key=key_prefix + "_human_comparison_why",
            placeholder="Optional",
        )
    else:
        responses["human_comparison_why"] = ""
    return responses


def render_workload_block(key_prefix: str = "tlx", keys=DEFAULT_TLX_KEYS) -> Dict[str, Any]:
    """NASA-TLX, unweighted. `keys` selects which subscales are administered."""
    st.subheader("How much did it take out of you?")
    st.caption("Drag each slider to wherever feels right. There is no correct position.")

    responses: Dict[str, Any] = {}
    for item in TLX_ITEMS:
        if item.key not in keys:
            continue
        st.markdown("**" + item.title + "**")
        st.caption(item.description)
        responses[item.key] = st.slider(
            item.title,
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key=key_prefix + "_" + item.key,
            label_visibility="collapsed",
        )
        left, right = st.columns(2)
        left.caption(item.low)
        right.markdown(
            '<div style="text-align:right"><span class="mh-muted">' + item.high + "</span></div>",
            unsafe_allow_html=True,
        )
    return responses


def render_open_block(key_prefix: str = "open") -> Dict[str, Any]:
    """The document's open-ended items. All optional -- these are the first thing
    a rushed participant abandons, and a forced text box causes break-off."""
    st.subheader("Anything else?")
    responses: Dict[str, Any] = {}
    for item in OPEN_ITEMS:
        if not item.include:
            responses[item.key] = ""
            continue
        responses[item.key] = st.text_area(
            item.text,
            key=key_prefix + "_" + item.key,
            placeholder="Optional",
            height=80,
        )
    return responses


# --------------------------------------------------------------------- scoring


def score_battery(
    fidelity: Dict[str, Any],
    experience: Dict[str, Any],
    workload: Dict[str, Any],
    open_text: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten and score, reproducing the paper's headline statistics per session."""
    scored: Dict[str, Any] = {}

    # --- Extraction fidelity
    fidelity_scores = []
    for key, item in fidelity.items():
        scored["fidelity_" + key + "_construct"] = item.get("construct", "")
        scored["fidelity_" + key + "_score"] = item.get("score")
        scored["fidelity_" + key + "_summary"] = item.get("summary", "")
        if item.get("score") is not None:
            fidelity_scores.append(item["score"])
    scored["fidelity_n_items"] = len(fidelity_scores)
    scored["fidelity_mean"] = _mean(fidelity_scores)
    # "75% agreed the summary accurately captured their input" -- the paper's
    # headline is the proportion of items rated 4 or 5, so it is computed the same way.
    scored["fidelity_agreement_rate"] = (
        round(sum(1 for s in fidelity_scores if s >= 4) / len(fidelity_scores), 4)
        if fidelity_scores
        else ""
    )

    # --- Experience, raw and aligned
    aligned: Dict[str, Optional[int]] = {}
    for item in EXPERIENCE_ITEMS:
        raw = experience.get(item.key)
        scored["ux_" + item.key + "_raw"] = raw
        aligned_value = _align(raw, item.reverse)
        scored["ux_" + item.construct + "_aligned"] = aligned_value
        aligned[item.construct] = aligned_value

    scored["ux_human_comparison"] = experience.get("human_comparison")
    scored["ux_human_comparison_why"] = experience.get("human_comparison_why", "")

    ux_facets = [
        aligned.get("ease"),
        aligned.get("visual_appeal"),
        aligned.get("coherence"),
        aligned.get("non_intrusiveness"),
    ]
    scored["ux_aligned_mean"] = _mean([v for v in ux_facets if v is not None])

    # --- The paper's central quantity, computed per session
    fidelity_mean = scored["fidelity_mean"]
    non_intrusiveness = aligned.get("non_intrusiveness")
    scored["fidelity_fluidity_gap"] = (
        round(fidelity_mean - non_intrusiveness, 3)
        if isinstance(fidelity_mean, float) and non_intrusiveness is not None
        else ""
    )

    # --- Workload
    tlx_values = []
    for item in TLX_ITEMS:
        value = workload.get(item.key)
        scored["tlx_" + item.key] = value
        if value is not None:
            tlx_values.append(value)
    scored["tlx_raw_mean"] = _mean(tlx_values)
    scored["tlx_n_items"] = len(tlx_values)
    administered = tuple(item.key for item in TLX_ITEMS if workload.get(item.key) is not None)
    scored["tlx_variant"] = "rtlx_6item" if set(administered) == set(RTLX_KEYS) else (
        "reduced_%ditem" % len(administered)
    )
    scored["tlx_subscales"] = ",".join(administered)

    # --- Open text. Every defined item gets a column whether or not it was shown,
    # so a configuration change mid-study does not change the table's shape.
    for item in OPEN_ITEMS:
        scored["open_" + item.key] = open_text.get(item.key, "")

    # Which battery configuration produced this row.
    scored["battery_experience_items"] = ",".join(i.key for i in active_experience_items())
    scored["battery_n_rated_items"] = (
        scored["fidelity_n_items"] + len(active_experience_items()) + 1 + len(administered)
    )

    return scored


def unanswered_required(fidelity: Dict[str, Any], experience: Dict[str, Any]) -> List[str]:
    """Which required items are still blank, so the participant can be told once."""
    missing = []
    for key, item in fidelity.items():
        if item.get("score") is None:
            missing.append("Accuracy of: " + str(item.get("construct", key)))
    for item in active_experience_items():
        if item.required and experience.get(item.key) is None:
            missing.append(item.text)
    return missing


def _mean(values: List[Any]) -> Any:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return ""
    return round(sum(numeric) / len(numeric), 3)

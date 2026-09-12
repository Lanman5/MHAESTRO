"""Participant information, age verification and informed consent.

The wording below is reproduced from the approved ethics documents
(`ParticipationInformationSection.docx` and `Consent Section.docx`). Only the
bracketed placeholders the approved text marks for substitution are filled in --
the conversation length and the use-case description. Nothing else is
paraphrased, reordered or abridged, because the approved wording is what
participants were granted consent against.

Three outcomes are possible, and they are enforced in this order:

    consented  18 or over, information read, consent box ticked -> the session
               runs with full data collection.
    explore    Under 18 -> the tool is usable so nobody is turned away at a public
               event, but **no data whatsoever is recorded or transmitted by us**.
               The participant is told plainly that their words still reach a
               third-party AI provider, because that is true regardless of whether
               we keep anything.
    declined   Consent withheld -> nothing runs and nothing is recorded.

The age gate precedes the consent form deliberately: an under-18 visitor is never
shown a consent form they cannot validly give.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

PIS_VERSION = "lboro-pis-2026-01"
CONSENT_VERSION = "lboro-consent-2026-01"

STATUS_PENDING = "pending"
STATUS_CONSENTED = "consented"
STATUS_EXPLORE = "explore"
STATUS_DECLINED = "declined"


@dataclass
class ConsentRecord:
    status: str = STATUS_PENDING
    over_18: Optional[bool] = None
    consent_given: bool = False
    pis_version: str = PIS_VERSION
    consent_version: str = CONSENT_VERSION
    recorded_utc: str = ""

    @property
    def may_collect_data(self) -> bool:
        """The single gate every write in the app must pass through."""
        return self.status == STATUS_CONSENTED and self.over_18 is True and self.consent_given

    @property
    def may_use_tool(self) -> bool:
        return self.status in (STATUS_CONSENTED, STATUS_EXPLORE)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "over_18": self.over_18,
            "consent_given": self.consent_given,
            "pis_version": self.pis_version,
            "consent_version": self.consent_version,
            "recorded_utc": self.recorded_utc,
        }


# --------------------------------------------------------------- approved text

CONTACTS_HTML = """
<div class="mh-contact">
Alan Naky (Student Investigator) &mdash; A.Naky@lboro.ac.uk<br>
Dr. Yanning Yang (Responsible Investigator) &mdash; Y.Yang2@lboro.ac.uk &middot; 01509 228055<br>
Dr. Sara Saravi (Staff Investigator) &mdash; S.Saravi@lboro.ac.uk &middot; 01509 223987<br>
Dr. Firat Batmaz (Staff Investigator) &mdash; F.Batmaz@lboro.ac.uk &middot; 01509 222699<br>
School of Science, Loughborough University, Loughborough, Leicestershire, LE11 3TU
</div>
"""


def information_sheet_markdown(use_case: str, survey_minutes: int, feedback_minutes: int) -> str:
    """The approved information sheet with its two marked placeholders filled."""
    return f"""
Thank you for considering taking part in this study.

Before you decide we would like you to understand why the research is being done
and what it would involve for you. Please contact one of the investigators using
the contact details below if you have any questions.
"""


def information_sheet_body(use_case: str, survey_minutes: int, feedback_minutes: int) -> str:
    return f"""
<div class="mh-legal">
<p>The purpose of this study is to evaluate the AI-driven knowledge elicitation tool
developed by the student investigator during a Talent Match summer internship
project, and understand how effectively and appropriately it can be used to gather
people&rsquo;s knowledge, experiences, perspectives, reflections and feedback in
different contexts.</p>

<p><strong>You must be over the age of 18 and have the capacity to fully understand
and consent to this research.</strong></p>

<p>You will be asked to interact with the knowledge elicitation tool through a
conversational, interview-style exchange, in which the tool will ask you questions
about an activity or experience you just had or a particular topic that you just
learned, and may ask follow-up questions based on your responses. The conversation
will be no longer than <strong>{survey_minutes} minutes</strong>
({use_case}). At the end of the conversation, the tool will generate summaries of
your views, you will be asked to agree or disagree with these summaries, allowing us
to assess how accurately the system has represented participants&rsquo; perspectives.
You will also be asked to provide feedback on your experience of using the tool
itself, this part will be no longer than <strong>{feedback_minutes} minutes</strong>.
Your interaction with the tool will be transcribed and analysed for common themes,
meta data regarding the interaction (e.g. time taken) will also be collected for
performance and quality analysis and enhancement of the tool. Your responses will be
sent to an LLM, please do not include any personal or sensitive information in your
responses.</p>

<p>This is a low-risk activity, and no disadvantages or risks have been identified in
association with participating.</p>

<p>Loughborough University is responsible for this research and for looking after
your information. No identifiable information will be collected and so your
participation in the study will be confidential. No individual will be identifiable
in any report, presentation or publication. All information collected will be
securely stored on the University&rsquo;s computer systems. We will keep your study
data for a maximum of three years. The study data will then be securely archived or
destroyed.</p>

<p>After you have read this information and asked any questions you may have, if you
are happy to participate, please read the consent page and confirm your consent by
checking the tick box at the bottom of the page. You can withdraw at any time by
closing the browser. However, as the participation is anonymous once you have
submitted the feedback it will not be possible to withdraw your data from the
study.</p>

<p><strong>What if I am not happy with how the research was conducted?</strong><br>
If you are not happy with how the research was conducted, please contact the
Secretary of the Ethics Review Sub-Committee, Research &amp; Innovation Office,
Hazlerigg Building, Loughborough University, Epinal Way, Loughborough, LE11 3TU.
Tel: 01509 222423. Email: researchpolicy@lboro.ac.uk</p>

<p>The University also has policies relating to Research Misconduct and Whistle
Blowing which are available online at
<a href="https://www.lboro.ac.uk/internal/research-ethics-integrity/research-integrity/">
https://www.lboro.ac.uk/internal/research-ethics-integrity/research-integrity/</a>.</p>
</div>
"""


CONSENT_STATEMENTS = [
    "The purpose and details of this study have been explained to me.",
    "I understand that this study is designed to further scientific knowledge and that all "
    "procedures have received a favourable decision from the Loughborough University Ethics "
    "Review Sub-Committee.",
    "I have read and understood the information sheet and this consent form.",
    "I have had an opportunity to ask questions about my participation.",
    "I understand that taking part in this study is anonymous.",
    "I understand that I am under no obligation to take part in the study and can withdraw by "
    "closing the browser but will not be able to withdraw once my responses have been submitted.",
    "I understand that information I provide will be used for research publication.",
]

CONSENT_TICKBOX = "I voluntarily agree to take part in this study."


# --------------------------------------------------------------------- the gate


def consent_gate(
    *,
    study_title: str,
    use_case: str,
    survey_minutes: int = 5,
    feedback_minutes: int = 5,
    state_key: str = "consent_record",
) -> ConsentRecord:
    """Render the information / age / consent flow and return the current record.

    Call this before anything else in the app. While the returned record's
    `may_use_tool` is False, the caller must render nothing else and stop.
    """
    record: ConsentRecord = st.session_state.setdefault(state_key, ConsentRecord())
    if record.status != STATUS_PENDING:
        return record

    st.markdown('<div class="mh-eyebrow">Loughborough University &middot; School of Science</div>', unsafe_allow_html=True)
    st.title("Information Section")
    st.subheader(study_title)

    st.markdown(information_sheet_markdown(use_case, survey_minutes, feedback_minutes))
    st.markdown(CONTACTS_HTML, unsafe_allow_html=True)
    st.markdown(information_sheet_body(use_case, survey_minutes, feedback_minutes), unsafe_allow_html=True)

    st.divider()

    # ---- Age gate. Asked before the consent form, never after.
    st.subheader("Before you continue")
    age_choice = st.radio(
        "Please confirm your age.",
        options=["I am 18 years old or over", "I am under 18"],
        index=None,
        key="consent_age_choice",
    )

    if age_choice is None:
        st.stop()

    if age_choice == "I am under 18":
        _render_under_18(record, state_key)
        st.stop()

    # ---- Informed consent, for over-18s only.
    st.divider()
    st.subheader("Informed Consent Section")
    st.markdown(
        '<div class="mh-legal"><ul>'
        + "".join("<li>" + statement + "</li>" for statement in CONSENT_STATEMENTS)
        + "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Consent to Participate**")
    agreed = st.checkbox(CONSENT_TICKBOX, key="consent_tickbox")

    left, right = st.columns([1, 1])
    if left.button("Begin", type="primary", width="stretch", disabled=not agreed):
        record.status = STATUS_CONSENTED
        record.over_18 = True
        record.consent_given = True
        record.recorded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        st.session_state[state_key] = record
        st.rerun()

    if right.button("I do not wish to take part", width="stretch"):
        record.status = STATUS_DECLINED
        record.over_18 = True
        record.consent_given = False
        record.recorded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        st.session_state[state_key] = record
        st.rerun()

    if not agreed:
        st.caption("Tick the box above to continue.")

    st.stop()
    return record


def _render_under_18(record: ConsentRecord, state_key: str) -> None:
    """Offer the tool without the study: usable, but nothing is collected."""
    st.info(
        "**You can still try the tool, but you will not be taking part in the study.**\n\n"
        "This research is approved for adults only, so if you are under 18 we will not "
        "collect, store or analyse anything from your session. Nothing you type is saved "
        "by us, nothing is sent to the research team, and your session forms no part of "
        "the results."
    )
    st.warning(
        "**One thing you should know before you start.**\n\n"
        "The tool works by sending what you type to an external AI provider (such as "
        "OpenAI, Anthropic or Google) so it can write its reply. That company receives "
        "your messages and may keep them on its own systems under its own terms, and we "
        "cannot delete them on your behalf. That happens whether or not we keep anything "
        "ourselves.\n\n"
        "**Please do not type your name, your school, your contact details, or anything "
        "personal or private.**"
    )

    understood = st.checkbox(
        "I understand that my session is not part of the study, that nothing will be "
        "collected by the research team, and that what I type is sent to an external AI "
        "provider.",
        key="under18_ack",
    )
    if st.button("Try the tool", type="primary", disabled=not understood, width="stretch"):
        record.status = STATUS_EXPLORE
        record.over_18 = False
        record.consent_given = False
        record.recorded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        st.session_state[state_key] = record
        st.rerun()


def render_declined() -> None:
    st.title("Thank you")
    st.write(
        "No information has been collected. You can close this page. If you change your "
        "mind, reopen the link and you will be asked again."
    )


def explore_banner() -> None:
    """Persistent reminder, shown on every screen of an under-18 session."""
    st.warning(
        "**Practice mode -- nothing is being collected.** You are not taking part in the "
        "study. Remember that what you type is still sent to an external AI provider, so "
        "please do not include anything personal.",
        icon="⚠️",
    )

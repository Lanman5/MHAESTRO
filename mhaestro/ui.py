"""Shared presentation layer.

The published prototype rendered its chat by concatenating escaped HTML into a
`components.html` iframe with a fixed 400px height. That is why the interface
scored only moderately on visual appeal (M=3.50): it does not resize, does not
respect the reader's theme, is not reachable by a screen reader, and scrolls
independently of the page. Section 6.3 treats accessibility as part of validity,
not decoration, so the chat here is built from native `st.chat_message` elements
and the type scale is set once, in one place.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import streamlit as st

BRAND_CSS = """
<style>
  .block-container {max-width: 50rem; padding-top: 2.2rem; padding-bottom: 4rem;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  h1 {font-size: 1.85rem !important; font-weight: 650;}
  h2 {font-size: 1.3rem !important; font-weight: 620;}
  h3 {font-size: 1.05rem !important; font-weight: 600;}
  [data-testid="stChatMessage"] {padding: 0.55rem 0.2rem;}
  .mh-eyebrow {
    text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem;
    font-weight: 650; opacity: 0.62; margin-bottom: 0.15rem;
  }
  .mh-legal {font-size: 0.92rem; line-height: 1.62;}
  .mh-legal p {margin-bottom: 0.75rem;}
  .mh-legal ul {margin-top: 0.2rem;}
  .mh-contact {
    font-size: 0.88rem; line-height: 1.55; opacity: 0.88;
    border-left: 3px solid rgba(128,128,128,0.35); padding-left: 0.8rem; margin: 0.5rem 0 1rem;
  }
  .mh-muted {font-size: 0.85rem; opacity: 0.68;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
"""


def page_setup(title: str, icon: str, *, wide: bool = False) -> None:
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide" if wide else "centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown('<div class="mh-eyebrow">' + eyebrow + "</div>", unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_chat(
    messages: Iterable[Dict[str, str]],
    *,
    interviewer_avatar: str = "🎙️",
    participant_avatar: str = "🙂",
) -> None:
    """Render the visible transcript. Hidden control messages are never passed here."""
    for message in messages:
        if message.get("role") == "assistant":
            with st.chat_message("assistant", avatar=interviewer_avatar):
                st.markdown(message.get("content", ""))
        elif message.get("role") == "user":
            with st.chat_message("user", avatar=participant_avatar):
                st.markdown(message.get("content", ""))


def progress_bar(fraction: float, label: str) -> None:
    st.progress(max(0.0, min(1.0, fraction)), text=label)


def likert(
    key: str,
    question: str,
    *,
    labels: Optional[List[str]] = None,
    help_text: str = "",
) -> Optional[int]:
    """One 1-5 agree item, rendered as segmented control.

    All rating items in the study use the same 1-5 agree scale with the anchors
    visible, matching Section 4.3.3. Nothing is pre-selected: a default of 3 would
    silently record neutral agreement for anyone who skipped the item.
    """
    labels = labels or ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    st.markdown("**" + question + "**")
    if help_text:
        st.caption(help_text)
    choice = st.segmented_control(
        question,
        options=[1, 2, 3, 4, 5],
        format_func=lambda v: labels[v - 1],
        key=key,
        label_visibility="collapsed",
        default=None,
    )
    return choice


def scale_item(
    key: str,
    question: str,
    *,
    low: str,
    high: str,
    steps: int = 21,
    help_text: str = "",
) -> int:
    """A 0-100 magnitude item with named endpoints (the NASA-TLX response format)."""
    st.markdown("**" + question + "**")
    if help_text:
        st.caption(help_text)
    value = st.slider(
        question,
        min_value=0,
        max_value=100,
        value=50,
        step=int(100 / (steps - 1)) or 5,
        key=key,
        label_visibility="collapsed",
    )
    left, right = st.columns(2)
    left.markdown('<span class="mh-muted">' + low + "</span>", unsafe_allow_html=True)
    right.markdown(
        '<div style="text-align:right"><span class="mh-muted">' + high + "</span></div>",
        unsafe_allow_html=True,
    )
    return value


def status_row(items: Dict[str, str]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items.items()):
        column.metric(label, value)

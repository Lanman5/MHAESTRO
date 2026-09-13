"""Append-only audit log and session record.

The paper names its own biggest gap (Sections 4.5, 6.2, 6.7): "the current sheet
records question text but not node identifiers, timestamps, or prompt-template
versions. As a result, Results can verify coverage rather than exact sequence."
and "Extending the log schema with node IDs, timestamps, prompt/model versions,
and a short hash of each summary will enable direct sequence and provenance
auditing."

This module is that extended schema. Every agent call, every participant turn and
every governance decision becomes one immutable row carrying who acted, on which
node, with which model and prompt version, when, how long it took, and why. Two
views are exported: a long `events` table (one row per event, for sequence and
provenance auditing) and a wide one-row `session` record (for pooled analysis
across participants).

Nothing here stores a name, an email address or free text the participant did not
type into the survey itself -- participation is anonymous, so the join key is a
generated pseudonym.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

APP_VERSION = "mhaestro-2.0"

# ----------------------------------------------------------------- event types

EV_SESSION_START = "session_start"
EV_CONSENT = "consent_recorded"
EV_ARM_ASSIGNED = "arm_assigned"
EV_POLICY_LOADED = "policy_loaded"
EV_QUESTION = "agent_a_question"
EV_REPLY = "participant_reply"
EV_ADEQUACY = "agent_b_adequacy"
EV_TRAVERSAL = "agent_c_traversal"
EV_REPROBE = "reprobe_issued"
EV_SOFT_CAP = "soft_cap_reached"
EV_SKIP = "participant_skip"
EV_WHY_HINT = "why_hint_shown"
EV_ADVANCE = "node_advance"
EV_SAFEGUARD = "safeguard_flag"
EV_SUMMARY = "summary_generated"
EV_RATING = "rating_submitted"
EV_SESSION_END = "session_end"
EV_DELIVERY = "delivery_attempt"
EV_ERROR = "error"

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENTENCE = re.compile(r"[.!?]+")

# The canonical per-turn column set, written for every arm whether or not that arm
# populates each one. Keeping it fixed is what lets turn files be concatenated
# across participants who were in different arms.
TURN_COLUMNS: List[str] = [
    "session_id",
    "pseudonym",
    "arm_id",
    "kind",
    "turn_index",
    "t_elapsed_s",
    # Graph arms only.
    "node_id",
    "node_topic",
    "node_priority",
    # The re-probe dose: 0 on a first attempt at a node, 1+ on a re-probe.
    "reprobe_index",
    "outcome",
    # Participant behaviour.
    "reply_latency_s",
    "input_mode",
    "asr_provider",
    "asr_model",
    "asr_fell_back",
    "asr_attempts",
    "reply_chars",
    "reply_words",
    "reply_sentences",
    "reply_unique_words",
    "reply_type_token_ratio",
    "reply_mean_word_length",
    "reply_mean_sentence_words",
    # Adequacy arms only.
    "adequacy_checked",
    "adequate",
    "adequacy_on_topic",
    "adequacy_wants_to_move_on",
    "adequacy_missing",
    "adequacy_confidence",
    "adequacy_soft_capped",
    "adequacy_fallback_used",
    "adequacy_reason",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_pseudonym() -> str:
    """Anonymous join key. Never derived from anything the participant typed."""
    return "P-" + uuid.uuid4().hex[:10].upper()


def text_metrics(text: str) -> Dict[str, Any]:
    """Response-elaboration measures.

    Length and lexical variety are the standard proxies for engagement and
    break-off risk in survey methodology; collecting them per turn lets the
    analysis test whether governed re-probing shortens later answers (the
    disengagement signature the paper's intrusiveness finding predicts).
    """
    text = text or ""
    words = _WORD.findall(text.lower())
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    unique = len(set(words))
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "unique_words": unique,
        "type_token_ratio": round(unique / len(words), 4) if words else 0.0,
        "mean_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
        "mean_sentence_words": round(len(words) / len(sentences), 2) if sentences else float(len(words)),
    }


@dataclass
class SessionLog:
    """One session's complete, append-only record."""

    tool: str = "elicitor"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pseudonym: str = field(default_factory=new_pseudonym)
    started_iso: str = field(default_factory=utc_now_iso)
    started_monotonic: float = field(default_factory=time.perf_counter)
    meta: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    ratings: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Dict[str, str]] = field(default_factory=list)
    _seq: int = 0

    # ------------------------------------------------------------------ writing

    def elapsed(self) -> float:
        return round(time.perf_counter() - self.started_monotonic, 3)

    def set_meta(self, **values: Any) -> None:
        self.meta.update(values)

    def log(
        self,
        event_type: str,
        *,
        node_id: str = "",
        agent_role: str = "",
        detail: Optional[Dict[str, Any]] = None,
        llm: Any = None,
        prompt_id: str = "",
        prompt_hash: str = "",
        ok: bool = True,
        error: str = "",
    ) -> Dict[str, Any]:
        """Append one immutable event row and return it."""
        self._seq += 1
        row: Dict[str, Any] = {
            "seq": self._seq,
            "session_id": self.session_id,
            "pseudonym": self.pseudonym,
            "tool": self.tool,
            "ts_utc": utc_now_iso(),
            "t_elapsed_s": self.elapsed(),
            "event_type": event_type,
            "node_id": node_id,
            "agent_role": agent_role,
            "prompt_template_id": prompt_id,
            "prompt_template_hash": prompt_hash,
            "provider": "",
            "model": "",
            "latency_ms": "",
            "input_tokens": "",
            "output_tokens": "",
            "llm_attempts": "",
            "llm_repaired_json": "",
            "llm_degraded": "",
            "ok": ok,
            "error": error,
            "detail_json": json.dumps(detail or {}, ensure_ascii=False, sort_keys=True),
        }
        if llm is not None:
            row.update(
                {
                    "provider": getattr(llm, "provider", ""),
                    "model": getattr(llm, "model", ""),
                    "latency_ms": getattr(llm, "latency_ms", ""),
                    "input_tokens": getattr(llm, "input_tokens", "") or "",
                    "output_tokens": getattr(llm, "output_tokens", "") or "",
                    "llm_attempts": getattr(llm, "attempts", ""),
                    "llm_repaired_json": getattr(llm, "repaired_json", ""),
                    # Request features the provider rejected for this model and
                    # which were therefore dropped. Blank is the normal case; a
                    # value here means this turn ran on a weaker request than
                    # intended, which the analysis needs to be able to see.
                    "llm_degraded": ",".join(getattr(llm, "degraded", []) or []),
                    # A row is a failure if *either* the transport failed or the
                    # caller judged the reply unusable. A traversal agent that
                    # returns an undeclared label has made a successful API call
                    # and an invalid decision; the caller's verdict must not be
                    # overwritten by the SDK's.
                    "ok": bool(ok) and bool(getattr(llm, "ok", True)),
                    "error": " | ".join(
                        part for part in (error, getattr(llm, "error", "") or "") if part
                    ),
                }
            )
        self.events.append(row)
        return row

    def add_turn(self, turn: Dict[str, Any]) -> None:
        self.turns.append(turn)

    def add_message(self, role: str, content: str) -> None:
        """Transcript as shown to the participant. Hidden control messages never appear here."""
        self.transcript.append({"role": role, "content": content, "ts_utc": utc_now_iso()})

    # ------------------------------------------------------------------ reading

    def transcript_text(self, interviewer: str = "Interviewer", participant: str = "Participant") -> str:
        lines = []
        for msg in self.transcript:
            who = interviewer if msg["role"] == "assistant" else participant
            lines.append(who + ": " + msg["content"])
        return "\n\n".join(lines)

    def events_csv(self) -> str:
        if not self.events:
            return ""
        fieldnames = list(self.events[0].keys())
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.events)
        return buffer.getvalue()

    def turns_csv(self) -> str:
        """One row per participant turn, with a column set that does not vary by arm.

        The arms record different things -- only the adequacy arms have a checker
        verdict, only the graph arms have a node -- so emitting just the keys that
        happen to be present would produce a differently-shaped file per arm, and
        concatenating them across participants would silently misalign. The
        canonical columns are always written, blank where they do not apply.
        """
        if not self.turns:
            return ""
        fieldnames: List[str] = list(TURN_COLUMNS)
        for turn in self.turns:
            for key in turn:
                if key not in fieldnames:
                    fieldnames.append(key)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.turns)
        return buffer.getvalue()

    # ------------------------------------------------------- derived aggregates

    def derived(self) -> Dict[str, Any]:
        """Session-level measures computed from the event and turn logs."""
        replies = [t for t in self.turns if t.get("kind") == "reply"]
        reply_words = [t.get("reply_words", 0) for t in replies]
        latencies = [t.get("reply_latency_s", 0.0) for t in replies if t.get("reply_latency_s")]
        agent_latencies = [
            e["latency_ms"] for e in self.events if isinstance(e.get("latency_ms"), int)
        ]

        reprobes = sum(1 for t in replies if t.get("reprobe_index", 0) > 0)
        adequacy_events = [e for e in self.events if e["event_type"] == EV_ADEQUACY]
        adequate_true = sum(
            1 for e in adequacy_events if json.loads(e["detail_json"] or "{}").get("adequate") is True
        )

        derived: Dict[str, Any] = {
            "duration_s": self.elapsed(),
            "n_participant_turns": len(replies),
            "n_events": len(self.events),
            "n_reprobes": reprobes,
            "reprobe_rate": round(reprobes / len(replies), 4) if replies else 0.0,
            "n_adequacy_checks": len(adequacy_events),
            "adequacy_pass_rate": round(adequate_true / len(adequacy_events), 4) if adequacy_events else "",
            "n_soft_cap_hits": sum(1 for e in self.events if e["event_type"] == EV_SOFT_CAP),
            "n_skips": sum(1 for e in self.events if e["event_type"] == EV_SKIP),
            "n_why_hints": sum(1 for e in self.events if e["event_type"] == EV_WHY_HINT),
            "n_safeguard_flags": sum(1 for e in self.events if e["event_type"] == EV_SAFEGUARD),
            "n_agent_errors": sum(1 for e in self.events if not e.get("ok", True)),
            # Questions the participant saw in the policy's own wording because
            # Agent [a] could not be reached. Non-zero means this session is a
            # partially scripted interview and should be reported as such, or
            # excluded -- the phrasing is part of what the tool does.
            "n_scripted_questions": sum(
                1
                for e in self.events
                if e["event_type"] == EV_QUESTION
                and json.loads(e["detail_json"] or "{}").get("scripted_fallback") is True
            ),
            "total_input_tokens": sum(e["input_tokens"] for e in self.events if isinstance(e.get("input_tokens"), int)),
            "total_output_tokens": sum(e["output_tokens"] for e in self.events if isinstance(e.get("output_tokens"), int)),
            "reply_words_total": sum(reply_words),
            "reply_words_mean": round(statistics.mean(reply_words), 2) if reply_words else 0.0,
            "reply_words_median": round(statistics.median(reply_words), 2) if reply_words else 0.0,
            "reply_latency_mean_s": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "reply_latency_median_s": round(statistics.median(latencies), 2) if latencies else 0.0,
            "agent_latency_mean_ms": round(statistics.mean(agent_latencies), 1) if agent_latencies else 0.0,
            "agent_latency_total_ms": sum(agent_latencies),
        }

        # Elaboration trend: does the participant write less as the session wears on?
        # A negative slope alongside high re-probing is the disengagement signature
        # the fidelity-fluidity trade-off predicts.
        derived["reply_words_slope"] = _slope(reply_words)
        derived["reply_words_first"] = reply_words[0] if reply_words else ""
        derived["reply_words_last"] = reply_words[-1] if reply_words else ""
        return derived

    def session_row(self) -> Dict[str, Any]:
        """The wide, one-row-per-session record. Concatenate these for analysis."""
        row: Dict[str, Any] = {
            "session_id": self.session_id,
            "pseudonym": self.pseudonym,
            "tool": self.tool,
            "app_version": APP_VERSION,
            "started_utc": self.started_iso,
            "ended_utc": utc_now_iso(),
        }
        for key, value in self.meta.items():
            row["meta_" + key] = _flatten_value(value)
        for key, value in self.derived().items():
            row[key] = value
        # Outcome keys are already namespaced by `feedback.score_battery`
        # (fidelity_*, ux_*, tlx_*, open_*) and cannot collide with the derived or
        # meta_* columns, so they are written under their own names. An extra
        # prefix here would only make every analysis script rename them back.
        for key, value in self.ratings.items():
            row[key] = _flatten_value(value)
        return row

    def session_csv(self) -> str:
        row = self.session_row()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
        return buffer.getvalue()

    def bundle_json(self) -> str:
        """Everything, losslessly -- the archival artefact behind the two CSVs."""
        return json.dumps(
            {
                "session_id": self.session_id,
                "pseudonym": self.pseudonym,
                "tool": self.tool,
                "app_version": APP_VERSION,
                "started_utc": self.started_iso,
                "ended_utc": utc_now_iso(),
                "meta": self.meta,
                "derived": self.derived(),
                "ratings": self.ratings,
                "turns": self.turns,
                "events": self.events,
                "transcript": self.transcript,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def _flatten_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _slope(values: List[float]) -> float:
    """Ordinary least-squares slope against turn index; 0.0 when undefined."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0 or math.isclose(denominator, 0.0):
        return 0.0
    numerator = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    return round(numerator / denominator, 4)

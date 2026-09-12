"""Getting a finished session out of the browser and into the dataset.

Streamlit Community Cloud has no durable filesystem -- anything written to disk
can vanish on the next restart -- so the session record has to leave the process
the moment it is complete. Email is the transport, with three properties that
matter when a hundred people pass through a stand in an afternoon:

*   **Failure is visible.** A send that fails leaves the session in a retryable
    state with an on-screen warning, and the failure is itself a logged event.
    A silently lost participant is worse than a visibly failed one.
*   **There is always a second route.** The download buttons are rendered whether
    or not the email succeeded, so a session can be rescued by hand.
*   **Nothing leaves without consent.** Every path through this module checks the
    consent record first; an under-18 practice session cannot reach the network
    even if a caller asks it to.
"""
from __future__ import annotations

import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple

from .config import get_secret
from .telemetry import EV_DELIVERY, SessionLog

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 465


@dataclass
class Attachment:
    filename: str
    content: str
    subtype: str = "csv"

    @property
    def bytes(self) -> bytes:
        return self.content.encode("utf-8")


@dataclass
class DeliveryState:
    attempts: int = 0
    delivered: bool = False
    last_error: str = ""
    attachments: List[Attachment] = field(default_factory=list)


def build_attachments(log: SessionLog) -> List[Attachment]:
    """The four artefacts that make one session analysable and auditable."""
    stem = log.tool + "_" + log.pseudonym
    attachments = [
        # One row. Concatenate these across participants to get the analysis table.
        Attachment(stem + "_session.csv", log.session_csv(), "csv"),
        # One row per participant turn: the per-turn measures.
        Attachment(stem + "_turns.csv", log.turns_csv(), "csv"),
        # One row per event: the provenance and sequence audit trail.
        Attachment(stem + "_events.csv", log.events_csv(), "csv"),
        # Lossless archival copy, including the transcript.
        Attachment(stem + "_bundle.json", log.bundle_json(), "json"),
    ]
    return [a for a in attachments if a.content.strip()]


def smtp_configured() -> bool:
    return bool(get_secret("REPORT_FROM_EMAIL")) and bool(get_secret("REPORT_EMAIL_PASSWORD"))


def deliver(
    log: SessionLog,
    *,
    may_collect: bool,
    attachments: Optional[List[Attachment]] = None,
    max_attempts: int = 3,
) -> Tuple[bool, str]:
    """Email the session record. Returns (delivered, message_for_the_screen)."""
    if not may_collect:
        # Not an error: an under-18 practice session is supposed to end here.
        return False, "No data was collected for this session, so nothing was sent."

    attachments = attachments or build_attachments(log)
    if not attachments:
        return False, "There was nothing to send."

    sender = get_secret("REPORT_FROM_EMAIL")
    password = get_secret("REPORT_EMAIL_PASSWORD")
    recipient = get_secret("REPORT_TO_EMAIL", "A.Naky@lboro.ac.uk")
    server = get_secret("SMTP_SERVER", DEFAULT_SMTP_SERVER)
    port = int(get_secret("SMTP_PORT", DEFAULT_SMTP_PORT) or DEFAULT_SMTP_PORT)

    if not sender or not password:
        message = "Email is not configured (REPORT_FROM_EMAIL / REPORT_EMAIL_PASSWORD)."
        log.log(EV_DELIVERY, detail={"ok": False, "reason": "not_configured"}, ok=False, error=message)
        return False, message

    arm = log.meta.get("arm_id", "")
    subject = "[MHAESTRO %s] %s %s%s" % (
        log.tool,
        log.pseudonym,
        ("arm " + str(arm) + " ") if arm else "",
        log.session_id[:8],
    )

    body_lines = [
        "Session record attached.",
        "",
        "Pseudonym:   " + log.pseudonym,
        "Session id:  " + log.session_id,
        "Tool:        " + log.tool,
        "Arm:         " + str(arm or "n/a"),
        "Started:     " + log.started_iso,
        "Policy hash: " + str(log.meta.get("policy_hash", "n/a")),
        "Prompt hash: " + str(log.meta.get("prompt_bundle_hash", "n/a")),
        "",
        "Attachments: " + ", ".join(a.filename for a in attachments),
    ]

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            message = EmailMessage()
            message["From"] = sender
            message["To"] = recipient
            message["Subject"] = subject
            message.set_content("\n".join(body_lines))
            for attachment in attachments:
                message.add_attachment(
                    attachment.bytes,
                    maintype="text" if attachment.subtype != "json" else "application",
                    subtype=attachment.subtype,
                    filename=attachment.filename,
                )

            with smtplib.SMTP_SSL(server, port, timeout=30) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)

            log.log(
                EV_DELIVERY,
                detail={
                    "ok": True,
                    "attempt": attempt,
                    "recipient": recipient,
                    "files": [a.filename for a in attachments],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return True, "Your responses have been submitted. Thank you."

        except Exception as exc:  # noqa: BLE001 -- every failure is logged and retried
            last_error = type(exc).__name__ + ": " + str(exc)
            log.log(
                EV_DELIVERY,
                detail={"ok": False, "attempt": attempt, "recipient": recipient},
                ok=False,
                error=last_error,
            )
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 8))

    return False, (
        "We could not submit your responses automatically. Please use the download "
        "button below and hand the file to a member of the research team."
    )

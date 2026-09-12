"""Experimental arms and their randomisation.

The paper's own stated next step (Sections 6.5 and 6.7) is "an A/B evaluation that
compares the current governed flow against the agency-enhanced variant, holding
coverage as a fixed constraint and tracking non-intrusiveness, human-equivalence,
and time/interaction measures."

This study generalises that to a 2x2 between-participants factorial crossing the
two things MHAESTRO actually contributes:

    structure   -- does a pre-validated decision graph dictate what is asked next?
    governance  -- does an adequacy checker block progression until the current
                   node's objective is met?

    A  tree      + adequacy   MHAESTRO as published
    B  tree      + no check   single pass through the graph, never re-probes
    C  free-form + adequacy   an adequacy checker over an unstructured interviewer
    D  free-form + no check   the naive LLM interviewer

Crossing them separates the paper's confound: the published run cannot say whether
its high extraction fidelity came from the graph, from the checker, or from both,
nor which of the two produced the 2.45-point fidelity-fluidity gap. Arm A is the
published configuration; arm D is the baseline everyone else ships.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

ARM_IDS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    structure: str  # "tree" | "freeform"
    governance: str  # "adequacy" | "none"
    description: str

    @property
    def uses_policy_graph(self) -> bool:
        return self.structure == "tree"

    @property
    def uses_adequacy_check(self) -> bool:
        return self.governance == "adequacy"


ARMS: Dict[str, Arm] = {
    "A": Arm(
        "A",
        "Graph + adequacy (MHAESTRO)",
        "tree",
        "adequacy",
        "Deterministic traversal of the validated decision graph, with the steering "
        "agent blocking progression until the current node's objective is met.",
    ),
    "B": Arm(
        "B",
        "Graph, no adequacy",
        "tree",
        "none",
        "Deterministic traversal of the same graph, but a single pass: whatever the "
        "participant says, the session advances to the next node.",
    ),
    "C": Arm(
        "C",
        "Free-form + adequacy",
        "freeform",
        "adequacy",
        "No graph. The interviewer works from the survey's topic list, with the same "
        "adequacy checker gating each turn.",
    ),
    "D": Arm(
        "D",
        "Free-form, no adequacy",
        "freeform",
        "none",
        "No graph and no checker: a conventional LLM interviewer given the topic list.",
    ),
}


def get_arm(arm_id: str) -> Arm:
    return ARMS.get((arm_id or "").strip().upper(), ARMS["A"])


class BlockRandomiser:
    """Permuted-block allocation so the four arms stay balanced as people arrive.

    Simple per-session coin-flipping drifts badly at the sample sizes an open day
    produces; permuted blocks of four guarantee the arms never differ by more than
    one participant at any block boundary. The randomiser lives in one Streamlit
    cache_resource object, so it is shared by every participant hitting the same
    app instance. If the app restarts mid-event the sequence restarts too -- that
    is a re-seed, not a bias, and the assignment method is recorded per session.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._queue: List[str] = []
        self.assigned = 0
        self.block_index = 0
        self.counts: Dict[str, int] = {arm: 0 for arm in ARM_IDS}

    def next_arm(self) -> Dict[str, object]:
        if not self._queue:
            block = list(ARM_IDS)
            self._rng.shuffle(block)
            self._queue = block
            self.block_index += 1
        arm_id = self._queue.pop(0)
        self.assigned += 1
        self.counts[arm_id] += 1
        return {
            "arm_id": arm_id,
            "method": "permuted_block",
            "block_index": self.block_index,
            "position_in_block": 4 - len(self._queue),
            "sequence_number": self.assigned,
        }


def independent_arm() -> Dict[str, object]:
    """Fallback allocation when no shared randomiser is available."""
    return {
        "arm_id": random.choice(ARM_IDS),
        "method": "simple_random",
        "block_index": "",
        "position_in_block": "",
        "sequence_number": "",
    }

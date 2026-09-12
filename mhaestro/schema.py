"""The governed survey policy artefact: schema, validation, and provenance.

The paper specifies the artefact precisely (Section 3.2.4): each node carries a
stable `id`, a `question` field, an `objective`, and an *ordered* `children`
array in which each child is a transition object with a categorical `condition`
label and a `target_id`. Labels used at run time must match those `condition`
values exactly or the Elicitor raises a traversal error. Joins are permitted, so
the structure is a rooted directed acyclic graph, not a strict tree (Section
4.2); the word "tree" is kept as shorthand.

The original prototype stored nodes *nested* inside one another, which cannot
express a join and made the traversal code guess between dict- and list-shaped
children. This module defines the flat node map the paper describes, validates
it before a session may start ("the Elicitor consumes only JSON that passes
schema and consistency checks"), and can up-convert a legacy nested artefact so
the published tree still loads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION = "mhaestro-policy-2"

# Node priority. Coverage of every `core` node is the study's non-negotiable
# constraint; `optional` nodes are depth probes that may be dropped when the
# session is running over its time budget.
CORE = "core"
OPTIONAL = "optional"

TERMINAL_CONDITION = "__end__"


@dataclass
class ValidationReport:
    """Outcome of checking a policy artefact. `ok` gates whether a session may run."""

    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def canonical_json(policy: dict) -> str:
    """Stable serialisation -- the basis of the tree hash recorded in every log row."""
    return json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def policy_hash(policy: dict) -> str:
    """Short content hash of the artefact (the paper's "decision-tree commit hash")."""
    return hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------- normalisation


def _normalise_node(raw: dict, node_id: str) -> dict:
    """Coerce one node into the canonical shape, tolerating the prototype's spellings."""
    question = raw.get("question") or raw.get("prompt") or ""

    objective = raw.get("objective")
    if objective is None:
        objectives = raw.get("objectives")
        if isinstance(objectives, list):
            objective = "; ".join(str(o) for o in objectives)
        elif isinstance(objectives, str):
            objective = objectives
        else:
            objective = ""

    children: List[dict] = []
    raw_children = raw.get("children")
    if isinstance(raw_children, list):
        for index, child in enumerate(raw_children):
            if not isinstance(child, dict):
                continue
            condition = child.get("condition") or child.get("label") or ("option_%d" % (index + 1))
            target = child.get("target_id") or child.get("id") or child.get("target")
            if target:
                children.append(
                    {
                        "condition": str(condition),
                        "target_id": str(target),
                        "description": str(child.get("description", "")),
                    }
                )

    priority = str(raw.get("priority", CORE)).lower()
    if priority not in (CORE, OPTIONAL):
        priority = CORE

    adequacy = raw.get("adequacy") if isinstance(raw.get("adequacy"), dict) else {}
    # The synthesis agent emits `max_reprobes` at the top level of a node; the
    # canonical form keeps it inside `adequacy` alongside the other gate settings.
    if "max_reprobes" not in adequacy and raw.get("max_reprobes") is not None:
        adequacy = dict(adequacy)
        adequacy["max_reprobes"] = raw["max_reprobes"]

    try:
        max_reprobes = max(0, min(2, int(adequacy.get("max_reprobes", 1))))
    except (TypeError, ValueError):
        max_reprobes = 1

    return {
        "id": node_id,
        "question": str(question),
        "objective": str(objective),
        "priority": priority,
        "topic": str(raw.get("topic", "")),
        "adequacy": {
            "require_stance": bool(adequacy.get("require_stance", True)),
            "require_reason": bool(adequacy.get("require_reason", True)),
            # The paper's recommended "soft cap on repeated adequacy checks": after
            # this many re-probes the turn advances regardless, and the shortfall is
            # logged rather than pressed.
            "max_reprobes": max_reprobes,
        },
        "children": children,
    }


def _flatten_nested(node: dict, nodes: Dict[str, dict], order: List[str]) -> Optional[str]:
    """Convert the prototype's nested `children` dict into a flat node map."""
    if not isinstance(node, dict):
        return None
    node_id = str(node.get("id") or ("node_%d" % (len(nodes) + 1)))

    raw_children = node.get("children") or {}
    if isinstance(raw_children, dict):
        child_items = list(raw_children.items())
    elif isinstance(raw_children, list):
        child_items = [(str(i), c) for i, c in enumerate(raw_children)]
    else:
        child_items = []

    transitions = []
    for key, child in child_items:
        child_id = _flatten_nested(child, nodes, order)
        if child_id:
            # The legacy format had no condition labels; the child's own id is the
            # most faithful stand-in and keeps labels unique within the node.
            transitions.append({"condition": str(child.get("id") or key), "target_id": child_id})

    flat = dict(node)
    flat["children"] = transitions
    nodes[node_id] = _normalise_node(flat, node_id)
    order.append(node_id)
    return node_id


def normalise_policy(raw: dict) -> dict:
    """Return a canonical policy artefact from either the flat or the legacy shape."""
    if not isinstance(raw, dict):
        raise ValueError("Policy artefact must be a JSON object.")

    if isinstance(raw.get("nodes"), list):
        # The synthesis agent emits nodes as an array (a JSON Schema cannot
        # constrain a map with arbitrary keys, so the list form is what the
        # providers' structured-output modes can actually validate).
        nodes = {}
        for index, node in enumerate(raw["nodes"]):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or ("node_%d" % (index + 1)))
            nodes[node_id] = _normalise_node(node, node_id)
        root_id = str(raw.get("root_id") or next(iter(nodes), ""))
        meta_source = raw
    elif isinstance(raw.get("nodes"), dict):
        nodes = {
            str(node_id): _normalise_node(node or {}, str(node_id))
            for node_id, node in raw["nodes"].items()
        }
        root_id = str(raw.get("root_id") or next(iter(nodes), ""))
        meta_source = raw
    else:
        # Legacy nested artefact (the published nanaBanana.json).
        nodes = {}
        order: List[str] = []
        root_id = _flatten_nested(raw, nodes, order) or ""
        meta_source = raw

    return {
        "schema_version": SCHEMA_VERSION,
        "survey_id": str(meta_source.get("survey_id", "") or "untitled_survey"),
        "title": str(meta_source.get("title", "") or "Conversational survey"),
        "description": str(meta_source.get("description", "")),
        "scope_note": str(meta_source.get("scope_note", "")),
        "steering_note": str(meta_source.get("steering_note", "")),
        "target_minutes": float(meta_source.get("target_minutes", 5) or 5),
        "summary_questions": list(meta_source.get("summary_questions", []) or []),
        "root_id": root_id,
        "nodes": nodes,
    }


# ------------------------------------------------------------------ validation


def validate_policy(policy: dict) -> ValidationReport:
    """Check the artefact is runnable. A failing report must block the session."""
    report = ValidationReport()
    nodes: Dict[str, dict] = policy.get("nodes") or {}
    root_id = policy.get("root_id") or ""

    if not nodes:
        report.error("The policy contains no nodes.")
        return report
    if root_id not in nodes:
        report.error("Root id %r is not present in the node map." % root_id)
        return report

    for node_id, node in nodes.items():
        if node.get("id") != node_id:
            report.error("Node %r has mismatched internal id %r." % (node_id, node.get("id")))
        if not str(node.get("question", "")).strip():
            report.error("Node %r has no question text." % node_id)
        if not str(node.get("objective", "")).strip():
            report.warn("Node %r has no objective; the adequacy check has nothing to test against." % node_id)

        seen_conditions: Set[str] = set()
        for child in node.get("children", []):
            condition = child.get("condition", "")
            target = child.get("target_id", "")
            if condition in seen_conditions:
                report.error(
                    "Node %r has duplicate condition label %r; traversal could not resolve it."
                    % (node_id, condition)
                )
            seen_conditions.add(condition)
            if target not in nodes:
                report.error(
                    "Node %r routes condition %r to unknown node %r." % (node_id, condition, target)
                )

    reachable = reachable_from(policy, root_id)
    for node_id in nodes:
        if node_id not in reachable:
            report.warn("Node %r is unreachable from the root and will never be asked." % node_id)

    cycle = find_cycle(policy)
    if cycle:
        report.error("The policy contains a cycle (%s); traversal would not terminate." % " -> ".join(cycle))

    terminals = [n for n in reachable if not nodes[n].get("children")]
    if not terminals:
        report.error("No terminal node is reachable; the session could never finish.")

    # The coverage guarantee. A core topic some routes bypass is not a guarantee,
    # and coverage completeness is this study's fixed constraint. Raised as a
    # warning rather than an error so a live event is never halted by it, but
    # surfaced prominently to the expert at review time.
    bypassable = bypassable_core_nodes(policy)
    for node_id in bypassable:
        report.warn(
            "Core node %r can be bypassed: some routes never reach it, so coverage is not "
            "guaranteed. Mark it optional, or route every branch through it." % node_id
        )

    core_nodes = [n for n in reachable if nodes[n].get("priority") == CORE]
    report.stats = {
        "bypassable_core": bypassable,
        "node_count": len(nodes),
        "reachable_count": len(reachable),
        "core_count": len(core_nodes),
        "optional_count": len(reachable) - len(core_nodes),
        "terminal_count": len(terminals),
        "max_depth": max_depth(policy),
        "max_branching": max((len(nodes[n].get("children", [])) for n in nodes), default=0),
        "join_count": sum(1 for n, deg in in_degrees(policy).items() if deg > 1),
    }
    return report


def reachable_from(policy: dict, start_id: str) -> Set[str]:
    nodes = policy.get("nodes") or {}
    seen: Set[str] = set()
    stack = [start_id]
    while stack:
        current = stack.pop()
        if current in seen or current not in nodes:
            continue
        seen.add(current)
        for child in nodes[current].get("children", []):
            stack.append(child.get("target_id", ""))
    return seen


def bypassable_core_nodes(policy: dict) -> List[str]:
    """Core nodes that some route through the graph avoids.

    A node lies on *every* root-to-terminal path exactly when deleting it makes
    every terminal unreachable from the root. So for each core node, walk the graph
    from the root with that node removed and see whether any terminal survives.
    """
    nodes = policy.get("nodes") or {}
    root_id = policy.get("root_id", "")
    reachable = reachable_from(policy, root_id)
    terminals = {n for n in reachable if not nodes[n].get("children")}
    if not terminals:
        return []

    bypassable = []
    for node_id in sorted(n for n in reachable if nodes[n].get("priority") == CORE):
        if node_id == root_id:
            continue
        seen: Set[str] = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in seen or current == node_id or current not in nodes:
                continue
            seen.add(current)
            for child in nodes[current].get("children", []):
                stack.append(child.get("target_id", ""))
        if seen & terminals:
            bypassable.append(node_id)
    return bypassable


def in_degrees(policy: dict) -> Dict[str, int]:
    nodes = policy.get("nodes") or {}
    degrees = {node_id: 0 for node_id in nodes}
    for node in nodes.values():
        for child in node.get("children", []):
            target = child.get("target_id", "")
            if target in degrees:
                degrees[target] += 1
    return degrees


def find_cycle(policy: dict) -> Optional[List[str]]:
    """Return one cycle as a node-id path, or None if the graph is acyclic."""
    nodes = policy.get("nodes") or {}
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node_id: WHITE for node_id in nodes}
    path: List[str] = []

    def visit(node_id: str) -> Optional[List[str]]:
        colour[node_id] = GREY
        path.append(node_id)
        for child in nodes[node_id].get("children", []):
            target = child.get("target_id", "")
            if target not in nodes:
                continue
            if colour[target] == GREY:
                return path[path.index(target) :] + [target]
            if colour[target] == WHITE:
                found = visit(target)
                if found:
                    return found
        colour[node_id] = BLACK
        path.pop()
        return None

    for node_id in nodes:
        if colour[node_id] == WHITE:
            found = visit(node_id)
            if found:
                return found
    return None


def max_depth(policy: dict) -> int:
    """Longest path length from the root, in nodes. Safe on a DAG with joins."""
    nodes = policy.get("nodes") or {}
    root_id = policy.get("root_id") or ""
    memo: Dict[str, int] = {}

    def depth(node_id: str, seen: Tuple[str, ...]) -> int:
        if node_id in seen or node_id not in nodes:
            return 0
        if node_id in memo:
            return memo[node_id]
        children = nodes[node_id].get("children", [])
        if not children:
            memo[node_id] = 1
            return 1
        best = 1 + max(depth(c.get("target_id", ""), seen + (node_id,)) for c in children)
        memo[node_id] = best
        return best

    return depth(root_id, ())


def estimate_longest_seconds(policy: dict, *, core_seconds: float = 38.0, optional_seconds: float = 22.0) -> int:
    """Worst-case session length in seconds, used to check the time budget.

    Core nodes carry the full per-node cost because they may attract a re-probe;
    optional depth probes are single-turn by construction and cost less.
    """
    nodes = policy.get("nodes") or {}
    memo: Dict[str, float] = {}

    def cost(node_id: str) -> float:
        node = nodes.get(node_id) or {}
        return optional_seconds if node.get("priority") == OPTIONAL else core_seconds

    def walk(node_id: str, seen: Tuple[str, ...]) -> float:
        if node_id not in nodes or node_id in seen:
            return 0.0
        if node_id in memo:
            return memo[node_id]
        children = nodes[node_id].get("children", [])
        if not children:
            memo[node_id] = cost(node_id)
            return memo[node_id]
        best = cost(node_id) + max(walk(c.get("target_id", ""), seen + (node_id,)) for c in children)
        memo[node_id] = best
        return best

    return int(round(walk(policy.get("root_id", ""), ())))


def shortest_remaining(policy: dict, node_id: str) -> int:
    """Fewest further nodes to a terminal -- drives the honest progress indicator."""
    nodes = policy.get("nodes") or {}
    memo: Dict[str, int] = {}

    def walk(current: str, seen: Tuple[str, ...]) -> int:
        if current not in nodes or current in seen:
            return 0
        if current in memo:
            return memo[current]
        children = nodes[current].get("children", [])
        if not children:
            memo[current] = 0
            return 0
        best = 1 + min(walk(c.get("target_id", ""), seen + (current,)) for c in children)
        memo[current] = best
        return best

    return walk(node_id, ())


def get_node(policy: dict, node_id: str) -> Optional[dict]:
    return (policy.get("nodes") or {}).get(node_id)


def child_options(policy: dict, node_id: str) -> List[dict]:
    node = get_node(policy, node_id)
    return list(node.get("children", [])) if node else []


def path_questions(policy: dict, path: List[str]) -> List[str]:
    nodes = policy.get("nodes") or {}
    return [nodes[n]["question"] for n in path if n in nodes]


def core_nodes(policy: dict) -> List[str]:
    nodes = policy.get("nodes") or {}
    return [n for n, node in nodes.items() if node.get("priority") == CORE]


def coverage(policy: dict, visited: List[str]) -> Dict[str, Any]:
    """Coverage completeness -- the study's fixed constraint, reported per session.

    Only *reachable core* nodes count: an optional depth probe that was skipped for
    time, or a branch the participant's own answers routed away from, is not a
    coverage failure. What matters is that every core topic on the path taken was
    actually asked.
    """
    nodes = policy.get("nodes") or {}
    visited_set = set(visited)
    on_path_core = [n for n in visited_set if nodes.get(n, {}).get("priority") == CORE]
    all_core = set(core_nodes(policy)) & reachable_from(policy, policy.get("root_id", ""))
    return {
        # Structural coverage is only defined for a session that traversed a graph.
        # The free-form arms have no node path, and reporting 0-of-6 for them would
        # read in a pooled analysis as "covered nothing" rather than "not measured
        # this way" -- use the coded coverage, which is computed for every arm.
        "applicable": True,
        "nodes_visited": len(visited_set),
        "core_visited": len(on_path_core),
        "core_total_reachable": len(all_core),
        "core_rate": round(len(on_path_core) / len(all_core), 4) if all_core else "",
        "visited_ids": sorted(visited_set),
    }


def coverage_not_applicable(reason: str) -> Dict[str, Any]:
    """The structural-coverage record for an arm that never traversed a graph."""
    return {
        "applicable": False,
        "reason": reason,
        "nodes_visited": "",
        "core_visited": "",
        "core_total_reachable": "",
        "core_rate": "",
        "visited_ids": [],
    }

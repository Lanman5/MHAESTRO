"""Rendering a policy graph so a human can actually check it.

Section 3.2.3 rests the whole reproducibility claim on the graph being "easy for
experts to review", and Table 1 lists human oversight as a governance control --
but the published pipeline handed the expert a JSON blob. An expert cannot see
from JSON that a branch dead-ends, that a topic they asked for is unreachable, or
that the longest path is twice the time budget. They can see all three in a
drawing.

DOT is emitted as a plain string and rendered client-side by `st.graphviz_chart`,
so no Graphviz binary is needed on the host.
"""
from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Tuple

from . import schema as S

_CORE_FILL = "#e8eefc"
_CORE_LINE = "#2f5bd7"
_OPTIONAL_FILL = "#f4f4f5"
_OPTIONAL_LINE = "#9a9aa2"
_TERMINAL_FILL = "#e6f6ec"
_TERMINAL_LINE = "#2f8f52"
_PATH_LINE = "#c2410c"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _wrap(text: str, width: int = 34, max_lines: int = 4) -> str:
    lines = textwrap.wrap(text or "", width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: width - 1] + "…"]
    return "\\n".join(_escape(line) for line in lines)


def policy_to_dot(
    policy: dict,
    *,
    highlight_path: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str, str]]] = None,
    show_objectives: bool = False,
) -> str:
    """Render the graph as DOT.

    `highlight_path` marks the nodes one session visited. Highlight the edges with
    `highlight_edges` -- `(source_id, condition, target_id)` triples, taken from the
    traversal events -- rather than inferring them from the node path: several
    conditions at one node often route to the same target, so a node pair does not
    identify which branch was actually taken.
    """
    nodes: Dict[str, dict] = policy.get("nodes") or {}
    root_id = policy.get("root_id", "")
    path = highlight_path or []
    hot_edges = set(highlight_edges or [])

    lines = [
        "digraph policy {",
        '  rankdir="TB";',
        '  bgcolor="transparent";',
        # Branch labels sit on the edges, so the ranks need room or the labels of
        # sibling branches overlap and the drawing stops being checkable -- which
        # is the only reason it exists.
        "  ranksep=0.85;",
        "  nodesep=0.55;",
        "  splines=true;",
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 margin="0.18,0.12"];',
        '  edge [fontname="Helvetica" fontsize=8 color="#8a8a92" labeldistance=1.4];',
    ]

    for node_id, node in nodes.items():
        is_terminal = not node.get("children")
        if is_terminal:
            fill, line = _TERMINAL_FILL, _TERMINAL_LINE
        elif node.get("priority") == S.OPTIONAL:
            fill, line = _OPTIONAL_FILL, _OPTIONAL_LINE
        else:
            fill, line = _CORE_FILL, _CORE_LINE

        label = "<b>" + _escape(node_id) + "</b>"
        parts = [_escape(node_id), _wrap(node.get("question", ""))]
        if show_objectives and node.get("objective"):
            parts.append(_wrap("↳ " + node["objective"], width=38, max_lines=2))
        label = "\\n".join(p for p in parts if p)

        penwidth = "2.4" if node_id in path else ("2.0" if node_id == root_id else "1.1")
        border = _PATH_LINE if node_id in path else line
        lines.append(
            '  "%s" [label="%s" fillcolor="%s" color="%s" penwidth=%s];'
            % (_escape(node_id), label, fill, border, penwidth)
        )

    for node_id, node in nodes.items():
        for child in node.get("children", []):
            target = child.get("target_id", "")
            if target not in nodes:
                continue
            condition = str(child.get("condition", ""))
            on_path = (node_id, condition, target) in hot_edges
            lines.append(
                '  "%s" -> "%s" [label="%s"%s];'
                % (
                    _escape(node_id),
                    _escape(target),
                    _wrap(condition, width=18, max_lines=3),
                    ' color="%s" penwidth=2.2 fontcolor="%s"' % (_PATH_LINE, _PATH_LINE) if on_path else "",
                )
            )

    lines.append("}")
    return "\n".join(lines)


def policy_outline(policy: dict) -> List[Dict[str, object]]:
    """A flat, sortable table of the graph -- the review view that is not a picture."""
    nodes: Dict[str, dict] = policy.get("nodes") or {}
    reachable = S.reachable_from(policy, policy.get("root_id", ""))
    degrees = S.in_degrees(policy)

    rows = []
    for node_id, node in nodes.items():
        rows.append(
            {
                "id": node_id,
                "priority": node.get("priority", ""),
                "topic": node.get("topic", ""),
                "question": node.get("question", ""),
                "objective": node.get("objective", ""),
                "branches": len(node.get("children", [])),
                "conditions": ", ".join(c.get("condition", "") for c in node.get("children", [])),
                "max_reprobes": node.get("adequacy", {}).get("max_reprobes", 1),
                "in_degree": degrees.get(node_id, 0),
                "reachable": node_id in reachable,
                "terminal": not node.get("children"),
            }
        )
    rows.sort(key=lambda r: (not r["reachable"], r["priority"] != S.CORE, r["id"]))
    return rows

"""Pool the emailed session CSVs into one analysis table.

Each completed session arrives as a one-row `*_session.csv`. Save them all to a
folder and run:

    python analysis/pool_sessions.py path/to/folder -o pooled.csv

Produces the pooled table plus the per-arm and marginal summaries for the study's
primary outcomes -- extraction fidelity, aligned non-intrusiveness, perceived
human-equivalence, workload, and the fidelity-fluidity gap -- reported the same
way the paper reports them (Sections 5.3-5.8), so the new numbers sit directly
alongside the published baseline.

Deliberately stdlib-only: this has to run on a university machine with nothing
installed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

# The paper's published baseline, for side-by-side comparison (Table 2).
BASELINE = {
    "fidelity_mean": 4.20,
    "ux_ease_aligned": 4.13,
    "ux_visual_appeal_aligned": 3.50,
    "ux_coherence_aligned": 3.50,
    "ux_non_intrusiveness_aligned": 1.75,
    "ux_human_equivalence_aligned": 2.75,
    "fidelity_fluidity_gap": 2.45,
}

PRIMARY = [
    "fidelity_mean",
    "fidelity_agreement_rate",
    "ux_non_intrusiveness_aligned",
    "ux_human_equivalence_aligned",
    "fidelity_fluidity_gap",
    "tlx_raw_mean",
]

SECONDARY = [
    "ux_ease_aligned",
    "ux_visual_appeal_aligned",
    "ux_coherence_aligned",
    "ux_non_repetitiveness_aligned",
    "ux_comfort_aligned",
    "ux_clarity_aligned",
    "ux_agency_aligned",
]

PROCESS = [
    "duration_s",
    "n_participant_turns",
    "n_reprobes",
    "reprobe_rate",
    "n_soft_cap_hits",
    "n_skips",
    "n_why_hints",
    "reply_words_mean",
    "reply_words_slope",
    "reply_latency_median_s",
    "agent_latency_mean_ms",
    "meta_coverage_coded_rate",
]


def load(folder: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    pattern = os.path.join(folder, "**", "*_session.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = os.path.basename(path)
                rows.append(row)
    return rows


def numeric(rows: List[Dict[str, str]], field: str) -> List[float]:
    out = []
    for row in rows:
        value = (row.get(field) or "").strip()
        if value == "":
            continue
        try:
            out.append(float(value))
        except ValueError:
            continue
    return out


def describe(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": "", "median": "", "sd": ""}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def table(title: str, groups: Dict[str, List[Dict[str, str]]], fields: List[str]) -> None:
    print("\n" + title)
    print("-" * len(title))
    names = list(groups)
    header = "%-34s %6s" % ("measure", "base") + "".join("%16s" % ("%s (n=%d)" % (n, len(groups[n]))) for n in names)
    print(header)
    for field in fields:
        base = BASELINE.get(field)
        line = "%-34s %6s" % (field[:34], ("%.2f" % base) if base is not None else "-")
        for name in names:
            stats = describe(numeric(groups[name], field))
            line += "%16s" % (
                "" if stats["n"] == 0 else "%.2f (%.2f)" % (stats["mean"], stats["sd"])
            )
        print(line)
    print("\n(cell = mean (SD); 'base' = the published N=8 baseline where one exists)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder containing the *_session.csv attachments.")
    parser.add_argument("-o", "--out", default="pooled_sessions.csv", help="Where to write the pooled table.")
    args = parser.parse_args()

    rows = load(args.folder)
    if not rows:
        print("No *_session.csv files found under " + args.folder, file=sys.stderr)
        return 1

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    elicitor = [r for r in rows if r.get("tool") == "elicitor"]
    keng = [r for r in rows if r.get("tool") == "keng"]

    print("Pooled %d session rows -> %s" % (len(rows), args.out))
    print("  Elicitor sessions: %d   K-Eng sessions: %d" % (len(elicitor), len(keng)))

    if not elicitor:
        return 0

    by_arm = {}
    for arm in ("A", "B", "C", "D"):
        subset = [r for r in elicitor if r.get("meta_arm_id") == arm]
        if subset:
            by_arm[arm] = subset
    incomplete = [r for r in elicitor if r.get("meta_arm_id") not in by_arm]
    if incomplete:
        print("  %d rows have no recognised arm and are excluded from the arm tables." % len(incomplete))

    if by_arm:
        table("Primary outcomes by arm", by_arm, PRIMARY)
        table("Secondary experience facets by arm", by_arm, SECONDARY)
        table("Process and interaction measures by arm", by_arm, PROCESS)

        # Marginal means: the point of the 2x2 is the two main effects.
        marginals = {
            "graph": [r for r in elicitor if r.get("meta_arm_structure") == "tree"],
            "free-form": [r for r in elicitor if r.get("meta_arm_structure") == "freeform"],
            "adequacy": [r for r in elicitor if r.get("meta_arm_governance") == "adequacy"],
            "no check": [r for r in elicitor if r.get("meta_arm_governance") == "none"],
        }
        marginals = {k: v for k, v in marginals.items() if v}
        if marginals:
            table("Marginal means (main effects of structure and of governance)", marginals, PRIMARY)

        print("\nCoverage completeness -- the fixed constraint, which must hold in every arm:")
        for arm, subset in by_arm.items():
            coded = numeric(subset, "meta_coverage_coded_rate")
            print(
                "  arm %s: coded coverage mean %s over %d sessions"
                % (arm, ("%.3f" % statistics.mean(coded)) if coded else "n/a", len(subset))
            )

    if keng:
        print("\nK-Eng (expert) sessions:")
        for field in ("compile_seconds", "compile_repairs_needed", "duration_s", "tlx_raw_mean"):
            key = field if field.startswith("tlx") or field == "duration_s" else "meta_" + field
            stats = describe(numeric(keng, key))
            print("  %-28s %s" % (field, stats))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

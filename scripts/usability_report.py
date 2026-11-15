"""Usability study report (interface phase F5): SUS scores and task metrics for the paper.

Usage: python scripts/usability_report.py --sus sus.csv --logs study_logs/*.jsonl
       [--out docs/paper/usability_results.md]

``sus.csv``: one row per participant, columns ``participant,q1..q10`` (answers 1-5, in the
order of the questionnaire in ``docs/07_usability_protocol.md``). Logs come from
``rausnet-desktop --study <code>``; the participant code is the file name prefix.
Acceptance (``docs/06_interface_plan.md`` F5): mean SUS >= 68 and every click budget met (median).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

from rausnet.ui.study import SUS_PASS, mean_ci, read_log, summary_table, sus_score, task_results


def build(sus_rows: list[dict], logs: list[str]) -> tuple[str, bool]:
    lines = ["# Usability study results", ""]
    ok = True
    scores = {}
    for row in sus_rows:
        scores[row["participant"]] = sus_score([int(row[f"q{i}"]) for i in range(1, 11)])
    if scores:
        m, lo, hi = mean_ci(list(scores.values()))
        ci = f" (95 % CI {lo:.1f} to {hi:.1f})" if not math.isnan(lo) else ""
        passed = m >= SUS_PASS
        ok &= passed
        lines += [f"**SUS:** mean {m:.1f}{ci}, n = {len(scores)}; target ≥ {SUS_PASS:.0f}: "
                  f"{'met' if passed else 'NOT met'}.", "",
                  "| Participant | SUS |", "|---|---|"]
        lines += [f"| {p} | {s:.1f} |" for p, s in sorted(scores.items())]
        lines.append("")
    results = []
    for path in logs:
        participant = Path(path).stem.split("_")[0]
        results += task_results(read_log(path), participant)
    if results:
        rows = summary_table(results)
        lines += ["| Task | Completed | Median actions | Budget | Budget met | Median time (s) |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            ok &= r["budget_met"]
            med = "-" if r["median_actions"] is None else f"{r['median_actions']:g}"
            secs = "-" if r["median_seconds"] is None else f"{r['median_seconds']:.0f}"
            lines.append(f"| {r['task']} | {r['completed']}/{r['participants']} | {med} | {r['budget']} | "
                         f"{'yes' if r['budget_met'] else 'no'} | {secs} |")
        lines += ["", "Actions = clicks, counted keys (shortcuts, Tab, Enter, Space) and drops between the "
                  "facilitator's task mark and the task's goal; typing is not counted."]
    lines += ["", f"**Acceptance:** {'met' if ok and scores and results else 'not met'}."]
    return "\n".join(lines) + "\n", bool(ok and scores and results)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sus", required=True)
    parser.add_argument("--logs", nargs="*", default=[])
    parser.add_argument("--out", default="docs/paper/usability_results.md")
    args = parser.parse_args(argv)
    with open(args.sus, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logs = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    text, ok = build(rows, logs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

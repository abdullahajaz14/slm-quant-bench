"""scripts/analysis.py: paired degradation analysis (the Chapter 5 core).

Reads a quality-run JSONL and computes, per (model, task, metric):
  - the mean at each precision with a bootstrap 95% CI, and
  - the PAIRED per-item delta against the FP16 reference for each
    quantised precision, with a bootstrap CI over the item-level
    deltas (seed 42), which is what Chapter 3 promises: every
    degradation claim carries its uncertainty.

Pairing joins items by item_id within (model, task); seed-controlled
sampling guarantees both precisions saw identical items.

Outputs:
  results/degradation.csv   one row per (model, task, metric, precision)
                            with mean, ci, and delta-vs-fp16 columns.

The public benchmarks and the curated corpus are produced by separate
runs into separate files, so --jsonl takes one or more paths and merges
them. They must be analysed together: the corpus tasks are part of the
same per-model comparison and appear in the same tables and figures.

Run from the repository root with the venv active:
  python scripts/analysis.py --jsonl results/quality-colab.jsonl \
      results/ukps-colab.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

from slmbench.results import bootstrap_ci

REFERENCE = "fp16"


def distinguishable(lo: float, hi: float) -> bool:
    """Does the paired-delta interval exclude zero?

    True only when the whole interval sits strictly on one side of
    zero. Computed once here rather than re-derived per table, because
    the obvious shorthand "the endpoints share a sign" gets the
    degenerate case backwards: where a quantised precision reproduces
    the reference output on every item, every paired delta is exactly
    zero and the interval collapses to [0, 0]. That is the strongest
    possible evidence of NO difference, yet the shorthand reports it as
    a difference. Such cases are real in this study and are identified
    by n_identical equalling n_paired.
    """
    return lo > 0.0 or hi < 0.0


def load(paths: list[str]) -> dict:
    """(model, task, metric) -> {precision: {item_id: value}}

    Item identifiers are unique within a task and tasks do not span
    files, so merging several run files cannot collide.
    """
    data: dict = defaultdict(lambda: defaultdict(dict))
    skipped = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error"):
                    skipped += 1
                    continue
                for metric, value in rec["scores"].items():
                    key = (rec["model"], rec["task"], metric)
                    data[key][rec["precision"]][rec["item_id"]] = value
    if skipped:
        # Errored evaluations are excluded from the means but their
        # count is reported, never silently absorbed: an analysis that
        # quietly drops items misstates what it measured.
        print(f"[analysis] {skipped} errored records excluded")
    return data


def analyse(data: dict) -> list[dict]:
    rows = []
    for (model, task, metric) in sorted(data):
        per_precision = data[(model, task, metric)]
        reference = per_precision.get(REFERENCE, {})
        for precision in sorted(per_precision):
            values = list(per_precision[precision].values())
            mean, lo, hi = bootstrap_ci(values)
            row = {"model": model, "task": task, "metric": metric,
                   "precision": precision, "n": len(values),
                   "mean": mean, "ci_lo": lo, "ci_hi": hi,
                   "delta_vs_fp16": "", "delta_lo": "", "delta_hi": "",
                   "n_paired": "", "distinguishable": "",
                   "n_identical": ""}
            if precision != REFERENCE and reference:
                paired_ids = sorted(set(per_precision[precision])
                                    & set(reference))
                deltas = [per_precision[precision][i] - reference[i]
                          for i in paired_ids]
                if deltas:
                    d_mean, d_lo, d_hi = bootstrap_ci(deltas)
                    row |= {"delta_vs_fp16": d_mean, "delta_lo": d_lo,
                            "delta_hi": d_hi, "n_paired": len(deltas),
                            "distinguishable": distinguishable(d_lo, d_hi),
                            "n_identical": sum(1 for d in deltas if d == 0)}
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, nargs="+")
    parser.add_argument("--out", default="results/degradation.csv")
    args = parser.parse_args()

    rows = analyse(load(args.jsonl))
    fields = ["model", "task", "metric", "precision", "n", "mean",
              "ci_lo", "ci_hi", "delta_vs_fp16", "delta_lo", "delta_hi",
              "n_paired", "distinguishable", "n_identical"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float)
                                 and not math.isnan(v) else v)
                             for k, v in row.items()})
    print(f"[analysis] {len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()

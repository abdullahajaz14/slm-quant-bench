"""scripts/pilot.py: the gate before full runs (Chapter 3, 3.5).

The pilot exercises every pipeline stage on small subsets and produces
results/pilot_report.md, which IS the protocol-freeze record: the file
Chapter 3's "planned values validated in the pilot" sentence points at,
and the trigger for the one pre-agreed mechanical update to Chapter 3.

What it establishes, per the chapter: per-item runtimes in the current
environment (projected to full-run cost), correct template rendering
per model, sane scorer outputs, and whether the planned sample sizes
deliver acceptably tight confidence intervals.

Run from the repository root with the package installed
(pip install -e .):
  python scripts/pilot.py --pilot-jsonl results/pilot-device.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict

from slmbench.results import bootstrap_ci

# Planned sample sizes from Chapter 3 Table 3.2 (protocol values the
# pilot validates; adjustments are reported with rationale).
PLANNED: dict[str, int] = {
    "cuad": 200,
    "hotpotqa": 300,
    "cnndm": 150,
    "truthfulqa_mc1": 817,
    "truthfulqa_mc2": 817,
    "ukps_qa": 60,
    "ukps_sum": 25,
}

_PRECISION_PREFERENCE = ["fp16", "q8_0", "q4_k_m"]


def _read_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def project_runtime(pilot_jsonl: str, planned: dict[str, int]) -> list[dict]:
    """Median per-item total_s per (model, precision, task), multiplied
    out to the planned sample size. MC records carry total_s summed
    over their option calls (the runner records them that way)."""
    times: dict[tuple, list[float]] = defaultdict(list)
    backends: dict[tuple, str] = {}
    for rec in _read_records(pilot_jsonl):
        if rec.get("error") or "total_s" not in rec.get("timings", {}):
            continue
        key = (rec["model"], rec["precision"], rec["task"])
        times[key].append(rec["timings"]["total_s"])
        backends[key] = rec["env"].get("backend", "?")
    rows = []
    for key in sorted(times):
        model, precision, task = key
        median_s = statistics.median(times[key])
        planned_n = planned.get(task, len(times[key]))
        rows.append({
            "model": model, "precision": precision, "task": task,
            "backend": backends[key],
            "n_pilot": len(times[key]),
            "median_s": median_s,
            "planned_n": planned_n,
            "projected_min": median_s * planned_n / 60.0,
        })
    return rows


def project_ci_width(pilot_jsonl: str, planned: dict[str, int]) -> list[dict]:
    """CI width at pilot n, projected to planned n via the standard
    planning approximation width ~ 1/sqrt(n) (valid for means, since
    the standard error of a mean scales with 1/sqrt(n))."""
    by_task_metric: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for rec in _read_records(pilot_jsonl):
        if rec.get("error"):
            continue
        for metric, value in rec.get("scores", {}).items():
            by_task_metric[(rec["task"], metric)][rec["precision"]].append(
                value)
    rows = []
    for (task, metric) in sorted(by_task_metric):
        per_precision = by_task_metric[(task, metric)]
        precision = next((p for p in _PRECISION_PREFERENCE
                          if p in per_precision), None)
        values = per_precision[precision]
        n_pilot = len(values)
        _, lo, hi = bootstrap_ci(values)
        width_pilot = hi - lo
        n_planned = planned.get(task, n_pilot)
        width_projected = width_pilot * math.sqrt(n_pilot / n_planned)
        in_unit_interval = all(0.0 <= v <= 1.0 for v in values)
        verdict = ("OK" if in_unit_interval and width_projected <= 0.10
                   else "FLAG")
        rows.append({
            "task": task, "metric": metric, "precision_used": precision,
            "n_pilot": n_pilot, "width_pilot": width_pilot,
            "n_planned": n_planned, "width_projected": width_projected,
            "verdict": verdict,
        })
    return rows


def render_check(models: list[str]) -> list[str]:
    """One rendered CUAD prompt and one TruthfulQA choice prompt per
    model, for eyeballing in the report (the spec's per-model template
    check)."""
    from slmbench import prompts
    from slmbench.adapters.base import Item

    qa_item = Item(task="cuad", item_id="render-check-qa",
                   context="EXAMPLE CONTRACT EXCERPT.",
                   question="EXAMPLE QUESTION?", references=[])
    mc_item = Item(task="truthfulqa_mc1", item_id="render-check-mc",
                   context=None, question="EXAMPLE QUESTION?",
                   references=[], choices=["yes"], choice_labels=[1])
    blocks = []
    for model in models:
        blocks.append(f"### {model}: extractive_qa\n```\n"
                      f"{prompts.render(qa_item, 'extractive_qa', model)}"
                      f"\n```")
        blocks.append(f"### {model}: mc_question\n```\n"
                      f"{prompts.mc_prompt(mc_item, model)}\n```")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-jsonl", required=True)
    parser.add_argument("--out", default="results/pilot_report.md")
    parser.add_argument("--findings", default="results/pilot_findings.md",
                        help="markdown of protocol findings and decisions, "
                             "folded into the report if present")
    parser.add_argument("--models", nargs="+",
                        default=["phi3-mini", "gemma2-2b",
                                 "llama32-3b", "mistral7b"])
    args = parser.parse_args()

    runtime_rows = project_runtime(args.pilot_jsonl, PLANNED)
    ci_rows = project_ci_width(args.pilot_jsonl, PLANNED)
    render_blocks = render_check(args.models)

    lines = ["# Pilot report (protocol-freeze record)", ""]

    if os.path.exists(args.findings):
        with open(args.findings) as f:
            # the findings document carries its own top-level heading,
            # so it is demoted one level when folded in
            body = f.read().replace("\n# ", "\n## ").replace("\n## ", "\n### ", 0)
        lines += [body.replace("# Pilot findings", "## Pilot findings", 1),
                  "", "---", ""]

    lines += ["## 1. Runtime projection", "",
              "| model | precision | task | backend | n_pilot | "
              "median_s | planned_n | projected_min |",
              "|---|---|---|---|---|---|---|---|"]
    per_backend: dict[str, float] = defaultdict(float)
    for r in runtime_rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['task']} | "
                     f"{r['backend']} | {r['n_pilot']} | "
                     f"{r['median_s']:.2f} | {r['planned_n']} | "
                     f"{r['projected_min']:.1f} |")
        per_backend[r["backend"]] += r["projected_min"]
    lines.append("")
    for backend, minutes in sorted(per_backend.items()):
        lines.append(f"Projected total on {backend}: "
                     f"{minutes / 60:.1f} hours")

    lines += ["", "## 2. Confidence-interval projection", "",
              "Planning approximation: CI width scales with 1/sqrt(n) "
              "(standard for means).", "",
              "| task | metric | precision | n_pilot | width_pilot | "
              "n_planned | width_projected | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for r in ci_rows:
        lines.append(f"| {r['task']} | {r['metric']} | "
                     f"{r['precision_used']} | {r['n_pilot']} | "
                     f"{r['width_pilot']:.3f} | {r['n_planned']} | "
                     f"{r['width_projected']:.3f} | {r['verdict']} |")

    lines += ["", "## 3. Template rendering check", ""] + render_blocks

    lines += ["", "## Freeze checklist", "",
              "- [ ] sample sizes confirmed or adjusted (say which, and why)",
              "- [ ] decoding settings confirmed (greedy, max tokens per task)",
              "- [ ] battery.jsonl committed and frozen",
              "- [ ] data/ukps/items.jsonl committed and frozen",
              "- [ ] any deviation from Chapter 3 planned values, listed "
              "for the mechanical update", ""]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"pilot report written to {args.out}")


if __name__ == "__main__":
    main()

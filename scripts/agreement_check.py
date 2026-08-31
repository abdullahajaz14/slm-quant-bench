"""scripts/agreement_check.py: validate the split-measurement design.

Chapter 3 promises that the quality-on-Colab / efficiency-on-device
split is verified, not assumed: greedy decoding should make outputs
backend-independent, and this script measures how true that is. The
number it prints goes straight into Chapter 4.

This is an ANALYSIS join, not a new experiment: the device side comes
from configs/runs/agreement_device.yaml, the Colab side from the full
quality run (a superset). Joining on (model, precision, task, item_id)
compares like with like because sampling is seed-deterministic.

Interpretation guide for Chapter 4 (keep with the results): exact rates
near 1.0 with metric deltas at zero support the design; small output
divergences with near-zero SCORE deltas still support it, since scoring
is what the claims rest on. Anything larger: investigate before
writing a word.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

from slmbench.results import bootstrap_ci


def load_records(path: str) -> dict[tuple, dict]:
    """JSONL -> {(model, precision, task, item_id): record}. Error
    records skipped (count logged); duplicate keys keep the LAST record
    (a resumed run can duplicate the boundary item)."""
    records: dict[tuple, dict] = {}
    n_errors = 0
    n_dupes = 0
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
                n_errors += 1
                continue
            key = (rec["model"], rec["precision"], rec["task"],
                   rec["item_id"])
            if key in records:
                n_dupes += 1
            records[key] = rec
    if n_errors:
        print(f"[agreement] {path}: skipped {n_errors} error record(s)")
    if n_dupes:
        print(f"[agreement] {path}: {n_dupes} duplicate key(s), "
              f"kept the last")
    return records


def _as_loglik_vector(output: str) -> list[float] | None:
    """Multiple-choice records store the per-option log-likelihood
    vector in `output`, not generated text. Returns it when that is
    what this output is, otherwise None."""
    text = output.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and parsed and \
            all(isinstance(v, (int, float)) for v in parsed):
        return parsed
    return None


def outputs_agree(device_output: str, colab_output: str) -> bool:
    """Do the two backends agree on this item?

    For generated text, string equality is the strictest and right
    test. For multiple choice it is the wrong test entirely: the stored
    output is a vector of log-likelihoods whose low-order digits differ
    between backends on essentially every item, so string equality
    reports zero agreement while the backends in fact chose the same
    option every time. Scoring uses the argmax, so the argmax is what
    agreement means here, and comparing raw strings would put a
    spurious "0 per cent agreement" for TruthfulQA into Chapter 4
    beside a metric delta of zero.
    """
    dev_vector = _as_loglik_vector(device_output)
    if dev_vector is None:
        return device_output == colab_output
    colab_vector = _as_loglik_vector(colab_output)
    if colab_vector is None or len(colab_vector) != len(dev_vector):
        return False
    return (dev_vector.index(max(dev_vector))
            == colab_vector.index(max(colab_vector)))


def compare(device: dict[tuple, dict], colab: dict[tuple, dict]) -> dict:
    joined = sorted(set(device) & set(colab))
    counts = {"device_only": len(set(device) - set(colab)),
              "colab_only": len(set(colab) - set(device)),
              "joined": len(joined)}
    if device and counts["joined"] < 0.9 * len(device):
        print(f"[agreement] WARNING: joined {counts['joined']} is under "
              f"90% of the {len(device)} device keys; check that both "
              f"runs used the same seeds and configs")

    per_task: dict[str, dict] = {}
    buckets: dict[str, list[tuple]] = defaultdict(list)
    for key in joined:
        buckets[key[2]].append(key)
    for task, keys in sorted(buckets.items()):
        exact = sum(1 for k in keys
                    if outputs_agree(device[k]["output"],
                                     colab[k]["output"]))
        metric_deltas: dict[str, list[float]] = defaultdict(list)
        for k in keys:
            for metric, dev_value in device[k]["scores"].items():
                if metric in colab[k]["scores"]:
                    metric_deltas[metric].append(
                        abs(dev_value - colab[k]["scores"][metric]))
        # Two different quantities, and conflating them misleads in
        # both directions. The absolute delta measures per-item
        # reproducibility: how often the two backends produce a
        # different score for the same item. The SIGNED delta measures
        # bias: whether those differences push the mean one way or
        # cancel out. This study's claims are about means, and per-item
        # scatter inflates the variance of a mean without shifting it,
        # so only the signed component can move a reported effect.
        #
        # CNN/DailyMail shows why both are needed: outputs differ on
        # most items, yet the mean ROUGE-L differs by under 0.001,
        # because the differences are symmetric. Judging it by scatter
        # alone would discard sound results. CUAD shows the converse,
        # scatter and bias of nearly equal size, meaning its
        # disagreements mostly point one way and genuinely can move a
        # mean.
        signed: dict[str, list[float]] = defaultdict(list)
        for k in keys:
            for metric, dev_value in device[k]["scores"].items():
                if metric in colab[k]["scores"]:
                    signed[metric].append(dev_value
                                          - colab[k]["scores"][metric])
        metrics = {}
        for m, d in sorted(metric_deltas.items()):
            bias, lo, hi = bootstrap_ci(signed[m])
            metrics[m] = {
                "mean_abs_delta": sum(d) / len(d),
                "max_abs_delta": max(d),
                "bias": bias,
                "bias_lo": lo,
                "bias_hi": hi,
                # The half-width of the bias interval is what a
                # reported effect has to clear: within it, the effect
                # could be produced by the choice of backend alone.
                "bias_floor": max(abs(lo), abs(hi)),
                "biased": lo > 0 or hi < 0,
            }
        per_task[task] = {
            "n": len(keys),
            # strictest possible statement: raw string equality
            "exact_output_rate": exact / len(keys),
            "metrics": metrics,
        }
    return {"counts": counts, "per_task": per_task}


def resolvability(summary: dict, degradation_csv: str,
                  min_n: int = 50) -> list[dict]:
    """Set each reported quantisation effect against the backend floor.

    An agreement rate on its own does not say whether the split design
    is safe, because "how much do the backends differ" only means
    something next to "how large are the effects being claimed". If a
    reported degradation is smaller than the disagreement between the
    two backends measuring it, that effect cannot be attributed to
    quantisation rather than to where it was measured, however tight
    its confidence interval looks. The interval describes sampling
    variation across items; it is silent about the backend.

    So for each task and metric this pairs the mean absolute
    cross-backend delta, treated as a noise floor, with the magnitude
    of each quantisation effect on that same task and metric, and marks
    the effect resolvable only when it clears the floor. Effects that
    do not clear it are still reported in the results, and reported as
    not separable from backend variation, which is the honest reading
    rather than a quiet omission.
    """
    # A floor is only usable once its task has been measured in full.
    # Part-way through a run a task can show perfect agreement simply
    # because the few items compared so far happened to agree, and a
    # floor of zero certifies every effect as resolvable. That would
    # turn an incomplete check into a blanket endorsement, which is the
    # opposite of what it is for, so short tasks are excluded and the
    # exclusion is reported.
    floors: dict[tuple[str, str], float] = {}
    for task, stats in summary["per_task"].items():
        if stats["n"] < min_n:
            print(f"[agreement] {task}: only {stats['n']} joined items, "
                  f"under the {min_n} needed to establish a floor; "
                  f"effects on this task are left unjudged")
            continue
        for metric, d in stats["metrics"].items():
            floors[(task, metric)] = d["bias_floor"]

    rows: list[dict] = []
    with open(degradation_csv) as f:
        for row in csv.DictReader(f):
            if not row["delta_vs_fp16"]:
                continue
            key = (row["task"], row["metric"])
            if key not in floors:
                continue  # task not covered by the agreement subset
            effect = abs(float(row["delta_vs_fp16"]))
            floor = floors[key]
            rows.append({
                "model": row["model"], "task": row["task"],
                "metric": row["metric"], "precision": row["precision"],
                "effect": float(row["delta_vs_fp16"]),
                "backend_floor": floor,
                "distinguishable": row.get("distinguishable") == "True",
                "resolvable": effect > floor,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True,
                        help="results/agreement-device.jsonl")
    parser.add_argument("--colab", required=True,
                        help="results/quality-colab.jsonl")
    parser.add_argument("--degradation", default=None,
                        help="results/degradation.csv; when given, each "
                             "reported effect is set against the "
                             "cross-backend floor for its task and metric")
    parser.add_argument("--min-floor-n", type=int, default=50,
                        help="joined items a task needs before its "
                             "cross-backend floor is used to judge effects")
    parser.add_argument("--out", default="results/agreement_summary.json")
    args = parser.parse_args()

    summary = compare(load_records(args.device), load_records(args.colab))
    print(json.dumps(summary, indent=2))

    if args.degradation:
        rows = resolvability(summary, args.degradation,
                             args.min_floor_n)
        summary["resolvability"] = rows
        at_risk = [r for r in rows if r["distinguishable"]
                   and not r["resolvable"]]
        print(f"\n{'model':<11} {'task':<16} {'metric':<7} {'prec':<7} "
              f"{'effect':>8} {'floor':>8}  verdict")
        for r in sorted(rows, key=lambda r: (r["task"], r["model"])):
            verdict = ("resolvable" if r["resolvable"]
                       else "within backend variation")
            flag = " <-- REPORTED AS DISTINGUISHABLE" if (
                r["distinguishable"] and not r["resolvable"]) else ""
            print(f"{r['model']:<11} {r['task']:<16} {r['metric']:<7} "
                  f"{r['precision']:<7} {r['effect']:>+8.4f} "
                  f"{r['backend_floor']:>8.4f}  {verdict}{flag}")
        if at_risk:
            print(f"\n[agreement] {len(at_risk)} effect(s) reported as "
                  f"distinguishable do not clear the cross-backend floor "
                  f"for their task. These must be stated as not separable "
                  f"from backend variation, not as quantisation effects.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[agreement] summary written to {args.out}")


if __name__ == "__main__":
    main()

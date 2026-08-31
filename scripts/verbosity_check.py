"""scripts/verbosity_check.py: is a measured change real, or just form?

Token-overlap F1 rewards an answer whose wording is close to the gold
span. On short-answer extraction the gold span is a few words, so an
answer that is correct but phrased as a full sentence is penalised for
the words it adds, while a terser answer carrying the same fact scores
higher. The score therefore mixes two things: whether the model got the
answer right, and how verbosely it chose to say so.

That matters here because quantisation changes output length. If a
quantised model becomes terser, its F1 rises without it having become
any more accurate, and a practitioner reading the F1 column alone would
conclude that compression improved the model.

This script separates the two. Alongside F1 it computes CONTAINMENT: 1
if any gold answer appears as a substring of the normalised output, 0
otherwise. Containment ignores added words, so it is insensitive to
verbosity while remaining sensitive to correctness. Comparing the paired
FP16 deltas under both scorers says which of the two a measured change
was:

  F1 moves, containment does not   -> a change in form, not in accuracy
  both move together               -> a real change in accuracy
  containment moves, F1 does not   -> a real change F1 is masking

Containment is deliberately a coarser scorer than F1 and is not proposed
as a replacement for it: an output that contains the gold span inside a
wrong or contradictory sentence still counts as correct here. It is used
only as a control, to test whether a difference already measured by F1
survives when verbosity is held constant.

Applies to the short-answer extraction tasks only; ROUGE and the
multiple-choice metrics are not token-F1 and are unaffected.

Run from the repository root with the venv active:
  python scripts/verbosity_check.py --jsonl results/quality-colab.jsonl \
      results/ukps-colab.jsonl --items data/ukps/items.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

from slmbench.results import bootstrap_ci

# Short-answer extraction tasks, the ones scored by token F1 against a
# span of a few words. Summarisation and multiple choice are excluded.
EXTRACTION_TASKS = ("cuad", "hotpotqa", "ukps_qa")
REFERENCE = "fp16"


def normalise(text: str) -> str:
    """SQuAD-style normalisation: casefold, drop articles and
    punctuation, collapse whitespace. Curly apostrophes are folded to
    straight ones first, since model outputs and the corpus disagree
    about which they use and the difference is not meaningful."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_gold(output: str, answers: list[str]) -> float:
    norm_output = normalise(output)
    return float(any(normalise(a) in norm_output
                     for a in answers if str(a).strip()))


def classify(f1_delta: float, f1_lo: float, f1_hi: float,
             ct_delta: float, ct_lo: float, ct_hi: float) -> str:
    """Compare the two scorers' paired deltas.

    The tempting rule, "F1 moved and containment did not, so the change
    was only in form", is wrong, because a containment interval can
    fail to exclude zero simply for want of power. Containment is
    binary where F1 is continuous, so on a few hundred items its
    interval is much the wider of the two, and an effect of the very
    size F1 reports can sit comfortably inside it. Read naively, that
    absence of evidence becomes evidence of absence and a real
    degradation gets written up as a harmless change of phrasing.

    So attributing a change to form requires containment to positively
    rule the effect out: its interval must exclude zero AND exclude the
    F1 estimate. Where the interval is too wide to do either, the
    honest answer is that the check was inconclusive, and the case is
    reported as such rather than resolved in whichever direction
    happens to suit.
    """
    f1_sig = f1_lo > 0 or f1_hi < 0
    ct_sig = ct_lo > 0 or ct_hi < 0
    if f1_sig and ct_sig:
        return ("real" if (f1_delta > 0) == (ct_delta > 0)
                else "contradictory")
    if ct_sig and not f1_sig:
        return "masked by F1"
    if f1_sig and not ct_sig:
        # Does containment actually rule out an effect the size F1 claims?
        if not (ct_lo <= f1_delta <= ct_hi):
            return "form only"
        return "inconclusive"
    return "no change detected"


def load_gold(paths: list[str]) -> dict[str, list[str]]:
    """item_id -> accepted answers, from the run records' own gold
    fields where present and from the curated corpus items file."""
    gold: dict[str, list[str]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = rec.get("id") or rec.get("item_id")
                answers = rec.get("answers") or rec.get("gold")
                if key and answers:
                    gold[key] = (answers if isinstance(answers, list)
                                 else [answers])
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, nargs="+")
    parser.add_argument("--items", nargs="*", default=[],
                        help="item files supplying gold answers")
    parser.add_argument("--out", default="results/verbosity_check.csv")
    args = parser.parse_args()

    gold = load_gold(args.items)

    # (model, task, precision) -> {item_id: record}
    runs: dict[tuple, dict] = defaultdict(dict)
    for path in args.jsonl:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec["task"] not in EXTRACTION_TASKS:
                    continue
                key = (rec["model"], rec["task"], rec["precision"])
                runs[key][rec["item_id"]] = rec

    rows = []
    models = sorted({k[0] for k in runs})
    tasks = sorted({k[1] for k in runs})
    for model in models:
        for task in tasks:
            base = runs.get((model, task, REFERENCE))
            if not base:
                continue
            missing = [i for i in base if i not in gold]
            if missing:
                print(f"[verbosity] {model}/{task}: no gold answers for "
                      f"{len(missing)} of {len(base)} items; skipped")
                continue
            for precision in sorted({k[2] for k in runs
                                     if k[0] == model and k[1] == task}):
                cur = runs[(model, task, precision)]
                ids = sorted(set(cur) & set(base))
                f1 = [cur[i]["scores"]["f1"] for i in ids]
                contains = [contains_gold(cur[i]["output"], gold[i])
                            for i in ids]
                length = [len(cur[i]["output"].split()) for i in ids]
                row = {"model": model, "task": task, "precision": precision,
                       "n": len(ids),
                       "mean_f1": statistics.mean(f1),
                       "mean_contains": statistics.mean(contains),
                       "mean_words": statistics.mean(length),
                       "delta_f1": "", "delta_f1_lo": "", "delta_f1_hi": "",
                       "delta_contains": "", "delta_contains_lo": "",
                       "delta_contains_hi": "", "verdict": ""}
                if precision != REFERENCE:
                    d_f1 = [cur[i]["scores"]["f1"] - base[i]["scores"]["f1"]
                            for i in ids]
                    d_ct = [contains_gold(cur[i]["output"], gold[i])
                            - contains_gold(base[i]["output"], gold[i])
                            for i in ids]
                    m1, lo1, hi1 = bootstrap_ci(d_f1)
                    m2, lo2, hi2 = bootstrap_ci(d_ct)
                    verdict = classify(m1, lo1, hi1, m2, lo2, hi2)
                    row |= {"delta_f1": m1, "delta_f1_lo": lo1,
                            "delta_f1_hi": hi1, "delta_contains": m2,
                            "delta_contains_lo": lo2,
                            "delta_contains_hi": hi2, "verdict": verdict}
                rows.append(row)

    fields = list(rows[0]) if rows else []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    print(f"\n{'model':<11} {'task':<9} {'prec':<7} {'F1':>7} {'cont':>7} "
          f"{'words':>6} {'dF1':>9} {'dContainment (95% CI)':>26}  verdict")
    for r in rows:
        d1 = d2 = ""
        if r["delta_f1"] != "":
            star = "*" if (r["delta_f1_lo"] > 0
                           or r["delta_f1_hi"] < 0) else " "
            d1 = f"{r['delta_f1']:+.4f}{star}"
            d2 = (f"{r['delta_contains']:+.4f} "
                  f"[{r['delta_contains_lo']:+.3f},"
                  f"{r['delta_contains_hi']:+.3f}]")
        print(f"{r['model']:<11} {r['task']:<9} {r['precision']:<7} "
              f"{r['mean_f1']:>7.4f} {r['mean_contains']:>7.4f} "
              f"{r['mean_words']:>6.1f} {d1:>9} {d2:>26}  {r['verdict']}")
    print("\n* = F1 delta interval excludes zero. A verdict of 'form "
          "only' additionally requires the\ncontainment interval to "
          "exclude the F1 estimate, so it cannot be reached merely "
          "because\ncontainment lacked the power to detect the effect.")
    print(f"\n[verbosity] {len(rows)} rows written to {args.out}")


if __name__ == "__main__":
    main()

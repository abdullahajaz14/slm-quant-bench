"""scripts/export_gold.py: recover the gold answers for a sampled task.

The run records store each model's output and its scores, but not the
reference answers the scores were computed against. Any later analysis
that needs to rescore an existing run, rather than simply re-read its
numbers, therefore has nothing to compare against. The verbosity
control in verbosity_check.py is the case in point.

Re-running the models to capture the references would be wasteful and
would not reproduce the same outputs anyway. It is unnecessary: the
sampling is seed-controlled and stratified by a fixed rule, so loading
a task through its adapter a second time yields exactly the same items
in the same order. This script does that and writes out the item
identifiers with their references. No model is loaded and no inference
is run.

The output is the same shape as the curated corpus items file, so both
can be passed to --items together.

Note that this downloads the source datasets if they are not already
cached, which is disk and network activity: do not run it on the
measurement machine while an efficiency battery is in progress, since
the battery reports cold-start load times and peak memory.

Verification: the exported identifiers are checked against those
present in a results file when one is supplied, so a silent mismatch
between the reconstruction and the actual run cannot pass unnoticed.

Run from the repository root with the venv active:
  python scripts/export_gold.py --tasks configs/tasks/cuad.yaml \
      configs/tasks/hotpotqa.yaml --out results/gold.jsonl \
      --verify-against results/quality-colab.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import slmbench.adapters  # noqa: F401  (registers every adapter)
from slmbench.adapters import base as ab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, nargs="+",
                        help="task config YAML files")
    parser.add_argument("--out", default="results/gold.jsonl")
    parser.add_argument("--verify-against", default=None,
                        help="results JSONL whose item ids must match")
    args = parser.parse_args()

    exported: dict[str, set[str]] = defaultdict(set)
    written = 0
    with open(args.out, "w") as out:
        for task_path in args.tasks:
            cfg = ab.TaskConfig.from_yaml(task_path)
            # Same resolution the runner uses, so the adapter and the
            # sampling are identical to the ones that produced the run.
            adapter_cls = ab.ADAPTERS[cfg.extra.get("adapter", cfg.name)]
            items = adapter_cls().load(cfg)
            for item in items:
                out.write(json.dumps({
                    "id": item.item_id,
                    "task": item.task,
                    "answers": item.references,
                }, ensure_ascii=False) + "\n")
                exported[item.task].add(item.item_id)
                written += 1
            print(f"[gold] {cfg.name}: {len(items)} items")

    if args.verify_against:
        seen: dict[str, set[str]] = defaultdict(set)
        with open(args.verify_against) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["task"] in exported:
                    seen[rec["task"]].add(rec["item_id"])
        ok = True
        for task, ids in seen.items():
            missing = ids - exported[task]
            extra = exported[task] - ids
            status = "ok" if not missing and not extra else "MISMATCH"
            if missing or extra:
                ok = False
            print(f"[gold] {task}: {status} "
                  f"({len(ids)} in results, {len(exported[task])} exported, "
                  f"{len(missing)} unmatched, {len(extra)} unused)")
        if not ok:
            raise SystemExit(
                "[gold] reconstruction does not match the run: the sample "
                "is not reproducing, so any rescoring built on it would "
                "compare against the wrong answers")

    print(f"[gold] {written} items written to {args.out}")


if __name__ == "__main__":
    main()

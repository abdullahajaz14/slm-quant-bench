"""slmbench.results: result records, resumability, bootstrap summaries.

One JSONL line per item per configuration; summaries are derived, never
primary. The bootstrap confidence interval is Chapter 3's statistical
commitment; its algorithm follows the contract exactly.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any


def sha256_of(obj: Any) -> str:
    """Stable hash of any JSON-serialisable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


@dataclass
class Record:
    run_id: str
    ts: float
    env: dict[str, Any]
    model: str
    precision: str
    task: str
    item_id: str
    prompt_sha256: str
    output: str | None
    scores: dict[str, float]
    timings: dict[str, float]
    config_sha256: str
    error: str | None = None


# pip distribution names for the packages recorded in every env snapshot
_TRACKED_PACKAGES = {
    "llama_cpp_python": "llama-cpp-python",
    "datasets": "datasets",
    "rouge_score": "rouge-score",
}


def capture_env(backend_tag: str) -> dict[str, Any]:
    """Environment snapshot stored on every record.

    Contract (fixed): {host, os, python, backend, versions{}}; absent
    packages recorded as null, never a crash here.
    """
    versions: dict[str, str | None] = {}
    for key, dist in _TRACKED_PACKAGES.items():
        try:
            versions[key] = importlib.metadata.version(dist)
        except Exception:
            versions[key] = None
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "backend": backend_tag,
        "versions": versions,
    }


class ResultsWriter:
    """Append-only JSONL with resumability."""

    def __init__(self, out_dir: str, run_id: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{run_id}.jsonl")

    def done_keys(self) -> set[tuple[str, str, str, str]]:
        """Set of (model, precision, task, item_id) already on disk.
        Corrupt trailing lines (a crash mid-write leaves one) are
        skipped with a logged warning."""
        done: set[tuple[str, str, str, str]] = set()
        if not os.path.exists(self.path):
            return done
        with open(self.path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[results] WARNING: skipping corrupt line "
                          f"{lineno} in {self.path}")
                    continue
                done.add((rec["model"], rec["precision"],
                          rec["task"], rec["item_id"]))
        return done

    def write(self, rec: Record) -> None:
        # flush per record: crash resumability depends on it
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f.flush()


def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                 seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile bootstrap 95% CI. Returns (mean, lo, hi).

    Contract (fixed): rng = random.Random(seed); each resample is
    len(values) draws WITH replacement; percentile indices are
    sorted_means[int(0.025 * n)] and sorted_means[int(0.975 * n) - 1]
    (the simple index method, noted here as the contract allows).
    Empty values -> (nan, nan, nan); single value -> (v, v, v).
    """
    if not values:
        return (math.nan, math.nan, math.nan)
    if len(values) == 1:
        v = values[0]
        return (v, v, v)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(
        sum(rng.choices(values, k=k)) / k for _ in range(n_resamples))
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    mean = sum(values) / len(values)
    return (mean, lo, hi)


def summarise(jsonl_path: str, out_csv: str) -> None:
    """Derive summary.csv from the raw records.

    One row per (model, precision, task, metric) with columns:
    model, precision, task, metric, n, mean, ci_lo, ci_hi. Records with
    error != null are EXCLUDED and their count printed per
    configuration (they still live in the JSONL).
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    errors: dict[tuple, int] = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cfg_key = (rec["model"], rec["precision"], rec["task"])
            if rec.get("error"):
                errors[cfg_key] += 1
                continue
            for metric, value in rec["scores"].items():
                groups[cfg_key + (metric,)].append(value)
    for cfg_key in sorted(errors):
        print(f"[summarise] {cfg_key}: {errors[cfg_key]} error record(s) "
              f"excluded")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "task", "metric",
                         "n", "mean", "ci_lo", "ci_hi"])
        for key in sorted(groups):
            values = groups[key]
            mean, lo, hi = bootstrap_ci(values)
            writer.writerow([*key, len(values),
                             f"{mean:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

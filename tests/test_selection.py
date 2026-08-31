"""The selection stages, pinned to the study's frozen records.

The golden test reproduces the dissertation's Section 5.5 result from
results/ alone: six feasible configurations, of which exactly
llama32-3b at q8_0 and phi3-mini at q4_k_m are Pareto-dominated,
leaving a frontier of four with mistral7b q4_k_m the most accurate.
"""

import csv
import json
import os

from slmbench.selection import (apply_preferences, dominates,
                                exceeds_budget, memory_lower_bound,
                                split_frontier)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")

PUBLIC_PRIMARY = {"cuad": "f1", "hotpotqa": "f1", "cnndm": "rougeL",
                  "truthfulqa_mc1": "mc1", "truthfulqa_mc2": "mc2"}


def _frozen_configs():
    quality = {}
    with open(os.path.join(RESULTS, "quality-colab-summary.csv")) as f:
        for row in csv.DictReader(f):
            if PUBLIC_PRIMARY.get(row["task"]) == row["metric"]:
                key = (row["model"], row["precision"])
                quality.setdefault(key, []).append(float(row["mean"]))
    agg = {k: sum(v) / len(v) for k, v in quality.items() if len(v) == 5}

    configs = []
    with open(os.path.join(RESULTS, "efficiency-device.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e.get("offload") != "full" or not e.get("feasible"):
                continue
            key = (e["model"], e["precision"])
            configs.append({
                "model": e["model"], "precision": e["precision"],
                "quality": agg[key],
                "decode": e["decode_tps_median"],
                "memory": memory_lower_bound(e["peak_rss_gb"],
                                             e["file_gb"]),
            })
    return configs


def test_memory_lower_bound_takes_the_larger_component():
    assert memory_lower_bound(3.01, 4.07) == 4.07   # Mistral 7B Q4_K_M
    assert memory_lower_bound(3.77, 2.23) == 3.77   # Phi-3 Mini Q4_K_M


def test_dominates_requires_strictness():
    a = {"quality": 0.4, "decode": 10.0, "memory": 2.0}
    assert not dominates(a, dict(a))
    assert dominates({**a, "decode": 11.0}, a)
    assert not dominates({**a, "decode": 11.0, "memory": 2.5}, a)


def test_frozen_records_reproduce_the_dissertation_frontier():
    configs = _frozen_configs()
    assert len(configs) == 6, "six feasible standard configurations"

    frontier, dominated = split_frontier(configs)
    dominated_keys = {(c["model"], c["precision"]) for c in dominated}
    assert dominated_keys == {("llama32-3b", "q8_0"),
                              ("phi3-mini", "q4_k_m")}
    assert len(frontier) == 4

    best = max(frontier, key=lambda c: c["quality"])
    assert (best["model"], best["precision"]) == ("mistral7b", "q4_k_m")
    assert round(best["quality"], 3) == 0.438


def test_preferences_filter_floor_and_budget():
    frontier, _ = split_frontier(_frozen_configs())
    all_pass = apply_preferences(frontier, 0.0, 8.0)
    assert len(all_pass) == 4
    tight = apply_preferences(frontier, 0.42, 3.5)
    assert {(c["model"], c["precision"]) for c in tight} == {
        ("gemma2-2b", "q8_0")}


# --------------------------------------------------------------------
# The dashboard builds a frontier per task from degradation.csv, not
# from the aggregate and not from the per-run summaries. These tests
# read the same file by the same route, so the tested path is the
# shipped path. Writing them surfaced two things worth recording: the
# curated-corpus tasks are absent from quality-colab-summary.csv, and
# the per-task frontiers genuinely differ from the aggregate frontier
# of Section 5.5, which is why testing the aggregate alone was not
# enough.
# --------------------------------------------------------------------

PRIMARY_METRIC = {"cuad": "f1", "hotpotqa": "f1", "cnndm": "rougeL",
                  "truthfulqa_mc1": "mc1", "truthfulqa_mc2": "mc2",
                  "ukps_qa": "f1", "ukps_sum": "rougeL"}


def _per_task_configs(task):
    """Rebuild what the dashboard builds for one task family."""
    metric = PRIMARY_METRIC[task]
    means = {}
    with open(os.path.join(RESULTS, "degradation.csv")) as f:
        for row in csv.DictReader(f):
            if row["task"] == task and row["metric"] == metric:
                means[(row["model"], row["precision"])] = float(row["mean"])

    configs = []
    with open(os.path.join(RESULTS, "efficiency-device.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e.get("offload") != "full" or not e.get("feasible"):
                continue
            key = (e["model"], e["precision"])
            if key not in means:
                continue
            configs.append({
                "model": e["model"], "precision": e["precision"],
                "quality": means[key],
                "decode": float(e["decode_tps_median"]),
                "memory": memory_lower_bound(float(e["peak_rss_gb"]),
                                             float(e["file_gb"])),
            })
    return configs


def test_every_task_yields_a_well_formed_frontier():
    for task in PRIMARY_METRIC:
        configs = _per_task_configs(task)
        assert len(configs) == 6, f"{task}: six feasible configurations"
        frontier, dominated = split_frontier(configs)
        assert frontier, f"{task}: frontier must not be empty"
        assert len(frontier) + len(dominated) == 6
        for d in dominated:
            assert any(dominates(o, d) for o in frontier), (
                f"{task}: each dominated configuration is dominated by "
                f"one on the frontier")


def test_per_task_frontier_differs_from_the_aggregate():
    """Recorded because it is easy to assume otherwise: the dominated
    pair on CNN/DailyMail is not the pair Section 5.5 reports for the
    aggregate across the public benchmarks."""
    _, cnndm_dominated = split_frontier(_per_task_configs("cnndm"))
    assert {(c["model"], c["precision"]) for c in cnndm_dominated} == {
        ("phi3-mini", "q4_k_m"), ("gemma2-2b", "q8_0")}

    _, aggregate_dominated = split_frontier(_frozen_configs())
    assert {(c["model"], c["precision"]) for c in aggregate_dominated} == {
        ("llama32-3b", "q8_0"), ("phi3-mini", "q4_k_m")}


def test_cnndm_at_the_settings_shown_in_the_dissertation_figure():
    """The dashboard figure shows CNN/DailyMail at a 90 per cent floor
    and a 6 GB budget."""
    frontier, _ = split_frontier(_per_task_configs("cnndm"))
    assert len(frontier) == 4
    fp16_best = 0.236314        # best FP16 CNN/DailyMail ROUGE-L
    kept = apply_preferences(frontier, fp16_best * 0.90, 6.0)
    assert len(kept) == 4


def test_budget_only_rejects_never_certifies():
    """A lower bound above the budget excludes; one below does not
    certify, so the helper must not claim the survivors fit."""
    assert exceeds_budget(4.07, 3.0) is True
    assert exceeds_budget(2.09, 3.0) is False
    frontier = [{"model": "m", "precision": "p", "quality": 0.5,
                 "decode": 10.0, "memory": 4.07}]
    assert apply_preferences(frontier, 0.0, 3.0) == []
    assert len(apply_preferences(frontier, 0.0, 5.0)) == 1

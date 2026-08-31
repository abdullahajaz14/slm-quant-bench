"""tests/test_adapters.py: contract enforcement for the samplers.

Written in full deliberately: these tests encode MY contract from
adapters/base.py, and your implementation has to satisfy them. They use
synthetic items only; nothing here downloads a dataset. The one
integration test at the bottom is skipped unless you opt in with
RUN_DATA_TESTS=1, so pytest stays fast and offline by default.
"""

import os

import pytest

from slmbench.adapters.base import Item, deterministic_sample, stratified_sample


def make_items(spec: dict[str, int]) -> list[Item]:
    """spec: {stratum_name: count} -> synthetic items with meta.category."""
    items = []
    for stratum, count in spec.items():
        for i in range(count):
            items.append(Item(
                task="synthetic",
                item_id=f"{stratum}-{i:03d}",
                context=None,
                question="q",
                references=["r"],
                meta={"category": stratum},
            ))
    return items


# ------------------------------------------------- deterministic_sample --

def test_deterministic_sample_repeatable():
    items = make_items({"a": 50})
    ids1 = [x.item_id for x in deterministic_sample(items, 10, seed=42)]
    ids2 = [x.item_id for x in deterministic_sample(items, 10, seed=42)]
    assert ids1 == ids2
    assert len(ids1) == 10


def test_deterministic_sample_sorted_by_id():
    items = make_items({"a": 50})
    out = deterministic_sample(items, 10, seed=42)
    ids = [x.item_id for x in out]
    assert ids == sorted(ids)


def test_deterministic_sample_passthrough():
    items = make_items({"a": 5})
    assert deterministic_sample(items, -1, seed=42) == items
    assert deterministic_sample(items, 5, seed=42) == items
    assert deterministic_sample(items, 99, seed=42) == items


def test_different_seed_different_sample():
    items = make_items({"a": 200})
    ids42 = [x.item_id for x in deterministic_sample(items, 20, seed=42)]
    ids43 = [x.item_id for x in deterministic_sample(items, 20, seed=43)]
    assert ids42 != ids43


# --------------------------------------------------- stratified_sample --

def test_stratified_repeatable():
    items = make_items({"easy": 60, "medium": 90, "hard": 50})
    ids1 = [x.item_id for x in stratified_sample(items, 40, 42, "meta.category")]
    ids2 = [x.item_id for x in stratified_sample(items, 40, 42, "meta.category")]
    assert ids1 == ids2
    assert len(ids1) == 40


def test_stratified_proportions_largest_remainder():
    # 60/90/50 of 200 total, n=40 -> quotas 12.0 / 18.0 / 10.0 exactly.
    items = make_items({"easy": 60, "medium": 90, "hard": 50})
    out = stratified_sample(items, 40, 42, "meta.category")
    counts = {}
    for x in out:
        counts[x.meta["category"]] = counts.get(x.meta["category"], 0) + 1
    assert counts == {"easy": 12, "medium": 18, "hard": 10}


def test_stratified_remainder_distribution():
    # 10/10/10 of 30, n=10 -> quotas 3.333.. each; floors 3+3+3=9;
    # one remainder goes to the tie-break winner: sorted stratum name
    # order means "a" gets it.
    items = make_items({"a": 10, "b": 10, "c": 10})
    out = stratified_sample(items, 10, 42, "meta.category")
    counts = {}
    for x in out:
        counts[x.meta["category"]] = counts.get(x.meta["category"], 0) + 1
    assert counts == {"a": 4, "b": 3, "c": 3}


def test_stratified_output_sorted_by_id():
    items = make_items({"a": 30, "b": 30})
    out = stratified_sample(items, 20, 42, "meta.category")
    ids = [x.item_id for x in out]
    assert ids == sorted(ids)


# ------------------------------------------------------- integration ----

@pytest.mark.skipif(os.environ.get("RUN_DATA_TESTS") != "1",
                    reason="set RUN_DATA_TESTS=1 to run dataset tests")
def test_hotpotqa_loads_and_conforms():
    """Opt-in smoke: loads the real HotpotQA config and checks schema."""
    from slmbench.adapters import hotpotqa  # noqa: F401  (registration)
    from slmbench.adapters.base import ADAPTERS, TaskConfig
    cfg = TaskConfig.from_yaml("configs/tasks/hotpotqa.yaml")
    items = ADAPTERS["hotpotqa"]().load(cfg)
    assert len(items) == 300
    for it in items:
        assert it.context and it.question and it.references
        assert it.meta.get("level") in {"easy", "medium", "hard"}

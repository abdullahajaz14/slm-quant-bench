"""slmbench.adapters.base: item schema, task config, sampling, adapter ABC.

Item and TaskConfig are fixed schema. The two samplers are
protocol-critical: Chapter 3 promises seed-controlled, stratified
samples, so their behaviour is contracted precisely and enforced by
tests/test_adapters.py.
"""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Item:
    task: str
    item_id: str
    context: str | None
    question: str | None
    references: list[str]
    choices: list[str] = field(default_factory=list)
    choice_labels: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    name: str
    hf_dataset: str
    split: str
    sample_size: int          # -1 means take everything
    seed: int
    template: str
    max_output_tokens: int
    scorer: str
    stratify_by: str | None = None
    hf_config: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TaskConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = {k: raw.pop(k) for k in list(raw)
                 if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known, extra=raw)


def _lookup(item: Item, dotted: str) -> Any:
    """Resolve a dotted path like "meta.category" against an Item:
    attributes first, then mapping keys."""
    obj: Any = item
    for part in dotted.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            obj = obj[part]
    return obj


def deterministic_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    """Simple seed-controlled sample.

    Contract (fixed):
      - if n < 0 or n >= len(items): return items unchanged (same order).
      - else: random.Random(seed).sample(items, n), then sort the result
        by item_id so output order never depends on sampling internals.
    """
    if n < 0 or n >= len(items):
        return items
    picked = random.Random(seed).sample(items, n)
    return sorted(picked, key=lambda it: it.item_id)


def stratified_sample(items: list[Item], n: int, seed: int, key: str) -> list[Item]:
    """Proportional stratified sample with largest-remainder allocation.

    `key` is a dotted path into the item, e.g. "meta.category".

    Contract (fixed, exactly reproducible):
      1. group items by stratum value; iterate strata in SORTED order.
      2. quota_s = n * len(stratum) / N; base allocation = floor(quota_s).
      3. distribute the remaining (n - sum of floors) one each to strata
         by largest fractional part; ties broken by sorted stratum name.
      4. within each stratum: ONE rng instance created before the loop,
         reused across strata (sorted visit order keeps this
         deterministic).
      5. concatenate, then sort by item_id.
      6. log: strata count, min/max allocation, any stratum with 0.
    """
    if n < 0 or n >= len(items):
        return items
    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(str(_lookup(it, key)), []).append(it)
    names = sorted(groups)
    total = len(items)
    quotas = {s: n * len(groups[s]) / total for s in names}
    alloc = {s: math.floor(quotas[s]) for s in names}
    remaining = n - sum(alloc.values())
    by_fraction = sorted(names, key=lambda s: (-(quotas[s] - alloc[s]), s))
    for s in by_fraction[:remaining]:
        alloc[s] += 1
    rng = random.Random(seed)
    out: list[Item] = []
    for s in names:
        out.extend(rng.sample(groups[s], alloc[s]))
    out.sort(key=lambda it: it.item_id)
    counts = [alloc[s] for s in names]
    zeros = [s for s in names if alloc[s] == 0]
    msg = (f"[stratified] {len(names)} strata, "
           f"alloc min={min(counts)} max={max(counts)}")
    if zeros:
        msg += f", empty allocation for: {zeros}"
    print(msg)
    return out


class TaskAdapter(abc.ABC):
    """One adapter per dataset. Subclasses set NAME and implement _load_raw
    and _to_item; load() below is the shared pipeline.
    """

    NAME: str = "abstract"

    def load(self, cfg: TaskConfig) -> list[Item]:
        """Fixed pipeline: raw rows -> Items -> drop malformed -> sample.

        _to_item may return an Item, a list of Items (TruthfulQA emits
        two per row), or None (counted as dropped).
        """
        rows = self._load_raw(cfg)
        items: list[Item] = []
        dropped = 0
        for row in rows:
            converted = self._to_item(row, cfg)
            if converted is None:
                dropped += 1
            elif isinstance(converted, list):
                items.extend(converted)
            else:
                items.append(converted)
        print(f"[{self.NAME}] kept {len(items)}, dropped {dropped}")
        if cfg.stratify_by:
            return stratified_sample(items, cfg.sample_size, cfg.seed,
                                     cfg.stratify_by)
        return deterministic_sample(items, cfg.sample_size, cfg.seed)

    @abc.abstractmethod
    def _load_raw(self, cfg: TaskConfig):
        ...

    @abc.abstractmethod
    def _to_item(self, row: Any, cfg: TaskConfig):
        ...


# Filled in by each adapter module at import time:
ADAPTERS: dict[str, type[TaskAdapter]] = {}


def register(cls: type[TaskAdapter]) -> type[TaskAdapter]:
    ADAPTERS[cls.NAME] = cls
    return cls

"""slmbench.adapters.ukps: the curated UK public-sector corpus (OGL).

One adapter, two task configs: ukps_qa.yaml and ukps_sum.yaml both
point here and select rows via extra["variant"] ("qa" | "summary").
Items come from data/ukps/items.jsonl, authored under
data/ukps/PROTOCOL.md.

Design point: this adapter ENFORCES the construction protocol rather
than trusting it. QA answers must appear verbatim in their passage
(the span-based rule from Chapter 3, which is what makes token F1
meaningful), and the adapter fails loudly listing every offending id.
A protocol that code checks is a protocol an examiner can believe.
"""

from __future__ import annotations

import json
import os

from .base import Item, TaskAdapter, TaskConfig, register

_ITEMS_PATH = os.path.join("data", "ukps", "items.jsonl")


@register
class UkpsAdapter(TaskAdapter):
    NAME = "ukps"

    def load(self, cfg: TaskConfig) -> list[Item]:
        self._violations: list[str] = []
        items = super().load(cfg)
        if self._violations:
            raise ValueError(
                "UKPS protocol violations (answer not verbatim in "
                f"passage, or empty answers): {self._violations}")
        return items

    def _load_raw(self, cfg: TaskConfig):
        if not os.path.exists(_ITEMS_PATH):
            raise FileNotFoundError(
                f"{_ITEMS_PATH} not found: author the corpus per "
                "data/ukps/PROTOCOL.md before running UKPS tasks")
        variant = cfg.extra["variant"]
        wanted = "qa" if variant == "qa" else "summary"
        with open(_ITEMS_PATH) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if r["type"] == wanted]

    def _to_item(self, row, cfg: TaskConfig) -> Item | None:
        if row["type"] == "qa":
            answers = row.get("answers") or []
            bad = not answers or any(a not in row["passage"] for a in answers)
            if bad:
                # collected, then raised at the end of load(): never
                # silently dropped, silent drops would hide protocol
                # failures
                self._violations.append(row["id"])
                return None
            return Item(
                task="ukps_qa",
                item_id=row["id"],
                context=row["passage"],
                question=row["question"],
                references=answers,
                meta={"source_doc": row.get("source_doc", "")},
            )
        # summary rows
        reference = row["reference_summary"]
        n_words = len(reference.split())
        if not 40 <= n_words <= 120:
            # warn, not fail: borderline references are reviewable,
            # missing spans are not
            print(f"[ukps] WARNING: {row['id']} reference summary is "
                  f"{n_words} words (protocol band 40-120)")
        return Item(
            task="ukps_sum",
            item_id=row["id"],
            context=row["passage"],
            question=None,
            references=[reference],
            meta={"source_doc": row.get("source_doc", "")},
        )

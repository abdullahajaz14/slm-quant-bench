"""slmbench.adapters.truthfulqa: TruthfulQA multiple-choice (MC1 + MC2).

Design: each dataset row becomes TWO Items, one per metric, because MC1
and MC2 are defined over different option sets (mc1_targets has exactly
one true option; mc2_targets has several). Separate items keep the
runner and results schema uniform: one record per item per
configuration, no special cases downstream. All 817 questions are used
(sample_size -1); choice scoring is cheap, so no sampling is needed.

This task runs in choice mode: truthfulqa.yaml carries mode: choice,
and the runner's task split keys off task_cfg.extra["mode"].
"""

from __future__ import annotations

from .base import Item, TaskAdapter, TaskConfig, register


@register
class TruthfulQAAdapter(TaskAdapter):
    NAME = "truthfulqa"

    def _load_raw(self, cfg: TaskConfig):
        import datasets
        ds = datasets.load_dataset(cfg.hf_dataset, cfg.hf_config,
                                   split=cfg.split)
        # rows are wrapped as (idx, row) so _to_item can build a stable
        # id; base.load() passes each tuple through unchanged.
        return enumerate(ds)

    def _to_item(self, row, cfg: TaskConfig) -> list[Item]:
        idx, record = row
        out: list[Item] = []
        for kind in ("mc1", "mc2"):
            targets = record[f"{kind}_targets"]
            choices = list(targets["choices"])
            labels = list(targets["labels"])
            # sanity asserts, fail loudly
            assert len(choices) == len(labels), \
                f"tqa{idx}: {kind} choices/labels length mismatch"
            if kind == "mc1":
                assert sum(labels) == 1, f"tqa{idx}: mc1 labels must sum to 1"
            else:
                assert sum(labels) >= 1, f"tqa{idx}: mc2 labels must sum >= 1"
            out.append(Item(
                task=f"truthfulqa_{kind}",
                item_id=f"tqa{idx:03d}-{kind}",
                context=None,
                question=record["question"],
                references=[],
                choices=choices,
                choice_labels=labels,
                meta={"category": record.get("category", "")},
            ))
        return out

"""slmbench.adapters.cuad: CUAD legal contract QA (SQuAD format).

Protocol note (carried into the pilot-freeze record): CUAD contexts are
full contracts, far beyond the 4096-token window. This adapter uses an
ANSWER-CENTRED CHARACTER WINDOW (window_chars in cuad.yaml, default
8000). Justification: it emulates the retrieval stage of enterprise
document QA, so the experiment isolates reading and extraction under
quantisation rather than long-context search, which no 4k model could
do anyway. The window is deterministic (centred on the first gold
span), identical across all twelve configurations, so comparisons
remain fair.
"""

from __future__ import annotations

from .base import Item, TaskAdapter, TaskConfig, register


@register
class CuadAdapter(TaskAdapter):
    NAME = "cuad"

    def load(self, cfg: TaskConfig) -> list[Item]:
        self._categories: set[str] = set()
        self._lost_spans = 0
        items = super().load(cfg)
        print(f"[cuad] distinct clause categories: {len(self._categories)} "
              f"(expect around 41); items that lost window-straddling "
              f"spans: {self._lost_spans}")
        return items

    def _load_raw(self, cfg: TaskConfig):
        # The CUAD repository ships a legacy loading script, which the
        # datasets library no longer executes, so the run config points
        # at the Hub's parquet export via `revision`. Recorded in the
        # README as the dataset id actually used.
        import datasets
        kwargs = {"split": cfg.split}
        if cfg.extra.get("revision"):
            kwargs["revision"] = cfg.extra["revision"]
        return datasets.load_dataset(cfg.hf_dataset, **kwargs)

    def _to_item(self, row, cfg: TaskConfig) -> Item | None:
        answers = list(row["answers"]["text"])
        if not answers:
            return None  # unanswerable; counted as dropped, per spec
        first_span_start = row["answers"]["answer_start"][0]
        context = row["context"]
        window_chars = cfg.extra.get("window_chars", 8000)
        start = max(0, first_span_start - window_chars // 2)
        end = min(len(context), start + window_chars)
        if end - start < window_chars:
            start = max(0, end - window_chars)  # keep the window full width
        window = context[start:end]
        kept = [a for a in answers if a in window]
        if len(kept) < len(answers):
            self._lost_spans += 1
        if not kept:
            return None
        category = row["id"].rsplit("__", 1)[-1]
        self._categories.add(category)
        return Item(
            task="cuad",
            item_id=row["id"],
            context=window,
            question=row["question"],
            references=kept,
            meta={"category": category},
        )

"""slmbench.adapters.cnndm: CNN/DailyMail summarisation (non-anonymised)."""

from __future__ import annotations

from .base import Item, TaskAdapter, TaskConfig, register


@register
class CnnDmAdapter(TaskAdapter):
    NAME = "cnndm"

    def load(self, cfg: TaskConfig) -> list[Item]:
        self._truncated = 0
        items = super().load(cfg)
        # CNN articles are lead-biased, so tail truncation is defensible;
        # if the pilot shows more than ~10% truncated, flag it for the
        # freeze note.
        print(f"[cnndm] articles truncated to max_context_chars: "
              f"{self._truncated}")
        return items

    def _load_raw(self, cfg: TaskConfig):
        import datasets
        return datasets.load_dataset(cfg.hf_dataset, cfg.hf_config,
                                     split=cfg.split)

    def _to_item(self, row, cfg: TaskConfig) -> Item | None:
        article = row.get("article")
        highlights = row.get("highlights")
        if not article or not highlights:
            return None
        max_chars = cfg.extra.get("max_context_chars", 14000)
        if len(article) > max_chars:
            self._truncated += 1
            article = article[:max_chars]
        return Item(
            task="cnndm",
            item_id=row["id"],
            context=article,
            question=None,
            references=[highlights],  # newline-separated bullet sentences
            meta={},
        )

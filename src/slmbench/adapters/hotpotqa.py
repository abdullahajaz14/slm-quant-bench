"""slmbench.adapters.hotpotqa: HotpotQA multi-hop QA, distractor setting."""

from __future__ import annotations

from .base import Item, TaskAdapter, TaskConfig, register

_MAX_CONTEXT_CHARS = 14000  # keeps the rendered prompt inside 4096 tokens


@register
class HotpotQAAdapter(TaskAdapter):
    NAME = "hotpotqa"

    def _load_raw(self, cfg: TaskConfig):
        import datasets
        return datasets.load_dataset(cfg.hf_dataset, cfg.hf_config,
                                     split=cfg.split)

    def _to_item(self, row, cfg: TaskConfig) -> Item | None:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        # paragraph order exactly as given: the distractor mix is part
        # of the benchmark
        paragraphs = [f"Title: {title}\n{' '.join(sents)}"
                      for title, sents in zip(titles, sentences)]
        context = "\n\n".join(paragraphs)
        if len(context) > _MAX_CONTEXT_CHARS:
            print(f"[hotpotqa] truncating context of {row['id']}")
            context = context[:_MAX_CONTEXT_CHARS]
        return Item(
            task="hotpotqa",
            item_id=row["id"],
            context=context,
            question=row["question"],
            references=[row["answer"]],  # includes yes/no, scored as text
            meta={"level": row["level"], "type": row["type"]},
        )

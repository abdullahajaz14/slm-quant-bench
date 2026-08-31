"""slmbench.scoring.rouge: ROUGE-1/2/L via the rouge-score package.

Decision (stated once in Chapter 4): plain rougeL on raw strings, not
rougeLsum with sentence splitting. Absolute values are therefore not
directly comparable with leaderboard tables, but the study's estimand
is the DIFFERENCE between precisions within model and task, and one
simple, deterministic scoring rule held constant across all twelve
configurations keeps those comparisons internally valid.
"""

from __future__ import annotations

from rouge_score import rouge_scorer

_SCORER = None  # module-level singleton; construction is not free


def rouge(pred: str, refs: list[str]) -> dict[str, float]:
    global _SCORER
    if _SCORER is None:
        _SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    if not pred.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    # ARGUMENT ORDER: scorer.score(target, prediction) -- the reference
    # goes FIRST; reversing silently swaps precision and recall.
    scores = _SCORER.score(refs[0], pred)
    return {"rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure}

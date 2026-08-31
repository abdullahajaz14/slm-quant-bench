"""slmbench.scoring.qa: exact match and token F1, SQuAD rules.

These formulas match the published SQuAD scoring exactly, or the
numbers stop being comparable with the literature. Golden tests in
tests/test_scoring.py pin the behaviour.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    """SQuAD answer normalisation, in this exact order:
      1. lowercase
      2. remove punctuation (every char in string.punctuation)
      3. remove articles: r"\\b(a|an|the)\\b" -> " "
      4. collapse whitespace
    Order matters: removing punctuation before articles means "the,"
    becomes "the" and is then removed.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def em(pred: str, refs: list[str]) -> int:
    """1 if normalize(pred) equals normalize(r) for ANY r in refs, else 0."""
    return int(any(normalize(pred) == normalize(r) for r in refs))


def f1(pred: str, refs: list[str]) -> float:
    """Max token-level F1 over references (SQuAD convention)."""
    def one(ref: str) -> float:
        p_toks = normalize(pred).split()
        r_toks = normalize(ref).split()
        if not p_toks and not r_toks:
            return 1.0          # both blank counts as match
        if not p_toks or not r_toks:
            return 0.0
        common = Counter(p_toks) & Counter(r_toks)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(p_toks)
        recall = n_common / len(r_toks)
        return 2 * precision * recall / (precision + recall)

    return max(one(r) for r in refs)

"""tests/test_scoring.py: golden tests for every scorer.

Expected values are hand-computed from the contracts in the scoring
modules (worked examples inline), so the implementation and the
computed expectation verify each other.

Workings for the filled values:
  partial overlap: pred tokens [london, england], ref [london];
    common=1, P=1/2, R=1/1, F1=2*(1/2*1)/(3/2)=2/3; EM=0.
  multiple refs: "42 percent" normalises equal to the second ref, so
    EM=1 and max-F1=1.0.
  articles/punctuation: "an apple!" -> "apple" (punctuation stripped,
    article removed) == "apple"; EM=1, F1=1.0.
  mc2 stability: subtract max 1000 -> exps [1, e^-1]=[1, 0.367879],
    z=1.367879, mass on label 1 = 1/1.367879 = 0.731059.
  mc2 split: equal logliks, three options, two true -> 2/3.
"""

import math

import pytest

from slmbench.scoring import qa, choice
from slmbench.scoring.rouge import rouge
from slmbench.results import bootstrap_ci


# ---------------------------------------------------------------- QA ----

def test_qa_exact_match_normalised():
    # WORKED EXAMPLE: "The Prime Minister" -> lowercase -> punctuation
    # unchanged -> article "the" removed -> "prime minister".
    # Reference normalises identically, so EM=1 and F1=1.0.
    assert qa.em("The Prime Minister", ["the prime minister"]) == 1
    assert qa.f1("The Prime Minister", ["the prime minister"]) == 1.0


def test_qa_partial_overlap():
    # pred tokens after normalisation: ["london", "england"]
    # ref tokens: ["london"]; common = 1
    # precision 1/2, recall 1/1, F1 = 2*(1/2)*(1)/(3/2) = 2/3
    assert qa.em("London, England", ["London"]) == 0
    assert qa.f1("London, England", ["London"]) == pytest.approx(2 / 3)


def test_qa_no_overlap():
    assert qa.em("Paris", ["Berlin"]) == 0
    assert qa.f1("Paris", ["Berlin"]) == 0.0


def test_qa_multiple_refs_takes_max():
    # refs: one poor match, one exact; contract says max over refs.
    assert qa.em("42 percent", ["42", "42 percent"]) == 1
    assert qa.f1("42 percent", ["42", "42 percent"]) == 1.0


def test_qa_articles_and_punctuation():
    # "an apple!" vs "apple" -- articles and punctuation both vanish.
    assert qa.em("an apple!", ["apple"]) == 1
    assert qa.f1("an apple!", ["apple"]) == 1.0


# ------------------------------------------------------------- ROUGE ----

def test_rouge_identity():
    # WORKED EXAMPLE: identical strings -> all three F-measures 1.0.
    s = "the committee approved the budget for next year"
    r = rouge(s, [s])
    assert r["rouge1"] == pytest.approx(1.0)
    assert r["rouge2"] == pytest.approx(1.0)
    assert r["rougeL"] == pytest.approx(1.0)


def test_rouge_disjoint():
    # no shared tokens -> all zeros
    r = rouge("alpha beta gamma", ["delta epsilon zeta"])
    assert r["rouge1"] == 0.0
    assert r["rouge2"] == 0.0
    assert r["rougeL"] == 0.0


def test_rouge_empty_prediction_guard():
    # Contract: empty pred returns zeros rather than raising.
    r = rouge("", ["anything at all"])
    assert r == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ------------------------------------------------------------ choice ----

def test_mc1_argmax_on_true_option():
    # WORKED EXAMPLE: argmax is index 0 which carries label 1 -> 1.0.
    assert choice.mc1([-1.0, -5.0, -9.0], [1, 0, 0]) == 1.0


def test_mc1_argmax_on_false_option():
    assert choice.mc1([-9.0, -1.0], [1, 0]) == 0.0


def test_mc2_stability_large_logliks():
    # Designed to overflow a naive softmax: exp(1000) is inf.
    # By hand with the stable formula (subtract max = 1000):
    # exps = [exp(0), exp(-1)] = [1.0, 0.36788]; z = 1.36788
    # mass on label 1 = 1.0 / 1.36788 = 0.7311 (4 dp)
    assert choice.mc2([1000.0, 999.0], [1, 0]) == pytest.approx(0.7311,
                                                                abs=1e-4)


def test_mc2_mass_splits_over_true_options():
    # Equal logliks, two of three options true -> 2/3 of the mass.
    assert choice.mc2([0.0, 0.0, 0.0], [1, 1, 0]) == pytest.approx(2 / 3)


def test_choice_dispatcher_raises_on_unknown_task():
    class FakeItem:
        task = "cuad"
        choices = ["a"]
        choice_labels = [1]
    with pytest.raises(KeyError):
        choice.score(FakeItem(), [0.0])


# --------------------------------------------------------- bootstrap ----
# Structural tests: these enforce the contract without leaking any
# number the implementation should be computing.

def test_bootstrap_deterministic_and_ordered():
    vals = [0.2, 0.4, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7]
    a = bootstrap_ci(vals, seed=42)
    b = bootstrap_ci(vals, seed=42)
    assert a == b                       # same seed, same interval
    mean, lo, hi = a
    assert lo <= mean <= hi


def test_bootstrap_single_value():
    assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_bootstrap_empty():
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)

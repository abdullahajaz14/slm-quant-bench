"""The selection framework's decision stages, as functions.

Chapter 5 of the dissertation derives a three-stage procedure:
feasibility first, Pareto dominance second, preferences third. The
dashboard applies these stages through this module, and the tests pin
the dominance outcome on the study's own frozen records.

Memory is compared through a lower bound rather than peak RSS alone.
Peak RSS is a processor-side working-set figure: with weights
memory-mapped and offloaded to Metal it understates the total
footprint and is not comparable across models (Mistral 7B at Q4_K_M
reports 3.01 GB of RSS against 4.07 GB of weights). The weights must
reside in unified memory for the model to run at all, so
max(peak RSS, file size) is a defensible lower bound on what a
configuration needs, and it is what the budget stage compares.
"""

from __future__ import annotations


def memory_lower_bound(peak_rss_gb: float, file_gb: float) -> float:
    """A conservative lower bound on a configuration's memory need."""
    return max(peak_rss_gb, file_gb)


def dominates(a: dict, b: dict) -> bool:
    """True if configuration ``a`` Pareto-dominates ``b``.

    ``a`` dominates ``b`` when it is at least as good on every axis
    (quality and decode rate higher-is-better, memory lower-is-better)
    and strictly better on at least one.
    """
    at_least = (a["quality"] >= b["quality"]
                and a["decode"] >= b["decode"]
                and a["memory"] <= b["memory"])
    strictly = (a["quality"] > b["quality"]
                or a["decode"] > b["decode"]
                or a["memory"] < b["memory"])
    return at_least and strictly


def split_frontier(configs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stage 2: split feasible configurations into (frontier, dominated).

    Each configuration is a mapping with at least ``quality``,
    ``decode`` and ``memory`` keys; other keys pass through untouched.
    """
    frontier, dominated = [], []
    for c in configs:
        if any(dominates(o, c) for o in configs if o is not c):
            dominated.append(c)
        else:
            frontier.append(c)
    return frontier, dominated


def exceeds_budget(memory_lb_gb: float, budget_gb: float) -> bool:
    """True when a configuration certainly does not fit the budget.

    The memory figure is a *lower* bound, so it can only rule a
    configuration out. A lower bound under the budget does not show
    that actual consumption fits: the true footprint may still exceed
    it. This asymmetry is why the budget is applied as a rejection
    test and never as a certificate of acceptance.
    """
    return memory_lb_gb > budget_gb


def apply_preferences(frontier: list[dict], quality_floor: float,
                      memory_budget_gb: float) -> list[dict]:
    """Stage 3: drop frontier configurations the preferences exclude.

    Returns those not excluded, rather than those certified to fit.
    The quality floor is a genuine threshold on a measured mean; the
    memory budget only removes configurations whose lower bound
    already exceeds it.
    """
    return [c for c in frontier
            if c["quality"] >= quality_floor
            and not exceeds_budget(c["memory"], memory_budget_gb)]

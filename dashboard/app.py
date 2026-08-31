"""slm-quant-bench interactive dashboard (dissertation deliverable O5).

Explore the quality-efficiency trade-off space and apply the selection
framework: given a task family, a quality floor and a memory budget,
which configurations qualify on the evidence.

Run from the repository root (requires streamlit + pandas, installed
with the analysis extras):
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

# The selection stages live in the framework package so they can be
# unit-tested; the dashboard only wires them to the widgets. Run from
# the repository root so src/ is importable.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slmbench.selection import (exceeds_budget, memory_lower_bound,  # noqa: E402
                                split_frontier)

RESULTS_DIR = "results"
PRIMARY_METRIC = {"cuad": "f1", "hotpotqa": "f1", "cnndm": "rougeL",
                  "truthfulqa_mc1": "mc1", "truthfulqa_mc2": "mc2",
                  "ukps_qa": "f1", "ukps_sum": "rougeL"}

st.set_page_config(page_title="slm-quant-bench", layout="wide")
st.title("slm-quant-bench: quantised SLM selection evidence")
st.caption("Quality measured on the reference GPU environment; "
           "efficiency and feasibility measured on the 8 GB consumer "
           "endpoint. Means carry 95% bootstrap confidence intervals.")


@st.cache_data
def load_degradation() -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, "degradation.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_efficiency() -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, "efficiency-device.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


quality = load_degradation()
efficiency = load_efficiency()

if quality is None:
    st.warning("results/degradation.csv not found. Run the quality "
               "sweep, then scripts/analysis.py, and reload.")
    st.stop()

# The battery records the study's standard configuration, full offload,
# and also supplementary rows measured at reduced offload wherever that
# standard configuration failed. Only the former describes what the
# study evaluated. Mixing them is not cosmetic: Phi-3 Mini at Q8_0 fails
# at full offload and loads at zero offload generating 0.37 tokens per
# second, so a view that takes the last matching row reports it as
# running, contradicting the feasibility table in the dissertation and
# telling a user they can deploy something they cannot.
STANDARD = (efficiency[efficiency["offload"] == "full"]
            if efficiency is not None else None)

# ---------------------------------------------------------- feasibility
st.header("1. Deployment feasibility on the 8 GB endpoint")
if STANDARD is not None:
    matrix = STANDARD.pivot_table(index="model", columns="precision",
                                  values="feasible", aggfunc="last")
    st.dataframe(matrix.replace({True: "runs", False: "does not fit"}))
    st.caption("Standard configuration: all layers offloaded to Metal. "
               "A configuration marked 'does not fit' did not load or "
               "did not complete the battery.")
    with st.expander("Failure modes"):
        failed = STANDARD[~STANDARD["feasible"].fillna(False)]
        st.dataframe(failed[["model", "precision", "failure_mode"]])
    supp = efficiency[(efficiency["offload"] != "full")
                      & (efficiency["feasible"] == True)]  # noqa: E712
    if not supp.empty:
        with st.expander("Reduced offload (not deployable, recorded for "
                         "completeness)"):
            st.caption("These configurations failed at full offload and "
                       "were re-measured with layers on the CPU. They "
                       "produce output, far too slowly to deploy, and "
                       "are excluded from every other view here.")
            st.dataframe(supp[["model", "precision", "offload",
                               "decode_tps_median"]])
else:
    st.info("results/efficiency-device.jsonl not found yet.")

# -------------------------------------------------------------- quality
st.header("2. Task quality by precision")
task = st.selectbox("Task", sorted(quality["task"].unique()))
metric = st.selectbox(
    "Metric", sorted(quality[quality["task"] == task]["metric"].unique()),
    index=0)
view = quality[(quality["task"] == task)
               & (quality["metric"] == metric)].copy()
st.dataframe(view[["model", "precision", "n", "mean", "ci_lo", "ci_hi",
                   "delta_vs_fp16", "delta_lo", "delta_hi"]]
             .sort_values(["model", "precision"]),
             use_container_width=True)
chart = view.pivot_table(index="model", columns="precision",
                         values="mean")
st.bar_chart(chart)

# ----------------------------------------------------------- efficiency
if STANDARD is not None:
    st.header("3. Efficiency on the endpoint")
    feasible = STANDARD[STANDARD["feasible"] == True]  # noqa: E712
    feasible = feasible.assign(
        mem_lb_gb=[memory_lower_bound(r, f) for r, f in
                   zip(feasible["peak_rss_gb"], feasible["file_gb"])])
    st.dataframe(feasible[["model", "precision", "file_gb", "load_s",
                           "peak_rss_gb", "mem_lb_gb",
                           "prefill_tps_median", "prefill_tps_p95",
                           "decode_tps_median", "decode_tps_p95"]],
                 use_container_width=True)
    st.caption("Peak resident memory is a processor-side working-set "
               "figure. With weights memory-mapped and offloaded to "
               "Metal it understates total footprint and is not "
               "comparable across models: Mistral 7B at Q4_K_M reports "
               "3.01 GB against 4.07 GB of weights. mem_lb_gb is the "
               "larger of peak RSS and file size, a conservative lower "
               "bound on what the configuration needs, and it is the "
               "figure the selection stages compare.")

# ------------------------------------------------- selection framework
st.header("4. Selection helper")
st.caption("The framework's three stages, in order. Stage 1, "
           "feasibility: only configurations that run on the device "
           "are considered. Stage 2, dominance: a configuration beaten "
           "on task quality, decode rate and memory at once is "
           "discarded, since no preference can make it the right "
           "choice. Stage 3, preferences: the survivors are filtered "
           "by your quality floor and memory budget. The floor is "
           "relative to the best FP16 result for the task, a reference "
           "point rather than an option: FP16 runs on none of these "
           "models on this hardware. The memory figure is a lower "
           "bound, so the budget removes configurations that certainly "
           "do not fit; it cannot certify that the survivors do.")
sel_task = st.selectbox("Workload task family",
                        sorted(quality["task"].unique()), key="sel")
sel_metric = PRIMARY_METRIC.get(sel_task,
                                quality[quality["task"] == sel_task]
                                ["metric"].iloc[0])
floor = st.slider("Quality floor (% of best FP16 mean)", 50, 100, 90)
budget = st.slider("Memory budget (GB), compared against the lower "
                   "bound", 1.0, 8.0, 6.0, 0.5)

q = quality[(quality["task"] == sel_task)
            & (quality["metric"] == sel_metric)]
fp16_best = q[q["precision"] == "fp16"]["mean"].max()

if STANDARD is None:
    st.info("results/efficiency-device.jsonl not found; the selection "
            "stages need the device battery.")
else:
    # Stage 1: feasibility.
    eff = STANDARD[STANDARD["feasible"] == True]  # noqa: E712
    means = q.set_index(["model", "precision"])["mean"]
    configs = []
    for _, e in eff.iterrows():
        key = (e["model"], e["precision"])
        if key not in means.index:
            continue
        configs.append({
            "model": e["model"], "precision": e["precision"],
            "quality": float(means[key]),
            "decode": float(e["decode_tps_median"]),
            "memory": memory_lower_bound(float(e["peak_rss_gb"]),
                                         float(e["file_gb"])),
        })

    # Stage 2: Pareto dominance on this task's quality, decode rate
    # and the memory lower bound.
    frontier, dominated = split_frontier(configs)

    # Stage 3: the user's preferences.
    chosen = [c for c in frontier
              if c["quality"] >= fp16_best * floor / 100
              and not exceeds_budget(c["memory"], budget)]

    if dominated:
        names = ", ".join(f"{c['model']} {c['precision']}"
                          for c in dominated)
        st.caption(f"Stage 2 discards {len(dominated)} dominated "
                   f"configuration(s) on this task: {names}.")
    if not chosen:
        st.error("Every frontier configuration is excluded by the "
                 "floor or the budget; relax one constraint.")
    else:
        st.success(f"{len(chosen)} configuration(s) not excluded "
                   f"(quality floor {floor}% of best FP16 "
                   f"{sel_metric}={fp16_best:.3f}; budget {budget} GB "
                   f"applied to the memory lower bound, which rules "
                   f"configurations out rather than in)")
        st.dataframe(pd.DataFrame(chosen)
                     .sort_values("quality", ascending=False),
                     use_container_width=True)

"""scripts/figures.py: generate the Chapter 4/5 figures from results.

Requires matplotlib (installed with the analysis extras, not part of
the core run environment). Figures are written as PDF into --outdir,
which defaults to figures/ in the repository root.

Figures produced (matching the Chapter Writing Plan):
  ch4_quality_<task>.pdf   per-task grouped bars, mean with 95% CI,
                           model x precision
  ch4_efficiency.pdf       decode tokens/s (median, p95 whisker) per
                           feasible configuration
  ch5_heatmap.pdf          Q4_K_M degradation delta vs FP16 (primary
                           metric) per model x task
  ch5_pareto_memory.pdf    quality vs the memory lower bound,
                           max(peak RSS, file size), on-device configs
  ch5_pareto_speed.pdf     quality vs decode rate, on-device configs

Run from the repository root; every argument has a working default:
  python scripts/figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Typography matched to the dissertation, which is set in a Times-like
# serif face, so figures do not read as foreign objects on the page.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
})

PRECISIONS = ["fp16", "q8_0", "q4_k_m"]
PRECISION_LABELS = {"fp16": "FP16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}

# Categorical hues assigned in fixed order, one per precision, never
# cycled and never reassigned when a subset is plotted, so a precision
# keeps its colour across every figure. Taken from the Okabe-Ito set and
# checked for colour-vision deficiency separation: worst adjacent pair
# dE 11.4 under protanopia, 24.2 under normal vision. Identity is never
# carried by colour alone: every figure also has a legend, and the same
# values appear in the chapter tables.
PRECISION_COLOURS = {"fp16": "#0072B2", "q8_0": "#E69F00",
                     "q4_k_m": "#009E73"}

TASK_LABELS = {
    "cuad": "CUAD (legal QA)",
    "hotpotqa": "HotpotQA (multi-hop)",
    "cnndm": "CNN/DailyMail",
    "truthfulqa_mc1": "TruthfulQA MC1",
    "truthfulqa_mc2": "TruthfulQA MC2",
    "ukps_qa": "Curated corpus (QA)",
    "ukps_sum": "Curated corpus (summary)",
}
METRIC_LABELS = {"em": "exact match", "f1": "token F1",
                 "rouge1": "ROUGE-1", "rouge2": "ROUGE-2",
                 "rougeL": "ROUGE-L", "mc1": "MC1", "mc2": "MC2"}
MODEL_LABELS = {"phi3-mini": "Phi-3 Mini", "gemma2-2b": "Gemma 2 2B",
                "llama32-3b": "Llama 3.2 3B", "mistral7b": "Mistral 7B"}

PRIMARY_METRIC = {"cuad": "f1", "hotpotqa": "f1", "cnndm": "rougeL",
                  "truthfulqa_mc1": "mc1", "truthfulqa_mc2": "mc2",
                  "ukps_qa": "f1", "ukps_sum": "rougeL"}


def read_degradation(path: str) -> list[dict]:
    with open(path) as f:
        return [row for row in csv.DictReader(f)]


def read_efficiency(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def quality_bars(rows: list[dict], outdir: str) -> None:
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        metric = PRIMARY_METRIC.get(task)
        sub = [r for r in rows if r["task"] == task
               and r["metric"] == metric]
        if not sub:
            continue
        models = sorted({r["model"] for r in sub})
        # 6.3 x 4.2 rather than 6.3 x 3.2: at full text width the
        # flatter aspect left the bars short and the charts looking
        # squeezed, particularly where several run consecutively.
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        # A small gap between adjacent bars so groups read as groups.
        slot = 0.8 / len(PRECISIONS)
        width = slot * 0.88
        for j, precision in enumerate(PRECISIONS):
            xs, ys, lo, hi = [], [], [], []
            for i, model in enumerate(models):
                match = [r for r in sub if r["model"] == model
                         and r["precision"] == precision]
                if not match:
                    continue
                r = match[0]
                mean = float(r["mean"])
                xs.append(i + (j - (len(PRECISIONS) - 1) / 2) * slot)
                ys.append(mean)
                lo.append(mean - float(r["ci_lo"]))
                hi.append(float(r["ci_hi"]) - mean)
            if xs:
                ax.bar(xs, ys, width=width,
                       color=PRECISION_COLOURS[precision],
                       label=PRECISION_LABELS[precision],
                       yerr=[lo, hi], capsize=2,
                       error_kw={"linewidth": 0.8, "ecolor": "#333333"})
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{TASK_LABELS.get(task, task)}: mean "
                     f"{METRIC_LABELS.get(metric, metric)}, "
                     f"95% bootstrap confidence interval")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=len(PRECISIONS), loc="upper center",
                  bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"ch4_quality_{task}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[figures] ch4_quality_{task}.pdf")


def heatmap(rows: list[dict], outdir: str,
            precision: str = "q4_k_m") -> None:
    tasks = sorted({r["task"] for r in rows})
    models = sorted({r["model"] for r in rows})
    grid = []
    for model in models:
        line = []
        for task in tasks:
            metric = PRIMARY_METRIC.get(task)
            match = [r for r in rows if r["model"] == model
                     and r["task"] == task and r["metric"] == metric
                     and r["precision"] == precision
                     and r["delta_vs_fp16"] != ""]
            line.append(float(match[0]["delta_vs_fp16"]) if match
                        else float("nan"))
        grid.append(line)
    # Deltas are signed, so the scale is diverging about zero: two hues
    # with a neutral midpoint, symmetric about zero so that a given
    # distance means the same in either direction. Not a rainbow scale,
    # which would imply an ordering the data does not have, and not a
    # hue at the midpoint, which would make "no change" look like a
    # value. Every cell is also labelled, so the reading never depends
    # on colour alone, in print or for a colour-vision-deficient reader.
    #
    # The colour scale is clipped at the 90th percentile of absolute
    # change, with a floor, rather than at the largest value. A single
    # outlying cell would otherwise set the limit and compress every
    # other cell towards the neutral midpoint, so the figure would show
    # a real and consistent pattern as though it were no pattern at
    # all. Clipping costs no information here because every cell is
    # labelled with its own value; cells beyond the scale are outlined
    # so that the clipping is visible rather than silent, and the
    # caption states the limit.
    finite = sorted(abs(v) for row in grid for v in row if v == v)
    limit = 0.05
    if finite:
        limit = max(0.02, finite[min(len(finite) - 1,
                                     int(0.9 * len(finite)))])
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    im = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks],
                       rotation=28, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_xticks([x - 0.5 for x in range(1, len(tasks))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    clipped = 0
    for i in range(len(models)):
        for j in range(len(tasks)):
            v = grid[i][j]
            if v != v:  # NaN
                continue
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=7.5)
            if abs(v) > limit:
                clipped += 1
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=1.4))
    cbar = fig.colorbar(im, fraction=0.025, pad=0.02, extend="both")
    cbar.set_label(f"change vs FP16 ({PRECISION_LABELS[precision]})",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    title = (f"Change in the primary metric at "
             f"{PRECISION_LABELS[precision]} relative to FP16")
    if clipped:
        title += (f"\nscale clipped at $\\pm${limit:.3f}; "
                  f"{clipped} outlined cell"
                  f"{'s' if clipped > 1 else ''} beyond it")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ch5_heatmap.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("[figures] ch5_heatmap.pdf")


def efficiency_and_pareto(quality_rows: list[dict],
                          efficiency_rows: list[dict],
                          outdir: str) -> None:
    # Only the study's standard configuration belongs on these charts.
    # The battery also records supplementary rows measured at reduced
    # offload, taken where the standard configuration failed, and those
    # are feasible in the narrow sense that they produce output. Plotting
    # them alongside would state that Phi-3 Mini runs at Q8_0 on this
    # device, at 0.4 tokens per second, when in the configuration this
    # study evaluates it does not run at all. Chapter 4 keeps "cannot
    # run" and "runs far too slowly to deploy" apart, and a chart that
    # merges them contradicts its own text.
    feasible = [r for r in efficiency_rows
                if r.get("feasible") and r.get("offload") == "full"
                and r.get("decode_tps_p95") is not None]
    if feasible:
        feasible = sorted(feasible, key=lambda r: (r["model"],
                                                   r["precision"]))
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        med = [r["decode_tps_median"] for r in feasible]
        p95 = [r["decode_tps_p95"] for r in feasible]
        # Colour carries precision, consistently with every other
        # figure, so a precision is recognisable across the chapter.
        ax.bar(range(len(feasible)), med,
               color=[PRECISION_COLOURS[r["precision"]] for r in feasible],
               width=0.7)
        ax.errorbar(range(len(feasible)), med,
                    yerr=[[0] * len(feasible),
                          [abs(b - a) for a, b in zip(med, p95)]],
                    fmt="none", capsize=3, ecolor="#333333", elinewidth=0.8)
        # Model and precision both appear in the tick label. Colour
        # repeats the precision rather than carrying it: two adjacent
        # bars for the same model would otherwise be distinguishable
        # only by hue, which fails in greyscale, in print, and for a
        # colour-vision-deficient reader.
        ax.set_xticks(range(len(feasible)))
        ax.set_xticklabels(
            [f"{MODEL_LABELS.get(r['model'], r['model'])}\n"
             f"{PRECISION_LABELS[r['precision']]}" for r in feasible],
            fontsize=7.5)
        ax.set_ylabel("decode tokens/s (median, p95 whisker)")
        ax.set_title("Decode rate for the configurations that run on the "
                     "8 GB endpoint")
        ax.set_axisbelow(True)
        present = [p for p in PRECISIONS
                   if any(r["precision"] == p for r in feasible)]
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1,
                                         color=PRECISION_COLOURS[p],
                                         label=PRECISION_LABELS[p])
                           for p in present],
                  frameon=False, ncol=len(present), loc="upper center",
                  bbox_to_anchor=(0.5, -0.22))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ch4_efficiency.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
        print("[figures] ch4_efficiency.pdf")

    # Pareto: mean quality (averaged over the four public benchmark
    # primary metrics) vs memory and vs speed, on-device configs only.
    quality_at: dict[tuple, list[float]] = defaultdict(list)
    for r in quality_rows:
        if (r["metric"] == PRIMARY_METRIC.get(r["task"])
                and not r["task"].startswith("ukps")):
            quality_at[(r["model"], r["precision"])].append(float(r["mean"]))
    # The memory axis is max(peak RSS, model file size), the same
    # lower bound the selection stages compare against. Peak RSS alone
    # understates the footprint wherever weights are memory-mapped and
    # offloaded to Metal, and is not comparable across models, so the
    # figure and the implemented rule would otherwise disagree.
    for r in feasible:
        r["mem_lb_gb"] = max(r["peak_rss_gb"], r["file_gb"])
    for x_key, x_label, name in [
            ("mem_lb_gb", "memory lower bound, max(peak RSS, file) (GB)",
             "ch5_pareto_memory.pdf"),
            ("decode_tps_median", "decode tokens/s (median)",
             "ch5_pareto_speed.pdf")]:
        fig, ax = plt.subplots(figsize=(6.3, 4))
        for r in feasible:
            key = (r["model"], r["precision"])
            if key not in quality_at:
                continue
            x = r[x_key]
            y = sum(quality_at[key]) / len(quality_at[key])
            ax.scatter(x, y, s=55, zorder=3,
                       color=PRECISION_COLOURS[r["precision"]],
                       edgecolor="white", linewidth=1.2)
            ax.annotate(f"{MODEL_LABELS.get(r['model'], r['model'])} "
                        f"{PRECISION_LABELS[r['precision']]}",
                        (x, y), fontsize=7,
                        textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean primary quality (public benchmarks)")
        ax.set_title("Deployable configurations only; FP16 runs on none "
                     "of them", fontsize=9)
        ax.set_axisbelow(True)
        present = [p for p in PRECISIONS
                   if any(r["precision"] == p for r in feasible)]
        ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="none",
                                      color=PRECISION_COLOURS[p],
                                      label=PRECISION_LABELS[p])
                           for p in present],
                  frameon=False, ncol=len(present), loc="upper center",
                  bbox_to_anchor=(0.5, -0.15))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), bbox_inches="tight")
        plt.close(fig)
        print(f"[figures] {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", default="results/degradation.csv")
    parser.add_argument("--efficiency",
                        default="results/efficiency-device.jsonl")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    quality_rows = read_degradation(args.degradation)
    quality_bars(quality_rows, args.outdir)
    heatmap(quality_rows, args.outdir)
    if os.path.exists(args.efficiency):
        efficiency_and_pareto(quality_rows,
                              read_efficiency(args.efficiency),
                              args.outdir)
    else:
        print(f"[figures] {args.efficiency} not found; efficiency and "
              f"Pareto figures skipped")


if __name__ == "__main__":
    main()

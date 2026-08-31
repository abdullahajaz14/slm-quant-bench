"""slmbench.measure: the on-device efficiency battery (Chapter 3, 3.5).

Run on the Air only, mains power, lid open, background apps closed.
Feasibility is a RESULT here, not an error state: a configuration that
cannot load on 8 GB produces a row saying so, with the failure mode
captured verbatim, and that row feeds the fits-or-does-not table in
Chapter 4.

Operational rule (fixed): hard OOM on macOS can kill the whole process,
not just raise, so full sweeps must survive that. Two defences:
config-level resumability (done configs are skipped on rerun) and a
--only model:precision flag so risky rows, all FP16 ones, run one per
invocation between download and delete.

Run from the repository root:
  python -m slmbench.measure --config configs/runs/efficiency_device.yaml
  python -m slmbench.measure --config ... --only mistral7b:fp16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time

import psutil
import yaml

from .backend import LlamaCppBackend
from . import prompts


def load_battery(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class MemorySampler:
    """Peak RSS of this process, sampled on a background thread.

    Samples psutil RSS every 0.2 s between start() and stop(); exposes
    peak_gb. Sampling, not before/after deltas, because peak usage
    occurs DURING generation (KV cache growth) and is gone by the time
    a call returns.
    """

    def __init__(self) -> None:
        self.peak_gb = 0.0
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss_gb = self._proc.memory_info().rss / 1024**3
            if rss_gb > self.peak_gb:
                self.peak_gb = rss_gb
            self._stop.wait(0.2)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def percentile(values: list[float], p: float) -> float:
    """Simple index method, used everywhere and noted once in Chapter
    4's table caption: sort ascending; index = min(len-1,
    int(p / 100 * len))."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(p / 100 * len(ordered)))
    return ordered[index]


def run_config(model_cfg: dict, precision: str, battery: list[dict],
               reps: int, n_gpu_layers: int = -1,
               gguf_path: str | None = None,
               budget_s: float | None = None) -> dict:
    """One efficiency row; infeasibility is data, not an error.

    `n_gpu_layers` defaults to -1, the study's standard configuration
    (offload every layer to Metal). The pilot established that some
    configurations fail at full offload on 8 GB while still loading at
    reduced offload, so main() re-runs this function at the fallback
    levels and records those as supplementary rows. That distinguishes
    "cannot run" from "runs, but too slowly to deploy", which is a
    distinction a deploying organisation needs.

    `gguf_path` overrides the path in the model config, so the caller
    can measure a copy staged on internal storage.
    """
    name = model_cfg["name"]
    gguf = gguf_path or model_cfg["gguf"][precision]
    row: dict = {"model": name, "precision": precision, "gguf_path": gguf,
                 "n_gpu_layers": n_gpu_layers}
    if not os.path.exists(gguf):
        row |= {"file_gb": None, "feasible": False,
                "failure_mode": f"gguf file not present: {gguf}"}
        return row
    row["file_gb"] = os.path.getsize(gguf) / 1024**3

    sampler = MemorySampler()
    sampler.start()
    try:
        bk = LlamaCppBackend(gguf, n_ctx=model_cfg.get("n_ctx", 4096),
                             n_gpu_layers=n_gpu_layers,
                             seed=model_cfg.get("seed", 42))
    except Exception as e:
        sampler.stop()
        row |= {"feasible": False, "failure_mode": repr(e),
                "peak_rss_gb": sampler.peak_gb}
        return row
    row["load_s"] = bk.load_s

    prefills: list[float] = []
    decodes: list[float] = []
    budget_exhausted = False
    started = time.perf_counter()
    try:
        for _rep in range(reps):
            for b in battery:
                # A configuration that survives loading can still be far
                # too slow to complete the battery: the pilot measured
                # one at roughly a twentieth of a token per second, at
                # which rate the full battery would take days. The
                # budget stops such a configuration once its rate is
                # established, records what it achieved, and lets the
                # sweep continue. Running out of budget is itself a
                # finding, and is reported as one.
                if budget_s is not None and \
                        time.perf_counter() - started > budget_s:
                    budget_exhausted = True
                    break
                # battery measures deployment-realistic chat inference,
                # not raw completion
                prompt = prompts.CHAT_FORMATS[name].format(user=b["prompt"])
                r = bk.generate(prompt, b["max_tokens"],
                                stop=prompts.STOP[name])
                prefills.append(r.prefill_tps)
                decodes.append(r.decode_tps)
            if budget_exhausted:
                break
    except Exception as e:
        sampler.stop()
        bk.close()
        row |= {"feasible": False, "failure_mode": repr(e),
                "peak_rss_gb": sampler.peak_gb,
                "prefill_tps_partial": prefills,
                "decode_tps_partial": decodes,
                "n_generations": len(decodes)}
        return row

    sampler.stop()
    bk.close()
    if budget_exhausted:
        # Loaded and generated, but too slowly to complete the battery
        # inside its budget. Not feasible for deployment, and for a
        # different reason than failing to load: recorded distinctly.
        row |= {"feasible": False,
                "failure_mode": f"exceeded time budget of {budget_s:.0f}s "
                                f"after {len(decodes)} generations",
                "budget_exhausted": True,
                "peak_rss_gb": sampler.peak_gb,
                "prefill_tps_median": percentile(prefills, 50) if prefills else None,
                "decode_tps_median": percentile(decodes, 50) if decodes else None,
                "n_generations": len(decodes)}
        return row
    row |= {"feasible": True,
            "peak_rss_gb": sampler.peak_gb,
            "prefill_tps_median": percentile(prefills, 50),
            "prefill_tps_p95": percentile(prefills, 95),
            "decode_tps_median": percentile(decodes, 50),
            "decode_tps_p95": percentile(decodes, 95),
            "n_generations": len(decodes)}
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", default=None,
                        help='run a single "model:precision" row')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    battery = load_battery(cfg["battery"])
    reps = cfg.get("repetitions", 3)
    cooldown_s = cfg.get("cooldown_s", 60)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out_path = os.path.join(cfg["out_dir"], f"{cfg['run_id']}.jsonl")

    done: set[tuple[str, str]] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done.add((rec["model"], rec["precision"]))
                    except json.JSONDecodeError:
                        pass
    if done:
        print(f"[measure] resuming: {sorted(done)} already measured")

    pairs = [(m, p) for m in cfg["models"] for p in cfg["precisions"]]
    if args.only:
        model_key, precision = args.only.split(":")
        pairs = [(model_key, precision)]

    stage_dir = os.path.expanduser(cfg["stage_dir"]) if cfg.get("stage_dir") \
        else None
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
    fallbacks = cfg.get("fallback_gpu_layers", [])

    def emit(row: dict) -> None:
        with open(out_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    first = True
    for model_key, precision in pairs:
        if (model_key, precision) in done:
            continue
        if not first:
            print(f"[measure] cooldown {cooldown_s}s (thermal carry-over)")
            time.sleep(cooldown_s)
        first = False
        if precision == "fp16":
            print("[measure] REMINDER: fp16 rows are download -> test -> "
                  "delete; check free disk (df -h) before each one and "
                  "prefer --only per row")
        with open(os.path.join("configs", "models",
                               f"{model_key}.yaml")) as f:
            model_cfg = yaml.safe_load(f)

        source = model_cfg["gguf"][precision]
        measured_path = source
        staged = False
        if stage_dir and os.path.exists(source):
            # Measure from internal storage: the pilot found a 23x
            # cold-start difference between internal and external USB
            # storage for the same artefact (3.1 s against 70.5 s), and
            # load time is a reported metric, so the source disk would
            # otherwise dominate it. Throughput was unaffected, but load
            # time alone justifies staging.
            measured_path = os.path.join(stage_dir, os.path.basename(source))
            if not os.path.exists(measured_path):
                print(f"[measure] staging {os.path.basename(source)} "
                      f"to {stage_dir}")
                shutil.copy2(source, measured_path)
            staged = True

        print(f"[measure] running {model_key}:{precision} "
              f"({reps} reps x {len(battery)} prompts)")
        row = run_config(model_cfg, precision, battery, reps,
                         n_gpu_layers=-1, gguf_path=measured_path,
                         budget_s=cfg.get("budget_s"))
        row["offload"] = "full"
        emit(row)

        # Supplementary rows: where the standard full-offload
        # configuration fails, retry at reduced offload so the record
        # separates "cannot run at all" from "runs, but far too slowly
        # to deploy". Both outcomes are findings.
        if not row.get("feasible") and fallbacks:
            for n in fallbacks:
                print(f"[measure] {model_key}:{precision} failed at full "
                      f"offload; trying n_gpu_layers={n}")
                frow = run_config(model_cfg, precision,
                                  battery[:cfg.get("fallback_prompts", 3)],
                                  1, n_gpu_layers=n,
                                  gguf_path=measured_path,
                                  budget_s=cfg.get("fallback_budget_s", 600))
                frow["offload"] = f"partial:{n}"
                frow["supplementary"] = True
                emit(frow)
                if frow.get("feasible"):
                    break

        if staged and cfg.get("cleanup_staged", True):
            os.remove(measured_path)
            print(f"[measure] removed staged copy of "
                  f"{os.path.basename(source)}")

    # mini table over everything on disk
    print(f"\n{'model':<12} {'precision':<8} {'feasible':<9} "
          f"{'peak_gb':<8} {'decode_med':<10}")
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            print(f"{r['model']:<12} {r['precision']:<8} "
                  f"{str(r.get('feasible')):<9} "
                  f"{r.get('peak_rss_gb', float('nan')) or 0:<8.2f} "
                  f"{r.get('decode_tps_median', float('nan')) or 0:<10.1f}")
    print(f"[measure] rows written to {out_path}")


if __name__ == "__main__":
    main()

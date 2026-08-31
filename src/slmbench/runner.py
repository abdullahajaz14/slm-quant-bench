"""slmbench.runner: orchestrates model x precision x task x item.

LOOP ORDER (fixed): model outermost, then precision, constructing ONE
backend per (model, precision) and running ALL generation tasks through
it before closing. Each GGUF loads exactly once per run; on an 8 GB
machine, reloading multi-gigabyte files per task would dominate wall
time. TruthfulQA MC needs logits (choice_mode=True), which costs
memory, so MC tasks run through a SECOND backend constructed after the
generation backend is closed: two loads per (model, precision) when MC
tasks are present, never both resident.

Resumability (fixed): before running, load writer.done_keys(); skip any
(model, precision, task, item_id) already on disk. A crashed overnight
run resumes by re-running the same command.

Per-item error policy (fixed): catch exceptions per item, write a
Record with error set and scores empty, log, continue. Errors are data
(they feed the robustness story), and one poison item must not kill a
batch.

Run from the repository root: paths to configs/ are relative.
Entry point: scripts/run_quality.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import yaml

from .adapters import base as ab
from . import prompts
from .backend import LlamaCppBackend
from .results import Record, ResultsWriter, capture_env, sha256_of, summarise


def get_scorer(name: str):
    """Scorer dispatch. Returns a callable
    (item, output_or_logliks) -> dict[str, float]."""
    if name == "qa_em_f1":
        from .scoring import qa
        return lambda item, out: {"em": float(qa.em(out, item.references)),
                                  "f1": qa.f1(out, item.references)}
    if name == "rouge":
        from .scoring.rouge import rouge
        return lambda item, out: rouge(out, item.references)
    if name == "truthfulqa_mc":
        from .scoring import choice
        return lambda item, logliks: choice.score(item, logliks)
    raise KeyError(f"unknown scorer {name!r}")


def load_run_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ["run_id", "env", "backend", "models", "precisions",
                "tasks", "out_dir"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(
            f"run config {path} is missing required keys: {missing}")
    return cfg


def run_generation_task(bk, task_cfg, items, model, precision, run_cfg,
                        writer, env, done, config_sha):
    scorer = get_scorer(task_cfg.scorer)
    total = len(items)
    processed = 0
    first_metric_values: list[float] = []
    for item in items:
        key = (model, precision, task_cfg.name, item.item_id)
        if key in done:
            continue
        prompt = prompts.render(item, task_cfg.template, model)
        try:
            r = bk.generate(prompt, task_cfg.max_output_tokens,
                            stop=prompts.STOP[model])
            scores = scorer(item, r.text)
            rec = Record(
                run_id=run_cfg["run_id"], ts=time.time(), env=env,
                model=model, precision=precision, task=task_cfg.name,
                item_id=item.item_id, prompt_sha256=sha256_of(prompt),
                output=r.text, scores=scores,
                timings={"prefill_tps": r.prefill_tps,
                         "decode_tps": r.decode_tps,
                         "total_s": r.total_s},
                config_sha256=config_sha)
            first_metric_values.append(next(iter(scores.values())))
        except Exception as e:
            rec = Record(
                run_id=run_cfg["run_id"], ts=time.time(), env=env,
                model=model, precision=precision, task=task_cfg.name,
                item_id=item.item_id, prompt_sha256=sha256_of(prompt),
                output=None, scores={}, timings={},
                config_sha256=config_sha, error=repr(e))
            print(f"[{task_cfg.name}] ERROR on {item.item_id}: {e!r}")
        writer.write(rec)
        processed += 1
        if processed % 25 == 0:
            mean = (sum(first_metric_values) / len(first_metric_values)
                    if first_metric_values else float("nan"))
            print(f"[{task_cfg.name}] {processed}/{total} done, "
                  f"running mean (first metric) {mean:.3f}")


def run_mc_task(bk, task_cfg, items, model, precision, run_cfg,
                writer, env, done, config_sha):
    scorer = get_scorer(task_cfg.scorer)
    total = len(items)
    processed = 0
    for item in items:
        key = (model, precision, item.task, item.item_id)
        if key in done:
            continue
        mc_prompt = prompts.mc_prompt(item, model)
        try:
            t0 = time.perf_counter()
            logliks = [bk.choice_loglik(mc_prompt, " " + opt)
                       for opt in item.choices]
            total_s = time.perf_counter() - t0  # per ITEM: sum over options
            scores = scorer(item, logliks)
            rec = Record(
                run_id=run_cfg["run_id"], ts=time.time(), env=env,
                model=model, precision=precision, task=item.task,
                item_id=item.item_id, prompt_sha256=sha256_of(mc_prompt),
                output=json.dumps(logliks),  # every number re-derivable
                scores=scores, timings={"total_s": total_s},
                config_sha256=config_sha)
        except Exception as e:
            rec = Record(
                run_id=run_cfg["run_id"], ts=time.time(), env=env,
                model=model, precision=precision, task=item.task,
                item_id=item.item_id, prompt_sha256=sha256_of(mc_prompt),
                output=None, scores={}, timings={},
                config_sha256=config_sha, error=repr(e))
            print(f"[{item.task}] ERROR on {item.item_id}: {e!r}")
        writer.write(rec)
        processed += 1
        if processed % 25 == 0:
            print(f"[{task_cfg.name}(mc)] {processed}/{total} done")


def _pending(done, model, precision, task_pairs) -> int:
    n = 0
    for task_cfg, items in task_pairs:
        for item in items:
            task_name = item.task if item.choices else task_cfg.name
            if (model, precision, task_name, item.item_id) not in done:
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="items per task (overrides the config)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="override the config model list "
                             "(one Colab session per model)")
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="override the config precision list, so a "
                             "caller can run one precision at a time and "
                             "checkpoint between them")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if args.models:
        cfg["models"] = args.models
    if args.precisions:
        cfg["precisions"] = args.precisions
    limit = args.limit if args.limit is not None else cfg.get("limit")

    env = capture_env(cfg["backend"])
    writer = ResultsWriter(cfg["out_dir"], cfg["run_id"])
    done = writer.done_keys()
    if done:
        print(f"[runner] resuming: {len(done)} records already on disk")

    # load all task configs and items ONCE up front
    gen_tasks, mc_tasks = [], []
    for task_name in cfg["tasks"]:
        task_cfg = ab.TaskConfig.from_yaml(
            os.path.join("configs", "tasks", f"{task_name}.yaml"))
        adapter_cls = ab.ADAPTERS[task_cfg.extra.get("adapter",
                                                     task_cfg.name)]
        items = adapter_cls().load(task_cfg)
        if limit:
            items = items[:limit]
        if task_cfg.extra.get("mode") == "choice":
            mc_tasks.append((task_cfg, items))
        else:
            gen_tasks.append((task_cfg, items))

    for model in cfg["models"]:
        with open(os.path.join("configs", "models", f"{model}.yaml")) as f:
            model_cfg = yaml.safe_load(f)
        for precision in cfg["precisions"]:
            gguf = model_cfg["gguf"][precision]
            if cfg["env"] == "colab" and precision == "fp16":
                gpu_layers = model_cfg["gpu_layers"].get(
                    "colab_fp16", model_cfg["gpu_layers"]["colab"])
            else:
                gpu_layers = model_cfg["gpu_layers"].get(cfg["env"], -1)

            def config_sha(task_cfg):
                return sha256_of({"run": {k: v for k, v in cfg.items()},
                                  "model": model_cfg,
                                  "task": dataclasses.asdict(task_cfg)})

            if gen_tasks and _pending(done, model, precision, gen_tasks):
                print(f"[runner] {model} {precision}: loading {gguf} "
                      f"(gpu_layers={gpu_layers})")
                bk = LlamaCppBackend(gguf, n_ctx=model_cfg.get("n_ctx", 4096),
                                     n_gpu_layers=gpu_layers,
                                     seed=model_cfg.get("seed", 42))
                print(f"[runner] loaded in {bk.load_s:.1f}s")
                for task_cfg, items in gen_tasks:
                    run_generation_task(bk, task_cfg, items, model,
                                        precision, cfg, writer, env, done,
                                        config_sha(task_cfg))
                bk.close()
            if mc_tasks and _pending(done, model, precision, mc_tasks):
                print(f"[runner] {model} {precision}: loading {gguf} "
                      f"in choice mode")
                bk2 = LlamaCppBackend(gguf,
                                      n_ctx=model_cfg.get("n_ctx", 4096),
                                      n_gpu_layers=gpu_layers,
                                      seed=model_cfg.get("seed", 42),
                                      choice_mode=True)
                for task_cfg, items in mc_tasks:
                    run_mc_task(bk2, task_cfg, items, model, precision,
                                cfg, writer, env, done, config_sha(task_cfg))
                bk2.close()

    out_csv = os.path.join(cfg["out_dir"], f"{cfg['run_id']}-summary.csv")
    summarise(writer.path, out_csv)
    print(f"[runner] raw records: {writer.path}")
    print(f"[runner] summary:     {out_csv}")


# Without this, `python -m slmbench.runner` imports the module, defines
# main, and exits 0 having run nothing at all: no output, no results
# file, and a success code. scripts/run_quality.py supplied the guard,
# so the documented module invocation was the only one that silently
# did nothing, and it did so in the manner hardest to notice.
if __name__ == "__main__":
    main()

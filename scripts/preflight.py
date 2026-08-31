"""scripts/preflight.py: validate the configuration surface before a run.

A quality sweep takes hours and a device battery takes an evening, so a
configuration error should be caught in seconds beforehand rather than
discovered part-way through. This checks every run configuration against
the model and task configurations, the prompt layer and the scorer
registry, and reports everything wrong at once rather than failing on
the first problem.

Checks performed:
  - every model named by a run has a model configuration, and that
    configuration declares a GGUF path for every precision the run uses;
  - every model has a chat format and a stop list in the prompt layer,
    since a missing wrapper silently degrades every downstream number;
  - every task named by a run has a task configuration;
  - every task's template exists in the prompt layer, and its scorer is
    one the runner can dispatch;
  - choice-mode tasks declare the mode flag the runner splits on;
  - artefacts referenced by the run are present on disk (reported as a
    warning, since they may be built later in the session).

Run from the repository root:
  python scripts/preflight.py                    # every run config
  python scripts/preflight.py configs/runs/quality_colab.yaml
"""

from __future__ import annotations

import glob
import os
import sys

import yaml

sys.path.insert(0, "src")
from slmbench import prompts                      # noqa: E402
from slmbench.adapters import base as ab          # noqa: E402
from slmbench.adapters import (                   # noqa: E402,F401
    cnndm, cuad, hotpotqa, truthfulqa, ukps)

KNOWN_SCORERS = {"qa_em_f1", "rouge", "truthfulqa_mc"}

errors: list[str] = []
warnings: list[str] = []


def check_run(path: str) -> None:
    where = os.path.basename(path)
    with open(path) as f:
        run = yaml.safe_load(f)

    for key in ("run_id", "env", "backend", "models", "precisions",
                "out_dir"):
        if key not in run:
            errors.append(f"{where}: missing required key '{key}'")
    if "tasks" not in run and "battery" not in run:
        errors.append(f"{where}: declares neither 'tasks' nor 'battery'")

    for model in run.get("models", []):
        mpath = os.path.join("configs", "models", f"{model}.yaml")
        if not os.path.exists(mpath):
            errors.append(f"{where}: no model config for '{model}'")
            continue
        with open(mpath) as f:
            mcfg = yaml.safe_load(f)
        for precision in run.get("precisions", []):
            gguf = mcfg.get("gguf", {}).get(precision)
            if not gguf:
                errors.append(f"{where}: {model} declares no GGUF for "
                              f"'{precision}'")
            elif not os.path.exists(gguf):
                warnings.append(f"{where}: {model} {precision} artefact "
                                f"not on disk yet ({gguf})")
        if model not in prompts.CHAT_FORMATS:
            errors.append(f"{model}: no chat format in the prompt layer")
        if model not in prompts.STOP:
            errors.append(f"{model}: no stop list in the prompt layer")

    for task in run.get("tasks", []):
        tpath = os.path.join("configs", "tasks", f"{task}.yaml")
        if not os.path.exists(tpath):
            errors.append(f"{where}: no task config for '{task}'")
            continue
        tcfg = ab.TaskConfig.from_yaml(tpath)
        if tcfg.template not in prompts.TASK_TEMPLATES:
            errors.append(f"{task}: template '{tcfg.template}' is not in "
                          f"the prompt layer")
        if tcfg.scorer not in KNOWN_SCORERS:
            errors.append(f"{task}: scorer '{tcfg.scorer}' is not one the "
                          f"runner can dispatch")
        adapter = tcfg.extra.get("adapter", tcfg.name)
        if adapter not in ab.ADAPTERS:
            errors.append(f"{task}: no adapter registered as '{adapter}'")
        if tcfg.scorer == "truthfulqa_mc" and \
                tcfg.extra.get("mode") != "choice":
            errors.append(f"{task}: uses the choice scorer but does not "
                          f"declare mode: choice, so the runner would "
                          f"route it through free generation")

    battery = run.get("battery")
    if battery and not os.path.exists(battery):
        errors.append(f"{where}: battery file not found ({battery})")


def main() -> None:
    targets = sys.argv[1:] or sorted(glob.glob("configs/runs/*.yaml"))
    for path in targets:
        check_run(path)

    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR:   {e}")
    print(f"\n{len(targets)} run configuration(s) checked: "
          f"{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

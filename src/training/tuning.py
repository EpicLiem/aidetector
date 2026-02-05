import copy
import itertools
import json
import os
import random
from typing import Any, Dict, Iterable, List

import yaml

from src.training.trainer import train_entry


def tune(config: Dict[str, Any]) -> Dict[str, Any]:
    tuning_cfg = config.get("tuning", {})
    params = tuning_cfg.get("params", {})
    if not params:
        raise ValueError("Tuning params are required. Set tuning.params in config.")

    strategy = str(tuning_cfg.get("strategy", "grid")).lower()
    seed = int(tuning_cfg.get("seed", config.get("seed", 42)))
    random.seed(seed)

    output_dir = tuning_cfg.get("output_dir", "outputs/tuning")
    os.makedirs(output_dir, exist_ok=True)

    trials = _build_trials(params, strategy, tuning_cfg.get("max_trials"), seed)
    metric = str(tuning_cfg.get("metric", "f1"))
    direction = str(tuning_cfg.get("direction", "maximize")).lower()

    results = []
    for idx, overrides in enumerate(trials, start=1):
        trial_config = copy.deepcopy(config)
        for key, value in overrides.items():
            _set_nested(trial_config, key, value)

        trial_dir = os.path.join(output_dir, f"trial_{idx:03d}")
        trial_config.setdefault("training", {})["output_dir"] = trial_dir

        training_cfg = trial_config.get("training", {})
        if str(training_cfg.get("device", "auto")).lower() == "xla" and bool(
            training_cfg.get("xla_distributed", False)
        ):
            raise ValueError(
                "XLA distributed is not supported for tuning. "
                "Set training.xla_distributed=false for tuning runs."
            )

        metrics = train_entry(trial_config)
        results.append(
            {
                "trial": idx,
                "overrides": overrides,
                "metrics": metrics,
                "output_dir": trial_dir,
            }
        )

    best = _select_best(results, metric, direction)
    _write_results(output_dir, results, best, config)
    return {"best": best, "results": results, "output_dir": output_dir}


def _build_trials(
    params: Dict[str, List[Any]], strategy: str, max_trials: int | None, seed: int
) -> Iterable[Dict[str, Any]]:
    for key, values in params.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Tuning param '{key}' must be a non-empty list.")

    if strategy == "random":
        rng = random.Random(seed)
        total = int(max_trials or 10)
        keys = sorted(params.keys())
        return [
            {key: rng.choice(params[key]) for key in keys} for _ in range(total)
        ]

    grid = list(_grid_trials(params))
    if max_trials is not None:
        grid = grid[: int(max_trials)]
    return grid


def _grid_trials(params: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = sorted(params.keys())
    values_list = [params[key] for key in keys]
    for values in itertools.product(*values_list):
        yield dict(zip(keys, values))


def _set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _select_best(
    results: List[Dict[str, Any]], metric: str, direction: str
) -> Dict[str, Any]:
    if not results:
        return {}

    def score(result: Dict[str, Any]) -> float:
        metrics = result.get("metrics", {}) or {}
        return float(metrics.get(metric, float("-inf")))

    reverse = direction != "minimize"
    return sorted(results, key=score, reverse=reverse)[0]


def _write_results(
    output_dir: str,
    results: List[Dict[str, Any]],
    best: Dict[str, Any],
    base_config: Dict[str, Any],
) -> None:
    results_path = os.path.join(output_dir, "tuning_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not best:
        return

    best_config = copy.deepcopy(base_config)
    overrides = best.get("overrides", {})
    for key, value in overrides.items():
        _set_nested(best_config, key, value)

    best_config_path = os.path.join(output_dir, "best_config.yaml")
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)

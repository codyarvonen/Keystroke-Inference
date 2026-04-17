#!/usr/bin/env python3
"""
Optuna hyperparameter search for Stage-1 Chronos training (in-process ``run_training``).

Expects a YAML config that is valid for ``train_stage1_chronos.py`` plus an ``hpo:`` block
(ignored by the trainer when run directly). See ``configs/experiments/exp_conv1d_phase2_500_500_hpo.yaml``.

Requires: pip install optuna pyyaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

JsonScalar = Union[str, int, float, bool, None]


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def _suggest(trial: Any, name: str, spec: Mapping[str, Any]) -> JsonScalar:
    st = str(spec["type"])
    if st == "log_float":
        return float(
            trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        )
    if st == "float":
        return float(trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=False))
    if st == "int":
        step = int(spec.get("step", 1))
        lo = int(spec["low"])
        hi = int(spec["high"])
        return int(trial.suggest_int(name, lo, hi, step=step))
    if st == "categorical":
        ch = spec["choices"]
        if not isinstance(ch, (list, tuple)) or not ch:
            raise ValueError(f"categorical {name}: non-empty choices required")
        return trial.suggest_categorical(name, list(ch))
    raise ValueError(f"Unknown search type {st!r} for param {name!r}")


def _bool_cli(flag_base: str, value: bool) -> List[str]:
    # e.g. freeze_encoder -> --freeze-encoder / --no-freeze-encoder
    dash = "--" + flag_base.replace("_", "-")
    if value:
        return [dash]
    return [f"--no-{flag_base.replace('_', '-')}"]


def _kv_cli(flag_base: str, value: JsonScalar) -> List[str]:
    if isinstance(value, bool):
        return _bool_cli(flag_base, value)
    dash = "--" + flag_base.replace("_", "-")
    return [dash, str(value)]


def _suggested_to_argv(suggested: Mapping[str, JsonScalar]) -> List[str]:
    out: List[str] = []
    for k, v in suggested.items():
        out.extend(_kv_cli(k, v))
    return out


def _make_pruner(
    name: str,
    *,
    n_startup_trials: int,
) -> Optional[Any]:
    import optuna

    n = str(name).lower().strip()
    if n in ("none", "off", ""):
        return None
    if n == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(n_startup_trials),
            n_warmup_steps=0,
        )
    raise ValueError(f"Unknown pruner {name!r} (use median or none)")


def main() -> None:
    import optuna
    from train_stage1_chronos import parse_args, run_training

    p = argparse.ArgumentParser(description="Optuna HPO for train_stage1_chronos (in-process).")
    p.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="YAML with training keys + hpo: block (see exp_conv1d_phase2_500_500_hpo.yaml)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override hpo.n_trials from YAML",
    )
    args_cli = p.parse_args()

    cfg_path = Path(args_cli.config).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    raw = _load_yaml(cfg_path)
    hpo = raw.get("hpo")
    if not isinstance(hpo, dict):
        raise SystemExit("YAML must contain an 'hpo:' mapping with enabled, study_name, search, etc.")
    if not bool(hpo.get("enabled", False)):
        raise SystemExit("hpo.enabled is false; set to true to run search.")

    search = hpo.get("search")
    if not isinstance(search, dict) or not search:
        raise SystemExit("hpo.search must be a non-empty mapping of param_name -> spec dicts")

    study_name = str(hpo.get("study_name", "stage1_hpo"))
    storage = hpo.get("storage")
    n_trials = int(args_cli.n_trials if args_cli.n_trials is not None else hpo.get("n_trials", 20))
    direction = str(hpo.get("direction", "minimize")).lower()
    if direction not in ("minimize", "maximize"):
        raise ValueError("hpo.direction must be minimize or maximize")

    metric = str(hpo.get("metric", "val_loss"))
    if metric not in ("val_loss", "val_top1"):
        raise ValueError("hpo.metric must be val_loss or val_top1")

    pruner_name = str(hpo.get("pruner", "median"))
    pruner_n_startup = int(hpo.get("pruner_n_startup_trials", 5))
    min_epochs_before_prune = int(hpo.get("min_epochs_before_prune", 2))
    if min_epochs_before_prune < 1:
        raise ValueError("hpo.min_epochs_before_prune must be >= 1")

    trial_root = Path(str(hpo.get("trial_out_root", "checkpoints/hpo_optuna_trials"))).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)
    final_eval_enabled = bool(hpo.get("evaluate_test_for_best_trial", True))
    final_eval_split = str(hpo.get("best_trial_eval_split", "test"))
    if final_eval_split not in ("train", "val", "test"):
        raise ValueError("hpo.best_trial_eval_split must be one of train/val/test")

    if isinstance(storage, str) and storage.startswith("sqlite:///"):
        rel = storage[len("sqlite:///") :]
        sqlite_path = Path(rel)
        if not sqlite_path.is_absolute():
            sqlite_path = Path.cwd() / sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    norm_seed = hpo.get("norm_stats_seed")
    norm_seed_path = Path(norm_seed).resolve() if norm_seed else (trial_root / "shared_norm_stats.json")
    shared_log_file = Path(str(hpo.get("shared_log_file", trial_root / "hpo_trials.log"))).resolve()
    shared_log_file.parent.mkdir(parents=True, exist_ok=True)

    pruner = _make_pruner(pruner_name, n_startup_trials=pruner_n_startup)

    def objective(trial: optuna.Trial) -> float:
        suggested: Dict[str, JsonScalar] = {}
        for pname, pspec in search.items():
            if not isinstance(pspec, dict):
                raise TypeError(f"hpo.search.{pname} must be a dict, got {type(pspec)}")
            suggested[str(pname)] = _suggest(trial, str(pname), pspec)

        argv: List[str] = ["--config", str(cfg_path)]
        argv.extend(_suggested_to_argv(suggested))

        targs = parse_args(argv)
        if str(targs.early_stopping_metric) != metric:
            targs.early_stopping_metric = metric  # type: ignore[misc]

        out_dir = trial_root / f"trial_{trial.number:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        targs.out_dir = str(out_dir)
        targs.log_file = str(shared_log_file)
        targs.norm_stats_path = str(norm_seed_path)
        targs.eval_test_split = False

        try:
            metrics = run_training(
                targs,
                optuna_trial=trial,
                optuna_prune_min_epochs=min_epochs_before_prune,
            )
        except optuna.TrialPruned:
            raise
        trial.set_user_attr("test_top1", metrics.get("test_top1"))
        trial.set_user_attr("test_loss", metrics.get("test_loss"))
        trial.set_user_attr("best_epoch", metrics.get("best_epoch"))
        trial.set_user_attr("out_dir", metrics.get("out_dir"))
        return float(metrics["best_metric_value"])

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=str(storage),
            load_if_exists=True,
            direction=direction,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    try:
        best = study.best_trial
        final_eval_payload: Dict[str, Any] = {}
        if final_eval_enabled:
            best_out_dir_raw = str(best.user_attrs.get("out_dir") or "")
            if not best_out_dir_raw:
                raise RuntimeError("best trial missing out_dir user attr; cannot run final eval")
            best_out_dir = Path(best_out_dir_raw)
            head_ckpt = best_out_dir / "head.pt"
            meta_path = best_out_dir / "train_meta.json"
            eval_out_dir = best_out_dir / f"eval_{final_eval_split}_from_hpo_best"
            eval_cmd = [
                sys.executable,
                "evaluate_stage1_chronos.py",
                "--head-ckpt",
                str(head_ckpt),
                "--meta-path",
                str(meta_path),
                "--norm-stats",
                str(norm_seed_path),
                "--split",
                final_eval_split,
                "--out-dir",
                str(eval_out_dir),
            ]
            subprocess.run(eval_cmd, check=True)
            eval_metrics_path = eval_out_dir / "metrics.json"
            if eval_metrics_path.is_file():
                final_eval_payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
            final_eval_payload["out_dir"] = str(eval_out_dir.resolve())
        best_payload = {
            "best_trial_number": int(best.number),
            "best_value": float(best.value) if best.value is not None else None,
            "best_params": dict(best.params),
            "best_user_attrs": dict(best.user_attrs),
            "final_eval": final_eval_payload if final_eval_enabled else None,
        }
    except ValueError:
        best_payload = {
            "best_trial_number": None,
            "best_value": None,
            "best_params": {},
            "best_user_attrs": {},
            "note": "No completed trials (all pruned or failed).",
        }
    summary = {
        "study_name": study_name,
        "storage": storage,
        "n_trials": n_trials,
        **best_payload,
    }
    summ_path = trial_root / "hpo_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {summ_path}")


if __name__ == "__main__":
    main()

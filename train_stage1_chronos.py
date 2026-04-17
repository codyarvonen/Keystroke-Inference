#!/usr/bin/env python3
"""
Stage-1 training: Chronos-2 encoder + configurable classification head (frozen encoder by default, or last-N-block fine-tuning).

Requires: pip install "chronos-forecasting>=2.0" pyyaml

Uses Chronos2Pipeline.embed() when the encoder is frozen; with ``--encoder-finetune-last-n N`` uses ``model.encode`` for gradients on the last N encoder blocks.
With embedding cache, stores pre-pool tokens (B×V×L×D); default storage is on-disk memmap
(low RAM). Heads consume full tokens (MLP, attention pool, Conv1d, BiLSTM, etc.) or legacy
pooled (B×D) for linear heads.

Defaults can be loaded from a YAML file: --config configs/stage1_chronos.defaults.yaml
(CLI flags override). See TRAINING_DEFAULTS and configs/stage1_chronos.defaults.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data_loader.config import Stage1ExportConfig
from data_loader.stage1 import load_stage1_vocab
from data_loader.stage1_dataset import Stage1IMUKeyDataset
from data_loader.stage1_norm import (
    collate_stack_lr_pad_batch,
    compute_global_norm_stats,
    load_norm_stats,
    normalize_mv,
    pad_crop_temporal,
    save_norm_stats,
)
from models.stage1_heads import (
    STAGE1_HEAD_TYPES,
    build_stage1_head,
    head_config_from_args,
    trim_encoder_outputs,
)
from models.chronos_finetune import (
    chronos_encode_multivariate_grad,
    configure_chronos_encoder_finetune,
    set_encoder_finetune_train_mode,
)
from models.stage1_losses import (
    bincount_labels,
    build_stage1_criterion,
    criterion_forward,
)


def _require_chronos():
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
    except ImportError as e:
        print(
            "Missing chronos-forecasting. Install with:\n"
            '  pip install "chronos-forecasting>=2.0"',
            file=sys.stderr,
        )
        raise e


# Single source of truth for argparse dest names / YAML keys (snake_case).
TRAINING_DEFAULTS: Dict[str, Any] = {
    "export_dir": "stage1_export",
    "data_dir": "data",
    "chronos_model": "autogluon/chronos-2-small",
    "device": "cuda:1",
    "epochs": 5,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "num_workers": 4,
    "context_length": 256,
    "left_ms": 700,
    "right_ms": 150,
    "target_rate_hz": 100.0,
    "out_dir": "checkpoints/stage1_chronos",
    "log_file": None,
    "norm_stats_path": None,
    "eval_test_split": True,
    "recompute_norm": False,
    "seed": 42,
    "patience": 0,
    "min_delta": 0.0,
    "early_stopping_metric": "val_loss",
    "freeze_encoder": True,
    "use_embedding_cache": True,
    "embedding_cache_dir": None,
    "recompute_embedding_cache": False,
        "cache_test_embeddings": True,
        # "memmap": stream cache to disk (low RAM); "ram" legacy: full tensor in .pt (OOM on large splits)
        "embedding_cache_storage": "memmap",
    # Classification head (encoder frozen)
    "head_type": "mlp",
    "encoder_num_variates": 12,
    "head_hidden_dim": 256,
    "head_num_layers": 2,
    "head_dropout": 0.1,
    "head_conv_channels": 256,
    "head_conv_d_proj": 128,
    "head_conv_kernel_size": 3,
    "head_conv_num_layers": 2,
    "head_lstm_hidden_dim": 256,
    "head_lstm_num_layers": 1,
    # Trim encoder (B,V,L,D) before head: drop trailing Chronos-2 specials + optional leading time patches
    # (from left NaN pad) using per-sample window lengths when available.
    "encoder_output_trim": False,
    "encoder_output_drop_last_specials": 2,
    # None = use model chronos_config input_patch_size when trimming leading patches
    "chronos_input_patch_size": None,
    # Encoder fine-tuning: last N Chronos2Encoder blocks (+ final_layer_norm); 0 = frozen encoder
    "encoder_finetune_last_n": 0,
    "lr_encoder": 1.0e-5,
    # Loss / class balance
    "loss_type": "weighted_ce",
    "class_weight_mode": "inverse_sqrt",
    "focal_gamma": 2.0,
    "class_balance_beta": 0.9999,
}


def _flatten_training_yaml(raw: Any, allowed: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept flat keys or one level of grouping (e.g. paths:, training:). Unknown keys warn."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping (dictionary).")
    keys = frozenset(allowed.keys())
    out: Dict[str, Any] = {}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if k in ("hpo",):
                continue
            if k in keys:
                out[k] = v
            elif isinstance(v, dict):
                walk(v)
            else:
                warnings.warn(f"Ignoring unknown training config key {k!r}", stacklevel=3)

    walk(raw)
    return out


def _deprecated_encoder_trim_from_raw(raw: Any) -> Tuple[Optional[bool], Optional[int]]:
    """Read legacy YAML keys (encoder_output_masking, encoder_output_drop_last_tokens)."""
    mask: Optional[bool] = None
    drop: Optional[int] = None

    def walk(node: Any) -> None:
        nonlocal mask, drop
        if not isinstance(node, dict):
            return
        if "encoder_output_masking" in node:
            mask = bool(node["encoder_output_masking"])
        if "encoder_output_drop_last_tokens" in node:
            drop = int(node["encoder_output_drop_last_tokens"])
        for v in node.values():
            walk(v)

    walk(raw)
    return mask, drop


def load_training_config_yaml(path: Path, allowed: Mapping[str, Any] = TRAINING_DEFAULTS) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        )
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    out = _flatten_training_yaml(loaded, allowed)
    dep_mask, dep_drop = _deprecated_encoder_trim_from_raw(loaded)
    if dep_mask is not None and "encoder_output_trim" not in out:
        out["encoder_output_trim"] = dep_mask
    if dep_drop is not None and "encoder_output_drop_last_specials" not in out:
        out["encoder_output_drop_last_specials"] = dep_drop
    return out


def dump_default_training_yaml() -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    # Comment header is in the example file; this is for --print-default-config / machine use.
    return yaml.safe_dump(
        TRAINING_DEFAULTS,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", type=str, default=None, metavar="PATH", help="YAML training defaults")
    pre.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print default settings as YAML and exit",
    )
    cli = sys.argv[1:] if argv is None else argv
    pre_args, rest = pre.parse_known_args(cli)
    if pre_args.print_default_config:
        if yaml is None:
            print("PyYAML is not installed; printing Python dict instead:\n", file=sys.stderr)
            print(json.dumps(TRAINING_DEFAULTS, indent=2, default=str))
            sys.exit(0)
        print(dump_default_training_yaml(), end="")
        sys.exit(0)

    merged: Dict[str, Any] = {**TRAINING_DEFAULTS}
    if pre_args.config:
        path = Path(pre_args.config)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        loaded = load_training_config_yaml(path)
        merged.update(loaded)

    m = merged
    p = argparse.ArgumentParser(description="Train Stage-1 keystroke classifier (Chronos-2 + head).")
    p.add_argument(
        "--config",
        "-c",
        type=str,
        default=pre_args.config,
        metavar="PATH",
        help="YAML file with training defaults (CLI flags override)",
    )
    p.add_argument("--export-dir", type=str, default=m["export_dir"], help="Dir with vocab.json + *.jsonl")
    p.add_argument("--data-dir", type=str, default=m["data_dir"], help="Raw IMU/PKL (must match export)")
    p.add_argument(
        "--chronos-model",
        type=str,
        default=m["chronos_model"],
        help="HF model id (default: autogluon/chronos-2-small)",
    )
    p.add_argument("--device", type=str, default=m["device"], help="e.g. cuda:1 or cpu")
    p.add_argument("--epochs", type=int, default=m["epochs"])
    p.add_argument("--batch-size", type=int, default=m["batch_size"])
    p.add_argument("--lr", type=float, default=m["lr"])
    p.add_argument("--weight-decay", type=float, default=m["weight_decay"])
    p.add_argument("--num-workers", type=int, default=m["num_workers"])
    p.add_argument(
        "--context-length",
        type=int,
        default=m["context_length"],
        help="Fixed time steps passed to Chronos (pad left / crop tail)",
    )
    p.add_argument("--left-ms", type=int, default=m["left_ms"])
    p.add_argument("--right-ms", type=int, default=m["right_ms"])
    p.add_argument("--target-rate-hz", type=float, default=m["target_rate_hz"])
    p.add_argument("--out-dir", type=str, default=m["out_dir"], help="Norm stats + head weights")
    p.add_argument(
        "--log-file",
        type=str,
        default=m["log_file"],
        help="Training log path (default: <out-dir>/train_<timestamp>.log, new file each run)",
    )
    p.add_argument(
        "--recompute-norm",
        action=argparse.BooleanOptionalAction,
        default=m["recompute_norm"],
        help="Recompute train norm stats even if file exists (default: from config or false)",
    )
    p.add_argument(
        "--norm-stats-path",
        type=str,
        default=m["norm_stats_path"],
        help="Optional shared norm stats JSON path (default: <out-dir>/norm_stats.json)",
    )
    p.add_argument(
        "--eval-test-split",
        action=argparse.BooleanOptionalAction,
        default=m["eval_test_split"],
        help="Run final test split evaluation after training (disable for HPO speed).",
    )
    p.add_argument("--seed", type=int, default=m["seed"])
    p.add_argument(
        "--patience",
        type=int,
        default=m["patience"],
        help="Stop after this many epochs without val improvement (0 = disabled; train all --epochs)",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=m["min_delta"],
        help="Minimum change in the monitored metric to count as improvement",
    )
    p.add_argument(
        "--early-stopping-metric",
        type=str,
        default=m["early_stopping_metric"],
        choices=("val_loss", "val_top1"),
        help="val_loss: lower is better; val_top1: higher is better",
    )
    p.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=m["freeze_encoder"],
        help="Freeze Chronos encoder parameters (recommended for linear-head training)",
    )
    p.add_argument(
        "--use-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=m["use_embedding_cache"],
        help="Cache pre-pool (V×L×D) encoder tokens; train applies pooling on the fly when encoder is frozen",
    )
    p.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=m["embedding_cache_dir"],
        help="Override embedding cache directory (default: <out-dir>/embedding_cache)",
    )
    p.add_argument(
        "--recompute-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=m["recompute_embedding_cache"],
        help="Rebuild embedding cache even when a compatible cache already exists",
    )
    p.add_argument(
        "--cache-test-embeddings",
        action=argparse.BooleanOptionalAction,
        default=m["cache_test_embeddings"],
        help="Also cache test embeddings for faster final test evaluation",
    )
    p.add_argument(
        "--embedding-cache-storage",
        type=str,
        default=m["embedding_cache_storage"],
        choices=("memmap", "ram"),
        help="memmap: stream cache to disk (low RAM, fast training reads). "
        "ram: legacy single .pt tensor (small splits only).",
    )
    p.add_argument(
        "--head-type",
        type=str,
        default=m["head_type"],
        choices=STAGE1_HEAD_TYPES,
        help="Classifier on top of frozen encoder tokens (see models/stage1_heads.py)",
    )
    p.add_argument(
        "--encoder-num-variates",
        type=int,
        default=m["encoder_num_variates"],
        help="Chronos encoder variate count V (probe overrides; default 12 for 6+6 IMU)",
    )
    p.add_argument("--head-hidden-dim", type=int, default=m["head_hidden_dim"])
    p.add_argument("--head-num-layers", type=int, default=m["head_num_layers"], help="MLP depth (hidden layers)")
    p.add_argument("--head-dropout", type=float, default=m["head_dropout"])
    p.add_argument("--head-conv-channels", type=int, default=m["head_conv_channels"])
    p.add_argument("--head-conv-d-proj", type=int, default=m["head_conv_d_proj"])
    p.add_argument("--head-conv-kernel-size", type=int, default=m["head_conv_kernel_size"])
    p.add_argument("--head-conv-num-layers", type=int, default=m["head_conv_num_layers"])
    p.add_argument("--head-lstm-hidden-dim", type=int, default=m["head_lstm_hidden_dim"])
    p.add_argument("--head-lstm-num-layers", type=int, default=m["head_lstm_num_layers"])
    p.add_argument(
        "--encoder-output-trim",
        action=argparse.BooleanOptionalAction,
        default=m["encoder_output_trim"],
        help="Trim encoder (B,V,L,D) before head: drop Chronos-2 trailing specials + leading pad-only patches "
        "when per-sample lengths are available.",
    )
    p.add_argument(
        "--encoder-output-drop-last-specials",
        type=int,
        default=m["encoder_output_drop_last_specials"],
        metavar="K",
        help="Trailing L slots to remove when trimming (Chronos-2: 2 for REG + output-patch token).",
    )
    p.add_argument(
        "--chronos-input-patch-size",
        type=int,
        default=m["chronos_input_patch_size"],
        metavar="P",
        help="Patch size for leading trim (default: from loaded Chronos model chronos_config).",
    )
    p.add_argument(
        "--encoder-finetune-last-n",
        type=int,
        default=m["encoder_finetune_last_n"],
        metavar="N",
        help="Fine-tune the last N Chronos2 encoder transformer blocks (+ final_layer_norm). 0 = frozen encoder.",
    )
    p.add_argument(
        "--lr-encoder",
        type=float,
        default=m["lr_encoder"],
        help="Learning rate for unfrozen encoder parameters (when encoder_finetune_last_n > 0).",
    )
    p.add_argument(
        "--loss-type",
        type=str,
        default=m["loss_type"],
        choices=("ce", "weighted_ce", "focal", "weighted_focal"),
        help="ce=unweighted softmax; weighted_ce/focal* use --class-weight-mode",
    )
    p.add_argument(
        "--class-weight-mode",
        type=str,
        default=m["class_weight_mode"],
        choices=("none", "inverse", "inverse_sqrt", "effective"),
        help="Train-set frequency weights (effective = Class-Balanced beta)",
    )
    p.add_argument("--focal-gamma", type=float, default=m["focal_gamma"])
    p.add_argument(
        "--class-balance-beta",
        type=float,
        default=m["class_balance_beta"],
        help="Beta for effective-number class weights (only if --class-weight-mode effective)",
    )
    args = p.parse_args(rest)
    if args.early_stopping_metric not in ("val_loss", "val_top1"):
        raise ValueError(f"Invalid early_stopping_metric: {args.early_stopping_metric!r}")
    if args.loss_type == "ce" and args.class_weight_mode != "none":
        raise ValueError("loss_type=ce requires class_weight_mode=none (use weighted_ce or focal variants).")
    if args.loss_type in ("weighted_ce", "weighted_focal") and args.class_weight_mode == "none":
        raise ValueError(f"{args.loss_type} requires class_weight_mode != none")
    if int(args.encoder_output_drop_last_specials) < 0:
        raise ValueError("encoder_output_drop_last_specials must be >= 0")
    if int(args.encoder_finetune_last_n) < 0:
        raise ValueError("encoder_finetune_last_n must be >= 0")
    return args


def resolved_training_config(ns: argparse.Namespace) -> Dict[str, Any]:
    """Snapshot of all tunables (YAML + CLI) for train_meta.json."""
    return {k: getattr(ns, k) for k in TRAINING_DEFAULTS}


def setup_logging(log_path: Path) -> logging.Logger:
    """Console + file (same messages) for diagnosis."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_stage1_chronos")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def read_chronos_patch_info(pipeline: Any) -> Dict[str, Any]:
    """Best-effort read of input patch hyperparameters from the loaded model."""
    m = pipeline.model
    cfg = getattr(m, "config", m)
    cc = getattr(cfg, "chronos_config", None)
    if cc is None and hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
        nested = d.get("chronos_config")
        if isinstance(nested, dict):
            cc = nested
    out: Dict[str, Any] = {}
    if cc is None:
        return out
    keys = ("input_patch_size", "input_patch_stride", "context_length", "output_patch_size")
    for k in keys:
        if isinstance(cc, dict):
            v = cc.get(k)
        else:
            v = getattr(cc, k, None)
        if v is not None:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vocab_num_classes(vocab_path: Path) -> int:
    token_to_id, _ = load_stage1_vocab(vocab_path)
    return int(max(token_to_id.values())) + 1


def make_stage1_cfg(args: argparse.Namespace) -> Stage1ExportConfig:
    return Stage1ExportConfig(
        data_dir=args.data_dir,
        left_context_ms=args.left_ms,
        right_context_ms=args.right_ms,
        target_rate_hz=args.target_rate_hz,
    )


def compute_train_label_counts(train_ds: Stage1IMUKeyDataset, num_classes: int) -> torch.Tensor:
    """Per-class counts on the training split (from JSONL rows)."""
    ids = [int(r["label_id"]) for r in train_ds._rows]
    t = torch.tensor(ids, dtype=torch.long)
    return bincount_labels(t, num_classes)


def collate_chronos_batch(
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
) -> Any:
    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = collate_stack_lr_pad_batch(batch)
        x = normalize_mv(base["imu_mv"], mean, std)
        lengths = base["lengths"].clone()
        x = pad_crop_temporal(x, context_length)
        return {"imu_mv": x, "label_id": base["label_id"], "lengths": lengths}

    return _fn


def pool_embedding_list(emb_list: List[torch.Tensor]) -> torch.Tensor:
    """(n_variates, L, D) or similar -> mean pool -> (D,). Stack batch -> (B, D)."""
    pooled = []
    for emb in emb_list:
        e = emb.float()
        d = e.shape[-1]
        v = e.reshape(-1, d).mean(dim=0)
        pooled.append(v)
    return torch.stack(pooled, dim=0)


def stack_embeddings_from_embed_list(emb_list: List[torch.Tensor]) -> torch.Tensor:
    """Chronos embed() batch output: list length B of (V, L, D) -> (B, V, L, D)."""
    return torch.stack([e.float() for e in emb_list], dim=0)


def apply_stage1_encoder_trim(
    z: torch.Tensor,
    lengths: Optional[torch.Tensor],
    *,
    enabled: bool,
    context_length: int,
    input_patch_size: int,
    drop_last_specials: int,
) -> torch.Tensor:
    """Trim ``(B,V,L,D)`` for the head (specials + optional leading pad-only patches). See ``trim_encoder_outputs``."""
    return trim_encoder_outputs(
        z,
        enabled=enabled,
        context_length=context_length,
        input_patch_size=input_patch_size,
        lengths=lengths,
        drop_last_specials=drop_last_specials,
        leading_trim=True,
    )


def pool_stacked_encoder_embeddings(stacked: torch.Tensor) -> torch.Tensor:
    """
    Batch equivalent of pool_embedding_list: (B, V, L, D) -> (B, D) mean over variates and time.
    Use after loading pre-pool cached encoder tokens.
    """
    if stacked.ndim != 4:
        raise ValueError(f"Expected encoder stack (B, V, L, D), got shape {tuple(stacked.shape)}")
    b, v, l, d = stacked.shape
    return stacked.reshape(b, v * l, d).mean(dim=1)


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    c = logits.size(1)
    max_k = min(max(ks), c)
    _, pred = logits.topk(max_k, dim=1)
    targets = targets.view(-1, 1)
    for k in ks:
        kk = min(k, c)
        correct = pred[:, :kk].eq(targets)
        out[f"top{k}"] = float(correct.any(dim=1).float().mean().item())
    return out


@torch.no_grad()
def evaluate(
    pipeline: Any,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    criterion: nn.Module,
    *,
    encoder_output_trim: bool = False,
    encoder_output_drop_last_specials: int = 2,
    chronos_input_patch_size: int = 16,
    encoder_finetune_last_n: int = 0,
) -> Dict[str, float]:
    head.eval()
    if encoder_finetune_last_n > 0:
        set_encoder_finetune_train_mode(pipeline.model, encoder_finetune_last_n, training=False)
    totals = {"n": 0, "loss": 0.0}
    acc_sums = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    for batch in loader:
        imu_cpu = batch["imu_mv"].float()
        y = batch["label_id"].to(device)
        b = imu_cpu.shape[0]
        lengths = batch.get("lengths")
        if lengths is not None:
            lengths = lengths.to(device=device, dtype=torch.long)
        if encoder_finetune_last_n > 0:
            imu = imu_cpu.to(device)
            stacked = chronos_encode_multivariate_grad(
                pipeline, imu, context_length=context_length
            ).to(device, dtype=torch.float32)
        else:
            # Chronos embed() manages its own loader/pin-memory path; pass CPU tensors here.
            emb_list, _ = pipeline.embed(imu_cpu, batch_size=b, context_length=context_length)
            stacked = stack_embeddings_from_embed_list(emb_list).to(device, dtype=torch.float32)
        stacked = apply_stage1_encoder_trim(
            stacked,
            lengths,
            enabled=encoder_output_trim,
            context_length=context_length,
            input_patch_size=chronos_input_patch_size,
            drop_last_specials=encoder_output_drop_last_specials,
        )
        logits = head(stacked)
        loss = criterion_forward(criterion, logits, y)
        totals["loss"] += float(loss.item()) * b
        totals["n"] += b
        acc = topk_accuracy(logits, y, (1, 3, 5))
        acc_sums["top1"] += acc["top1"] * b
        acc_sums["top3"] += acc["top3"] * b
        acc_sums["top5"] += acc["top5"] * b
    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "top1": acc_sums["top1"] / n,
        "top3": acc_sums["top3"] / n,
        "top5": acc_sums["top5"] / n,
    }


class MemmapCachedEmbeddingDataset(Dataset):
    """Disk-backed fp16 (V,L,D) encoder tokens + int64 labels + optional raw window lengths."""

    def __init__(
        self,
        feat_path: Path,
        lab_path: Path,
        num_samples: int,
        shape_vld: Tuple[int, int, int],
        len_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._n = int(num_samples)
        v, l, d = shape_vld
        self._feat = np.memmap(
            feat_path,
            dtype=np.float16,
            mode="r",
            shape=(self._n, v, l, d),
        )
        self._lab = np.memmap(
            lab_path,
            dtype=np.int64,
            mode="r",
            shape=(self._n,),
        )
        self._len_mm: Optional[np.memmap]
        if len_path is not None and len_path.is_file():
            self._len_mm = np.memmap(
                len_path,
                dtype=np.int64,
                mode="r",
                shape=(self._n,),
            )
        else:
            self._len_mm = None

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self._feat[idx])).float()
        y = torch.tensor(int(self._lab[idx]), dtype=torch.long)
        if self._len_mm is not None:
            ln = torch.tensor(int(self._len_mm[idx]), dtype=torch.long)
        else:
            ln = torch.tensor(-1, dtype=torch.long)
        return x, y, ln


def _stable_json_hash(payload: Mapping[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_signature_payload(
    args: argparse.Namespace,
    split_path: Path,
    stats_path: Path,
    patch_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "split_path": str(split_path.resolve()),
        "split_mtime_ns": split_path.stat().st_mtime_ns,
        "norm_stats_path": str(stats_path.resolve()),
        "norm_stats_mtime_ns": stats_path.stat().st_mtime_ns,
        "chronos_model": args.chronos_model,
        "context_length": int(args.context_length),
        "left_ms": int(args.left_ms),
        "right_ms": int(args.right_ms),
        "target_rate_hz": float(args.target_rate_hz),
        "data_dir": str(Path(args.data_dir).resolve()),
        "patch_info": dict(patch_info),
        "cache_format": "encoder_tokens_pre_pool",
        "cache_storage_dtype": "float16",
    }


@torch.no_grad()
def _try_load_legacy_pt_cache(
    cache_path: Path,
    logger: logging.Logger,
    split_name: str,
) -> Optional[TensorDataset]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    h = payload["features"]
    y = payload["labels"]
    fmt = payload.get("cache_format", "legacy_pooled")
    ok = (
        y.ndim == 1
        and int(h.shape[0]) == int(y.shape[0])
        and (
            (fmt == "encoder_tokens_pre_pool" and h.ndim == 4)
            or (fmt == "legacy_pooled" and h.ndim == 2)
        )
    )
    if not ok:
        return None
    if h.ndim == 4:
        _, v, l, d = h.shape
        lens = payload.get("lengths")
        if lens is None or int(lens.shape[0]) != int(h.shape[0]):
            lens = torch.full((int(h.shape[0]),), -1, dtype=torch.long)
        logger.info(
            "Loaded RAM (.pt) embedding cache for %s: %s (n=%d, shape=V×L×D=%d×%d×%d, pre-pool)",
            split_name,
            cache_path,
            h.shape[0],
            v,
            l,
            d,
        )
        return TensorDataset(h, y, lens)
    logger.info(
        "Loaded legacy pooled RAM cache for %s: %s (n=%d, d_model=%d)",
        split_name,
        cache_path,
        h.shape[0],
        h.shape[1],
    )
    return TensorDataset(h, y)


@torch.no_grad()
def _build_or_load_cached_embeddings(
    *,
    split_name: str,
    split_path: Path,
    raw_loader: DataLoader,
    pipeline: Any,
    context_length: int,
    cache_dir: Path,
    stats_path: Path,
    patch_info: Mapping[str, Any],
    args: argparse.Namespace,
    logger: logging.Logger,
    force_recompute: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build or load cached encoder features. Returns (dataset, cache_meta) where dataset is
    MemmapCachedEmbeddingDataset or TensorDataset, both compatible with DataLoader default collate.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage = str(getattr(args, "embedding_cache_storage", "memmap"))
    if storage not in ("memmap", "ram"):
        raise ValueError(f"embedding_cache_storage must be memmap or ram, got {storage!r}")

    legacy_pt_path = cache_dir / f"{split_name}_embeddings.pt"
    feat_mm_path = cache_dir / f"{split_name}_embed_features.mmap"
    lab_mm_path = cache_dir / f"{split_name}_embed_labels.mmap"
    len_mm_path = cache_dir / f"{split_name}_embed_lengths.mmap"
    meta_path = cache_dir / f"{split_name}_cache_meta.json"

    sig_payload = _cache_signature_payload(args=args, split_path=split_path, stats_path=stats_path, patch_info=patch_info)
    sig_hash = _stable_json_hash(sig_payload)

    if (not force_recompute) and meta_path.is_file():
        try:
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cache_meta.get("signature_hash") == sig_hash:
                if feat_mm_path.is_file() and lab_mm_path.is_file():
                    n = int(cache_meta["num_samples"])
                    v_l_d = cache_meta.get("encoder_shape_V_L_D", cache_meta.get("encoder_shape_vld"))
                    if not v_l_d or len(v_l_d) != 3:
                        raise ValueError("cache meta missing encoder_shape_V_L_D")
                    v, l, d_model = (int(x) for x in v_l_d)
                    logger.info(
                        "Loaded memmap embedding cache for %s: %s (%d samples, V×L×D=%d×%d×%d)",
                        split_name,
                        feat_mm_path,
                        n,
                        v,
                        l,
                        d_model,
                    )
                    ds = MemmapCachedEmbeddingDataset(
                        feat_mm_path,
                        lab_mm_path,
                        n,
                        (v, l, d_model),
                        len_path=len_mm_path if len_mm_path.is_file() else None,
                    )
                    return ds, cache_meta
                if legacy_pt_path.is_file() and storage == "ram":
                    loaded = _try_load_legacy_pt_cache(legacy_pt_path, logger, split_name)
                    if loaded is not None:
                        return loaded, cache_meta
                if legacy_pt_path.is_file() and storage == "memmap":
                    logger.warning(
                        "Only legacy RAM cache %s exists; rebuilding on-disk memmap (one-time embed pass). "
                        "Remove the .pt file and use --recompute-embedding-cache to force.",
                        legacy_pt_path.name,
                    )
        except Exception as e:  # pragma: no cover
            logger.warning("Could not load existing embedding cache for %s (%s); rebuilding", split_name, e)

    # Rebuild: remove stale files for this split so we do not mix formats
    for p in (feat_mm_path, lab_mm_path, len_mm_path, legacy_pt_path):
        if p.is_file():
            p.unlink()

    logger.info(
        "Building embedding cache for %s (pre-pool encoder tokens, storage=%s) …",
        split_name,
        storage,
    )
    t0 = time.perf_counter()
    n_total = len(raw_loader.dataset)
    if n_total == 0:
        raise RuntimeError(f"Cannot build embedding cache for {split_name}: split has zero samples.")

    if storage == "memmap":
        offset = 0
        mm_f: Optional[np.memmap] = None
        mm_y: Optional[np.memmap] = None
        mm_ln: Optional[np.memmap] = None
        v_ld: Optional[Tuple[int, int, int]] = None
        for batch in raw_loader:
            imu = batch["imu_mv"].float()
            y = batch["label_id"].to(torch.long).cpu()
            lens = batch["lengths"].to(torch.long).cpu()
            b = int(imu.shape[0])
            emb_list, _ = pipeline.embed(imu, batch_size=b, context_length=context_length)
            stacked = stack_embeddings_from_embed_list(emb_list).to(dtype=torch.float32, device="cpu")
            if mm_f is None:
                v1, l1, d1 = (int(stacked.shape[i]) for i in (1, 2, 3))
                v_ld = (v1, l1, d1)
                mm_f = np.memmap(
                    feat_mm_path,
                    dtype=np.float16,
                    mode="w+",
                    shape=(n_total, v1, l1, d1),
                )
                mm_y = np.memmap(
                    lab_mm_path,
                    dtype=np.int64,
                    mode="w+",
                    shape=(n_total,),
                )
                mm_ln = np.memmap(
                    len_mm_path,
                    dtype=np.int64,
                    mode="w+",
                    shape=(n_total,),
                )
            assert mm_y is not None and mm_ln is not None and v_ld is not None
            mm_f[offset : offset + b] = stacked.numpy().astype(np.float16)
            mm_y[offset : offset + b] = y.numpy().astype(np.int64)
            mm_ln[offset : offset + b] = lens.numpy().astype(np.int64)
            offset += b
        if offset != n_total:
            raise RuntimeError(
                f"Embedding cache row count mismatch for {split_name}: wrote {offset}, expected {n_total}"
            )
        if mm_f is not None:
            mm_f.flush()
        if mm_y is not None:
            mm_y.flush()
        if mm_ln is not None:
            mm_ln.flush()
        del mm_f, mm_y, mm_ln

        v, l, d_model = v_ld  # type: ignore[misc]
        nbytes = int(n_total * v * l * d_model * 2)
        cache_meta = {
            "split": split_name,
            "storage": "memmap",
            "features_path": feat_mm_path.name,
            "labels_path": lab_mm_path.name,
            "lengths_path": len_mm_path.name,
            "created_at": datetime.now().isoformat(),
            "signature_hash": sig_hash,
            "signature_payload": sig_payload,
            "num_samples": n_total,
            "encoder_shape_V_L_D": [v, l, d_model],
            "d_model": d_model,
            "cache_format": "encoder_tokens_pre_pool",
            "approx_bytes_on_disk": nbytes + n_total * 8 + n_total * 8,
        }
        meta_path.write_text(json.dumps(cache_meta, indent=2, default=str), encoding="utf-8")
        logger.info(
            "Saved memmap embedding cache for %s: %s (n=%d, V×L×D=%d×%d×%d, fp16 disk ~%.1f MiB, %.1fs)",
            split_name,
            feat_mm_path,
            n_total,
            v,
            l,
            d_model,
            nbytes / (1024**2),
            time.perf_counter() - t0,
        )
        ds = MemmapCachedEmbeddingDataset(
            feat_mm_path, lab_mm_path, n_total, (v, l, d_model), len_path=len_mm_path
        )
        return ds, cache_meta

    # storage == "ram": legacy in-memory concat + single .pt
    h_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    len_chunks: List[torch.Tensor] = []
    for batch in raw_loader:
        imu = batch["imu_mv"].float()
        y = batch["label_id"].to(torch.long).cpu()
        lens = batch["lengths"].to(torch.long).cpu()
        b = int(imu.shape[0])
        emb_list, _ = pipeline.embed(imu, batch_size=b, context_length=context_length)
        stacked = stack_embeddings_from_embed_list(emb_list).to(dtype=torch.float32, device="cpu")
        h_chunks.append(stacked)
        y_chunks.append(y)
        len_chunks.append(lens)

    h_all = torch.cat(h_chunks, dim=0).contiguous()
    y_all = torch.cat(y_chunks, dim=0).contiguous()
    lens_all = torch.cat(len_chunks, dim=0).contiguous()
    if int(h_all.shape[0]) != int(y_all.shape[0]):
        raise RuntimeError("Embedding cache size mismatch between features and labels.")

    h_save = h_all.half()
    nbytes = int(h_save.numel() * h_save.element_size())
    torch.save(
        {
            "features": h_save,
            "labels": y_all,
            "lengths": lens_all,
            "cache_format": "encoder_tokens_pre_pool",
            "cache_storage_dtype": "float16",
        },
        legacy_pt_path,
    )
    v, l, d_model = int(h_all.shape[1]), int(h_all.shape[2]), int(h_all.shape[3])
    cache_meta = {
        "split": split_name,
        "storage": "ram",
        "cache_path": str(legacy_pt_path.resolve()),
        "created_at": datetime.now().isoformat(),
        "signature_hash": sig_hash,
        "signature_payload": sig_payload,
        "num_samples": int(h_all.shape[0]),
        "encoder_shape_V_L_D": [v, l, d_model],
        "d_model": d_model,
        "cache_format": "encoder_tokens_pre_pool",
        "approx_bytes_on_disk": nbytes
        + int(y_all.numel() * y_all.element_size())
        + int(lens_all.numel() * lens_all.element_size()),
    }
    meta_path.write_text(json.dumps(cache_meta, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Saved RAM (.pt) embedding cache for %s: %s (n=%d, V×L×D=%d×%d×%d, fp16 disk ~%.1f MiB, %.1fs)",
        split_name,
        legacy_pt_path,
        h_all.shape[0],
        v,
        l,
        d_model,
        nbytes / (1024**2),
        time.perf_counter() - t0,
    )
    return TensorDataset(h_all, y_all, lens_all), cache_meta


@torch.no_grad()
def evaluate_cached(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    context_length: int,
    encoder_output_trim: bool = False,
    encoder_output_drop_last_specials: int = 2,
    chronos_input_patch_size: int = 16,
) -> Dict[str, float]:
    head.eval()
    totals = {"n": 0, "loss": 0.0}
    acc_sums = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            emb_cpu, y_cpu, len_cpu = batch
            lengths = len_cpu.to(device=device, non_blocking=True, dtype=torch.long)
        else:
            emb_cpu, y_cpu = batch
            lengths = None
        emb = emb_cpu.to(device=device, non_blocking=True)
        if emb.dtype == torch.float16:
            emb = emb.float()
        if emb.ndim == 4:
            emb = apply_stage1_encoder_trim(
                emb,
                lengths,
                enabled=encoder_output_trim,
                context_length=context_length,
                input_patch_size=chronos_input_patch_size,
                drop_last_specials=encoder_output_drop_last_specials,
            )
            logits = head(emb)
        elif emb.ndim == 2:
            logits = head(emb)
        else:
            raise ValueError(f"Unexpected cached feature shape {tuple(emb.shape)}")
        y = y_cpu.to(device=device, non_blocking=True)
        b = int(logits.shape[0])
        loss = criterion_forward(criterion, logits, y)
        totals["loss"] += float(loss.item()) * b
        totals["n"] += b
        acc = topk_accuracy(logits, y, (1, 3, 5))
        acc_sums["top1"] += acc["top1"] * b
        acc_sums["top3"] += acc["top3"] * b
        acc_sums["top5"] += acc["top5"] * b
    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "top1": acc_sums["top1"] / n,
        "top3": acc_sums["top3"] / n,
        "top5": acc_sums["top5"] / n,
    }


def _clone_state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _build_head_ckpt(
    head: nn.Module,
    d_model: int,
    num_classes: int,
    context_length: int,
    chronos_model: str,
    args: argparse.Namespace,
    loss_meta: Dict[str, Any],
    *,
    pipeline: Optional[Any] = None,
    encoder_finetune_last_n: int = 0,
) -> Dict[str, Any]:
    hc = head_config_from_args(args)
    hc["d_model"] = d_model
    hc["num_classes"] = num_classes
    out: Dict[str, Any] = {
        "head_state_dict": head.state_dict(),
        "head_config": hc,
        "d_model": d_model,
        "num_classes": num_classes,
        "context_length": context_length,
        "chronos_model": chronos_model,
        "loss_config": loss_meta,
        "encoder_finetune_last_n": int(encoder_finetune_last_n),
    }
    if pipeline is not None and int(encoder_finetune_last_n) > 0:
        out["model_state_dict"] = pipeline.model.state_dict()
    return out


def run_training(
    args: argparse.Namespace,
    *,
    optuna_trial: Any = None,
    optuna_prune_min_epochs: int = 2,
) -> Dict[str, Any]:
    """Run one training job. If ``optuna_trial`` is set, reports val metrics and supports pruning."""
    _require_chronos()
    from chronos import Chronos2Pipeline
    finetune_n = int(getattr(args, "encoder_finetune_last_n", 0))
    if not args.freeze_encoder and finetune_n == 0:
        raise NotImplementedError(
            "Encoder fine-tuning: set --encoder-finetune-last-n N (N > 0) for last-N-block fine-tuning, "
            "or use --freeze-encoder (default) for head-only training."
        )
    set_seed(args.seed)
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = out_dir / f"train_{stamp}.log"
    logger = setup_logging(log_path)
    logger.info("Logging to %s", log_path.resolve())
    if args.config:
        logger.info("Config file: %s", Path(args.config).resolve())
    else:
        logger.info("Config file: (none; using built-in TRAINING_DEFAULTS unless overridden on CLI)")
    logger.info(
        "Run settings: epochs=%d batch_size=%d lr=%s patience=%d early_stopping_metric=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
        args.early_stopping_metric,
    )

    vocab_path = export_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Run export_stage1_data.py first.")

    num_classes = vocab_num_classes(vocab_path)
    stage_cfg = make_stage1_cfg(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        logger.info(
            "Training on CUDA device index %s (%s); resolved from args.device=%r",
            cuda_idx,
            torch.cuda.get_device_name(device),
            args.device,
        )

    logger.info("Loading Chronos: %s", args.chronos_model)
    t0 = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(args.chronos_model)
    pipeline.model.to(device)
    if finetune_n > 0:
        n_enc_layers = configure_chronos_encoder_finetune(pipeline.model, finetune_n)
        logger.info(
            "Encoder fine-tune: last %d transformer block(s) + final_layer_norm trainable (encoder depth=%d).",
            finetune_n,
            n_enc_layers,
        )
        if args.freeze_encoder:
            logger.warning(
                "encoder_finetune_last_n=%d: training last encoder blocks; ignoring freeze_encoder=True in logs/meta.",
                finetune_n,
            )
    else:
        for p in pipeline.model.parameters():
            p.requires_grad = False
    pipeline.model.eval()
    logger.info("Chronos loaded on %s in %.1fs", device, time.perf_counter() - t0)

    patch_info = read_chronos_patch_info(pipeline)
    if patch_info:
        logger.info("Chronos patch config: %s", patch_info)
        ips = patch_info.get("input_patch_size")
        if ips and args.target_rate_hz > 0:
            ms = 1000.0 * ips / args.target_rate_hz
            est = math.ceil(args.context_length / ips) if args.context_length else None
            logger.info(
                "At %.1f Hz, input_patch_size=%d steps ≈ %.1f ms per patch; "
                "context_length=%d → ~%s input patches along time (per series before specials).",
                args.target_rate_hz,
                ips,
                ms,
                args.context_length,
                est,
            )
    else:
        logger.info("Could not read chronos_config from model (patch info unavailable).")

    if getattr(args, "chronos_input_patch_size", None) is None:
        args.chronos_input_patch_size = int((patch_info or {}).get("input_patch_size", 16) or 16)
    else:
        args.chronos_input_patch_size = int(args.chronos_input_patch_size)

    train_ds = Stage1IMUKeyDataset(export_dir / "train.jsonl", stage_cfg)
    val_ds = Stage1IMUKeyDataset(export_dir / "val.jsonl", stage_cfg)
    test_ds = Stage1IMUKeyDataset(export_dir / "test.jsonl", stage_cfg)

    if args.norm_stats_path:
        stats_path = Path(args.norm_stats_path)
    else:
        stats_path = out_dir / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists() and not args.recompute_norm:
        mean, std, _eps = load_norm_stats(stats_path)
        logger.info("Loaded norm stats from %s", stats_path)
    else:
        logger.info("Computing train-only global norm stats …")
        mean, std = compute_global_norm_stats(train_ds, batch_size=256, num_workers=0)
        save_norm_stats(stats_path, mean, std, 1e-6)
        logger.info("Saved norm stats to %s", stats_path)

    collate_fn = collate_chronos_batch(mean, std, args.context_length)

    train_loader_raw = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader_raw = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader_raw = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Infer d_model from one batch
    probe = next(iter(train_loader_raw))
    imu0 = probe["imu_mv"].float()
    b0 = imu0.shape[0]
    n_nan = torch.isnan(imu0).sum().item()
    n_tot = imu0.numel()
    logger.info(
        "Probe batch: imu_mv shape=%s, nan_values=%d/%d (left-pad / align)",
        tuple(imu0.shape),
        n_nan,
        n_tot,
    )
    with torch.no_grad():
        emb0, _ = pipeline.embed(imu0, batch_size=b0, context_length=args.context_length)
    d_model = int(emb0[0].shape[-1])
    args.encoder_num_variates = int(emb0[0].shape[0])
    logger.info(
        "Embed probe: first series emb shape=%s, d_model=%d, V=%d, num_classes=%d",
        tuple(emb0[0].shape),
        d_model,
        args.encoder_num_variates,
        num_classes,
    )

    class_counts = compute_train_label_counts(train_ds, num_classes)
    criterion, loss_meta = build_stage1_criterion(
        loss_type=args.loss_type,
        num_classes=num_classes,
        class_counts=class_counts,
        class_weight_mode=args.class_weight_mode,
        focal_gamma=args.focal_gamma,
        class_balance_beta=args.class_balance_beta,
        device=device,
    )
    logger.info(
        "Classifier head: type=%s hidden=%d dropout=%s | loss=%s class_weight_mode=%s",
        args.head_type,
        args.head_hidden_dim,
        args.head_dropout,
        args.loss_type,
        args.class_weight_mode,
    )
    logger.info(
        "Encoder output trim: enabled=%s drop_last_specials=%d patch_size=%d (before head; cache stores full L + lengths sidecar)",
        bool(args.encoder_output_trim),
        int(args.encoder_output_drop_last_specials),
        int(args.chronos_input_patch_size),
    )

    head = build_stage1_head(
        args.head_type,
        d_model,
        num_classes,
        num_variates=args.encoder_num_variates,
        hidden_dim=args.head_hidden_dim,
        head_num_layers=args.head_num_layers,
        dropout=args.head_dropout,
        conv_channels=args.head_conv_channels,
        conv_d_proj=args.head_conv_d_proj,
        conv_kernel_size=args.head_conv_kernel_size,
        conv_num_layers=args.head_conv_num_layers,
        lstm_hidden_dim=args.head_lstm_hidden_dim,
        lstm_num_layers=args.head_lstm_num_layers,
    ).to(device)
    enc_trainable = [p for p in pipeline.model.parameters() if p.requires_grad]
    if finetune_n > 0:
        if not enc_trainable:
            raise RuntimeError(
                f"encoder_finetune_last_n={finetune_n} but no encoder parameters require gradients."
            )
        opt = torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": float(args.lr)},
                {"params": enc_trainable, "lr": float(args.lr_encoder)},
            ],
            weight_decay=float(args.weight_decay),
        )
        logger.info(
            "Optimizer: AdamW two groups (head lr=%s, encoder lr=%s, wd=%s).",
            args.lr,
            args.lr_encoder,
            args.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if finetune_n > 0 and args.use_embedding_cache:
        logger.warning("encoder_finetune_last_n=%d: embedding cache disabled (encoder gradients required).", finetune_n)
    cache_enabled = bool(args.freeze_encoder and args.use_embedding_cache) and finetune_n == 0
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else (out_dir / "embedding_cache")
    cache_meta_by_split: Dict[str, Any] = {}
    if cache_enabled:
        logger.info(
            "Embedding cache mode enabled (freeze_encoder=%s, storage=%s).",
            args.freeze_encoder,
            getattr(args, "embedding_cache_storage", "memmap"),
        )
        train_ds_cached, train_cache_meta = _build_or_load_cached_embeddings(
            split_name="train",
            split_path=export_dir / "train.jsonl",
            raw_loader=train_loader_raw,
            pipeline=pipeline,
            context_length=args.context_length,
            cache_dir=cache_dir,
            stats_path=stats_path,
            patch_info=patch_info,
            args=args,
            logger=logger,
            force_recompute=bool(args.recompute_embedding_cache),
        )
        val_ds_cached, val_cache_meta = _build_or_load_cached_embeddings(
            split_name="val",
            split_path=export_dir / "val.jsonl",
            raw_loader=val_loader_raw,
            pipeline=pipeline,
            context_length=args.context_length,
            cache_dir=cache_dir,
            stats_path=stats_path,
            patch_info=patch_info,
            args=args,
            logger=logger,
            force_recompute=bool(args.recompute_embedding_cache),
        )
        cache_meta_by_split["train"] = train_cache_meta
        cache_meta_by_split["val"] = val_cache_meta
        test_ds_cached: Optional[Dataset] = None
        if args.cache_test_embeddings:
            test_ds_cached, test_cache_meta = _build_or_load_cached_embeddings(
                split_name="test",
                split_path=export_dir / "test.jsonl",
                raw_loader=test_loader_raw,
                pipeline=pipeline,
                context_length=args.context_length,
                cache_dir=cache_dir,
                stats_path=stats_path,
                patch_info=patch_info,
                args=args,
                logger=logger,
                force_recompute=bool(args.recompute_embedding_cache),
            )
            cache_meta_by_split["test"] = test_cache_meta

        train_loader = DataLoader(
            train_ds_cached,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds_cached,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = None
        if test_ds_cached is not None:
            test_loader = DataLoader(
                test_ds_cached,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
    else:
        if args.use_embedding_cache and not args.freeze_encoder:
            logger.warning("Embedding cache requested but freeze_encoder=false; disabling cache.")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = val_loader_raw
        test_loader = test_loader_raw

    meta = {
        "chronos_model": args.chronos_model,
        "export_dir": str(export_dir),
        "context_length": args.context_length,
        "num_classes": num_classes,
        "d_model": d_model,
        "encoder_num_variates": int(args.encoder_num_variates),
        "device": str(device),
        "target_rate_hz": args.target_rate_hz,
        "chronos_patch_info": patch_info,
        "log_file": str(log_path.resolve()),
        "config_file": str(Path(args.config).resolve()) if args.config else None,
        "training_config": resolved_training_config(args),
        "head_config": head_config_from_args(args),
        "loss_config": loss_meta,
        "freeze_encoder": bool(args.freeze_encoder) and finetune_n == 0,
        "encoder_finetune_last_n": int(finetune_n),
        "lr_encoder": float(args.lr_encoder),
        "embedding_cache_enabled": bool(cache_enabled),
        "embedding_cache_dir": str(cache_dir.resolve()) if cache_enabled else None,
        "embedding_cache_metadata": cache_meta_by_split if cache_enabled else {},
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    minimize = args.early_stopping_metric == "val_loss"
    best_metric = float("inf") if minimize else float("-inf")
    best_epoch = 0
    best_val_snapshot: Dict[str, float] = {}
    best_head_state_cpu: Optional[Dict[str, torch.Tensor]] = None
    best_model_state_cpu: Optional[Dict[str, torch.Tensor]] = None  # set when encoder_finetune_last_n > 0
    epochs_no_improve = 0
    early_stopped = False
    head_best_path = out_dir / "head_best.pt"

    optuna_mod: Any = None
    if optuna_trial is not None:
        import optuna as optuna_mod  # type: ignore[no-redef]

    for epoch in range(1, args.epochs + 1):
        head.train()
        if finetune_n > 0:
            set_encoder_finetune_train_mode(pipeline.model, finetune_n, training=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_ep = time.perf_counter()
        running = 0.0
        n_seen = 0
        if cache_enabled:
            for batch_item in train_loader:
                if isinstance(batch_item, (list, tuple)) and len(batch_item) == 3:
                    emb_cpu, y_cpu, len_cpu = batch_item
                    lengths = len_cpu.to(device=device, non_blocking=True, dtype=torch.long)
                else:
                    emb_cpu, y_cpu = batch_item  # type: ignore[misc]
                    lengths = None
                emb = emb_cpu.to(device=device, non_blocking=True)
                if emb.dtype == torch.float16:
                    emb = emb.float()
                if emb.ndim not in (2, 4):
                    raise ValueError(f"Unexpected cached feature shape {tuple(emb.shape)}")
                if emb.ndim == 4:
                    emb = apply_stage1_encoder_trim(
                        emb,
                        lengths,
                        enabled=bool(args.encoder_output_trim),
                        context_length=int(args.context_length),
                        input_patch_size=int(args.chronos_input_patch_size),
                        drop_last_specials=int(args.encoder_output_drop_last_specials),
                    )
                y = y_cpu.to(device=device, non_blocking=True)
                b = int(emb.shape[0])
                opt.zero_grad(set_to_none=True)
                logits = head(emb)
                loss = criterion_forward(criterion, logits, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * b
                n_seen += b
        else:
            for batch in train_loader:
                imu_cpu = batch["imu_mv"].float()
                y = batch["label_id"].to(device)
                lengths = batch["lengths"].to(device=device, dtype=torch.long)
                b = imu_cpu.shape[0]
                opt.zero_grad(set_to_none=True)
                if finetune_n > 0:
                    imu = imu_cpu.to(device)
                    stacked = chronos_encode_multivariate_grad(
                        pipeline, imu, context_length=int(args.context_length)
                    )
                else:
                    with torch.no_grad():
                        # Keep embed() input on CPU; move returned stacked embeddings to training device.
                        emb_list, _ = pipeline.embed(
                            imu_cpu, batch_size=b, context_length=args.context_length
                        )
                    stacked = stack_embeddings_from_embed_list(emb_list).to(device, dtype=torch.float32)
                stacked = apply_stage1_encoder_trim(
                    stacked,
                    lengths,
                    enabled=bool(args.encoder_output_trim),
                    context_length=int(args.context_length),
                    input_patch_size=int(args.chronos_input_patch_size),
                    drop_last_specials=int(args.encoder_output_drop_last_specials),
                )
                logits = head(stacked)
                loss = criterion_forward(criterion, logits, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * b
                n_seen += b
        train_loss = running / max(n_seen, 1)
        if cache_enabled:
            val_m = evaluate_cached(
                head,
                val_loader,
                device,
                criterion,
                context_length=int(args.context_length),
                encoder_output_trim=bool(args.encoder_output_trim),
                encoder_output_drop_last_specials=int(args.encoder_output_drop_last_specials),
                chronos_input_patch_size=int(args.chronos_input_patch_size),
            )
        else:
            val_m = evaluate(
                pipeline,
                head,
                val_loader,
                device,
                args.context_length,
                criterion,
                encoder_output_trim=bool(args.encoder_output_trim),
                encoder_output_drop_last_specials=int(args.encoder_output_drop_last_specials),
                chronos_input_patch_size=int(args.chronos_input_patch_size),
                encoder_finetune_last_n=int(finetune_n),
            )
        if device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            logger.info("epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val top1/3/5=%.4f/%.4f/%.4f  (%ds)  cuda_mem_peak=%.0fMiB",
                epoch,
                args.epochs,
                train_loss,
                val_m["loss"],
                val_m["top1"],
                val_m["top3"],
                val_m["top5"],
                time.perf_counter() - t_ep,
                mem_mb,
            )
        else:
            logger.info(
                "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val top1/3/5=%.4f/%.4f/%.4f  (%ds)",
                epoch,
                args.epochs,
                train_loss,
                val_m["loss"],
                val_m["top1"],
                val_m["top3"],
                val_m["top5"],
                time.perf_counter() - t_ep,
            )

        cur = val_m["loss"] if minimize else val_m["top1"]
        if optuna_trial is not None and optuna_mod is not None:
            optuna_trial.report(float(cur), int(epoch) - 1)
            if int(epoch) >= int(optuna_prune_min_epochs) and optuna_trial.should_prune():
                raise optuna_mod.TrialPruned()

        if minimize:
            improved = cur < (best_metric - args.min_delta)
        else:
            improved = cur > (best_metric + args.min_delta)
        if improved:
            best_metric = cur
            best_epoch = epoch
            best_val_snapshot = {
                "val_loss": val_m["loss"],
                "val_top1": val_m["top1"],
                "val_top3": val_m["top3"],
                "val_top5": val_m["top5"],
            }
            best_head_state_cpu = _clone_state_dict_cpu(head)
            if finetune_n > 0:
                best_model_state_cpu = _clone_state_dict_cpu(pipeline.model)
            ckpt_best = _build_head_ckpt(
                head,
                d_model,
                num_classes,
                args.context_length,
                args.chronos_model,
                args,
                loss_meta,
                pipeline=pipeline,
                encoder_finetune_last_n=finetune_n,
            )
            torch.save(ckpt_best, head_best_path)
            logger.info(
                "  new best (%s=%.6f at epoch %d) → saved %s",
                args.early_stopping_metric,
                cur,
                epoch,
                head_best_path,
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                logger.info(
                    "Early stopping: no improvement on %s for %d epochs (patience=%d)",
                    args.early_stopping_metric,
                    args.patience,
                    args.patience,
                )
                early_stopped = True
                break

    if best_head_state_cpu is None:
        raise RuntimeError("No validation metrics recorded; val loader may be empty.")

    head.load_state_dict({k: v.to(device) for k, v in best_head_state_cpu.items()})
    if best_model_state_cpu is not None:
        pipeline.model.load_state_dict({k: v.to(device) for k, v in best_model_state_cpu.items()})

    test_m: Optional[Dict[str, float]] = None
    if bool(args.eval_test_split):
        if cache_enabled and test_loader is not None:
            test_m = evaluate_cached(
                head,
                test_loader,
                device,
                criterion,
                context_length=int(args.context_length),
                encoder_output_trim=bool(args.encoder_output_trim),
                encoder_output_drop_last_specials=int(args.encoder_output_drop_last_specials),
                chronos_input_patch_size=int(args.chronos_input_patch_size),
            )
        else:
            test_m = evaluate(
                pipeline,
                head,
                test_loader_raw,
                device,
                args.context_length,
                criterion,
                encoder_output_trim=bool(args.encoder_output_trim),
                encoder_output_drop_last_specials=int(args.encoder_output_drop_last_specials),
                chronos_input_patch_size=int(args.chronos_input_patch_size),
                encoder_finetune_last_n=int(finetune_n),
            )
        logger.info(
            "test  loss=%.4f  top1/3/5=%.4f/%.4f/%.4f",
            test_m["loss"],
            test_m["top1"],
            test_m["top3"],
            test_m["top5"],
        )
    else:
        logger.info("Skipping test split evaluation (--no-eval-test-split).")

    ckpt = _build_head_ckpt(
        head,
        d_model,
        num_classes,
        args.context_length,
        args.chronos_model,
        args,
        loss_meta,
        pipeline=pipeline,
        encoder_finetune_last_n=finetune_n,
    )
    head_path = out_dir / "head.pt"
    torch.save(ckpt, head_path)
    logger.info("Wrote best checkpoint to %s (best epoch %d by %s)", head_path, best_epoch, args.early_stopping_metric)

    meta.update(
        {
            "early_stopping_metric": args.early_stopping_metric,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "best_epoch": best_epoch,
            "best_metric_name": args.early_stopping_metric,
            "best_metric_value": best_metric,
            "best_val_metrics": best_val_snapshot,
            "early_stopped": early_stopped,
            "head_checkpoint": str(head_path.resolve()),
            "head_best_checkpoint": str(head_best_path.resolve()),
        }
    )
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {
        "best_epoch": int(best_epoch),
        "best_metric_name": str(args.early_stopping_metric),
        "best_metric_value": float(best_metric),
        "best_val_metrics": dict(best_val_snapshot),
        "test_loss": float(test_m["loss"]) if test_m is not None else None,
        "test_top1": float(test_m["top1"]) if test_m is not None else None,
        "test_top3": float(test_m["top3"]) if test_m is not None else None,
        "test_top5": float(test_m["top5"]) if test_m is not None else None,
        "early_stopped": bool(early_stopped),
        "out_dir": str(out_dir.resolve()),
        "log_file": str(log_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

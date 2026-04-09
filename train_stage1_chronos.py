#!/usr/bin/env python3
"""
Stage-1 training: frozen Chronos-2 encoder embeddings + linear classification head.

Requires: pip install "chronos-forecasting>=2.0"

Uses Chronos2Pipeline.embed() for encoder features (multivariate IMU as 12 variates).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-1 keystroke classifier (Chronos-2 + head).")
    p.add_argument("--export-dir", type=str, default="stage1_export", help="Dir with vocab.json + *.jsonl")
    p.add_argument("--data-dir", type=str, default="data", help="Raw IMU/PKL (must match export)")
    p.add_argument(
        "--chronos-model",
        type=str,
        default="autogluon/chronos-2-small",
        help="HF model id (default: autogluon/chronos-2-small)",
    )
    p.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1 or cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Fixed time steps passed to Chronos (pad left / crop tail)",
    )
    p.add_argument("--left-ms", type=int, default=700)
    p.add_argument("--right-ms", type=int, default=150)
    p.add_argument("--target-rate-hz", type=float, default=100.0)
    p.add_argument("--out-dir", type=str, default="checkpoints/stage1_chronos", help="Norm stats + head weights")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append training log here (default: <out-dir>/train.log)",
    )
    p.add_argument("--recompute-norm", action="store_true", help="Recompute train norm stats even if file exists")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def collate_chronos_batch(
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
) -> Any:
    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = collate_stack_lr_pad_batch(batch)
        x = normalize_mv(base["imu_mv"], mean, std)
        x = pad_crop_temporal(x, context_length)
        return {"imu_mv": x, "label_id": base["label_id"]}

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
) -> Dict[str, float]:
    head.eval()
    totals = {"n": 0, "loss": 0.0}
    acc_sums = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    for batch in loader:
        imu = batch["imu_mv"].float()
        y = batch["label_id"].to(device)
        b = imu.shape[0]
        emb_list, _ = pipeline.embed(imu, batch_size=b, context_length=context_length)
        h = pool_embedding_list(emb_list).to(device, dtype=torch.float32)
        logits = head(h)
        loss = F.cross_entropy(logits, y)
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


def main() -> None:
    _require_chronos()
    from chronos import Chronos2Pipeline

    args = parse_args()
    set_seed(args.seed)
    export_dir = Path(args.export_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else (out_dir / "train.log")
    logger = setup_logging(log_path)
    logger.info("Logging to %s", log_path.resolve())

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

    logger.info("Loading Chronos: %s", args.chronos_model)
    t0 = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(args.chronos_model)
    pipeline.model.to(device)
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

    train_ds = Stage1IMUKeyDataset(export_dir / "train.jsonl", stage_cfg)
    val_ds = Stage1IMUKeyDataset(export_dir / "val.jsonl", stage_cfg)
    test_ds = Stage1IMUKeyDataset(export_dir / "test.jsonl", stage_cfg)

    stats_path = out_dir / "norm_stats.json"
    if stats_path.exists() and not args.recompute_norm:
        mean, std, _eps = load_norm_stats(stats_path)
        logger.info("Loaded norm stats from %s", stats_path)
    else:
        logger.info("Computing train-only global norm stats …")
        mean, std = compute_global_norm_stats(train_ds, batch_size=256, num_workers=0)
        save_norm_stats(stats_path, mean, std, 1e-6)
        logger.info("Saved norm stats to %s", stats_path)

    collate_fn = collate_chronos_batch(mean, std, args.context_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # Infer d_model from one batch
    probe = next(iter(train_loader))
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
    logger.info(
        "Embed probe: first series emb shape=%s, d_model=%d, num_classes=%d",
        tuple(emb0[0].shape),
        d_model,
        num_classes,
    )

    head = nn.Linear(d_model, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    meta = {
        "chronos_model": args.chronos_model,
        "export_dir": str(export_dir),
        "context_length": args.context_length,
        "num_classes": num_classes,
        "d_model": d_model,
        "device": str(device),
        "target_rate_hz": args.target_rate_hz,
        "chronos_patch_info": patch_info,
        "log_file": str(log_path.resolve()),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        head.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_ep = time.perf_counter()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            imu = batch["imu_mv"].float()
            y = batch["label_id"].to(device)
            b = imu.shape[0]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                emb_list, _ = pipeline.embed(imu, batch_size=b, context_length=args.context_length)
            h = pool_embedding_list(emb_list).to(device, dtype=torch.float32)
            logits = head(h)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * b
            n_seen += b
        train_loss = running / max(n_seen, 1)
        val_m = evaluate(pipeline, head, val_loader, device, args.context_length)
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

    test_m = evaluate(pipeline, head, test_loader, device, args.context_length)
    logger.info(
        "test  loss=%.4f  top1/3/5=%.4f/%.4f/%.4f",
        test_m["loss"],
        test_m["top1"],
        test_m["top3"],
        test_m["top5"],
    )

    ckpt = {
        "head_state_dict": head.state_dict(),
        "d_model": d_model,
        "num_classes": num_classes,
        "context_length": args.context_length,
        "chronos_model": args.chronos_model,
    }
    torch.save(ckpt, out_dir / "head.pt")
    logger.info("Wrote %s", out_dir / "head.pt")


if __name__ == "__main__":
    main()

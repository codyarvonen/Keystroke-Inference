#!/usr/bin/env python3
"""
Evaluate a trained Stage-1 Chronos head on a split and export confusion/F1 artifacts.

Point --head-ckpt / --meta-path / --norm-stats at your run (defaults still point at example paths under checkpoints/stage1_chronos/).

Outputs (default --out-dir: <parent-of-head-ckpt>/eval_<split>/):
  - metrics.json
  - confusion_matrix.npy
  - confusion_matrix.csv
  - per_class_f1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.config import Stage1ExportConfig
from data_loader.stage1 import load_stage1_vocab, stage1_export_config_from_manifest
from data_loader.stage1_dataset import Stage1IMUKeyDataset
from data_loader.stage1_norm import (
    collate_stack_lr_pad_batch,
    load_norm_stats,
    normalize_mv,
    pad_crop_temporal,
)
from models.chronos_finetune import chronos_encode_multivariate_grad, set_encoder_finetune_train_mode
from models.stage1_heads import load_stage1_head_from_checkpoint
from train_stage1_chronos import (
    _require_chronos,
    apply_stage1_encoder_trim,
    stack_embeddings_from_embed_list,
    topk_accuracy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage-1 Chronos classifier and export confusion/F1.")
    p.add_argument("--head-ckpt", type=str, default="checkpoints/stage1_chronos/head.pt")
    p.add_argument("--meta-path", type=str, default="checkpoints/stage1_chronos/train_meta.json")
    p.add_argument("--norm-stats", type=str, default="checkpoints/stage1_chronos/norm_stats.json")
    p.add_argument("--export-dir", type=str, default=None, help="Override export dir (defaults to train_meta export_dir)")
    p.add_argument("--data-dir", type=str, default=None, help="Override raw data dir for Stage1 dataset")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default=None, help="Override device (default from train_meta)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--context-length", type=int, default=None, help="Override context length (default from train_meta)")
    p.add_argument("--chronos-model", type=str, default=None, help="Override Chronos model (default from train_meta)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to write metrics/CSVs (default: <head-ckpt parent>/eval_<split>/)",
    )
    p.add_argument("--topk-per-class", type=int, default=20, help="How many classes to print by support/F1")
    p.add_argument(
        "--show-random-sequence",
        action="store_true",
        help="Print one random contiguous sequence with GT token and top-5 predictions per sample",
    )
    p.add_argument(
        "--random-seq-min-len",
        type=int,
        default=20,
        help="Minimum random sequence length (inclusive)",
    )
    p.add_argument(
        "--random-seq-max-len",
        type=int,
        default=30,
        help="Maximum random sequence length (inclusive)",
    )
    p.add_argument(
        "--random-seq-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random sequence sampling",
    )
    return p.parse_args()


def default_eval_out_dir(head_ckpt: Path, split: str) -> Path:
    """Keep eval artifacts next to the checkpoint run, not a global shared folder."""
    return head_ckpt.resolve().parent / f"eval_{split}"


def _build_stage1_cfg(manifest: Dict[str, Any], data_dir_override: str | None) -> Stage1ExportConfig:
    return stage1_export_config_from_manifest(manifest, data_dir=data_dir_override)


def _collate_for_chronos(mean: np.ndarray, std: np.ndarray, context_length: int):
    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = collate_stack_lr_pad_batch(batch)
        x = normalize_mv(base["imu_mv"], mean, std)
        lengths = base["lengths"].clone()
        x = pad_crop_temporal(x, context_length)
        return {"imu_mv": x, "label_id": base["label_id"], "lengths": lengths}

    return _fn


def _f1_from_confusion(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)       # true counts
    pred_support = cm.sum(axis=0).astype(np.float64)  # predicted counts

    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def _write_confusion_csv(path: Path, cm: np.ndarray, id_to_token: List[str]) -> None:
    # Header uses "pred:<id>:<token>"; rows use "true:<id>:<token>".
    header = ["true\\pred"] + [f"pred:{i}:{id_to_token[i]!r}" for i in range(len(id_to_token))]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(cm.shape[0]):
            row_name = f"true:{i}:{id_to_token[i]!r}"
            w.writerow([row_name] + cm[i].tolist())


def _print_random_sequence_predictions(
    ds: Stage1IMUKeyDataset,
    pipeline: Any,
    head: nn.Module,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
    id_to_token: List[str],
    min_len: int,
    max_len: int,
    seed: int | None,
    *,
    encoder_output_trim: bool = False,
    encoder_output_drop_last_specials: int = 2,
    chronos_input_patch_size: int = 16,
    encoder_finetune_last_n: int = 0,
) -> None:
    n = len(ds)
    if n == 0:
        print("\nRandom sequence preview skipped: dataset split is empty.")
        return

    lo = max(1, int(min_len))
    hi = max(lo, int(max_len))
    hi = min(hi, n)
    lo = min(lo, hi)
    rng = random.Random(seed)
    seq_len = rng.randint(lo, hi)
    start = rng.randint(0, n - seq_len)
    idxs = list(range(start, start + seq_len))

    samples = [ds[i] for i in idxs]
    base = collate_stack_lr_pad_batch(samples)
    x = normalize_mv(base["imu_mv"], mean, std)
    lengths = base["lengths"].to(device=device, dtype=torch.long)
    x = pad_crop_temporal(x, context_length)
    y = base["label_id"].to(device)
    bsz = int(x.shape[0])

    with torch.no_grad():
        if encoder_finetune_last_n > 0:
            x_dev = x.to(device)
            set_encoder_finetune_train_mode(pipeline.model, encoder_finetune_last_n, training=False)
            stacked = chronos_encode_multivariate_grad(
                pipeline, x_dev.float(), context_length=context_length
            ).to(device, dtype=torch.float32)
        else:
            # Keep embed() input on CPU; Chronos embed internally handles batching/pin-memory.
            emb_list, _ = pipeline.embed(x.float(), batch_size=bsz, context_length=context_length)
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
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(k=min(5, probs.shape[1]), dim=1)

    top_p_np = top_p.detach().cpu().numpy()
    top_i_np = top_i.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    print(f"\nRandom sequential sample preview: start={start}, length={seq_len}")
    for j, sample in enumerate(samples):
        gt_id = int(y_np[j])
        gt_tok = id_to_token[gt_id] if 0 <= gt_id < len(id_to_token) else "<OOB>"
        preds = ", ".join(
            f"{id_to_token[int(k)]!r}:{float(p):.3f}"
            for p, k in zip(top_p_np[j], top_i_np[j])
        )
        print(
            f"  idx={idxs[j]:>6} session={sample['session_key']} ts={sample['key_press_ts']:.6f} "
            f"gt={gt_tok!r} top5=[{preds}]"
        )


def main() -> None:
    _require_chronos()
    from chronos import Chronos2Pipeline

    args = parse_args()
    head_ckpt_path = Path(args.head_ckpt)
    meta_path = Path(args.meta_path)
    norm_stats_path = Path(args.norm_stats)

    if not head_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {head_ckpt_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing train meta: {meta_path}")
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Missing norm stats: {norm_stats_path}")

    out_dir = Path(args.out_dir) if args.out_dir else default_eval_out_dir(head_ckpt_path, args.split)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    export_dir = Path(args.export_dir or meta["export_dir"])
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    vocab_path = export_dir / "vocab.json"
    _, id_to_token = load_stage1_vocab(vocab_path)
    num_classes = len(id_to_token)

    context_length = int(args.context_length or meta["context_length"])
    chronos_model = args.chronos_model or meta["chronos_model"]
    device_name = args.device or meta.get("device", "cuda:1")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    stage_cfg = _build_stage1_cfg(manifest, args.data_dir)
    split_path = export_dir / f"{args.split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split JSONL: {split_path}")

    ds = Stage1IMUKeyDataset(split_path, stage_cfg)
    mean, std, _eps = load_norm_stats(norm_stats_path)
    collate_fn = _collate_for_chronos(mean, std, context_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"Loading Chronos: {chronos_model}")
    pipeline = Chronos2Pipeline.from_pretrained(chronos_model)
    pipeline.model.to(device)
    pipeline.model.eval()
    for p in pipeline.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
    d_model = int(ckpt["d_model"])
    ckpt_num_classes = int(ckpt["num_classes"])
    ft_n = int(ckpt.get("encoder_finetune_last_n") or 0)
    if ft_n == 0:
        tc = meta.get("training_config") or {}
        ft_n = int(tc.get("encoder_finetune_last_n") or 0)
    if ckpt_num_classes != num_classes:
        raise ValueError(
            f"Class mismatch: checkpoint has {ckpt_num_classes}, vocab has {num_classes}. "
            "Use matching export/vocab/checkpoint artifacts."
        )
    head = load_stage1_head_from_checkpoint(ckpt, d_model=d_model, num_classes=num_classes, device=device)
    head.eval()

    md = ckpt.get("model_state_dict")
    if md:
        pipeline.model.load_state_dict({k: v.to(device) for k, v in md.items()}, strict=True)
    elif ft_n > 0:
        print(
            "WARNING: encoder_finetune_last_n>0 in meta/checkpoint but no model_state_dict in head checkpoint; "
            "using base Chronos weights (evaluation may not match training)."
        )

    hc = ckpt.get("head_config") or {}
    tc = meta.get("training_config") or {}
    enc_trim = bool(hc.get("encoder_output_trim", hc.get("encoder_output_masking", False)))
    enc_drop = int(
        hc.get("encoder_output_drop_last_specials", hc.get("encoder_output_drop_last_tokens", 2))
    )
    patch_sz = int(hc.get("chronos_input_patch_size") or tc.get("chronos_input_patch_size") or 16)
    if enc_trim:
        print(
            f"Encoder output trim (from checkpoint): enabled=True drop_last_specials={enc_drop} patch_size={patch_sz}"
        )
    if ft_n > 0:
        print(f"Encoder fine-tune eval: last {ft_n} block(s) use encode() path + loaded weights when present.")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    totals = {"n": 0, "loss_sum": 0.0, "top1_sum": 0.0, "top3_sum": 0.0, "top5_sum": 0.0}

    with torch.no_grad():
        for batch in loader:
            imu_cpu = batch["imu_mv"].float()
            y = batch["label_id"].to(device)
            lengths = batch["lengths"].to(device=device, dtype=torch.long)
            bsz = int(imu_cpu.shape[0])

            if ft_n > 0:
                imu = imu_cpu.to(device)
                set_encoder_finetune_train_mode(pipeline.model, ft_n, training=False)
                stacked = chronos_encode_multivariate_grad(
                    pipeline, imu, context_length=context_length
                ).to(device, dtype=torch.float32)
            else:
                # Chronos embed() expects CPU tensors for its internal DataLoader/pin-memory path.
                emb_list, _ = pipeline.embed(imu_cpu, batch_size=bsz, context_length=context_length)
                stacked = stack_embeddings_from_embed_list(emb_list).to(device, dtype=torch.float32)
            stacked = apply_stage1_encoder_trim(
                stacked,
                lengths,
                enabled=enc_trim,
                context_length=context_length,
                input_patch_size=patch_sz,
                drop_last_specials=enc_drop,
            )
            logits = head(stacked)
            loss = F.cross_entropy(logits, y)

            acc = topk_accuracy(logits, y, (1, 3, 5))
            pred = logits.argmax(dim=1)
            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            np.add.at(cm, (y_np, p_np), 1)

            totals["n"] += bsz
            totals["loss_sum"] += float(loss.item()) * bsz
            totals["top1_sum"] += acc["top1"] * bsz
            totals["top3_sum"] += acc["top3"] * bsz
            totals["top5_sum"] += acc["top5"] * bsz

    if args.show_random_sequence:
        _print_random_sequence_predictions(
            ds=ds,
            pipeline=pipeline,
            head=head,
            device=device,
            mean=mean,
            std=std,
            context_length=context_length,
            id_to_token=id_to_token,
            min_len=args.random_seq_min_len,
            max_len=args.random_seq_max_len,
            seed=args.random_seq_seed,
            encoder_output_trim=enc_trim,
            encoder_output_drop_last_specials=enc_drop,
            chronos_input_patch_size=patch_sz,
            encoder_finetune_last_n=ft_n,
        )

    n = max(totals["n"], 1)
    loss = totals["loss_sum"] / n
    top1 = totals["top1_sum"] / n
    top3 = totals["top3_sum"] / n
    top5 = totals["top5_sum"] / n

    precision, recall, f1, support = _f1_from_confusion(cm)
    tp_sum = float(np.diag(cm).sum())
    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * support).sum() / max(support.sum(), 1.0))
    micro_f1 = float(tp_sum / max(cm.sum(), 1))

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "confusion_matrix.npy", cm)
    _write_confusion_csv(out_dir / "confusion_matrix.csv", cm, id_to_token)

    per_class_rows: List[Dict[str, Any]] = []
    for i in range(num_classes):
        per_class_rows.append(
            {
                "class_id": i,
                "token": id_to_token[i],
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    per_class_rows.sort(key=lambda r: (-r["support"], r["class_id"]))
    with (out_dir / "per_class_f1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "token", "support", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(per_class_rows)

    metrics = {
        "split": args.split,
        "num_samples": int(cm.sum()),
        "num_classes": num_classes,
        "loss": loss,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "artifacts": {
            "confusion_matrix_npy": str((out_dir / "confusion_matrix.npy").as_posix()),
            "confusion_matrix_csv": str((out_dir / "confusion_matrix.csv").as_posix()),
            "per_class_f1_csv": str((out_dir / "per_class_f1.csv").as_posix()),
        },
        "checkpoint": str(head_ckpt_path),
        "export_dir": str(export_dir),
        "chronos_model": chronos_model,
        "context_length": context_length,
        "device": str(device),
        "out_dir": str(out_dir.resolve()),
        "encoder_finetune_last_n": int(ft_n),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Stage-1 Evaluation ===")
    print(f"out_dir:      {out_dir.resolve()}")
    print(f"split:        {args.split}")
    print(f"samples:      {int(cm.sum())}")
    print(f"loss:         {loss:.6f}")
    print(f"top1/top3/top5: {top1:.4f}/{top3:.4f}/{top5:.4f}")
    print(f"F1 micro/macro/weighted: {micro_f1:.4f}/{macro_f1:.4f}/{weighted_f1:.4f}")
    print(f"wrote: {out_dir / 'metrics.json'}")
    print(f"wrote: {out_dir / 'confusion_matrix.npy'}")
    print(f"wrote: {out_dir / 'confusion_matrix.csv'}")
    print(f"wrote: {out_dir / 'per_class_f1.csv'}")

    print(f"\nTop {args.topk_per_class} classes by support:")
    for r in per_class_rows[: args.topk_per_class]:
        print(
            f"  id={r['class_id']:>2} token={r['token']!r:<16} support={r['support']:>5} "
            f"P/R/F1={r['precision']:.3f}/{r['recall']:.3f}/{r['f1']:.3f}"
        )


if __name__ == "__main__":
    main()

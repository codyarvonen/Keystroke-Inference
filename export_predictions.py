#!/usr/bin/env python3
"""
Export per-keystroke top-k predictions for all data splits.

Loads a trained Stage-1 checkpoint, runs inference on train/val/test (or a
single split), and writes a JSONL file where each line is one keystroke with:
  - metadata  (session_key, split, key_press_ts, key_token, is_unk)
  - ground-truth label  (label_id)
  - top-k predictions  [{token, label_id, prob}, ...]
  - predicted token (top-1) and correctness flag

This output file is the input to decode_llm.py and decode_learned.py.

Usage
-----
# All splits (train + val + test) → predictions.jsonl
python export_predictions.py \\
    --head-ckpt checkpoints/stage1_chronos/head.pt \\
    --meta-path checkpoints/stage1_chronos/train_meta.json \\
    --norm-stats checkpoints/stage1_chronos/norm_stats.json \\
    --out predictions.jsonl

# Only test split
python export_predictions.py ... --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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
from models.chronos_finetune import set_encoder_finetune_train_mode
from models.stage1_heads import load_stage1_head_from_checkpoint
from train_stage1_chronos import (
    _require_chronos,
    apply_stage1_encoder_trim,
    stack_embeddings_from_embed_list,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export per-keystroke predictions to JSONL for LLM / learned decoder."
    )
    p.add_argument("--head-ckpt", default="checkpoints/stage1_chronos/head.pt")
    p.add_argument("--meta-path", default="checkpoints/stage1_chronos/train_meta.json")
    p.add_argument("--norm-stats", default="checkpoints/stage1_chronos/norm_stats.json")
    p.add_argument("--export-dir", default=None, help="Override export dir from train_meta")
    p.add_argument("--data-dir", default=None, help="Override raw data dir")
    p.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which split(s) to run. 'all' writes every split (default).",
    )
    p.add_argument("--device", default=None, help="Override device")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--chronos-model", default=None)
    p.add_argument("--top-k", type=int, default=5, help="Number of top predictions to save per keystroke")
    p.add_argument(
        "--out",
        default="predictions.jsonl",
        help="Output JSONL path (default: predictions.jsonl)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Collate: preserve per-sample metadata alongside tensors
# ---------------------------------------------------------------------------

def collate_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack IMU tensors; collect metadata as lists."""
    base = collate_stack_lr_pad_batch(batch)
    return {
        **base,
        "session_key": [s["session_key"] for s in batch],
        "split": [s["split"] for s in batch],
        "key_press_ts": [float(s["key_press_ts"]) for s in batch],
        "key_token": [str(s["key_token"]) for s in batch],
        "is_unk": [bool(s.get("is_unk", False)) for s in batch],
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_on_split(
    split_name: str,
    jsonl_path: Path,
    stage_cfg: Stage1ExportConfig,
    pipeline: Any,
    head: torch.nn.Module,
    id_to_token: List[str],
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    top_k: int,
    encoder_output_trim: bool,
    encoder_output_drop_last_specials: int,
    chronos_input_patch_size: int,
    encoder_finetune_last_n: int,
) -> List[Dict[str, Any]]:
    """Run inference on one split and return a list of prediction records."""
    if not jsonl_path.is_file():
        print(f"  [skip] {split_name}: {jsonl_path} not found")
        return []

    ds = Stage1IMUKeyDataset(jsonl_path, stage_cfg)
    if len(ds) == 0:
        print(f"  [skip] {split_name}: 0 samples")
        return []

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = collate_with_metadata(batch)
        x = normalize_mv(base["imu_mv"], mean, std)
        lengths = base["lengths"].clone()
        x = pad_crop_temporal(x, context_length)
        return {**base, "imu_mv": x, "lengths": lengths}

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    records: List[Dict[str, Any]] = []
    vocab_size = len(id_to_token)
    k = min(top_k, vocab_size)
    t0 = time.perf_counter()

    for batch in loader:
        imu_cpu = batch["imu_mv"].float()
        y = batch["label_id"].to(device)
        lengths = batch["lengths"].to(device=device, dtype=torch.long)
        bsz = int(imu_cpu.shape[0])

        if encoder_finetune_last_n > 0:
            imu = imu_cpu.to(device)
            set_encoder_finetune_train_mode(pipeline.model, encoder_finetune_last_n, training=False)
            from models.chronos_finetune import chronos_encode_multivariate_grad
            stacked = chronos_encode_multivariate_grad(
                pipeline, imu, context_length=context_length
            ).to(device, dtype=torch.float32)
        else:
            emb_list, _ = pipeline.embed(imu_cpu, batch_size=bsz, context_length=context_length)
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
        top_probs, top_ids = probs.topk(k, dim=1)  # (B, k)

        top_probs_np = top_probs.detach().cpu().numpy()
        top_ids_np = top_ids.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for i in range(bsz):
            gt_label_id = int(y_np[i])
            pred_label_id = int(top_ids_np[i, 0])
            gt_token = id_to_token[gt_label_id] if 0 <= gt_label_id < vocab_size else "<OOB>"
            pred_token = id_to_token[pred_label_id] if 0 <= pred_label_id < vocab_size else "<OOB>"

            top_k_list = [
                {
                    "token": id_to_token[int(top_ids_np[i, j])] if 0 <= int(top_ids_np[i, j]) < vocab_size else "<OOB>",
                    "label_id": int(top_ids_np[i, j]),
                    "prob": round(float(top_probs_np[i, j]), 6),
                }
                for j in range(k)
            ]

            records.append(
                {
                    "session_key": batch["session_key"][i],
                    "split": batch["split"][i],
                    "key_press_ts": batch["key_press_ts"][i],
                    "key_token": batch["key_token"][i],
                    "label_id": gt_label_id,
                    "is_unk": batch["is_unk"][i],
                    "pred_token": pred_token,
                    "pred_label_id": pred_label_id,
                    "correct": pred_token == batch["key_token"][i],
                    "top_k": top_k_list,
                }
            )

    elapsed = time.perf_counter() - t0
    n = len(records)
    correct = sum(1 for r in records if r["correct"] and not r["is_unk"])
    total_known = sum(1 for r in records if not r["is_unk"])
    acc = correct / max(total_known, 1)
    print(
        f"  {split_name}: {n} keystrokes | top-1 acc (non-UNK) = {acc:.4f} | {elapsed:.1f}s"
    )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _require_chronos()
    from chronos import Chronos2Pipeline

    args = parse_args()
    head_ckpt_path = Path(args.head_ckpt)
    meta_path = Path(args.meta_path)
    norm_stats_path = Path(args.norm_stats)

    for p in (head_ckpt_path, meta_path, norm_stats_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing: {p}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    export_dir = Path(args.export_dir or meta["export_dir"])
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    vocab_path = export_dir / "vocab.json"
    _, id_to_token = load_stage1_vocab(vocab_path)

    context_length = int(args.context_length or meta["context_length"])
    chronos_model = args.chronos_model or meta["chronos_model"]
    device_str = args.device or meta.get("device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    stage_cfg = stage1_export_config_from_manifest(manifest, data_dir=args.data_dir)

    ckpt = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
    d_model = int(ckpt["d_model"])
    num_classes = int(ckpt["num_classes"])
    ft_n = int(ckpt.get("encoder_finetune_last_n") or 0)
    if ft_n == 0:
        tc = meta.get("training_config") or {}
        ft_n = int(tc.get("encoder_finetune_last_n") or 0)

    hc = ckpt.get("head_config") or {}
    tc = meta.get("training_config") or {}
    enc_trim = bool(hc.get("encoder_output_trim", False))
    enc_drop = int(hc.get("encoder_output_drop_last_specials", 2))
    patch_sz = int(hc.get("chronos_input_patch_size") or tc.get("chronos_input_patch_size") or 16)

    print(f"Loading Chronos: {chronos_model}")
    pipeline = Chronos2Pipeline.from_pretrained(chronos_model)
    pipeline.model.to(device)
    pipeline.model.eval()
    for param in pipeline.model.parameters():
        param.requires_grad = False

    if ckpt.get("model_state_dict"):
        pipeline.model.load_state_dict(
            {k: v.to(device) for k, v in ckpt["model_state_dict"].items()},
            strict=True,
        )

    head = load_stage1_head_from_checkpoint(ckpt, d_model=d_model, num_classes=num_classes, device=device)
    head.eval()

    mean, std, _eps = load_norm_stats(norm_stats_path)

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]
    print(f"\nRunning inference on split(s): {splits_to_run}")
    print(f"top-k = {args.top_k}, device = {device}\n")

    all_records: List[Dict[str, Any]] = []
    for split_name in splits_to_run:
        jsonl_path = export_dir / f"{split_name}.jsonl"
        records = run_inference_on_split(
            split_name=split_name,
            jsonl_path=jsonl_path,
            stage_cfg=stage_cfg,
            pipeline=pipeline,
            head=head,
            id_to_token=id_to_token,
            mean=mean,
            std=std,
            context_length=context_length,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            top_k=args.top_k,
            encoder_output_trim=enc_trim,
            encoder_output_drop_last_specials=enc_drop,
            chronos_input_patch_size=patch_sz,
            encoder_finetune_last_n=ft_n,
        )
        all_records.extend(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(all_records)
    correct = sum(1 for r in all_records if r["correct"] and not r["is_unk"])
    total_known = sum(1 for r in all_records if not r["is_unk"])
    print(f"\nWrote {n} records to {out_path.resolve()}")
    print(f"Overall top-1 accuracy (non-UNK): {correct}/{total_known} = {correct / max(total_known, 1):.4f}")


if __name__ == "__main__":
    main()


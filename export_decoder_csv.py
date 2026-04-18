#!/usr/bin/env python3
"""
Export Stage-1 classifier softmax probabilities + labels to CSV for decoder training.

Writes:
  - train.csv   — rows from train.jsonl and val.jsonl (Stage-1 train+val), sorted by
                  (session_id, key_press_ts)
  - test.csv    — rows from test.jsonl, same sort
  - vocab.json  — copy of the Stage-1 export vocab (id ↔ label mapping)

CSV columns:
  split, session_id, key_press_ts, label, label_id, p0, p1, ..., p{C-1}

Requires the same environment as evaluate_stage1_chronos.py (Chronos, torch, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.config import Stage1ExportConfig
from data_loader.stage1 import load_stage1_jsonl_rows, load_stage1_vocab, stage1_export_config_from_manifest
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
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export per-key softmax CSVs for decoder training.")
    p.add_argument("--head-ckpt", type=str, required=True, help="Stage-1 head checkpoint (.pt)")
    p.add_argument("--meta-path", type=str, required=True, help="train_meta.json from the same run")
    p.add_argument("--norm-stats", type=str, required=True, help="norm_stats.json used for that run")
    p.add_argument("--export-dir", type=str, default=None, help="Override Stage-1 export dir (default from meta)")
    p.add_argument("--data-dir", type=str, default=None, help="Override raw data dir")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <head-ckpt parent>/decoder_csv)")
    p.add_argument("--device", type=str, default=None, help="cuda:0 / cpu (default from meta)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--context-length", type=int, default=None, help="Override context length (default from meta)")
    p.add_argument("--chronos-model", type=str, default=None, help="Override Chronos model id (default from meta)")
    p.add_argument(
        "--prob-decimals",
        type=int,
        default=8,
        help="Decimal places for probability columns (smaller = smaller files)",
    )
    return p.parse_args()


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


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (str(r["session_key"]), float(r["key_press_ts"])))


def _csv_header(num_classes: int) -> List[str]:
    return ["split", "session_id", "key_press_ts", "label", "label_id"] + [f"p{i}" for i in range(num_classes)]


def _export_one_csv(
    *,
    rows: List[Dict[str, Any]],
    jsonl_path_for_dataset: Path,
    out_path: Path,
    cfg: Stage1ExportConfig,
    pipeline: Any,
    head: torch.nn.Module,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
    num_classes: int,
    collate_fn,
    batch_size: int,
    num_workers: int,
    prob_decimals: int,
    enc_trim: bool,
    enc_drop: int,
    patch_sz: int,
    ft_n: int,
    id_to_token: List[str],
) -> None:
    rows = _sort_rows(rows)
    ds = Stage1IMUKeyDataset(jsonl_path_for_dataset, cfg, rows=rows)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{max(0, prob_decimals)}f}}"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_csv_header(num_classes))

        offset = 0
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
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                for j in range(bsz):
                    r = rows[offset + j]
                    split = str(r.get("split", ""))
                    session_id = str(r["session_key"])
                    ts = float(r["key_press_ts"])
                    lid = int(y_np[j])
                    lab = id_to_token[lid] if 0 <= lid < len(id_to_token) else str(r.get("key_token", ""))
                    row_out: List[Any] = [split, session_id, ts, lab, lid]
                    row_out.extend(fmt.format(float(x)) for x in probs[j].tolist())
                    w.writerow(row_out)

                offset += bsz

        if offset != len(rows):
            raise RuntimeError(f"Row count mismatch: wrote {offset}, expected {len(rows)}")


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

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    export_dir = Path(args.export_dir or meta["export_dir"])
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    vocab_path = export_dir / "vocab.json"
    _, id_to_token = load_stage1_vocab(vocab_path)
    num_classes = len(id_to_token)

    out_dir = Path(args.out_dir) if args.out_dir else head_ckpt_path.resolve().parent / "decoder_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    context_length = int(args.context_length or meta["context_length"])
    chronos_model = args.chronos_model or meta["chronos_model"]
    device_name = args.device or meta.get("device", "cuda:0")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    stage_cfg = _build_stage1_cfg(manifest, args.data_dir)
    train_path = export_dir / "train.jsonl"
    val_path = export_dir / "val.jsonl"
    test_path = export_dir / "test.jsonl"
    for p in (train_path, val_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing split JSONL: {p}")

    train_rows = load_stage1_jsonl_rows(train_path)
    val_rows = load_stage1_jsonl_rows(val_path)
    test_rows = load_stage1_jsonl_rows(test_path)
    train_and_val_rows = train_rows + val_rows

    mean, std, _eps = load_norm_stats(norm_stats_path)
    collate_fn = _collate_for_chronos(mean, std, context_length)

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
            f"Class mismatch: checkpoint has {ckpt_num_classes}, vocab has {num_classes}."
        )

    head = load_stage1_head_from_checkpoint(ckpt, d_model=d_model, num_classes=num_classes, device=device)
    head.eval()

    md = ckpt.get("model_state_dict")
    if md:
        pipeline.model.load_state_dict({k: v.to(device) for k, v in md.items()}, strict=True)
    elif ft_n > 0:
        print(
            "WARNING: encoder_finetune_last_n>0 but no model_state_dict in head checkpoint; "
            "using base Chronos weights (export may not match training)."
        )

    hc = ckpt.get("head_config") or {}
    tc = meta.get("training_config") or {}
    enc_trim = bool(hc.get("encoder_output_trim", hc.get("encoder_output_masking", False)))
    enc_drop = int(
        hc.get("encoder_output_drop_last_specials", hc.get("encoder_output_drop_last_tokens", 2))
    )
    patch_sz = int(hc.get("chronos_input_patch_size") or tc.get("chronos_input_patch_size") or 16)

    print(f"Writing train.csv ({len(train_and_val_rows)} rows: train+val)...")
    _export_one_csv(
        rows=train_and_val_rows,
        jsonl_path_for_dataset=train_path,
        out_path=out_dir / "train.csv",
        cfg=stage_cfg,
        pipeline=pipeline,
        head=head,
        device=device,
        mean=mean,
        std=std,
        context_length=context_length,
        num_classes=num_classes,
        collate_fn=collate_fn,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prob_decimals=int(args.prob_decimals),
        enc_trim=enc_trim,
        enc_drop=enc_drop,
        patch_sz=patch_sz,
        ft_n=ft_n,
        id_to_token=id_to_token,
    )

    print(f"Writing test.csv ({len(test_rows)} rows)...")
    _export_one_csv(
        rows=test_rows,
        jsonl_path_for_dataset=test_path,
        out_path=out_dir / "test.csv",
        cfg=stage_cfg,
        pipeline=pipeline,
        head=head,
        device=device,
        mean=mean,
        std=std,
        context_length=context_length,
        num_classes=num_classes,
        collate_fn=collate_fn,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prob_decimals=int(args.prob_decimals),
        enc_trim=enc_trim,
        enc_drop=enc_drop,
        patch_sz=patch_sz,
        ft_n=ft_n,
        id_to_token=id_to_token,
    )

    vocab_out = out_dir / "vocab.json"
    shutil.copy2(vocab_path, vocab_out)
    print(f"Copied vocab → {vocab_out}")

    manifest_out = {
        "schema": "decoder_csv_v1",
        "head_checkpoint": str(head_ckpt_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "norm_stats": str(norm_stats_path.resolve()),
        "export_dir": str(export_dir.resolve()),
        "chronos_model": chronos_model,
        "context_length": context_length,
        "num_classes": num_classes,
        "train_csv_rows": len(train_and_val_rows),
        "test_csv_rows": len(test_rows),
        "train_csv_includes": "train.jsonl + val.jsonl (sorted by session_id, key_press_ts)",
        "files": {
            "train": str((out_dir / "train.csv").resolve()),
            "test": str((out_dir / "test.csv").resolve()),
            "vocab": str(vocab_out.resolve()),
        },
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'export_manifest.json'}")
    print(f"Done. Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

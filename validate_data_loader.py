#!/usr/bin/env python3
"""
Validation and usage demo for the keystroke–IMU data loader.

This script:
- Builds the PyTorch datasets/dataloaders from DataConfig
- Reports timing information (tokenizer load, window building, dataloader construction, iteration)
- Computes dataset statistics (windows, tokens, characters, keystrokes, IMU window lengths)
- Estimates how many windows are dropped due to large IMU gaps
"""

import argparse
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
from transformers import GPT2TokenizerFast

from data_loader import DataConfig, make_dataloaders
from data_loader.sessions import discover_sessions, load_session_raw
from data_loader.windows import align_rings_to_grid, build_imu_windows


def _safe_stats(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def _print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and profile the keystroke–IMU data loader."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing IMU CSV and Macbook PKL files (default: data)",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["LOSO", "LOPO", "random"],
        default="LOSO",
        help="Split strategy: LOSO, LOPO, or random (default: LOSO)",
    )
    parser.add_argument(
        "--test-session",
        type=str,
        default=None,
        help='Test session key for LOSO (e.g., "003_005")',
    )
    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help='Test subject ID for LOPO (e.g., "003")',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=10,
        help="Max number of train batches to iterate when timing (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DataConfig(
        data_dir=args.data_dir,
        split_strategy=args.split_strategy,  # type: ignore[arg-type]
        test_session=args.test_session,
        test_subject=args.test_subject,
    )

    _print_header("Data Loader Validation")
    print("Configuration:")
    print(f"  data_dir       : {cfg.data_dir}")
    print(f"  split_strategy : {cfg.split_strategy}")
    print(f"  test_session   : {cfg.test_session}")
    print(f"  test_subject   : {cfg.test_subject}")
    print(f"  window_size_s  : {cfg.window_size_s}")
    print(f"  train_stride_s : {cfg.train_stride_s}")
    print(f"  test_stride_s  : {cfg.test_stride_s}")
    print(f"  max_tokens     : {cfg.max_tokens}")
    print(f"  target_variant : {cfg.target_variant}")
    print(f"  rings_used     : {cfg.rings_used}")

    # ------------------------------------------------------------
    # Timing: tokenizer + dataloaders
    # ------------------------------------------------------------
    _print_header("Timing: Tokenizer and Dataloaders")

    t0 = time.perf_counter()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    t1 = time.perf_counter()
    print(f"Tokenizer load time: {(t1 - t0):.3f} s")

    t2 = time.perf_counter()
    train_loader, val_loader, test_loader, windows = make_dataloaders(
        cfg, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers
    )
    t3 = time.perf_counter()
    print(f"Window build + split + dataloader time: {(t3 - t2):.3f} s")

    # ------------------------------------------------------------
    # Split sizes
    # ------------------------------------------------------------
    _print_header("Split Sizes")
    n_total = len(windows)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Total windows (after filtering): {n_total}")
    print(f"  Train windows: {n_train}")
    print(f"  Val windows  : {n_val}")
    print(f"  Test windows : {n_test}")

    # ------------------------------------------------------------
    # Dataset statistics (tokens, characters, keystrokes, IMU lengths)
    # ------------------------------------------------------------
    _print_header("Dataset Statistics")

    token_lengths = [w.token_length or 0 for w in windows if w.token_length is not None]
    clean_char_lengths = [
        len(w.clean_text) for w in windows if w.clean_text is not None
    ]
    raw_keystroke_counts = [
        len(w.raw_key_sequence)
        for w in windows
        if w.raw_key_sequence is not None
    ]

    # Token stats
    mean_tok, std_tok, min_tok, max_tok = _safe_stats(token_lengths)
    print("Tokens per window (clean_tokens):")
    print(f"  mean / std : {mean_tok:.2f} / {std_tok:.2f}")
    print(f"  min / max  : {min_tok:.0f} / {max_tok:.0f}")
    print(f"  total tokens (non-pad): {int(sum(token_lengths))}")

    # Simple token length histogram (bucketed)
    if token_lengths:
        buckets = Counter()
        for tl in token_lengths:
            if tl <= 10:
                buckets["0-10"] += 1
            elif tl <= 20:
                buckets["11-20"] += 1
            elif tl <= 30:
                buckets["21-30"] += 1
            elif tl <= 40:
                buckets["31-40"] += 1
            else:
                buckets["41-50"] += 1
        print("  length buckets (token_length):")
        for bucket in ["0-10", "11-20", "21-30", "31-40", "41-50"]:
            if bucket in buckets:
                print(f"    {bucket}: {buckets[bucket]}")

    # Clean-text character stats
    mean_cc, std_cc, min_cc, max_cc = _safe_stats(clean_char_lengths)
    print("\nClean text characters per window:")
    print(f"  mean / std : {mean_cc:.2f} / {std_cc:.2f}")
    print(f"  min / max  : {min_cc:.0f} / {max_cc:.0f}")
    print(f"  total clean characters: {int(sum(clean_char_lengths))}")

    # Raw keystroke stats (per-window key presses)
    mean_keys, std_keys, min_keys, max_keys = _safe_stats(raw_keystroke_counts)
    print("\nRaw keystrokes per window (number of key events):")
    print(f"  mean / std : {mean_keys:.2f} / {std_keys:.2f}")
    print(f"  min / max  : {min_keys:.0f} / {max_keys:.0f}")
    print(f"  total raw keystrokes: {int(sum(raw_keystroke_counts))}")

    # IMU window length stats (T dimension)
    imu_lengths = []
    for w in windows:
        if w.imu_L is not None:
            imu_lengths.append(w.imu_L.shape[0])
        elif w.imu_R is not None:
            imu_lengths.append(w.imu_R.shape[0])
    mean_T, std_T, min_T, max_T = _safe_stats(imu_lengths)
    print("\nIMU timesteps per window (T dimension):")
    print(f"  mean / std : {mean_T:.2f} / {std_T:.2f}")
    print(f"  min / max  : {min_T:.0f} / {max_T:.0f}")

    # ------------------------------------------------------------
    # Estimate windows dropped due to large IMU gaps
    # ------------------------------------------------------------
    _print_header("Windows Dropped Due to Large Gaps")

    session_files = discover_sessions(cfg)
    total_possible_windows = 0
    total_kept_windows = 0
    total_dropped_for_gaps = 0

    for key, files in sorted(session_files.items()):
        subject, session = key.split("_", 1)

        raw = load_session_raw(files)
        imu_L = raw.get("imu_L")
        imu_R = raw.get("imu_R")
        if imu_L is None or imu_R is None:
            continue

        gaps_L = raw.get("gaps_L") or []
        gaps_R = raw.get("gaps_R") or []
        bad_intervals = list(gaps_L) + list(gaps_R)

        time_grid, imu_L_grid, imu_R_grid = align_rings_to_grid(imu_L, imu_R)

        windows_all = build_imu_windows(
            time_grid=time_grid,
            imu_L_grid=imu_L_grid,
            imu_R_grid=imu_R_grid,
            subject=subject,
            session=session,
            cfg=cfg,
            is_test=False,  # stride choice doesn't matter for gap comparison
            bad_intervals=None,
        )

        windows_clean = build_imu_windows(
            time_grid=time_grid,
            imu_L_grid=imu_L_grid,
            imu_R_grid=imu_R_grid,
            subject=subject,
            session=session,
            cfg=cfg,
            is_test=False,
            bad_intervals=bad_intervals,
        )

        possible = len(windows_all)
        kept = len(windows_clean)
        dropped = possible - kept

        total_possible_windows += possible
        total_kept_windows += kept
        total_dropped_for_gaps += dropped

        if possible > 0:
            pct = 100.0 * dropped / possible
            print(f"Session {key}: possible={possible}, dropped_for_gaps={dropped} ({pct:.1f}%)")

    if total_possible_windows > 0:
        pct_total = 100.0 * total_dropped_for_gaps / total_possible_windows
        print(
            f"\nOverall: possible={total_possible_windows}, "
            f"dropped_for_gaps={total_dropped_for_gaps} ({pct_total:.1f}%)"
        )
    else:
        print("No windows found when estimating gap-related drops.")

    # ------------------------------------------------------------
    # Batch iteration timing + sample preview
    # ------------------------------------------------------------
    _print_header("Train Loader Iteration Timing & Sample Preview")

    if len(train_loader) == 0:
        print("Train loader is empty; skipping iteration timing.")
        return

    max_batches = max(1, args.max_train_batches)
    n_batches = 0
    n_samples = 0

    t_iter_start = time.perf_counter()
    first_batch = None

    for batch in train_loader:
        if first_batch is None:
            first_batch = batch
        bsz = len(batch["target"])
        n_batches += 1
        n_samples += bsz
        if n_batches >= max_batches:
            break

    t_iter_end = time.perf_counter()
    elapsed = t_iter_end - t_iter_start
    print(
        f"Iterated over {n_batches} batches "
        f"({n_samples} samples) in {elapsed:.3f} s "
        f"-> {n_samples / max(elapsed, 1e-6):.1f} windows/s"
    )

    # Preview shapes and a few labels from the first batch
    if first_batch is not None:
        imu_l = first_batch["imu_l"]
        imu_r = first_batch["imu_r"]
        target = first_batch["target"]

        print("\nFirst batch shapes:")
        if imu_l is not None:
            print(f"  imu_l: {tuple(imu_l.shape)} (dtype={imu_l.dtype})")
        else:
            print("  imu_l: None")
        if imu_r is not None:
            print(f"  imu_r: {tuple(imu_r.shape)} (dtype={imu_r.dtype})")
        else:
            print("  imu_r: None")

        if hasattr(target, "shape"):
            print(f"  target: {tuple(target.shape)} (dtype={target.dtype})")
        else:
            print(f"  target type: {type(target)}")

        # Show a few reconstructed clean_text strings
        print("\nSample clean_text from first batch:")
        clean_texts = first_batch.get("clean_text", [])
        for i in range(min(3, len(clean_texts))):
            txt = clean_texts[i]
            if txt is None:
                txt = ""
            preview = txt[:120].replace("\n", "\\n")
            print(f"  [{i}] {preview!r}")


if __name__ == "__main__":
    main()


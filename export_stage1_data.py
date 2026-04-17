#!/usr/bin/env python3
"""
Export Stage-1 JSONL shards + train-only vocab (M2).

Writes:
  out_dir/vocab.json
  out_dir/train.jsonl, val.jsonl, test.jsonl
  out_dir/manifest.json
"""

import argparse
import json
from pathlib import Path

from data_loader.config import Stage1ExportConfig
from data_loader.stage1 import export_stage1_to_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage-1 rows and vocabulary.")
    p.add_argument("--out-dir", type=str, default="exports/stage1_export", help="Output directory")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--left-ms", type=int, default=700)
    p.add_argument("--right-ms", type=int, default=150)
    p.add_argument("--target-rate-hz", type=float, default=100.0)
    p.add_argument(
        "--session-split-strategy",
        choices=["session_random", "session_holdout", "session_holdout_random_train_val"],
        default="session_random",
    )
    p.add_argument(
        "--holdout-test-random-train-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use explicit --test-sessions as session-held-out test set, then random "
        "train/val split by rows over all remaining non-test rows (respects --train-only-sessions).",
    )
    p.add_argument("--test-sessions", nargs="*", default=[])
    p.add_argument("--val-sessions", nargs="*", default=[])
    p.add_argument(
        "--train-only-sessions",
        nargs="*",
        default=[],
        metavar="SESSION_KEY",
        help='Sessions forced to train only (never val/test), e.g. 003_014 003_015 (must match filename-derived keys)',
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        metavar="F",
        help="session_random: fraction of pool sessions (after train_only) for test. "
        "If omitted, legacy rule: n_test = max(1, n_val) from --val-ratio only.",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--balance-val-test-by-session-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="session_random: assign val/test sessions to balance approximate row counts "
        "(extra pass over raw sessions; no leakage)",
    )
    p.add_argument(
        "--merge-letter-case",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lowercase single-letter alphabetic keys so 'A' and 'a' share one class (default: off)",
    )
    p.add_argument(
        "--coarse-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collapse to 32 classes: a–z, space, backspace, shift, punct, digits, other (recommended with --merge-letter-case)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session_split_strategy = args.session_split_strategy
    if args.holdout_test_random_train_val:
        session_split_strategy = "session_holdout_random_train_val"

    cfg = Stage1ExportConfig(
        data_dir=args.data_dir,
        left_context_ms=args.left_ms,
        right_context_ms=args.right_ms,
        target_rate_hz=args.target_rate_hz,
        session_split_strategy=session_split_strategy,  # type: ignore[arg-type]
        test_sessions=args.test_sessions,
        val_sessions=args.val_sessions,
        train_only_sessions=tuple(args.train_only_sessions),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        balance_val_test_by_session_rows=args.balance_val_test_by_session_rows,
        merge_letter_case=args.merge_letter_case,
        coarse_labels=args.coarse_labels,
    )
    out_dir = Path(args.out_dir)
    manifest = export_stage1_to_dir(cfg, out_dir)

    print(f"Wrote export under {out_dir.resolve()}")
    print(f"  vocab_size: {manifest['vocab_size']}")
    print(f"  rows_written: {manifest['rows_written']}")
    print(f"  unk_label_count_by_split: {manifest['unk_label_count_by_split']}")

    sbs = manifest.get("build_stats", {}).get("sessions_by_split")
    if isinstance(sbs, dict):
        print("  sessions_by_split (session_key → split assignment):")
        for split_name in ("train", "val", "test"):
            sessions = sbs.get(split_name) or []
            n = len(sessions)
            joined = ", ".join(sessions)
            print(f"    {split_name} ({n}): {joined}")

    vocab_path = out_dir / "vocab.json"
    v = json.loads(vocab_path.read_text(encoding="utf-8"))
    print(f"  special_tokens: {v.get('special_tokens')}")


if __name__ == "__main__":
    main()

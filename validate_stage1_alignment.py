#!/usr/bin/env python3
"""
Validate Stage-1 keypress-to-IMU alignment and export schema readiness.

Rebuilds rows from raw session files under --data-dir (same code path as export), not from
JSONL shards. Use --export-dir path/to/stage1_export to load manifest.json so settings
(including merge_letter_case) match that export.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from data_loader.config import Stage1ExportConfig
from data_loader.stage1 import build_stage1_rows, stage1_export_config_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage-1 keypress timestamp alignment and per-key IMU windows."
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Load Stage1ExportConfig from DIR/manifest.json (matches export; includes merge_letter_case).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Raw data root (default: 'data', or from manifest when using --export-dir)",
    )
    parser.add_argument("--left-ms", type=int, default=700)
    parser.add_argument("--right-ms", type=int, default=150)
    parser.add_argument("--target-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--session-split-strategy",
        choices=["session_random", "session_holdout"],
        default="session_random",
    )
    parser.add_argument(
        "--test-sessions",
        nargs="*",
        default=[],
        help='Used with session_holdout, e.g. --test-sessions 003_005 003_006',
    )
    parser.add_argument(
        "--val-sessions",
        nargs="*",
        default=[],
        help='Used with session_holdout, e.g. --val-sessions 003_007',
    )
    parser.add_argument(
        "--train-only-sessions",
        nargs="*",
        default=[],
        metavar="SESSION_KEY",
        help="Sessions forced to train only (ignored when using --export-dir; manifest wins)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        metavar="F",
        help="session_random only; omit to match legacy export behavior",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--balance-val-test-by-session-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--merge-letter-case",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lowercase single-letter keys so A/a share one label (must match export)",
    )
    parser.add_argument(
        "--coarse-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="32-class coarse labels (a–z + space/backspace/shift/punct/num/other); must match export",
    )
    parser.add_argument("--show-samples", type=int, default=8)
    parser.add_argument("--show-label-topk", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_dir:
        manifest_path = Path(args.export_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = stage1_export_config_from_manifest(manifest, data_dir=args.data_dir)
        print(
            f"Config from manifest: {manifest_path.resolve()} "
            f"(merge_letter_case={cfg.merge_letter_case}, coarse_labels={cfg.coarse_labels}, "
            f"train_only_sessions={list(cfg.train_only_sessions)}, "
            f"test_ratio={cfg.test_ratio}, balance_val_test_by_session_rows="
            f"{cfg.balance_val_test_by_session_rows})"
        )
    else:
        cfg = Stage1ExportConfig(
            data_dir=args.data_dir or "data",
            left_context_ms=args.left_ms,
            right_context_ms=args.right_ms,
            target_rate_hz=args.target_rate_hz,
            session_split_strategy=args.session_split_strategy,  # type: ignore[arg-type]
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

    rows, stats = build_stage1_rows(cfg)

    print("\n=== Stage-1 Alignment Validation ===")
    print(f"rows_kept                 : {len(rows)}")
    print(f"keys_total                : {stats['keys_total']}")
    print(f"keys_kept                 : {stats['keys_kept']}")
    print(f"dropped_outside_imu_range : {stats['dropped_outside_imu_range']}")
    print(f"dropped_no_imu_window     : {stats['dropped_no_imu_window']}")
    print(f"split_counts              : {stats['by_split']}")

    print("\nTop labels:")
    label_counts = Counter(stats["label_distribution"])
    for label, cnt in label_counts.most_common(args.show_label_topk):
        print(f"  {label!r}: {cnt}")

    print("\nSample rows:")
    for row in rows[: max(0, args.show_samples)]:
        print(
            f"  {row.session_key} [{row.split}] "
            f"press={row.key_press_ts:.6f} key={row.key_token!r} raw={row.key_raw!r} "
            f"window=[{row.window_start_ts:.6f}, {row.window_end_ts:.6f}] "
            f"n={row.n_imu_samples} duration={row.key_duration_s}"
        )

    schema_example = {
        "subject": "003",
        "session": "005",
        "session_key": "003_005",
        "split": "train",
        "key_press_ts": 1744600000.123,
        "key_release_ts": 1744600000.201,
        "key_duration_s": 0.078,
        "key_raw": "a",
        "key_token": "a",
        "window_start_ts": 1744599999.423,
        "window_end_ts": 1744600000.273,
        "imu_idx_start": 12000,
        "imu_idx_end": 12086,
        "n_imu_samples": 86,
        "imu_target_rate_hz": 100.0,
        "rings_used": "both",
        "label_id": 42,
        "is_unk": False,
    }
    print("\nRow schema (canonical):")
    print(json.dumps(schema_example, indent=2))


if __name__ == "__main__":
    main()

"""
Preprocess raw IMU data into Chronos embeddings for training.

Loads paired IMU (CSV) and keystroke (PKL) data from the data/ directory,
creates sliding-window samples with reconstructed ground-truth text, and
encodes IMU via a frozen Chronos model.

Data format:
    {subject}_{session}_DIBS-L_corrected.csv  — Left ring IMU (6 channels)
    {subject}_{session}_DIBS-R_corrected.csv  — Right ring IMU (6 channels)
    {subject}_{session}_Macbook.pkl           — Keystroke timestamps

Per-channel Chronos encoding: each IMU axis encoded independently, then
concatenated → d_chronos = 768 * n_channels
  - Single ring (--ring L or R): n_channels=6, d_chronos=4608
  - Both rings  (--ring both):   n_channels=12, d_chronos=9216

Usage:
    python preprocess.py \\
        --raw_dir ./data \\
        --output_file ./embeddings/train.pt \\
        --ring both \\
        --window_size 10.0 \\
        --step_size 5.0
"""

import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from utils.constants import IMU_COLS, CHRONOS_DEFAULT_MODEL, IMU_EPS
from utils.filename import parse_filename as _parse_filename
from utils.imu_io import load_imu_csv as _load_imu_csv
from utils.keystroke import (
    get_keystroke_events as _get_keystroke_events,
    parse_key_name as _parse_key_name,
    translate_to_text as _translate_to_text,
    post_process_text as _post_process_text,
)
from utils.chronos_encode import encode_with_chronos


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess IMU → Chronos embeddings")
    p.add_argument("--raw_dir", type=str, required=True,
                   help="Directory containing CSV and PKL data files")
    p.add_argument("--output_dir", type=str, default="./embeddings",
                   help="Directory to save train.pt, val.pt, test.pt")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of sessions held out for validation")
    p.add_argument("--test_split", type=float, default=0.1,
                   help="Fraction of sessions held out for testing")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for session shuffle")
    p.add_argument("--chronos_model", type=str, default=CHRONOS_DEFAULT_MODEL,
                   help="Chronos model ID from HuggingFace")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ring", type=str, default="both", choices=["L", "R", "both"],
                   help="Which ring(s) to encode. 'both' concatenates L+R channels.")
    p.add_argument("--window_size", type=float, default=10.0,
                   help="Sliding window duration in seconds")
    p.add_argument("--step_size", type=float, default=5.0,
                   help="Sliding window step size in seconds")
    p.add_argument("--min_text_len", type=int, default=5,
                   help="Minimum characters in a window to create a sample")
    p.add_argument("--target_hz", type=int, default=100,
                   help="Target sampling rate (Hz) to resample IMU data to")
    p.add_argument("--no_normalize", action="store_true",
                   help="Disable per-session per-channel z-score normalization "
                        "(normalization is ON by default)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Keystroke → text reconstruction (mirrors regenerate_text.py)
# ---------------------------------------------------------------------------

def _load_pkl(pkl_path) -> dict:
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# IMU loading & resampling
# ---------------------------------------------------------------------------

def _extract_imu_window(
    timestamps: np.ndarray,
    imu_array: np.ndarray,
    t_start: float,
    t_end: float,
    target_hz: int,
) -> np.ndarray | None:
    """
    Extract and resample an IMU window to a fixed sampling rate.

    Returns (n_samples, n_channels) at target_hz, or None if insufficient data.
    """
    mask = (timestamps >= t_start) & (timestamps < t_end)
    if mask.sum() < 2:
        return None
    ts = timestamps[mask]
    data = imu_array[mask]
    n_target = max(2, int((t_end - t_start) * target_hz))
    target_ts = np.linspace(ts[0], ts[-1], n_target)
    resampled = np.zeros((n_target, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        resampled[:, ch] = np.interp(target_ts, ts, data[:, ch])
    return resampled


# ---------------------------------------------------------------------------
# Per-session IMU normalization
# ---------------------------------------------------------------------------

def _normalize_imu(imu_array: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each IMU channel using session-wide statistics.

    Computing stats across all timesteps of the session (not per-window)
    so every window from the same session shares the same scale, while
    removing person-specific sensor offsets and magnitude differences
    across sessions/subjects.

    Args:
        imu_array: (n_timesteps, n_channels)

    Returns:
        Normalized array of same shape.
    """
    mean = imu_array.mean(axis=0, keepdims=True)   # (1, n_channels)
    std  = imu_array.std(axis=0, keepdims=True)    # (1, n_channels)
    std  = np.where(std < IMU_EPS, 1.0, std)        # avoid div-by-zero on static channels
    return (imu_array - mean) / std


# ---------------------------------------------------------------------------
# Main sample loader
# ---------------------------------------------------------------------------

def _load_raw_samples(
    raw_dir: str,
    window_size: float,
    step_size: float,
    min_text_len: int,
    ring: str,
    target_hz: int,
    normalize: bool = True,
) -> list[dict]:
    """
    Create IMU + text sample pairs from the CSV/PKL data directory.

    For each subject/session, a sliding window is applied over the keystroke
    timeline. Each window yields:
      - imu:  np.ndarray (n_timesteps, n_channels) resampled to target_hz
      - text: str reconstructed from keystrokes in the window

    Args:
        raw_dir:      Path to data/ directory with CSV and PKL files
        window_size:  Window duration in seconds
        step_size:    Step between consecutive windows in seconds
        min_text_len: Skip windows with fewer characters than this
        ring:         'L', 'R', or 'both' — which ring(s) to include
        target_hz:    Resample IMU to this rate before Chronos encoding
        normalize:    If True, z-score normalize each channel per session
                      before windowing (removes person-specific offsets)

    Returns:
        list of {'imu': np.ndarray, 'text': str}
    """
    data_path = Path(raw_dir)

    # --- Group files by (subject, session) ---
    sessions: dict = {}
    for f in sorted(data_path.glob("*.csv")):
        subject, session, ring_side = _parse_filename(f.name)
        if subject and session and ring_side:
            sessions.setdefault((subject, session), {})
            sessions[(subject, session)][f'csv_{ring_side}'] = f

    for f in sorted(data_path.glob("*_Macbook.pkl")):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            key = (parts[0], parts[1])
            if key in sessions:
                sessions[key]['pkl'] = f

    samples = {}  # (subject, session) → list of dicts
    for (subject, session), files in sorted(sessions.items()):
        if 'pkl' not in files:
            print(f"  [{subject}_{session}] no PKL file, skipping")
            continue

        rings_to_use = []
        if ring in ('L', 'both') and 'csv_L' in files:
            rings_to_use.append('L')
        if ring in ('R', 'both') and 'csv_R' in files:
            rings_to_use.append('R')
        if not rings_to_use:
            print(f"  [{subject}_{session}] no CSV for ring={ring}, skipping")
            continue

        ring_label = '+'.join(rings_to_use)
        print(f"  [{subject}_{session}] loading IMU ({ring_label}) + keystrokes...")

        # Load keystroke events
        events = _get_keystroke_events(_load_pkl(files['pkl']))
        if len(events) < 5:
            print(f"  [{subject}_{session}] too few events, skipping")
            continue

        # Load IMU CSV(s) and optionally normalize per session
        imu_data: dict[str, tuple] = {}
        for side in rings_to_use:
            ts, arr = _load_imu_csv(files[f'csv_{side}'])
            if normalize:
                arr = _normalize_imu(arr)
            imu_data[side] = (ts, arr)

        # Sliding window
        session_start = events[0]['timestamp']
        session_end = events[-1]['timestamp']
        session_samples = []
        t = session_start

        while t + window_size <= session_end:
            t_end = t + window_size

            window_events = [e for e in events if t <= e['timestamp'] < t_end]
            if len(window_events) < 3:
                t += step_size
                continue

            key_seq = [_parse_key_name(e['key']) for e in window_events]
            text = _post_process_text(_translate_to_text(key_seq))
            if len(text.strip()) < min_text_len:
                t += step_size
                continue

            # Extract resampled IMU windows for each ring
            imu_arrays = []
            for side in rings_to_use:
                ts, arr = imu_data[side]
                window_arr = _extract_imu_window(ts, arr, t, t_end, target_hz)
                if window_arr is not None:
                    imu_arrays.append(window_arr)

            if not imu_arrays:
                t += step_size
                continue

            # Concatenate rings along channel axis (both have same n_target rows)
            imu_combined = (np.concatenate(imu_arrays, axis=1)
                            if len(imu_arrays) > 1 else imu_arrays[0])

            session_samples.append({'imu': imu_combined, 'text': text})
            t += step_size

        samples[(subject, session)] = session_samples
        print(f"  [{subject}_{session}] created {len(session_samples)} samples")

    total = sum(len(v) for v in samples.values())
    print(f"Total: {total} samples across {len(samples)} sessions from {raw_dir}")
    return samples


def _split_sessions(
    session_map: dict,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Split sessions into train/val/test by session (not by window), so the
    model is evaluated on unseen typing sessions rather than unseen windows
    from the same session.

    Returns three flat lists of raw samples (dicts with 'imu' and 'text').
    """
    import random
    keys = list(session_map.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_test = max(1, round(n * test_split))
    n_val  = max(1, round(n * val_split))
    n_train = n - n_val - n_test

    train_keys = keys[:n_train]
    val_keys   = keys[n_train : n_train + n_val]
    test_keys  = keys[n_train + n_val :]

    def flatten(ks):
        out = []
        for k in ks:
            out.extend(session_map[k])
        return out

    train = flatten(train_keys)
    val   = flatten(val_keys)
    test  = flatten(test_keys)

    print(f"\nSplit ({n} sessions, seed={seed}):")
    print(f"  train: {len(train_keys)} sessions → {len(train)} samples  {[f'{s}_{se}' for s, se in train_keys]}")
    print(f"  val:   {len(val_keys)}  sessions → {len(val)} samples  {[f'{s}_{se}' for s, se in val_keys]}")
    print(f"  test:  {len(test_keys)} sessions → {len(test)} samples  {[f'{s}_{se}' for s, se in test_keys]}")
    return train, val, test


def main():
    args = get_args()

    session_map = _load_raw_samples(
        raw_dir=args.raw_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        min_text_len=args.min_text_len,
        ring=args.ring,
        target_hz=args.target_hz,
        normalize=not args.no_normalize,
    )

    if not session_map:
        print("No samples found. Check --raw_dir and that CSV/PKL files are present.")
        return

    first = next(s for v in session_map.values() for s in v)
    n_channels = first["imu"].shape[1]
    print(f"IMU channels per sample: {n_channels}  →  d_chronos = 768 * {n_channels} = {768 * n_channels}")

    train_raw, val_raw, test_raw = _split_sessions(
        session_map, args.val_split, args.test_split, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, raw_split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        if not raw_split:
            print(f"  [{split_name}] no samples, skipping")
            continue
        encoded = encode_with_chronos(
            raw_split, args.chronos_model, args.batch_size, args.device
        )
        out_path = output_dir / f"{split_name}.pt"
        torch.save(encoded, str(out_path))
        print(f"Saved {len(encoded)} {split_name} samples to {out_path}")


if __name__ == "__main__":
    main()

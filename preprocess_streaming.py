#!/usr/bin/env python3
"""
Memory-safe streaming preprocessing:
- Processes ONE session at a time
- Encodes immediately with Chronos
- Saves to disk
- Never stores full dataset in RAM
"""

import argparse
import pickle
from pathlib import Path
import gc

import torch
import numpy as np

from utils.filename import parse_filename as _parse_filename
from utils.imu_io import load_imu_csv as _load_imu_csv
from utils.keystroke import (
    get_keystroke_events as _get_keystroke_events,
    parse_key_name as _parse_key_name,
    translate_to_text as _translate_to_text,
    post_process_text as _post_process_text,
)
from utils.chronos_encode import encode_with_chronos


# ----------------------------
# Device (CUDA → MPS → CPU)
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    else:
        return torch.device("cpu"), "CPU"


# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--output_dir", default="./embeddings")
    p.add_argument("--chronos_model", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--window_size", type=float, default=10.0)
    p.add_argument("--step_size", type=float, default=5.0)
    p.add_argument("--min_text_len", type=int, default=5)
    p.add_argument("--target_hz", type=int, default=100)
    p.add_argument("--ring", default="both", choices=["L", "R", "both"])
    return p.parse_args()


# ----------------------------
# IMU window extraction
# ----------------------------
def extract_window(ts, arr, t0, t1, hz):
    mask = (ts >= t0) & (ts < t1)
    if mask.sum() < 2:
        return None

    ts_win = ts[mask]
    data = arr[mask]

    n = max(2, int((t1 - t0) * hz))
    target_ts = np.linspace(ts_win[0], ts_win[-1], n)

    out = np.zeros((n, data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        out[:, i] = np.interp(target_ts, ts_win, data[:, i])
    return out


# ----------------------------
# Stream sessions
# ----------------------------
def iter_sessions(raw_dir):
    raw_path = Path(raw_dir)
    sessions = {}

    for f in raw_path.glob("*.csv"):
        subject, session, ring = _parse_filename(f.name)
        if subject and session:
            sessions.setdefault((subject, session), {})
            sessions[(subject, session)][f"csv_{ring}"] = f

    for f in raw_path.glob("*_Macbook.pkl"):
        parts = f.stem.split("_")
        key = (parts[0], parts[1])
        if key in sessions:
            sessions[key]["pkl"] = f

    for key, files in sorted(sessions.items()):
        if "pkl" not in files:
            continue
        yield key, files


# ----------------------------
# Build windows (per session)
# ----------------------------
def build_samples(files, args):
    events = _get_keystroke_events(pickle.load(open(files["pkl"], "rb")))
    if len(events) < 5:
        return []

    rings = []
    if args.ring in ("L", "both") and "csv_L" in files:
        rings.append("L")
    if args.ring in ("R", "both") and "csv_R" in files:
        rings.append("R")

    imu_data = {}
    for r in rings:
        imu_data[r] = _load_imu_csv(files[f"csv_{r}"])

    start = events[0]["timestamp"]
    end = events[-1]["timestamp"]

    samples = []
    t = start

    while t + args.window_size <= end:
        t1 = t + args.window_size

        ev = [e for e in events if t <= e["timestamp"] < t1]
        if len(ev) < 3:
            t += args.step_size
            continue

        text = _post_process_text(
            _translate_to_text([_parse_key_name(e["key"]) for e in ev])
        )

        if len(text.strip()) < args.min_text_len:
            t += args.step_size
            continue

        imu_list = []
        for r in rings:
            ts, arr = imu_data[r]
            w = extract_window(ts, arr, t, t1, args.target_hz)
            if w is not None:
                imu_list.append(w)

        if imu_list:
            imu = np.concatenate(imu_list, axis=1) if len(imu_list) > 1 else imu_list[0]
            samples.append({"imu": imu, "text": text})

        t += args.step_size

    return samples


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = get_args()

    device, device_name = get_device()
    print(f"Using device: {device_name}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0

    for (subject, session), files in iter_sessions(args.raw_dir):
        print(f"\n[{subject}_{session}] Processing...")

        samples = build_samples(files, args)
        print(f"  Built {len(samples)} windows")

        if not samples:
            continue

        encoded = encode_with_chronos(
            samples,
            args.chronos_model,
            args.batch_size,
            device
        )

        out_path = out_dir / f"{subject}_{session}.pt"
        torch.save(encoded, out_path)

        print(f"  Saved → {out_path}")

        total_samples += len(encoded)

        #free memory
        del samples
        del encoded
        gc.collect()

    print(f"\nDone. Total samples processed: {total_samples}")


if __name__ == "__main__":
    main()

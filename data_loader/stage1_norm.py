"""Global per-channel normalization for Stage-1 IMU (fit on train only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Stage1ExportConfig
from .stage1_dataset import Stage1IMUKeyDataset


def imu_lr_to_multivariate(imu_l: torch.Tensor, imu_r: torch.Tensor) -> torch.Tensor:
    """(T, 6), (T, 6) -> (12, T) Chronos multivariate layout."""
    # Channels: L accel(3), L gyro(3), R accel(3), R gyro(3)
    x = torch.cat([imu_l, imu_r], dim=-1)  # (T, 12)
    return x.transpose(0, 1).contiguous()


def compute_global_norm_stats(
    train_dataset: Stage1IMUKeyDataset,
    batch_size: int = 256,
    num_workers: int = 0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std over all real timesteps and samples, per channel (12). No padding leakage."""

    def _collate_raw(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_raw,
    )
    n = 0
    mean = torch.zeros(12, dtype=torch.float64)
    m2 = torch.zeros(12, dtype=torch.float64)
    for samples in loader:
        for s in samples:
            mv = imu_lr_to_multivariate(s["imu_l"], s["imu_r"]).double()
            x_flat = mv.transpose(0, 1)
            for i in range(x_flat.shape[0]):
                n += 1
                row = x_flat[i]
                delta = row - mean
                mean = mean + delta / n
                m2 = m2 + delta * (row - mean)
    var = m2 / max(n - 1, 1)
    std = torch.sqrt(var + eps)
    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def collate_stack_lr_pad_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Align variable-length windows in-batch with **left NaN padding** (missing time)."""
    imu_mv_list = []
    label_ids = []
    for s in batch:
        mv = imu_lr_to_multivariate(s["imu_l"], s["imu_r"])
        imu_mv_list.append(mv)
        label_ids.append(s["label_id"])
    lengths = [x.shape[1] for x in imu_mv_list]
    max_t = max(lengths)
    padded = []
    for x in imu_mv_list:
        if x.shape[1] < max_t:
            pad_w = max_t - x.shape[1]
            pad = torch.full((12, pad_w), float("nan"), dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)
        padded.append(x)
    return {
        "imu_mv": torch.stack(padded, dim=0),
        "label_id": torch.stack(label_ids, dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def save_norm_stats(path: Path, mean: np.ndarray, std: np.ndarray, eps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"mean": mean.tolist(), "std": std.tolist(), "eps": eps},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_norm_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32), float(d.get("eps", 1e-6))


def normalize_mv(x_12t: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """x_12t: (12, T) or (B, 12, T), float32."""
    m = torch.as_tensor(mean, device=x_12t.device, dtype=x_12t.dtype).view(-1, 1)
    s = torch.as_tensor(std, device=x_12t.device, dtype=x_12t.dtype).view(-1, 1)
    return (x_12t - m) / s


def pad_crop_temporal(
    x_b12t: torch.Tensor,
    context_length: int,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Crop to the last `context_length` steps, or left-pad with **NaN** (missing data).

    Chronos uses NaN to build validity masks; zeros after z-score are ambiguous.
    """
    b, c, t = x_b12t.shape
    assert c == 12
    if t == context_length:
        return x_b12t
    out = torch.full(
        (b, c, context_length),
        float("nan"),
        device=x_b12t.device,
        dtype=x_b12t.dtype,
    )
    if t >= context_length:
        out[:] = x_b12t[:, :, -context_length:]
    else:
        out[:, :, context_length - t :] = x_b12t
    return out

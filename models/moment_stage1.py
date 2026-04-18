"""MOMENT encoder helpers for Stage-1 (B, V, L, D) tokens — same layout as Chronos heads."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F


def resample_imu_to_moment_seq(imu_bvt: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Linearly resample time axis: ``(B, V, T)`` → ``(B, V, seq_len)``."""
    if imu_bvt.dim() != 3:
        raise ValueError(f"Expected imu (B, V, T), got shape {tuple(imu_bvt.shape)}")
    b, v, t = imu_bvt.shape
    if t == seq_len:
        return imu_bvt.float()
    x = imu_bvt.reshape(b * v, 1, t).float()
    x = F.interpolate(x, size=seq_len, mode="linear", align_corners=False)
    return x.reshape(b, v, seq_len)


def moment_embed_stacked(
    pipeline: Any,
    imu_bvt: torch.Tensor,
    *,
    moment_seq_len: int,
    device: torch.device,
    grad: bool,
) -> torch.Tensor:
    """
    Run MOMENT ``embed(..., reduction='none')`` and return ``(B, V, L, D)``.

    ``imu_bvt`` may be on CPU or ``device``; encoding runs on ``device``.
    """
    x = resample_imu_to_moment_seq(imu_bvt, moment_seq_len)
    x = x.to(device=device, dtype=torch.float32)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = pipeline.embed(x_enc=x, reduction="none")
        emb = out.embeddings
    if emb.dim() != 4:
        raise ValueError(f"Expected MOMENT embeddings (B, V, L, D), got {tuple(emb.shape)}")
    return emb


def read_moment_patch_info(pipeline: Any) -> Dict[str, Any]:
    """Patch / sequence metadata for logging and embedding-cache signatures."""
    cfg = getattr(pipeline, "config", None)
    if cfg is None:
        return {}
    out: Dict[str, Any] = {}
    for k in ("seq_len", "patch_len", "patch_stride_len", "d_model"):
        v = getattr(cfg, k, None) if not isinstance(cfg, dict) else cfg.get(k)
        if v is not None:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def load_moment_pipeline(model_name: str, device: torch.device) -> Any:
    from momentfm import MOMENTPipeline

    pipeline = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )
    pipeline.init()
    pipeline = pipeline.to(device)
    return pipeline

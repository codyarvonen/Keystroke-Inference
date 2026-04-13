"""
Classification heads for frozen Chronos encoder tokens (B, V, L, D).

All heads accept either:
  - (B, V, L, D) pre-pooled encoder tokens (preferred), or
  - (B, D) legacy pooled embeddings (mean-pool fallback is applied only for linear/mlp).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE1_HEAD_TYPES: Tuple[str, ...] = (
    "linear",
    "mlp",
    "attention_pool",
    "conv1d",
    "lstm",
)


def _pool_mean_tokens(z: torch.Tensor) -> torch.Tensor:
    """(B, V, L, D) -> (B, D)."""
    b, v, l, d = z.shape
    return z.reshape(b, v * l, d).mean(dim=1)


def forward_stage1_head(head: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Dispatch to head.forward(z); z is (B, V, L, D) or (B, D)."""
    return head(z)


class LinearClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return self.fc(z)
        if z.dim() == 4:
            return self.fc(_pool_mean_tokens(z))
        raise ValueError(f"Expected z (B,D) or (B,V,L,D), got shape {tuple(z.shape)}")


class MLPClassifierHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: List[nn.Module] = []
        in_dim = d_model
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = _pool_mean_tokens(z)
        elif z.dim() != 2:
            raise ValueError(f"Expected z (B,D) or (B,V,L,D), got shape {tuple(z.shape)}")
        return self.net(z)


class AttentionPoolClassifierHead(nn.Module):
    """
    Single learnable query over flattened V×L positions (scaled dot-product), then linear to classes.
    """

    def __init__(self, d_model: int, num_classes: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.scale = 1.0 / math.sqrt(float(d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return self.fc(z)
        if z.dim() != 4:
            raise ValueError(f"Expected z (B,D) or (B,V,L,D), got shape {tuple(z.shape)}")
        b, v, l, d = z.shape
        x = z.reshape(b, v * l, d)
        x = self.norm(x)
        q = self.query.expand(b, -1, -1)
        scores = (x * q).sum(dim=-1) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        pooled = (attn.unsqueeze(-1) * x).sum(dim=1)
        return self.fc(pooled)


class Conv1DClassifierHead(nn.Module):
    """
    Per-position projection, then 1D conv over time L with C = num_variates * d_proj channels.
    `num_variates` must match Chronos encoder variate count (12 for 6+6 IMU channels).
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        num_variates: int = 12,
        d_proj: int = 128,
        conv_channels: int = 256,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_variates = int(num_variates)
        self.d_proj = int(d_proj)
        self.proj = nn.Linear(d_model, d_proj)
        in_ch = self.num_variates * self.d_proj
        self.conv_channels = int(conv_channels)
        self.num_conv_layers = int(num_conv_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        pad = kernel_size // 2
        convs: List[nn.Module] = []
        bns: List[nn.Module] = []
        c = in_ch
        for _ in range(self.num_conv_layers):
            out_c = self.conv_channels
            convs.append(nn.Conv1d(c, out_c, kernel_size=kernel_size, padding=pad))
            bns.append(nn.BatchNorm1d(out_c))
            c = out_c
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(
                f"conv1d head expects z (B,V,L,D); legacy (B,D) pooled cache is not supported. "
                f"Got shape {tuple(z.shape)}"
            )
        b, v, l, d = z.shape
        if v != self.num_variates:
            raise ValueError(f"Expected V={self.num_variates} variates, got V={v}")
        x = self.proj(z)
        x = x.reshape(b, v * self.d_proj, l)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h)
            h = bn(h)
            h = self.act(h)
            h = self.dropout(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.fc(h)


class LSTMClassifierHead(nn.Module):
    """
    Mean over variates -> (B, L, D), then BiLSTM over time, mean of outputs, classify.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(
                f"lstm head expects z (B,V,L,D); legacy (B,D) pooled cache is not supported. "
                f"Got shape {tuple(z.shape)}"
            )
        x = z.mean(dim=1)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


def build_stage1_head(
    head_type: str,
    d_model: int,
    num_classes: int,
    *,
    num_variates: int = 12,
    hidden_dim: int = 256,
    head_num_layers: int = 2,
    dropout: float = 0.1,
    conv_channels: int = 256,
    conv_d_proj: int = 128,
    conv_kernel_size: int = 3,
    conv_num_layers: int = 2,
    lstm_hidden_dim: int = 256,
    lstm_num_layers: int = 1,
) -> nn.Module:
    ht = head_type.lower().strip()
    if ht == "linear":
        return LinearClassifierHead(d_model, num_classes)
    if ht == "mlp":
        return MLPClassifierHead(
            d_model,
            num_classes,
            hidden_dim=hidden_dim,
            num_layers=head_num_layers,
            dropout=dropout,
        )
    if ht == "attention_pool":
        return AttentionPoolClassifierHead(d_model, num_classes, dropout=dropout)
    if ht == "conv1d":
        return Conv1DClassifierHead(
            d_model,
            num_classes,
            num_variates=num_variates,
            d_proj=conv_d_proj,
            conv_channels=conv_channels,
            num_conv_layers=conv_num_layers,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
    if ht == "lstm":
        return LSTMClassifierHead(
            d_model,
            num_classes,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown head_type {head_type!r}; choose from {STAGE1_HEAD_TYPES}")


def head_config_from_args(ns: Any) -> Dict[str, Any]:
    """Snapshot for checkpoints / reproducibility."""
    return {
        "head_type": getattr(ns, "head_type", "linear"),
        "num_variates": getattr(ns, "encoder_num_variates", 12),
        "hidden_dim": getattr(ns, "head_hidden_dim", 256),
        "head_num_layers": getattr(ns, "head_num_layers", 2),
        "dropout": getattr(ns, "head_dropout", 0.1),
        "conv_channels": getattr(ns, "head_conv_channels", 256),
        "conv_d_proj": getattr(ns, "head_conv_d_proj", 128),
        "conv_kernel_size": getattr(ns, "head_conv_kernel_size", 3),
        "conv_num_layers": getattr(ns, "head_conv_num_layers", 2),
        "lstm_hidden_dim": getattr(ns, "head_lstm_hidden_dim", 256),
        "lstm_num_layers": getattr(ns, "head_lstm_num_layers", 1),
    }


def load_stage1_head_from_checkpoint(
    ckpt: Mapping[str, Any],
    *,
    d_model: int,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Restore head from checkpoint dict. Supports legacy checkpoints (Linear only).
    """
    cfg = ckpt.get("head_config")
    if not cfg:
        head = build_stage1_head("linear", d_model, num_classes)
        head.load_state_dict(ckpt["head_state_dict"])
        return head.to(device)

    ht = str(cfg.get("head_type", "linear"))
    head = build_stage1_head(
        ht,
        int(cfg.get("d_model", d_model)),
        int(cfg.get("num_classes", num_classes)),
        num_variates=int(cfg.get("num_variates", 12)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        head_num_layers=int(cfg.get("head_num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        conv_channels=int(cfg.get("conv_channels", 256)),
        conv_d_proj=int(cfg.get("conv_d_proj", 128)),
        conv_kernel_size=int(cfg.get("conv_kernel_size", 3)),
        conv_num_layers=int(cfg.get("conv_num_layers", 2)),
        lstm_hidden_dim=int(cfg.get("lstm_hidden_dim", 256)),
        lstm_num_layers=int(cfg.get("lstm_num_layers", 1)),
    )
    head.load_state_dict(ckpt["head_state_dict"])
    return head.to(device)

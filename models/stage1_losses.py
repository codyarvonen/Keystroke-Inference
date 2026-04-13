"""Loss functions for Stage-1 classification (class imbalance)."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ClassWeightMode = Literal["none", "inverse", "inverse_sqrt", "effective"]
LossType = Literal["ce", "weighted_ce", "focal", "weighted_focal"]


def bincount_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Integer labels (N,) -> per-class counts (num_classes,) on CPU float."""
    if labels.dtype != torch.long:
        labels = labels.long()
    return torch.bincount(labels.clamp(min=0, max=num_classes - 1), minlength=num_classes).float()


def class_weights_from_counts(
    counts: torch.Tensor,
    mode: ClassWeightMode,
    *,
    beta: float = 0.9999,
) -> Optional[torch.Tensor]:
    """
    Map per-class sample counts to nonnegative weights; normalize to mean 1.
    Modes:
      none: no weighting (returns None for use with unweighted CE)
      inverse: w_c ∝ 1 / max(n_c, 1)
      inverse_sqrt: w_c ∝ 1 / sqrt(max(n_c, 1))
      effective: Class-Balanced Loss effective number of samples (Cui et al.), beta in (0,1)
    """
    if mode == "none":
        return None
    counts = counts.clone()
    counts = torch.clamp(counts, min=0.0)
    if mode == "inverse":
        w = 1.0 / torch.clamp(counts, min=1.0)
    elif mode == "inverse_sqrt":
        w = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
    elif mode == "effective":
        n = torch.clamp(counts, min=1.0)
        b = torch.tensor(beta, dtype=n.dtype, device=n.device)
        eff = (1.0 - torch.pow(b, n)) / (1.0 - b).clamp(min=1e-8)
        w = 1.0 / torch.clamp(eff, min=1e-8)
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    w = w / w.mean().clamp(min=1e-8)
    return w


class FocalLoss(nn.Module):
    """
    Multiclass focal loss on logits. Optional per-class alpha weights (same role as CE weight).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        *,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().float())
        else:
            self.alpha = None
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce.detach()).clamp(min=1e-8, max=1.0)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = self.alpha.to(device=logits.device, dtype=logits.dtype)[targets]
            loss = loss * at
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_stage1_criterion(
    *,
    loss_type: LossType,
    num_classes: int,
    class_counts: torch.Tensor,
    class_weight_mode: ClassWeightMode,
    focal_gamma: float = 2.0,
    class_balance_beta: float = 0.9999,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Returns (criterion, meta) where criterion(logits, targets) -> scalar.
    class_counts: (num_classes,) on any device; copied to weight tensor on `device`.
    """
    counts_cpu = class_counts.detach().float().cpu()
    w_cpu = class_weights_from_counts(counts_cpu, class_weight_mode, beta=class_balance_beta)
    meta: Dict[str, Any] = {
        "loss_type": loss_type,
        "class_weight_mode": class_weight_mode,
        "focal_gamma": focal_gamma,
        "class_counts_sum": int(counts_cpu.sum().item()),
    }
    if w_cpu is not None:
        meta["weight_vector"] = w_cpu.tolist()

    weight = w_cpu.to(device=device) if w_cpu is not None else None

    if loss_type == "ce":
        if weight is not None:
            raise ValueError("ce with class_weight_mode!=none is ambiguous; use weighted_ce")
        return nn.CrossEntropyLoss(), meta

    if loss_type == "weighted_ce":
        if weight is None:
            raise ValueError("weighted_ce requires class_weight_mode != none")
        return nn.CrossEntropyLoss(weight=weight), meta

    if loss_type == "focal":
        alpha = weight  # optional reweight inside focal
        return FocalLoss(gamma=focal_gamma, alpha=alpha), meta

    if loss_type == "weighted_focal":
        if weight is None:
            raise ValueError("weighted_focal requires class_weight_mode != none")
        return FocalLoss(gamma=focal_gamma, alpha=weight), meta

    raise ValueError(f"Unknown loss_type: {loss_type}")


def criterion_forward(criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropyLoss vs FocalLoss."""
    if isinstance(criterion, FocalLoss):
        return criterion(logits, targets)
    return criterion(logits, targets)

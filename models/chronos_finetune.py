"""
Helpers for fine-tuning a subset of Chronos-2 (last encoder blocks) for Stage-1 classification.

Uses ``Chronos2Model.encode`` (same path as ``Chronos2Pipeline.embed``) with gradients enabled.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


def chronos_encode_multivariate_grad(
    pipeline: Any,
    imu_bvt: torch.Tensor,
    *,
    context_length: int,
) -> torch.Tensor:
    """
    Differentiable encoder forward for multivariate inputs.

    Parameters
    ----------
    pipeline
        ``Chronos2Pipeline`` with ``.model`` set.
    imu_bvt
        ``(B, V, T)`` float tensor on the same device as the model (e.g. V=12 IMU channels).

    Returns
    -------
    torch.Tensor
        ``(B, V, L, D)`` encoder hidden states (same layout as ``stack_embeddings_from_embed_list``).
    """
    if imu_bvt.dim() != 3:
        raise ValueError(f"Expected imu (B, V, T), got shape {tuple(imu_bvt.shape)}")
    b, v, t = imu_bvt.shape
    dev = imu_bvt.device
    dtype = torch.float32
    context = imu_bvt.reshape(b * v, t).to(device=dev, dtype=dtype)
    group_ids = torch.arange(b, device=dev, dtype=torch.long).repeat_interleave(v)

    enc_out, _loc_scale, *_ = pipeline.model.encode(
        context=context,
        group_ids=group_ids,
    )
    hidden = getattr(enc_out, "last_hidden_state", None)
    if hidden is None:
        hidden = enc_out[0]
    # (B*V, L, D) -> (B, V, L, D)
    lv, ldim = int(hidden.shape[1]), int(hidden.shape[2])
    stacked = hidden.reshape(b, v, lv, ldim)
    return stacked


def freeze_all_chronos_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def configure_chronos_encoder_finetune(model: Any, unfreeze_last_n_blocks: int) -> int:
    """
    Freeze the full Chronos-2 ``model``, then enable gradients on the last ``unfreeze_last_n_blocks``
    ``Chronos2Encoder`` blocks and ``final_layer_norm``.

    Returns the number of encoder layers (for validation).
    """
    freeze_all_chronos_parameters(model)
    enc = model.encoder
    n_layers = len(enc.block)
    if unfreeze_last_n_blocks <= 0:
        return n_layers
    if unfreeze_last_n_blocks > n_layers:
        raise ValueError(
            f"encoder_finetune_last_n ({unfreeze_last_n_blocks}) exceeds encoder depth ({n_layers})"
        )
    start = n_layers - unfreeze_last_n_blocks
    for i in range(start, n_layers):
        for p in enc.block[i].parameters():
            p.requires_grad = True
    for p in enc.final_layer_norm.parameters():
        p.requires_grad = True
    return n_layers


def set_encoder_finetune_train_mode(model: Any, unfreeze_last_n_blocks: int, training: bool) -> None:
    """
    Set ``model.eval()`` globally, then set the last ``unfreeze_last_n_blocks`` encoder blocks
    (and ``final_layer_norm``) to train or eval mode so dropout matches train vs val.
    """
    model.eval()
    if unfreeze_last_n_blocks <= 0:
        return
    enc = model.encoder
    n_layers = len(enc.block)
    start = max(0, n_layers - unfreeze_last_n_blocks)
    for i in range(start, n_layers):
        enc.block[i].train(training)
    enc.final_layer_norm.train(training)

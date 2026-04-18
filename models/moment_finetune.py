"""
Last-N-block fine-tuning for MOMENT (T5 encoder backbone), mirroring ``chronos_finetune``.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def configure_moment_encoder_finetune(moment_model: nn.Module, unfreeze_last_n_blocks: int) -> int:
    """
    Freeze the full MOMENT module, then enable gradients on the last ``unfreeze_last_n_blocks``
    T5 encoder blocks and ``final_layer_norm``.

    Returns the number of encoder layers (for validation).
    """
    freeze_all_parameters(moment_model)
    enc = moment_model.encoder
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


def set_moment_encoder_finetune_train_mode(
    moment_model: nn.Module, unfreeze_last_n_blocks: int, training: bool
) -> None:
    """``model.eval()`` globally, then last-N blocks + ``final_layer_norm`` train or eval."""
    moment_model.eval()
    if unfreeze_last_n_blocks <= 0:
        return
    enc = moment_model.encoder
    n_layers = len(enc.block)
    start = max(0, n_layers - unfreeze_last_n_blocks)
    for i in range(start, n_layers):
        enc.block[i].train(training)
    enc.final_layer_norm.train(training)

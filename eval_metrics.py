"""Teacher-forced token top-k accuracy (same positions as cross-entropy loss)."""

from __future__ import annotations

import torch


def token_topk_update(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ks: list[int],
    accum: dict,
) -> None:
    """
    Accumulate hits for top-k accuracy on non-padded label positions.

    Args:
        logits: (B, S, V) next-token logits from the model
        labels: (B, S) with -100 for padding / ignored positions
        ks:     e.g. [1, 5, 10]; each k is capped by vocab size
        accum:  dict with keys n_tokens, top{k}_hits (updated in-place)
    """
    valid = labels != -100
    if not valid.any():
        return

    logits_v = logits[valid].float()
    targets_v = labels[valid]
    vocab = logits_v.size(-1)
    max_k = max(ks)
    max_k = min(max_k, vocab)
    if max_k < 1:
        return

    _, topi = logits_v.topk(max_k, dim=-1)
    n = int(targets_v.numel())
    accum["n_tokens"] = accum.get("n_tokens", 0) + n

    for k in ks:
        kk = min(k, max_k)
        hits = (topi[:, :kk] == targets_v.unsqueeze(-1)).any(dim=-1).sum().item()
        key = f"top{k}_hits"
        accum[key] = accum.get(key, 0) + hits


def finalize_token_topk(accum: dict, ks: list[int]) -> dict[str, float]:
    """Convert accumulated hits to accuracies in [0, 1]."""
    n = accum.get("n_tokens", 0)
    out: dict[str, float] = {}
    if n <= 0:
        for k in ks:
            out[f"top{k}_acc"] = float("nan")
        return out

    for k in ks:
        hits = accum.get(f"top{k}_hits", 0)
        out[f"top{k}_acc"] = hits / n
    return out


def empty_topk_accum(ks: list[int]) -> dict:
    d = {"n_tokens": 0}
    for k in ks:
        d[f"top{k}_hits"] = 0
    return d

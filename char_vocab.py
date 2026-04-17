"""
Character vocabulary for the CTC head.

Case-sensitive: the reconstructed text preserves shift-modified characters
because pynput emits the uppercase glyph directly (the Shift token is dropped
downstream, but its effect on the following character is already baked in).
"""

from __future__ import annotations

import torch

_SPECIALS = ["<blank>", "<unk>"]
_LOWER = [chr(c) for c in range(ord("a"), ord("z") + 1)]
_UPPER = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_DIGITS = [chr(c) for c in range(ord("0"), ord("9") + 1)]
_PUNCT = [" ", ".", ",", "'", "?", "!", "-", "\n"]

ITOS: list[str] = _SPECIALS + _LOWER + _UPPER + _DIGITS + _PUNCT
STOI: dict[str, int] = {c: i for i, c in enumerate(ITOS)}

BLANK_ID: int = STOI["<blank>"]
UNK_ID: int = STOI["<unk>"]
VOCAB_SIZE: int = len(ITOS)


def encode(text: str) -> list[int]:
    return [STOI.get(ch, UNK_ID) for ch in text]


def decode(ids: list[int]) -> str:
    return "".join(
        ITOS[i] for i in ids if 0 <= i < VOCAB_SIZE and i != BLANK_ID
    )


def ctc_greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """
    Collapse repeats and drop blanks.

    Args:
        logits: (B, T, V)
    Returns:
        list of length B; each element is a list of int token ids.
    """
    preds = logits.argmax(dim=-1).detach().cpu().tolist()
    out: list[list[int]] = []
    for seq in preds:
        collapsed: list[int] = []
        prev = -1
        for tok in seq:
            if tok != prev and tok != BLANK_ID:
                collapsed.append(tok)
            prev = tok
        out.append(collapsed)
    return out

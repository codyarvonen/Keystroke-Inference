"""
Coarse Stage-1 label scheme: 32 content classes (26 letters + space + backspace + shift
+ punctuation + digits + other), plus standard <PAD>/<UNK> in vocab.
"""

from __future__ import annotations

import string
import unicodedata
from typing import Dict, List, Tuple

# Keep in sync with data_loader.stage1
STAGE1_PAD_TOKEN = "<PAD>"
STAGE1_UNK_TOKEN = "<UNK>"

# Canonical coarse tokens (must match export vocab order after PAD/UNK).
COARSE_LETTERS: Tuple[str, ...] = tuple(chr(ord("a") + i) for i in range(26))
COARSE_SPACE = " "
COARSE_BACKSPACE = "<BACKSPACE>"
COARSE_SHIFT = "<SHIFT>"
COARSE_PUNCT = "<PUNCT>"
COARSE_NUM = "<NUM>"
COARSE_OTHER = "<OTHER>"

COARSE_32_ORDER: Tuple[str, ...] = (
    COARSE_LETTERS
    + (COARSE_SPACE, COARSE_BACKSPACE, COARSE_SHIFT, COARSE_PUNCT, COARSE_NUM, COARSE_OTHER)
)

assert len(COARSE_32_ORDER) == 32

# ASCII punctuation from the stdlib plus common symbols not always included.
_PUNCT_EXTRA = frozenset({"'", '"', "`"})
_PUNCT_CHARS = frozenset(string.punctuation) | _PUNCT_EXTRA


def map_key_token_to_coarse_32(key_token: str) -> str:
    """
    Map a key_token (after parse_key_name and optional merge_letter_case) to one of 32
    coarse classes. Non-English single letters and special keys collapse to <OTHER>.
    """
    if key_token in (COARSE_SPACE, COARSE_BACKSPACE, COARSE_SHIFT):
        return key_token

    if len(key_token) == 1:
        o = ord(key_token)
        if ord("a") <= o <= ord("z"):
            return key_token
        if key_token.isdigit():
            return COARSE_NUM
        if key_token in _PUNCT_CHARS or unicodedata.category(key_token).startswith("P"):
            return COARSE_PUNCT
        if key_token.isalpha():
            return COARSE_OTHER
        return COARSE_OTHER

    # Multi-character tokens: <Key.left>, <CTRL>, literal multi-char names, etc.
    return COARSE_OTHER


def build_fixed_coarse_32_vocab() -> Tuple[Dict[str, int], List[str]]:
    """
    Fixed vocabulary: id 0 = <PAD>, 1 = <UNK>, then 32 coarse classes in COARSE_32_ORDER.
    Total 34 ids (0..33).
    """
    id_to_token: List[str] = [STAGE1_PAD_TOKEN, STAGE1_UNK_TOKEN] + list(COARSE_32_ORDER)
    token_to_id: Dict[str, int] = {t: i for i, t in enumerate(id_to_token)}
    return token_to_id, id_to_token

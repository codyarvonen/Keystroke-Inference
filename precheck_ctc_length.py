"""
CTC length-feasibility precheck.

CTC requires input_length >= target_length after blank-collapsing; empirically
T ~> 2-3x L for stable training. This script loads a preprocessed .pt file and
reports the distribution of target lengths vs. encoder sequence length S.

Usage:
    python precheck_ctc_length.py moment
    python precheck_ctc_length.py chronos
    python precheck_ctc_length.py <path-to-.pt>
"""

from __future__ import annotations

import statistics
import sys

import torch

from char_vocab import encode


_DEFAULT_PATHS = {
    "moment": "./embeddings/train_moment.pt",
    "chronos": "./embeddings/train.pt",
}


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    path = _DEFAULT_PATHS.get(arg, arg)

    data = torch.load(path, map_location="cpu")
    if not data:
        raise RuntimeError(f"{path} is empty")

    emb0 = data[0]["embeddings"]
    S = emb0.shape[0]
    lens = [len(encode(s["text"])) for s in data]
    L_max = max(lens)
    L_mean = statistics.mean(lens)

    print(f"path={path}")
    print(f"  N={len(lens)}  S={S}  L_max={L_max}  L_mean={L_mean:.1f}")

    for label, thr in [("S", S), ("S/2", S // 2), ("S/3", S // 3)]:
        frac = sum(1 for L in lens if L >= thr) / len(lens)
        print(f"  P(L >= {label}={thr}) = {frac:.3f}")

    for up in (1, 2, 4, 8):
        T = S * up
        frac_bad = sum(1 for L in lens if L >= T // 3) / len(lens)
        print(f"  upsample={up}x  T_ctc={T}  P(L >= T_ctc/3) = {frac_bad:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Learned beam-search decoder for Stage-1 keystroke predictions.

Combines the per-keystroke character posteriors from the IMU encoder with a
character-level bigram language model trained on the training-split ground-truth
labels to find the most likely character sequence for each session.

The approach is "shallow fusion" (same as classic ASR hybrid decoding):
    score(c_t) = (1 - lm_weight) * log P_enc(c_t | IMU_t)
               +       lm_weight  * log P_lm(c_t | c_{t-1})

Outputs a JSONL where each line is one session sequence with:
  - session_key, split, n_keystrokes
  - ground_truth_text
  - encoder_text  (greedy top-1)
  - beam_text     (beam search output)
  - cer_encoder, cer_beam

Usage
-----
# Decode test split
python decode_learned.py \\
    --predictions predictions.jsonl \\
    --split test \\
    --beam-size 5 \\
    --lm-weight 0.3 \\
    --out beam_decoded.jsonl

# Evaluate all splits (LM fit on training rows only)
python decode_learned.py --predictions predictions.jsonl --split all
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text assembly from key token sequence
# ---------------------------------------------------------------------------

def _tokens_to_text(tokens: List[str]) -> str:
    """Convert raw key_token list to clean text (handles backspace etc.)."""
    out: List[str] = []
    for tok in tokens:
        if tok == " ":
            out.append(" ")
        elif tok == "\n":
            out.append("\n")
        elif tok == "\t":
            out.append("\t")
        elif tok == "<BACKSPACE>":
            if out:
                out.pop()
        elif tok in ("<SHIFT>", "<CTRL>", "<ALT>", "<CMD>", "<ESC>", "<PAD>", "<UNK>"):
            pass
        elif tok.startswith("<") and tok.endswith(">"):
            pass
        else:
            out.append(tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# CER utility
# ---------------------------------------------------------------------------

def _levenshtein(s: str, t: str) -> int:
    if not s:
        return len(t)
    if not t:
        return len(s)
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s[i - 1] == t[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, gt) / len(gt)


# ---------------------------------------------------------------------------
# Bigram language model
# ---------------------------------------------------------------------------

class CharBigramLM:
    """
    Maximum-likelihood bigram character LM with add-k smoothing.

    Trained from sequences of key_token strings.
    Special start/end tokens are added around each sequence.
    """

    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, k: float = 0.1):
        self.k = k                                        # smoothing
        self._vocab: set = set()
        self._bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._unigrams: Dict[str, int] = defaultdict(int)
        self._fitted = False

    def fit(self, sequences: List[List[str]]) -> None:
        """sequences: list of key_token lists (one per session/sentence)."""
        for seq in sequences:
            prev = self.BOS
            for tok in seq:
                self._bigrams[prev][tok] += 1
                self._unigrams[tok] += 1
                self._vocab.add(tok)
                prev = tok
            self._bigrams[prev][self.EOS] += 1
        self._vocab.add(self.BOS)
        self._vocab.add(self.EOS)
        self._fitted = True

    def log_prob(self, token: str, prev_token: Optional[str]) -> float:
        """log P(token | prev_token) with add-k smoothing."""
        if not self._fitted:
            return 0.0
        context = prev_token if prev_token is not None else self.BOS
        counts = self._bigrams.get(context, {})
        count_ct = counts.get(token, 0)
        total = sum(counts.values())
        vocab_size = max(len(self._vocab), 1)
        prob = (count_ct + self.k) / (total + self.k * vocab_size)
        return math.log(max(prob, 1e-30))


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def beam_search(
    timestep_candidates: List[List[Tuple[str, float]]],  # [(token, log_prob), ...] per timestep
    lm: CharBigramLM,
    beam_size: int,
    lm_weight: float,
) -> str:
    """
    Combine encoder log-probs with bigram LM to find best token sequence.

    Returns the decoded string (key_token sequence assembled into text).
    """
    if not timestep_candidates:
        return ""

    # Beam state: list of (cumulative_score, token_sequence_list)
    beams: List[Tuple[float, List[str]]] = [(0.0, [])]

    for t, candidates in enumerate(timestep_candidates):
        if not candidates:
            continue

        new_beams: List[Tuple[float, List[str]]] = []

        for beam_score, beam_tokens in beams:
            prev_token = beam_tokens[-1] if beam_tokens else None

            for token, enc_log_prob in candidates:
                lm_log_prob = lm.log_prob(token, prev_token)
                score = (
                    beam_score
                    + (1.0 - lm_weight) * enc_log_prob
                    + lm_weight * lm_log_prob
                )
                new_beams.append((score, beam_tokens + [token]))

        # Prune
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

    if not beams:
        return ""

    best_tokens = beams[0][1]
    return _tokens_to_text(best_tokens)


# ---------------------------------------------------------------------------
# Session grouping
# ---------------------------------------------------------------------------

def _group_into_sessions(
    rows: List[Dict[str, Any]],
    split_filter: str,
) -> Dict[str, List[Dict[str, Any]]]:
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if split_filter != "all" and row.get("split") != split_filter:
            continue
        key = str(row["session_key"])
        sessions.setdefault(key, []).append(row)
    for key in sessions:
        sessions[key].sort(key=lambda r: float(r["key_press_ts"]))
    return sessions


def _get_candidates(row: Dict[str, Any], top_k: int) -> List[Tuple[str, float]]:
    """Extract (token, log_prob) list for one keystroke."""
    candidates = []
    for c in row.get("top_k", [])[:top_k]:
        token = c["token"]
        prob = max(float(c["prob"]), 1e-30)
        candidates.append((token, math.log(prob)))
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bigram-LM beam search decoder for Stage-1 keystroke predictions."
    )
    p.add_argument("--predictions", default="predictions.jsonl")
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
    )
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument(
        "--lm-weight",
        type=float,
        default=0.3,
        help="Weight for bigram LM (0=pure encoder, 1=pure LM)",
    )
    p.add_argument("--lm-smoothing", type=float, default=0.1, help="Add-k smoothing for bigram LM")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many encoder candidates to consider per timestep during beam search",
    )
    p.add_argument("--out", default="beam_decoded.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    print(f"Loading predictions from {pred_path}...")
    rows: List[Dict[str, Any]] = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} keystroke records")

    # ---- Fit bigram LM on training split only ----
    print("Fitting bigram LM on training split...")
    train_sessions = _group_into_sessions(rows, "train")
    train_sequences = [
        [r["key_token"] for r in sess_rows]
        for sess_rows in train_sessions.values()
    ]
    lm = CharBigramLM(k=args.lm_smoothing)
    lm.fit(train_sequences)
    print(f"  LM vocab size: {len(lm._vocab)}  |  training sessions: {len(train_sequences)}")

    # ---- Decode target split ----
    target_sessions = _group_into_sessions(rows, args.split)
    session_keys = sorted(target_sessions.keys())
    print(f"\nDecoding {len(session_keys)} session(s) [split='{args.split}', beam={args.beam_size}, lm_weight={args.lm_weight}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cer_enc_total = 0.0
    cer_beam_total = 0.0
    n_ok = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, sk in enumerate(session_keys):
            sess_rows = target_sessions[sk]
            split_name = sess_rows[0].get("split", "?")

            # Ground truth
            gt_tokens = [r["key_token"] for r in sess_rows]
            gt_text = _tokens_to_text(gt_tokens)

            # Encoder greedy
            enc_tokens = [r.get("pred_token", "<UNK>") for r in sess_rows]
            enc_text = _tokens_to_text(enc_tokens)

            # Beam search candidates
            timestep_cands = [_get_candidates(r, args.top_k) for r in sess_rows]
            beam_text = beam_search(timestep_cands, lm, args.beam_size, args.lm_weight)

            cer_enc = cer(enc_text, gt_text)
            cer_beam = cer(beam_text, gt_text)
            cer_enc_total += cer_enc
            cer_beam_total += cer_beam
            n_ok += 1

            record = {
                "session_key": sk,
                "split": split_name,
                "n_keystrokes": len(sess_rows),
                "start_ts": float(sess_rows[0]["key_press_ts"]),
                "end_ts": float(sess_rows[-1]["key_press_ts"]),
                "ground_truth_text": gt_text,
                "encoder_text": enc_text,
                "beam_text": beam_text,
                "cer_encoder": round(cer_enc, 4),
                "cer_beam": round(cer_beam, 4),
                "cer_improvement": round(cer_enc - cer_beam, 4),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1}/{len(session_keys)}] {sk}  "
                    f"enc CER={cer_enc:.3f}  beam CER={cer_beam:.3f}"
                )

    print(f"\n=== Summary ({n_ok} sessions) ===")
    if n_ok:
        print(f"  Avg encoder CER : {cer_enc_total / n_ok:.4f}")
        print(f"  Avg beam CER    : {cer_beam_total / n_ok:.4f}")
        print(f"  Avg improvement : {(cer_enc_total - cer_beam_total) / n_ok:.4f}")
    print(f"  Output          : {out_path.resolve()}")


if __name__ == "__main__":
    main()


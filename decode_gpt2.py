#!/usr/bin/env python3
"""
GPT-2 reranking decoder for Stage-1 keystroke predictions.

Two-pass decoding:

  Pass 1 — encoder N-best beam search
    Generates the top-N candidate character sequences using ONLY the IMU
    encoder's per-keystroke posterior probabilities. No language model is used
    here, so the candidates are diverse and reflect what the sensors saw.

  Pass 2 — GPT-2 reranking
    Each candidate key-token sequence is assembled into a text string and scored
    with GPT-2 log-probability. The final answer is the candidate with the best
    combined score (length-normalised so short and long sequences are comparable):

        combined = (1 - lm_weight) * enc_avg_log_prob
                 +       lm_weight * gpt2_avg_log_prob

    GPT-2 log-prob is higher for grammatically natural, plausible English.

Optional: --online-word-scoring
    Instead of N-best + reranking at the end, score the GPT-2 contribution
    online at every word boundary (space character). The beam accumulates a
    GPT-2 running score as words complete. This is more expensive but helps
    when N-best lists collapse to similar candidates on long sessions.

Outputs a JSONL with one line per session:
  session_key, split, n_keystrokes, ground_truth_text, encoder_text,
  gpt2_text, cer_encoder, cer_gpt2, cer_improvement

Usage
-----
pip install transformers torch

# Decode test split (N-best reranking, default)
python decode_gpt2.py \\
    --predictions predictions.jsonl \\
    --split test \\
    --beam-size 10 \\
    --lm-weight 0.5 \\
    --out gpt2_decoded.jsonl

# Online word-boundary scoring (better for long sessions)
python decode_gpt2.py \\
    --predictions predictions.jsonl \\
    --split test \\
    --online-word-scoring \\
    --beam-size 5 \\
    --lm-weight 0.4 \\
    --out gpt2_decoded.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

def _tokens_to_text(tokens: List[str]) -> str:
    """Convert raw key_token list → clean rendered text."""
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
# CER
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
# GPT-2 scoring
# ---------------------------------------------------------------------------

def gpt2_total_log_prob(
    text: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 1024,
) -> Tuple[float, int]:
    """
    Returns (total_log_prob, n_tokens) for `text` under GPT-2.

    Uses teacher-forced next-token prediction (same as perplexity).
    Higher total_log_prob = more natural text.
    Empty or un-tokenisable text returns (-inf, 0).
    """
    if not text or not text.strip():
        return float("-inf"), 0

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    n_tokens = int(input_ids.shape[1])
    if n_tokens == 0:
        return float("-inf"), 0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    # outputs.loss = mean negative log-likelihood per token
    total_log_prob = -float(outputs.loss.item()) * n_tokens
    return total_log_prob, n_tokens


def gpt2_prefix_word_log_prob(
    word: str,
    prefix_ids: torch.Tensor,   # (1, L) already-scored prefix
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """
    Score a single word given a GPT-2 prefix. Returns (log_prob_sum, new_prefix_ids).

    Runs one forward pass over [prefix | word_ids] and sums the log-probs of
    the word tokens (not the prefix tokens).
    """
    # Tokenise word with a leading space so GPT-2 sees it as a mid-sentence word
    word_text = " " + word.strip() if prefix_ids.shape[1] > 0 else word.strip()
    word_ids = tokenizer.encode(word_text, add_special_tokens=False)
    if not word_ids:
        return 0.0, prefix_ids

    word_tensor = torch.tensor([word_ids], device=device)
    full_ids = torch.cat([prefix_ids, word_tensor], dim=1)

    if full_ids.shape[1] > 1024:
        # Keep only the last 1024 tokens to stay within GPT-2 context
        full_ids = full_ids[:, -1024:]

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, L, V)

    # The log-prob of word_ids[i] is at position (prefix_len + i - 1) in logits
    prefix_len = prefix_ids.shape[1]
    log_probs = torch.log_softmax(logits[0], dim=-1)

    total = 0.0
    for i, wid in enumerate(word_ids):
        pos = prefix_len + i - 1  # next-token prediction offset
        if 0 <= pos < log_probs.shape[0]:
            total += float(log_probs[pos, wid].item())

    new_prefix = torch.cat([prefix_ids, word_tensor], dim=1)
    if new_prefix.shape[1] > 1024:
        new_prefix = new_prefix[:, -1024:]

    return total, new_prefix


# ---------------------------------------------------------------------------
# Pass 1: encoder-only N-best beam search
# ---------------------------------------------------------------------------

def encoder_beam_search(
    timestep_candidates: List[List[Tuple[str, float]]],  # [(token, log_prob), ...] per step
    beam_size: int,
) -> List[Tuple[float, List[str]]]:
    """
    Pure encoder beam search — no LM.
    Returns list of (cumulative_enc_log_prob, token_list), best first.
    """
    # Beam state: (cumulative_score, token_list)
    beams: List[Tuple[float, List[str]]] = [(0.0, [])]

    for candidates in timestep_candidates:
        if not candidates:
            continue
        new_beams: List[Tuple[float, List[str]]] = []
        for score, tokens in beams:
            for token, enc_lp in candidates:
                new_beams.append((score + enc_lp, tokens + [token]))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

    return beams


# ---------------------------------------------------------------------------
# Pass 2a: N-best GPT-2 reranking (default)
# ---------------------------------------------------------------------------

def rerank_nbest(
    nbest: List[Tuple[float, List[str]]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    lm_weight: float,
    max_gpt2_length: int,
) -> str:
    """
    Score each candidate with GPT-2 and return the text of the best combined candidate.
    """
    best_text = ""
    best_score = float("-inf")

    for enc_score, tokens in nbest:
        text = _tokens_to_text(tokens)
        if not text.strip():
            continue

        n_chars = max(len(tokens), 1)
        enc_avg = enc_score / n_chars  # avg log prob per char

        gpt2_total, n_gpt2_toks = gpt2_total_log_prob(
            text, model, tokenizer, device, max_length=max_gpt2_length
        )
        if n_gpt2_toks == 0:
            continue
        gpt2_avg = gpt2_total / n_gpt2_toks  # avg log prob per GPT-2 token

        combined = (1.0 - lm_weight) * enc_avg + lm_weight * gpt2_avg

        if combined > best_score:
            best_score = combined
            best_text = text

    return best_text


# ---------------------------------------------------------------------------
# Pass 2b: Online word-boundary beam search with GPT-2
# ---------------------------------------------------------------------------

def online_word_boundary_beam_search(
    timestep_candidates: List[List[Tuple[str, float]]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    beam_size: int,
    lm_weight: float,
) -> str:
    """
    Beam search where GPT-2 scores are added at every space (word boundary).

    Beam state: (cum_score, token_list, current_word_chars, prefix_ids)
      - cum_score     : combined enc + LM score so far
      - token_list    : decoded key_tokens
      - current_word  : chars accumulated since last space (not yet GPT-2 scored)
      - prefix_ids    : GPT-2 token IDs for already-completed words
    """
    # Initial state
    empty_prefix = torch.zeros(1, 0, dtype=torch.long, device=device)
    # (cum_score, token_list, current_word, prefix_ids)
    BeamState = Tuple[float, List[str], str, torch.Tensor]
    beams: List[BeamState] = [(0.0, [], "", empty_prefix)]

    for candidates in timestep_candidates:
        if not candidates:
            continue

        new_beams: List[BeamState] = []

        for cum_score, tokens, current_word, prefix_ids in beams:
            for token, enc_lp in candidates:
                enc_contribution = (1.0 - lm_weight) * enc_lp
                new_tokens = tokens + [token]

                # Render what this token adds to the current word buffer
                rendered = _tokens_to_text([token])  # single-token rendering

                if rendered == " " or token in ("\n", "\t"):
                    # Word boundary: score current_word with GPT-2
                    if current_word.strip():
                        lm_lp, new_prefix = gpt2_prefix_word_log_prob(
                            current_word, prefix_ids, model, tokenizer, device
                        )
                        lm_contribution = lm_weight * lm_lp
                    else:
                        lm_contribution = 0.0
                        new_prefix = prefix_ids

                    new_beams.append((
                        cum_score + enc_contribution + lm_contribution,
                        new_tokens,
                        "",  # reset word buffer
                        new_prefix,
                    ))

                elif rendered:  # regular character
                    new_beams.append((
                        cum_score + enc_contribution,
                        new_tokens,
                        current_word + rendered,
                        prefix_ids,
                    ))

                else:
                    # Special token (backspace / modifier): apply to word buffer
                    new_word = current_word[:-1] if token == "<BACKSPACE>" and current_word else current_word
                    new_beams.append((
                        cum_score + enc_contribution,
                        new_tokens,
                        new_word,
                        prefix_ids,
                    ))

        # Prune
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

    # Score any trailing word in each beam
    final_beams: List[Tuple[float, List[str]]] = []
    for cum_score, tokens, current_word, prefix_ids in beams:
        if current_word.strip():
            lm_lp, _ = gpt2_prefix_word_log_prob(
                current_word, prefix_ids, model, tokenizer, device
            )
            final_score = cum_score + lm_weight * lm_lp
        else:
            final_score = cum_score
        final_beams.append((final_score, tokens))

    if not final_beams:
        return ""
    final_beams.sort(key=lambda x: x[0], reverse=True)
    return _tokens_to_text(final_beams[0][1])


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


def _get_candidates(
    row: Dict[str, Any], top_k: int
) -> List[Tuple[str, float]]:
    out = []
    for c in row.get("top_k", [])[:top_k]:
        prob = max(float(c["prob"]), 1e-30)
        out.append((c["token"], math.log(prob)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT-2 reranking decoder for Stage-1 keystroke predictions."
    )
    p.add_argument("--predictions", default="predictions.jsonl")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="N-best size for encoder beam search (default 10). "
             "In --online-word-scoring mode this is the active beam width.",
    )
    p.add_argument(
        "--lm-weight",
        type=float,
        default=0.5,
        help="Weight for GPT-2 score vs encoder score (0=pure encoder, 1=pure GPT-2)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Encoder top-k candidates to consider per keystroke",
    )
    p.add_argument(
        "--online-word-scoring",
        action="store_true",
        help="Score GPT-2 at every word boundary during beam search "
             "(slower but better for long sessions)",
    )
    p.add_argument(
        "--gpt2-model",
        default="gpt2",
        help="HuggingFace GPT-2 model name (gpt2 / gpt2-medium / gpt2-large / gpt2-xl)",
    )
    p.add_argument("--device", default=None, help="torch device (auto-detected if omitted)")
    p.add_argument("--batch-size", type=int, default=1, help="GPT-2 inference batch (currently 1)")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--max-gpt2-length", type=int, default=512, help="Max GPT-2 context length")
    p.add_argument("--out", default="gpt2_decoded.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load predictions ----
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

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Load GPT-2 ----
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"Loading GPT-2 model: {args.gpt2_model}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model)
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_model).to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Group sessions ----
    sessions = _group_into_sessions(rows, args.split)
    session_keys = sorted(sessions.keys())
    if args.max_sessions:
        session_keys = session_keys[: args.max_sessions]

    mode = "online word-boundary" if args.online_word_scoring else "N-best reranking"
    print(
        f"\nDecoding {len(session_keys)} session(s) "
        f"[split='{args.split}', mode={mode}, beam={args.beam_size}, lm_weight={args.lm_weight}]\n"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cer_enc_total = 0.0
    cer_gpt2_total = 0.0
    n_ok = 0
    t_total = 0.0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, sk in enumerate(session_keys):
            sess_rows = sessions[sk]
            split_name = sess_rows[0].get("split", "?")

            # Ground truth
            gt_tokens = [r["key_token"] for r in sess_rows]
            gt_text = _tokens_to_text(gt_tokens)

            # Encoder greedy (top-1)
            enc_tokens = [r.get("pred_token", "<UNK>") for r in sess_rows]
            enc_text = _tokens_to_text(enc_tokens)

            # Per-timestep encoder candidates
            timestep_cands = [_get_candidates(r, args.top_k) for r in sess_rows]

            t0 = time.perf_counter()

            if args.online_word_scoring:
                # Online word-boundary GPT-2 scoring
                gpt2_text = online_word_boundary_beam_search(
                    timestep_cands,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    beam_size=args.beam_size,
                    lm_weight=args.lm_weight,
                )
            else:
                # N-best encoder beam search, then GPT-2 reranking
                nbest = encoder_beam_search(timestep_cands, beam_size=args.beam_size)
                gpt2_text = rerank_nbest(
                    nbest,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    lm_weight=args.lm_weight,
                    max_gpt2_length=args.max_gpt2_length,
                )

            elapsed = time.perf_counter() - t0
            t_total += elapsed

            cer_enc = cer(enc_text, gt_text)
            cer_gpt2 = cer(gpt2_text, gt_text)
            cer_enc_total += cer_enc
            cer_gpt2_total += cer_gpt2
            n_ok += 1

            record = {
                "session_key": sk,
                "split": split_name,
                "n_keystrokes": len(sess_rows),
                "start_ts": float(sess_rows[0]["key_press_ts"]),
                "end_ts": float(sess_rows[-1]["key_press_ts"]),
                "ground_truth_text": gt_text,
                "encoder_text": enc_text,
                "gpt2_text": gpt2_text,
                "cer_encoder": round(cer_enc, 4),
                "cer_gpt2": round(cer_gpt2, 4),
                "cer_improvement": round(cer_enc - cer_gpt2, 4),
                "decode_time_s": round(elapsed, 2),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"  [{i+1}/{len(session_keys)}] {sk} ({len(sess_rows)} keys, {elapsed:.1f}s) | "
                f"enc CER={cer_enc:.3f}  gpt2 CER={cer_gpt2:.3f}"
            )

    print(f"\n=== Summary ({n_ok} sessions, total {t_total:.1f}s) ===")
    if n_ok:
        print(f"  Avg encoder CER : {cer_enc_total / n_ok:.4f}")
        print(f"  Avg GPT-2 CER   : {cer_gpt2_total / n_ok:.4f}")
        print(f"  Avg improvement : {(cer_enc_total - cer_gpt2_total) / n_ok:.4f}")
        print(f"  Avg time/session: {t_total / n_ok:.2f}s")
    print(f"  Output          : {out_path.resolve()}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Zero-shot LLM decoder for Stage-1 keystroke predictions using Groq API.

Reads the JSONL produced by export_predictions.py, groups keystrokes into
contiguous per-session sequences, and asks an LLM to reconstruct the most
likely text given the noisy per-character top-k predictions from the IMU
encoder.

Long sessions are split into chunks so each API call stays within
Groq's free-tier token limits.

Outputs a JSONL where each line is one session with:
  - session_key, split, start_ts, end_ts, n_keystrokes
  - ground_truth_text, encoder_text, llm_text
  - cer_encoder, cer_llm, cer_improvement

Usage
-----
export GROQ_API_KEY=gsk_...

python decode_llm.py \\
    --predictions predictions.jsonl \\
    --split test \\
    --out llm_decoded.jsonl

# Limit sessions for a quick test
python decode_llm.py \\
    --predictions predictions.jsonl \\
    --split test \\
    --max-sessions 4 \\
    --out llm_decoded.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _levenshtein(s: str, t: str) -> int:
    if not s:
        return len(t)
    if not t:
        return len(s)
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s[i - 1] == t[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, gt) / len(gt)


def _tokens_to_text(tokens: List[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        if tok in (" ", "\n", "\t"):
            out.append(tok)
        elif tok == "<BACKSPACE>":
            if out:
                out.pop()
        elif tok.startswith("<") and tok.endswith(">"):
            pass
        else:
            out.append(tok)
    return "".join(out)


def _format_sequence_for_llm(rows: List[Dict[str, Any]], top_k: int) -> str:
    parts = []
    for row in rows:
        candidates = row.get("top_k", [])[:top_k]
        display = [c for c in candidates
                   if not (c["token"].startswith("<") and c["token"].endswith(">"))]
        if not display:
            display = candidates
        if not display:
            parts.append("[?]")
            continue
        inner = " ".join(f"{c['token']}:{round(c['prob'] * 100):.0f}%" for c in display)
        parts.append(f"[{inner}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Session grouping
# ---------------------------------------------------------------------------

def _group_into_sessions(
    rows: List[Dict[str, Any]],
    split_filter: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if split_filter and split_filter != "all" and row.get("split") != split_filter:
            continue
        key = str(row["session_key"])
        sessions.setdefault(key, []).append(row)
    for key in sessions:
        sessions[key].sort(key=lambda r: float(r["key_press_ts"]))
    return sessions


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a keystroke recovery decoder. A smart-ring IMU sensor predicts which key was pressed \
for each keystroke; the predictions are noisy. Each bracket shows the top candidate characters \
with their confidence percentages for one keystroke.

Your task: reconstruct the most likely text the person was typing.
Rules:
- Use confidence percentages and natural language context to pick the right character.
- Return ONLY the reconstructed text — no explanation, no quotes, no extra formatting.
- If given a chunk mid-sentence, reconstruct just that portion naturally.
- Prefer common English words and phrases when ambiguous.\
"""


def call_llm(
    sequence_str: str,
    client: Any,
    model: str,
    max_tokens: int,
) -> str:
    user_msg = (
        "Keystroke predictions (one bracket per key, characters with confidence %):\n\n"
        f"{sequence_str}\n\n"
        "Reconstruct the text:"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def decode_session(
    sess_rows: List[Dict[str, Any]],
    client: Any,
    model: str,
    top_k: int,
    max_tokens: int,
    chunk_size: int,
    delay_s: float,
) -> str:
    """Decode one session, chunking if it exceeds chunk_size keystrokes."""
    chunks = [sess_rows[i:i + chunk_size] for i in range(0, len(sess_rows), chunk_size)]
    parts = []
    for j, chunk in enumerate(chunks):
        seq_str = _format_sequence_for_llm(chunk, top_k)
        text = call_llm(seq_str, client, model=model, max_tokens=max_tokens)
        parts.append(text)
        if delay_s > 0 and j < len(chunks) - 1:
            time.sleep(delay_s)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Groq LLM decoding of Stage-1 keystroke predictions.")
    p.add_argument("--predictions", default="predictions.jsonl")
    p.add_argument("--out", default="llm_decoded.jsonl")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--top-k", type=int, default=3, help="Candidates shown per keystroke in prompt")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--api-key", default=None, help="Groq API key (or set GROQ_API_KEY)")
    p.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model (default: llama-3.3-70b-versatile)",
    )
    p.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per LLM response")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Max keystrokes per API call (splits long sessions; default 300)",
    )
    p.add_argument(
        "--delay-s",
        type=float,
        default=1.0,
        help="Seconds between API calls (rate-limit safety; default 1.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable or use --api-key")

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install the Groq SDK: pip install groq")

    client = Groq(api_key=api_key)

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Not found: {pred_path}")

    print(f"Loading predictions from {pred_path}...")
    rows: List[Dict[str, Any]] = [
        json.loads(l) for l in pred_path.open(encoding="utf-8") if l.strip()
    ]
    print(f"Loaded {len(rows)} keystroke records")

    sessions = _group_into_sessions(rows, args.split)
    session_keys = sorted(sessions.keys())
    if args.max_sessions:
        session_keys = session_keys[: args.max_sessions]

    n_chunks_total = sum(
        max(1, -(-len(sessions[sk]) // args.chunk_size)) for sk in session_keys
    )
    print(
        f"Decoding {len(session_keys)} session(s) | "
        f"model={args.model} | chunk_size={args.chunk_size} | "
        f"~{n_chunks_total} API call(s)\n"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cer_enc_total = 0.0
    cer_llm_total = 0.0
    n_ok = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, sk in enumerate(session_keys):
            sess_rows = sessions[sk]
            split_name = sess_rows[0].get("split", "?")

            gt_text = _tokens_to_text([r["key_token"] for r in sess_rows])
            enc_text = _tokens_to_text([r.get("pred_token", "<UNK>") for r in sess_rows])

            try:
                llm_text = decode_session(
                    sess_rows,
                    client,
                    model=args.model,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    chunk_size=args.chunk_size,
                    delay_s=args.delay_s,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(session_keys)}] {sk}: API error — {e}")
                llm_text = enc_text

            cer_enc = cer(enc_text, gt_text)
            cer_llm = cer(llm_text, gt_text)
            cer_enc_total += cer_enc
            cer_llm_total += cer_llm
            n_ok += 1

            record = {
                "session_key": sk,
                "split": split_name,
                "n_keystrokes": len(sess_rows),
                "start_ts": float(sess_rows[0]["key_press_ts"]),
                "end_ts": float(sess_rows[-1]["key_press_ts"]),
                "ground_truth_text": gt_text,
                "encoder_text": enc_text,
                "llm_text": llm_text,
                "cer_encoder": round(cer_enc, 4),
                "cer_llm": round(cer_llm, 4),
                "cer_improvement": round(cer_enc - cer_llm, 4),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"  [{i+1}/{len(session_keys)}] {sk} ({len(sess_rows)} keys) | "
                f"enc CER={cer_enc:.3f}  llm CER={cer_llm:.3f}  "
                f"improvement={cer_enc - cer_llm:+.3f}"
            )

            if args.delay_s > 0 and i < len(session_keys) - 1:
                time.sleep(args.delay_s)

    print(f"\n=== Summary ({n_ok} sessions) ===")
    if n_ok:
        print(f"  Avg encoder CER : {cer_enc_total / n_ok:.4f}")
        print(f"  Avg LLM CER     : {cer_llm_total / n_ok:.4f}")
        print(f"  Avg improvement : {(cer_enc_total - cer_llm_total) / n_ok:.4f}")
    print(f"  Output          : {out_path.resolve()}")


if __name__ == "__main__":
    main()

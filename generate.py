"""
Inference script for generating text from IMU data.

Supports three modes:
    1. Embeddings file:  pre-computed Chronos embeddings (.pt)
    2. Raw data dir:     load CSV/PKL directly, run Chronos on-the-fly
    3. Demo:             random embeddings for integration testing

Usage:
    # From pre-computed embeddings
    python generate.py --adapter_path ./checkpoints/adapter_final.pt \
                       --input_file ./embeddings/test.pt

    # Directly from raw data directory
    python generate.py --adapter_path ./checkpoints/adapter_final.pt \
                       --raw_dir ./data

    # Demo (random embeddings, no data needed)
    python generate.py --adapter_path ./checkpoints/adapter_final.pt \
                       --demo
"""

import argparse
import torch
from model import RingToText
from utils.constants import CHRONOS_DEFAULT_MODEL


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RingToText inference")
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--llm", type=str, default=RingToText.__init__.__defaults__[0])
    p.add_argument("--d_chronos", type=int, default=9216,
                   help="Chronos output dim: 768 * n_channels (9216 for both rings, 4608 for one)")
    p.add_argument("--n_soft_tokens", type=int, default=32)
    p.add_argument("--n_resampler_layers", type=int, default=2)

    # LoRA (must match the settings used at training time)
    p.add_argument("--lora_rank", type=int, default=0,
                   help="LoRA rank used at training time; 0 if no LoRA (default: 0)")
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    p.add_argument("--adapter_dim", type=int, default=256,
                   help="Must match the value used at training time (default: 256)")

    # Input sources (mutually exclusive)
    p.add_argument("--input_file", type=str, default=None,
                   help="Pre-computed Chronos embeddings file (.pt)")
    p.add_argument("--raw_dir", type=str, default=None,
                   help="Raw data directory with CSV/PKL files; runs Chronos on-the-fly")
    p.add_argument("--demo", action="store_true",
                   help="Run with random embeddings for testing")

    # Raw-dir options (only used with --raw_dir)
    p.add_argument("--chronos_model", type=str, default=CHRONOS_DEFAULT_MODEL)
    p.add_argument("--ring", type=str, default="both", choices=["L", "R", "both"])
    p.add_argument("--window_size", type=float, default=10.0)
    p.add_argument("--step_size", type=float, default=5.0)
    p.add_argument("--min_text_len", type=int, default=5)
    p.add_argument("--target_hz", type=int, default=100)

    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit number of samples to generate (default: all)")

    return p.parse_args()


def _load_samples_from_raw(args) -> list[dict]:
    """Run preprocess pipeline on raw data and return encoded samples."""
    from preprocess import _load_raw_samples
    from utils.chronos_encode import encode_with_chronos
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session_map = _load_raw_samples(
        raw_dir=args.raw_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        min_text_len=args.min_text_len,
        ring=args.ring,
        target_hz=args.target_hz,
    )
    raw = [s for v in session_map.values() for s in v]
    return encode_with_chronos(raw, args.chronos_model, batch_size=32, device=device)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model ({args.llm})...")
    model = RingToText(
        llm_name=args.llm,
        d_chronos=args.d_chronos,
        n_soft_tokens=args.n_soft_tokens,
        n_resampler_layers=args.n_resampler_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        adapter_dim=args.adapter_dim,
    ).to(device)

    model.load_adapter(args.adapter_path, map_location=device, weights_only=True)
    model.eval()
    print(f"Adapter loaded from {args.adapter_path}")

    # ── demo mode ────────────────────────────────────────────────────────────
    if args.demo:
        print("\n--- Demo mode: random embeddings ---")
        fake_embeds = torch.randn(1, 50, args.d_chronos, device=device)
        results = model.generate(
            fake_embeds,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Generated: {results[0]!r}")
        return

    # ── load samples ─────────────────────────────────────────────────────────
    if args.raw_dir:
        print(f"\nLoading raw data from {args.raw_dir} (running Chronos)...")
        samples = _load_samples_from_raw(args)
    elif args.input_file:
        print(f"\nLoading embeddings from {args.input_file}...")
        samples = torch.load(args.input_file, weights_only=False)
    else:
        print("Provide --input_file, --raw_dir, or --demo.")
        return

    if args.n_samples is not None:
        samples = samples[:args.n_samples]

    # ── generate ─────────────────────────────────────────────────────────────
    for i, sample in enumerate(samples):
        embeds = sample["embeddings"].unsqueeze(0).to(device)
        gt = sample.get("text", "<no ground truth>")

        results = model.generate(
            embeds,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(f"\n[{i}]")
        print(f"  GT:   {gt!r}")
        print(f"  PRED: {results[0]!r}")


if __name__ == "__main__":
    main()

"""
Visualise Phase A per-frame keystroke predictions.

Loads a Phase A checkpoint (or any checkpoint with a keystroke_head) and plots
sigmoid(activity_logits) and sigmoid(onset_logits) against the ground-truth
keystroke_active / keystroke_onset masks for a handful of samples.

Phase A produces no text — these per-frame logits are the entire model output.

Usage:
    python visualize_phase_a.py \\
        --adapter_path ./checkpoints/phase_a/adapter_best.pt \\
        --input_file ./embeddings/val.pt \\
        --output_dir ./phase_a_viz \\
        --n_samples 6
"""

import argparse
from pathlib import Path

import torch

from model import RingToText


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise Phase A keystroke predictions")
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--input_file", type=str, required=True,
                   help="Embeddings .pt (e.g. ./embeddings/val.pt)")
    p.add_argument("--output_dir", type=str, default="./phase_a_viz")
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B",
                   help="LLM name used at Phase A training time. LLM is not loaded "
                        "(skip_llm=True) but its hidden size determines adapter.out_proj "
                        "shape — must match training to load the checkpoint.")
    p.add_argument("--d_chronos", type=int, default=None,
                   help="Adapter input dim. If omitted, inferred from adapter_path / "
                        "input_file name: 'moment' in the path → 12288 (MOMENT-1-large, "
                        "both rings), otherwise 9216 (Chronos, both rings).")
    p.add_argument("--n_soft_tokens", type=int, default=32)
    p.add_argument("--n_resampler_layers", type=int, default=2)
    p.add_argument("--adapter_dim", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for binarised overlay lines (default: 0.5)")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.d_chronos is None:
        is_moment = "moment" in args.adapter_path.lower() or "moment" in args.input_file.lower()
        args.d_chronos = 12288 if is_moment else 9216
        print(f"Inferred d_chronos={args.d_chronos} ({'MOMENT' if is_moment else 'Chronos'})")

    # Build a Phase-A model (skip_llm=True → no tokenizer/LLM loaded).
    # llm_name must match the training run so adapter.out_proj has the right
    # output dim (d_llm from model.py's _KNOWN_HIDDEN table).
    model = RingToText(
        llm_name=args.llm,
        d_chronos=args.d_chronos,
        n_soft_tokens=args.n_soft_tokens,
        n_resampler_layers=args.n_resampler_layers,
        adapter_dim=args.adapter_dim,
        skip_llm=True,
    ).to(device)
    model.load_adapter(args.adapter_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Loaded {args.adapter_path}")

    samples = torch.load(args.input_file, weights_only=False)
    samples = samples[: args.n_samples]
    print(f"Visualising {len(samples)} samples from {args.input_file}")

    # Defer matplotlib import so the script doesn't crash when matplotlib
    # isn't installed and the user only needs the numeric metrics.
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            emb = torch.as_tensor(sample["embeddings"]).float().unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(emb)
            act_prob = torch.sigmoid(out["keystroke_logits"][0].float()).cpu().numpy()
            on_prob = torch.sigmoid(out["onset_logits"][0].float()).cpu().numpy()

            gt_act = torch.as_tensor(sample.get("keystroke_active", torch.zeros(len(act_prob)))).float().numpy()
            gt_on = torch.as_tensor(sample.get("keystroke_onset", torch.zeros(len(act_prob)))).float().numpy()
            text = sample.get("text", "")

            S = len(act_prob)
            x = range(S)

            fig, (ax_a, ax_o) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

            ax_a.fill_between(x, 0, gt_act, color="tab:blue", alpha=0.25, step="mid", label="GT activity")
            ax_a.plot(x, act_prob, color="tab:blue", lw=1.3, label="pred activity prob")
            ax_a.axhline(args.threshold, color="gray", ls="--", lw=0.6)
            ax_a.set_ylim(-0.05, 1.05)
            ax_a.set_ylabel("activity")
            ax_a.legend(loc="upper right", fontsize=8)

            # Onsets are sparse impulses — use stem for GT, line for prediction.
            gt_on_idx = [t for t, v in enumerate(gt_on) if v > 0.5]
            if gt_on_idx:
                ax_o.vlines(gt_on_idx, 0, 1, color="tab:orange", alpha=0.5, lw=1.2, label="GT onset")
            ax_o.plot(x, on_prob, color="tab:red", lw=1.3, label="pred onset prob")
            ax_o.axhline(args.threshold, color="gray", ls="--", lw=0.6)
            ax_o.set_ylim(-0.05, 1.05)
            ax_o.set_xlabel("encoder frame")
            ax_o.set_ylabel("onset")
            ax_o.legend(loc="upper right", fontsize=8)

            title = text if len(text) <= 80 else text[:77] + "..."
            fig.suptitle(f"[{i}] {title!r}  (S={S})", fontsize=10)
            fig.tight_layout()

            out_path = out_dir / f"sample_{i:02d}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  [{i}] saved {out_path}  "
                  f"| pred>{args.threshold}: act={int((act_prob >= args.threshold).sum())} "
                  f"on={int((on_prob >= args.threshold).sum())} "
                  f"| GT: act={int(gt_act.sum())} on={int(gt_on.sum())}")

    print(f"\nDone. {len(samples)} plots in {out_dir}")


if __name__ == "__main__":
    main()

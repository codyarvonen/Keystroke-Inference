"""
Offline evaluation on a precomputed embeddings split (e.g. test.pt).

Computes validation loss, teacher-forced top-K token accuracy, and optional
sample generations + CER (same logic as training validation).

Usage:
    python evaluate.py --config configs/default.yaml --adapter_path ./checkpoints/adapter_best.pt
    python evaluate.py --adapter_path ./checkpoints/adapter_best.pt \\
        --data_file ./embeddings/test.pt --llm Qwen/Qwen2.5-1.5B --device cuda:1
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

import yaml

import torch
from torch.utils.data import DataLoader

from dataset import IMUTextDataset, collate_fn
from model import RingToText
from train import cer, validate


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RingToText adapter on embeddings")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config (defaults for any flag not set on CLI)")
    p.add_argument("--adapter_path", type=str, required=True,
                   help="Checkpoint from training (adapter_best.pt / adapter_final.pt)")

    p.add_argument("--data_file", type=str, default="./embeddings/test.pt",
                   help="Precomputed .pt list of {embeddings, text}")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda",
                   help="Device string, e.g. cuda, cuda:1, cpu")

    p.add_argument("--llm", type=str, default=RingToText.__init__.__defaults__[0])
    p.add_argument("--d_chronos", type=int, default=9216)
    p.add_argument("--n_soft_tokens", type=int, default=32)
    p.add_argument("--n_resampler_layers", type=int, default=2)
    p.add_argument("--prompt", type=str, default=None)

    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    p.add_argument("--adapter_dim", type=int, default=256)
    p.add_argument("--adapter_dropout", type=float, default=0.1)

    p.add_argument("--eval_topk", type=int, nargs="+", default=[1, 5, 10],
                   help="K values for teacher-forced top-K accuracy (empty list disables)")
    p.add_argument("--n_generate_samples", type=int, default=3,
                   help="Number of val samples to decode for CER (0 = skip generation)")
    p.add_argument("--control", type=str, default="none", choices=["none", "shuffle", "zero"],
                   help="Optional control baseline for IMU conditioning: "
                        "'shuffle' permutes IMU across samples in each batch; "
                        "'zero' zeros IMU embeddings.")
    p.add_argument("--gen_prefix_tokens", type=int, default=0,
                   help="For an additional qualitative check, feed this many ground-truth "
                        "tokens before free generation (0 disables this mode).")

    args = p.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cli_argv = set()
        for token in sys.argv[1:]:
            if token.startswith("--"):
                cli_argv.add(token.lstrip("-").split("=")[0])
        for key, value in cfg.items():
            if key not in cli_argv:
                setattr(args, key, value)

    # Training configs set data_file to train.pt; for this script use eval split unless
    # --data_file was passed explicitly.
    if "--data_file" not in cli_argv:
        args.data_file = getattr(args, "eval_data_file", None) or "./embeddings/test.pt"

    return args


def _build_control_loader(base_loader: DataLoader, mode: str) -> DataLoader:
    """
    Return a DataLoader that applies a control transform to chronos embeddings.
    """
    if mode == "none":
        return base_loader

    def control_collate(batch):
        out = collate_fn(batch)
        if mode == "zero":
            out["chronos_embeds"] = torch.zeros_like(out["chronos_embeds"])
            return out
        if mode == "shuffle":
            bsz = out["chronos_embeds"].size(0)
            perm = torch.randperm(bsz)
            out["chronos_embeds"] = out["chronos_embeds"][perm]
            out["chronos_mask"] = out["chronos_mask"][perm]
            return out
        raise ValueError(f"Unsupported control mode: {mode}")

    return DataLoader(
        base_loader.dataset,
        batch_size=base_loader.batch_size,
        shuffle=False,
        collate_fn=control_collate,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
    )


def _fmt_metric(v: float) -> str:
    return f"{v:.6f}" if not math.isnan(v) else "nan"


def _print_result_block(title: str, result: dict, topks: list[int], n_generate: int):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"loss:         {_fmt_metric(result['val_loss'])}")
    for k in topks:
        key = f"top{k}_acc"
        if key in result:
            print(f"{key:<12}{_fmt_metric(result[key])}")
    mean_cer = result["mean_cer"]
    if not math.isnan(mean_cer):
        print(f"mean_CER:     {_fmt_metric(mean_cer)} (over {n_generate} generated samples)")
    else:
        print("mean_CER:     skipped")


def _autocast_ctx(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def generate_with_target_prefix(
    model: RingToText,
    chronos_embeds: torch.Tensor,
    chronos_mask: torch.Tensor | None,
    target_ids: torch.Tensor,
    n_prefix_tokens: int,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """
    Generate text after priming with the first n ground-truth tokens.
    """
    n_prefix_tokens = max(0, min(n_prefix_tokens, target_ids.size(1)))
    prefix_ids = target_ids[:, :n_prefix_tokens] if n_prefix_tokens > 0 else None

    # Match adapter dtype (bf16 on CUDA in this project) to avoid dtype mismatch.
    adapter_dtype = next(model.adapter.parameters()).dtype
    chronos_embeds = chronos_embeds.to(dtype=adapter_dtype)

    with _autocast_ctx(chronos_embeds.device):
        soft_tokens, _, _ = model.adapter(chronos_embeds, chronos_mask)
        combined, _ = model._build_inputs(soft_tokens, prefix_ids)

    bsz = combined.size(0)
    generated_ids: list[list[int]] = [[] for _ in range(bsz)]
    finished = [False] * bsz

    with _autocast_ctx(chronos_embeds.device):
        out = model.llm(inputs_embeds=combined, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            mask = cumulative - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(1, torch.multinomial(sorted_probs, 1)).squeeze(-1)
        else:
            next_token = next_logits.argmax(dim=-1)

        for i in range(bsz):
            if not finished[i]:
                tid = next_token[i].item()
                generated_ids[i].append(tid)
                if tid == model.tokenizer.eos_token_id:
                    finished[i] = True

        if all(finished):
            break

        next_embeds = model._embed_layer(next_token.unsqueeze(1))
        with _autocast_ctx(chronos_embeds.device):
            out = model.llm(inputs_embeds=next_embeds, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

    return model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def main():
    args = get_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data_path = Path(args.data_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"data_file not found: {data_path.resolve()}")

    print(f"Loading LLM + adapter ({args.llm})...")
    model = RingToText(
        llm_name=args.llm,
        d_chronos=args.d_chronos,
        n_soft_tokens=args.n_soft_tokens,
        n_resampler_layers=args.n_resampler_layers,
        prompt=args.prompt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
    ).to(device)

    model.load_adapter(args.adapter_path, map_location=device, weights_only=True)
    print(f"Adapter loaded from {args.adapter_path}")

    ds = IMUTextDataset(
        data_file=args.data_file,
        tokenizer=model.tokenizer,
        max_text_len=args.max_text_len,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    print(f"Samples: {len(ds)}  |  batches: {len(loader)}")

    result_main = validate(
        model,
        loader,
        device,
        n_generate=args.n_generate_samples,
        logger=None,
        eval_topk=args.eval_topk,
    )

    print("\n" + "=" * 60)
    _print_result_block("MAIN (real IMU)", result_main, args.eval_topk, args.n_generate_samples)

    if args.control != "none":
        control_loader = _build_control_loader(loader, args.control)
        # For control comparisons, skip generation/CER to focus on teacher-forced metrics.
        result_ctrl = validate(
            model,
            control_loader,
            device,
            n_generate=0,
            logger=None,
            eval_topk=args.eval_topk,
        )
        _print_result_block(f"CONTROL ({args.control})", result_ctrl, args.eval_topk, 0)

        print("\nDELTA (main - control)")
        print("----------------------")
        print(f"loss_delta:   {_fmt_metric(result_main['val_loss'] - result_ctrl['val_loss'])}")
        for k in args.eval_topk:
            key = f"top{k}_acc"
            if key in result_main and key in result_ctrl:
                dv = result_main[key] - result_ctrl[key]
                print(f"{key}_delta:{' ' if len(key) < 8 else ''} {_fmt_metric(dv)}")

    if args.gen_prefix_tokens > 0 and args.n_generate_samples > 0:
        sample_batch = next(iter(loader))
        n = min(args.n_generate_samples, sample_batch["target_ids"].size(0))
        embeds = sample_batch["chronos_embeds"][:n].to(device)
        mask = sample_batch["chronos_mask"][:n].to(device)
        targets = sample_batch["target_ids"][:n].to(device)

        preds = generate_with_target_prefix(
            model,
            embeds,
            mask,
            targets,
            n_prefix_tokens=args.gen_prefix_tokens,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"\nPREFIX-GUIDED GENERATION (n_prefix_tokens={args.gen_prefix_tokens})")
        print("----------------------------------------------------------")
        for i in range(n):
            gt = model.tokenizer.decode(targets[i], skip_special_tokens=True)
            prefix = model.tokenizer.decode(targets[i, :args.gen_prefix_tokens], skip_special_tokens=True)
            pred = preds[i]
            print(f"sample {i+1}/{n}")
            print(f"  prefix: {prefix!r}")
            print(f"  gt:     {gt!r}")
            print(f"  pred:   {pred!r}")
            print(f"  cer:    {cer(pred, gt):.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()

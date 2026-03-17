"""
Training script for the RingToText adapter.

Only the adapter weights are updated. The LLM and Chronos stay frozen.
Uses mixed-precision (bf16), gradient accumulation, cosine LR schedule,
and periodic validation with sample generation.

Usage:
    python train.py --data_dir ./data/train --val_data_dir ./data/val
    python train.py --data_file ./data/train.pt --val_data_file ./data/val.pt
"""

import argparse
import logging
import math
import time
import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import RingToText
from dataset import IMUTextDataset, collate_fn


# --------------------------------------------------------------------------- #
#  Logging setup
# --------------------------------------------------------------------------- #

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ring2text")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# --------------------------------------------------------------------------- #
#  Args
# --------------------------------------------------------------------------- #

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RingToText adapter")

    # Data
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--val_data_file", type=str, default=None)

    # Model
    p.add_argument("--llm", type=str, default=RingToText.__init__.__defaults__[0],
                   help="HuggingFace model ID (e.g. gpt2, gpt2-medium, Qwen/Qwen2.5-1.5B)")
    p.add_argument("--d_chronos", type=int, default=9216,
                   help="Chronos output dim: 768 * n_channels (9216 for both rings, 4608 for one)")
    p.add_argument("--n_soft_tokens", type=int, default=32)
    p.add_argument("--n_resampler_layers", type=int, default=2)
    p.add_argument("--prompt", type=str, default=None)

    # LoRA (LLM fine-tuning)
    p.add_argument("--lora_rank", type=int, default=0,
                   help="LoRA rank; 0 disables LoRA. With adapter_dim=256 prefer rank<=4 "
                        "so LoRA params stay smaller than the adapter (default: disabled)")
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    p.add_argument("--adapter_dim", type=int, default=256,
                   help="Internal hidden dim of the perceiver adapter. "
                        "Must be divisible by 8 (n_heads). "
                        "Smaller = fewer params = less overfitting (default: 256)")
    p.add_argument("--adapter_dropout", type=float, default=0.5)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / saving
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--n_generate_samples", type=int, default=3,
                   help="Number of samples to generate during validation")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping: stop after this many val checks without improvement (0 = disabled)")

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  CER (Character Error Rate)
# --------------------------------------------------------------------------- #

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = edit_distance(hyp, ref) / max(len(ref), 1).
    Range [0, ∞) — values above 1 mean more insertions than reference chars.
    """
    return _edit_distance(hypothesis, reference) / max(len(reference), 1)


# --------------------------------------------------------------------------- #
#  LR schedule
# --------------------------------------------------------------------------- #

def cosine_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(
    model: RingToText,
    val_loader: DataLoader,
    device: torch.device,
    n_generate: int = 3,
    logger: logging.Logger = None,
) -> dict:
    log = logger.info if logger else print

    model.adapter.eval()
    if model.lora_enabled:
        model.llm.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        chronos_embeds = batch["chronos_embeds"].to(device)
        chronos_mask = batch["chronos_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        labels = batch["target_labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(chronos_embeds, target_ids, chronos_mask, labels)
        total_loss += out["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Generate samples for qualitative inspection
    generated_samples = []
    if n_generate > 0:
        sample_batch = next(iter(val_loader))
        embeds = sample_batch["chronos_embeds"][:n_generate].to(device)
        mask = sample_batch["chronos_mask"][:n_generate].to(device)
        targets = sample_batch["target_ids"][:n_generate]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model.generate(embeds, mask, max_new_tokens=64)

        for i in range(min(n_generate, len(predictions))):
            gt = model.tokenizer.decode(targets[i], skip_special_tokens=True)
            pred = predictions[i]
            sample_cer = cer(pred, gt)
            generated_samples.append({"ground_truth": gt, "predicted": pred, "cer": sample_cer})
            log(f"    sample {i+1}/{min(n_generate, len(predictions))}")
            log(f"      GT  ({len(gt):3d} chars): {gt!r}")
            log(f"      PRED({len(pred):3d} chars): {pred!r}")
            log(f"      CER: {sample_cer:.3f}")

    mean_cer = (sum(s["cer"] for s in generated_samples) / len(generated_samples)
                if generated_samples else float("nan"))

    model.adapter.train()
    if model.lora_enabled:
        model.llm.train()

    return {"val_loss": avg_loss, "mean_cer": mean_cer, "samples": generated_samples}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"gpu {alloc:.1f}/{reserved:.1f} GB"


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = save_dir / f"train_{timestamp}.log"
    logger = setup_logger(log_path)

    logger.info("=" * 70)
    logger.info("RingToText training run")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Device:   {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU:      {torch.cuda.get_device_name(0)}")
    logger.info("=" * 70)

    # ---- Hyperparameters ----
    logger.info("Hyperparameters:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k:<25} = {v}")
    logger.info("-" * 70)

    # ---- Model ----
    logger.info(f"Loading model: {args.llm}")
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

    total = model.total_parameters()
    trainable = model.trainable_parameters()
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    logger.info(f"LoRA enabled:     {model.lora_enabled}")
    if model.lora_enabled:
        lora_params = sum(
            p.numel() for n, p in model.llm.named_parameters() if p.requires_grad
        )
        adapter_params = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
        logger.info(f"  Adapter params: {adapter_params:,}")
        logger.info(f"  LoRA params:    {lora_params:,}")
    if torch.cuda.is_available():
        logger.info(f"After model load: {gpu_mem_str()}")
    logger.info("-" * 70)

    # ---- Data ----
    train_ds = IMUTextDataset(
        data_dir=args.data_dir,
        data_file=args.data_file,
        tokenizer=model.tokenizer,
        max_text_len=args.max_text_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Train samples: {len(train_ds):,}  |  batches/epoch: {len(train_loader)}")

    val_loader = None
    if args.val_data_dir or args.val_data_file:
        val_ds = IMUTextDataset(
            data_dir=args.val_data_dir,
            data_file=args.val_data_file,
            tokenizer=model.tokenizer,
            max_text_len=args.max_text_len,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"Val samples:   {len(val_ds):,}  |  batches: {len(val_loader)}")
    logger.info("-" * 70)

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    total_steps = args.epochs * len(train_loader) // args.grad_accum
    logger.info(f"Total steps: {total_steps:,}  |  warmup: {args.warmup_steps}")
    logger.info("=" * 70)

    # ---- Training loop ----
    global_step = 0
    best_val_loss = float("inf")
    no_improve_count = 0
    best_ckpt_path = save_dir / "adapter_best.pt"
    train_start = time.time()

    model.adapter.train()
    if model.lora_enabled:
        model.llm.train()

    stop_training = False
    for epoch in range(args.epochs):
        if stop_training:
            break

        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_epoch_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            chronos_embeds = batch["chronos_embeds"].to(device)
            chronos_mask = batch["chronos_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["target_labels"].to(device)

            # LR schedule
            lr = cosine_lr(global_step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(chronos_embeds, target_ids, chronos_mask, labels)
                loss = out["loss"] / args.grad_accum

            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                ).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_grad_norm += grad_norm
                n_epoch_steps += 1

            epoch_loss += out["loss"].item()

            # Step log
            if global_step % args.log_every == 0 and global_step > 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_gnorm = epoch_grad_norm / max(n_epoch_steps, 1)
                elapsed = time.time() - t0
                wall = time.time() - train_start
                mem = gpu_mem_str()
                logger.info(
                    f"[ep {epoch+1:02d}/{args.epochs}] "
                    f"step {global_step:5d} | "
                    f"loss {avg_loss:.4f} | "
                    f"grad_norm {avg_gnorm:.3f} | "
                    f"lr {lr:.2e} | "
                    f"epoch_t {elapsed:.1f}s | "
                    f"wall {wall/60:.1f}min"
                    + (f" | {mem}" if mem else "")
                )

            # Validation
            if (
                val_loader
                and global_step % args.val_every == 0
                and global_step > 0
            ):
                logger.info(f"--- Validation at step {global_step} ---")
                val_result = validate(
                    model, val_loader, device, args.n_generate_samples, logger
                )
                val_loss = val_result["val_loss"]
                improved = val_loss < best_val_loss

                mean_cer = val_result["mean_cer"]
                cer_str = f"  mean_CER: {mean_cer:.3f}" if not math.isnan(mean_cer) else ""

                if improved:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    model.save_adapter(str(best_ckpt_path))
                    logger.info(
                        f"  val_loss: {val_loss:.4f}{cer_str}  [BEST] saved {best_ckpt_path.name}"
                    )
                else:
                    no_improve_count += 1
                    logger.info(
                        f"  val_loss: {val_loss:.4f}{cer_str}  "
                        f"[no improvement {no_improve_count}/{args.patience}, "
                        f"best={best_val_loss:.4f}]"
                    )

                if args.patience > 0 and no_improve_count >= args.patience:
                    logger.info(
                        f"  Early stopping triggered: no improvement for "
                        f"{args.patience} consecutive val checks."
                    )
                    stop_training = True
                    break

            # Periodic checkpoint
            if global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = save_dir / f"adapter_step{global_step}.pt"
                model.save_adapter(str(ckpt_path))
                logger.info(f"  Checkpoint saved: {ckpt_path.name}")

        # End-of-epoch summary
        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        avg_epoch_gnorm = epoch_grad_norm / max(n_epoch_steps, 1)
        epoch_time = time.time() - t0
        logger.info(
            f"[ep {epoch+1:02d}/{args.epochs}] EPOCH DONE | "
            f"avg_loss {avg_epoch_loss:.4f} | "
            f"avg_grad_norm {avg_epoch_gnorm:.3f} | "
            f"time {epoch_time:.1f}s"
        )

    # Final save
    logger.info("=" * 70)
    total_wall = time.time() - train_start
    logger.info(f"Training complete in {total_wall/60:.1f} min ({total_wall:.0f}s)")
    final_path = save_dir / "adapter_final.pt"
    model.save_adapter(str(final_path))
    logger.info(f"Final adapter:    {final_path}")
    if val_loader:
        logger.info(f"Best val_loss:    {best_val_loss:.4f}")
        logger.info(f"Best checkpoint:  {best_ckpt_path}")
    logger.info(f"Log saved to:     {log_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    train(get_args())

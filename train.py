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

import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import RingToText
from dataset import IMUTextDataset, collate_fn
import char_vocab


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

    # Config file
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (e.g. configs/default.yaml). "
                        "Values are used as defaults; explicit CLI flags override them.")

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
    p.add_argument("--adapter_dropout", type=float, default=0.1,
                   help="Dropout in adapter layers (default: 0.1; was 0.5 but that killed discriminative gradients)")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Learning rate (default: 3e-5; was 1e-4 but caused exploding grad norms)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--max_grad_norm", type=float, default=0.5,
                   help="Gradient clipping norm (default: 0.5; was 1.0 but norms reached 17-18)")
    p.add_argument("--div_weight", type=float, default=10.0,
                   help="Weight for soft-token diversity loss. Penalises high cosine similarity "
                        "between soft tokens of different samples in the same batch, preventing "
                        "adapter collapse to a constant output (0 = disabled). "
                        "Needs to be large (~10) for Qwen2.5-1.5B whose LM gradient overwhelms "
                        "smaller weights; was 0.1 initially (ineffective), then 1.0 (effective "
                        "for GPT-2 but not Qwen).")
    p.add_argument("--recon_weight", type=float, default=100.0,
                   help="Weight for auxiliary reconstruction loss. Trains the adapter to make "
                        "the mean of its resampler output (adapter_dim) reconstruct the mean of "
                        "its projected input (adapter_dim). Provides an input-conditional "
                        "gradient even when LM loss is saturated. Zero extra params. "
                        "History: 1.0 had zero effect (recon stuck at 0.0078, orthogonal case "
                        "for 256-d unit vectors, gradient ~700x smaller than LM). 500.0 was "
                        "effective (recon 0.0078->0.0015, ~36deg alignment) but combined with "
                        "ct=2.0 drove grad norms to 60-144 and hurt val loss. 100.0 targets "
                        "~0.15 nat contribution (100 * 0.0015), ~2.7% of LM gradient.")
    p.add_argument("--contrast_weight", type=float, default=0.3,
                   help="Weight for contrastive (InfoNCE) loss. Pulls soft-token means toward "
                        "the LLM text-embedding mean of their own target text and pushes away "
                        "from other texts in the batch. Teaches the adapter to distinguish "
                        "inputs by content. Zero extra params. "
                        "History: 0.1 barely moved ct from random baseline (2.74 vs log(16)=2.77). "
                        "2.0 dominated training (2.0 * 2.6 = 5.2 nats, ~95% of LM gradient), "
                        "drove grad norms to 60-144, worst val loss yet. 0.3 targets ~0.78 nat "
                        "contribution (0.3 * 2.6), ~14% of LM gradient.")
    p.add_argument("--contrast_temp", type=float, default=0.1,
                   help="Temperature for InfoNCE contrastive loss (default: 0.1).")
    p.add_argument("--keystroke_weight", type=float, default=1.0,
                   help="Weight for binary keystroke-activity BCE loss on per-frame "
                        "logits attached to the adapter's projected features. This is "
                        "the primary content-grounding signal in Phase A — without it "
                        "the adapter has no dense frame-level supervision. 0 = disabled.")
    p.add_argument("--onset_weight", type=float, default=1.0,
                   help="Weight for the per-frame keystroke-onset focal-BCE loss. "
                        "Dedicated head predicts whether a keypress starts at each "
                        "frame; complements the dense activity target with a sparse "
                        "impulse target that encourages crisp rising edges. Set to 0 "
                        "to disable (reverts Phase A to activity-only supervision).")
    p.add_argument("--onset_focal_gamma", type=float, default=2.0,
                   help="Focusing exponent for onset focal loss. Higher values "
                        "down-weight easy negatives harder. 0 reduces focal loss "
                        "to plain weighted BCE.")
    p.add_argument("--onset_focal_alpha", type=float, default=0.75,
                   help="Class-balance weight for onset focal loss. alpha applies "
                        "to positives, (1 - alpha) to negatives. Raise above 0.5 "
                        "when onsets are sparse; per-event onsets sit around 1–3 "
                        "per 100 frames on this dataset.")
    p.add_argument("--skip_llm", action="store_true",
                   help="Phase A: skip loading and running the frozen LLM entirely. "
                        "Only the adapter + keystroke head train. Much faster and uses "
                        "less GPU memory. Use this to pretrain the adapter on the "
                        "keystroke-activity detection task before moving to Phase B.")
    p.add_argument("--no_keystroke", action="store_true",
                   help="Disable the keystroke head entirely.")
    p.add_argument("--adapter_init", type=str, default=None,
                   help="Path to a checkpoint to initialise the adapter (and "
                        "keystroke head, if present) from. Used to seed Phase B "
                        "with a Phase-A pretrained adapter.")
    p.add_argument("--freeze_resampler", action="store_true",
                   help="Freeze adapter.proj and adapter.resampler (leaving "
                        "out_proj trainable) so Phase B LoRA runs don't drift "
                        "the Phase-A grounded rhythm. Step 1 default.")
    p.add_argument("--char_ctc_weight", type=float, default=0.0,
                   help="Peak weight of the character-level CTC loss attached to "
                        "per-frame adapter features (0 = disabled). Linearly warmed "
                        "from char_ctc_start over char_ctc_warmup_steps.")
    p.add_argument("--char_ctc_start", type=float, default=0.1,
                   help="Initial (pre-warmup) CTC weight; warms up to --char_ctc_weight.")
    p.add_argument("--char_ctc_warmup_steps", type=int, default=2000,
                   help="Linear ramp length for the CTC loss weight.")
    p.add_argument("--ctc_upsample_factor", type=int, default=1,
                   help="ConvTranspose1d stride on proj_seq before the CTC head. "
                        "Use 4 for MOMENT (S=64) and 1 for Chronos (S~=513).")
    p.add_argument("--ctc_on_resampler", action="store_true",
                   help="Diagnostic: attach CTC head to the 32 compressed resampler "
                        "tokens instead of proj_seq_up. Tests whether character info "
                        "survives the perceiver bottleneck.")
    p.add_argument("--ctc_no_cross_attn", action="store_true",
                   help="Ablation: disable resampler cross-attention in CharCTCHead. "
                        "CTC runs on proj_seq_up with no KV from resampler_out. Tests "
                        "whether diffuse KV gradient is the cause of the 64-token regression.")
    p.add_argument("--char_seq2seq_weight", type=float, default=0.0,
                   help="Weight for the seq2seq character decoder loss (teacher-forced "
                        "cross-entropy on the 32 resampler soft tokens). 0 = disabled. "
                        "Unlike CTC, has no T≥L length constraint.")
    p.add_argument("--require_adapter_init", action="store_true",
                   help="Error out if --adapter_init is not provided. Set in "
                        "phase_b.yaml so Phase B never silently starts from a "
                        "random adapter (which would discard Phase A's work).")

    # Logging / saving
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--n_generate_samples", type=int, default=3,
                   help="Number of samples to generate during validation")
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping: stop after this many val checks without improvement (0 = disabled)")

    args = p.parse_args()

    # Load YAML config and use as defaults for any unset CLI flags
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Only override args that were not explicitly passed on the CLI
        import sys
        cli_argv = set()
        for token in sys.argv[1:]:
            if token.startswith("--"):
                cli_argv.add(token.lstrip("-").split("=")[0])
        known_keys = set(vars(args).keys())
        for key, value in cfg.items():
            if key not in known_keys:
                # A typo (e.g. `keystroke_wieght: 0.3`) would otherwise attach
                # a stray attribute and the YAML value would be silently
                # ignored in favour of the argparse default. Fail loudly.
                p.error(
                    f"Unknown key {key!r} in config {args.config}. "
                    f"Valid keys: {sorted(known_keys)}"
                )
            if key not in cli_argv:
                setattr(args, key, value)

    # Sanity: a Phase-A run that disables the keystroke head has no trainable loss.
    if args.skip_llm and args.no_keystroke:
        p.error("--skip_llm together with --no_keystroke leaves no loss to train on.")

    if args.skip_llm and args.keystroke_weight <= 0 and args.onset_weight <= 0:
        p.error(
            "--skip_llm requires keystroke_weight > 0 or onset_weight > 0 — "
            "Phase A has no other supervision."
        )

    if args.require_adapter_init and not args.adapter_init:
        p.error("--require_adapter_init was set (Phase B) but --adapter_init was not provided. "
                "Pass --adapter_init <path-to-phase-a-checkpoint>.")

    return args


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
    # Clamp so that lr decays to 0 and stays there past total_steps,
    # rather than oscillating back up because cos goes negative for x > pi.
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
#  Keystroke-activity loss + frame/onset metrics
# --------------------------------------------------------------------------- #


def _compute_keystroke_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    """
    Mean-over-valid-frames BCE loss for the per-frame keystroke head.

    Args:
        logits:     (B, S) raw logits from KeystrokeHead.
        targets:    (B, S) float in {0, 1} from collate_fn.
        mask:       (B, S) bool — True where the frame is a real (non-padded)
                    encoder position.
        pos_weight: optional scalar tensor scaling the loss on positive frames
                    to compensate for class imbalance (sparse keystrokes).
    """
    # The head emits one logit per encoder frame; targets/mask are padded in
    # collate_fn to the longest sample in the batch. These typically agree,
    # but clip to the common prefix so an off-by-one from a mismatched Chronos
    # stride cannot crash training silently.
    S = min(logits.size(1), targets.size(1), mask.size(1))
    logits = logits[:, :S]
    targets = targets[:, :S]
    mask = mask[:, :S]
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(),
        reduction="none",
        pos_weight=pos_weight,
    )
    m = mask.float()
    return (bce * m).sum() / m.sum().clamp(min=1.0)


def _compute_onset_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """
    Masked binary focal loss on per-frame onset logits.

    Onsets are sparse impulses (a few per 100 frames), so plain BCE with a
    positive-class weight over-weights the dense negatives that are already
    easy. Focal loss down-weights confident predictions (regardless of class)
    so the optimiser focuses on hard, low-confidence frames. Alpha separately
    up-weights the rare positive class.

    Args:
        logits:  (B, S) raw logits from the onset classifier.
        targets: (B, S) float in {0, 1}.
        mask:    (B, S) bool — True on real encoder frames.
        gamma:   focusing exponent; 0 reduces this to alpha-weighted BCE.
        alpha:   class weight; alpha for positives, (1 - alpha) for negatives.
    """
    S = min(logits.size(1), targets.size(1), mask.size(1))
    logits = logits[:, :S].float()
    targets = targets[:, :S].float()
    mask = mask[:, :S]

    # Numerically stable per-element BCE: log(sigmoid(x)) used directly avoids
    # instability from computing sigmoid then log separately.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # p_t = exp(-bce) is the probability of the true class at each frame.
    p_t = torch.exp(-bce)
    focal = ((1.0 - p_t) ** gamma) * bce
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = alpha_t * focal

    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def _char_ctc_weight(step: int, start: float, peak: float, warmup: int) -> float:
    """Linear ramp from `start` to `peak` over `warmup` optimizer steps."""
    if warmup <= 0:
        return peak
    frac = min(1.0, step / warmup)
    return start + (peak - start) * frac


def _compute_char_ctc_loss(
    logits: torch.Tensor,
    char_ids: torch.Tensor,
    char_lens: torch.Tensor,
    embed_lens: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
    """
    CTC loss on (B, T, V) per-frame character logits.

    Args:
        logits:         (B, T, V) — T is S or S * upsample_factor.
        char_ids:       (B, max_L) padded with BLANK_ID (ignored by CTC
                        beyond char_lens).
        char_lens:      (B,) actual target lengths.
        embed_lens:     (B,) encoder sequence lengths (pre-upsample).
        upsample_factor:int — multiplies embed_lens to get per-sample T.
    """
    # log_softmax over vocab, transpose to (T, B, V) as F.ctc_loss expects.
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    input_lens = (embed_lens.to(torch.long) * upsample_factor).clamp(
        max=log_probs.size(0)
    )
    return F.ctc_loss(
        log_probs,
        char_ids.to(torch.long),
        input_lens,
        char_lens.to(torch.long),
        blank=char_vocab.BLANK_ID,
        zero_infinity=True,
    )


def _ctc_cer_batch(
    logits: torch.Tensor,
    char_ids: torch.Tensor,
    char_lens: torch.Tensor,
) -> tuple[int, int]:
    """
    Greedy CTC decode + edit distance per sample. Returns (total_ed, total_L)
    so the caller can aggregate across batches into a single CER number.
    """
    decoded = char_vocab.ctc_greedy_decode(logits)
    ed_sum = 0
    L_sum = 0
    for i, pred_ids in enumerate(decoded):
        L = int(char_lens[i].item())
        tgt_ids = char_ids[i, :L].tolist()
        pred_str = char_vocab.decode(pred_ids)
        tgt_str = char_vocab.decode(tgt_ids)
        ed_sum += _edit_distance(pred_str, tgt_str)
        L_sum += max(len(tgt_str), 1)
    return ed_sum, L_sum


def _compute_char_seq2seq_loss(
    logits: torch.Tensor,
    char_ids: torch.Tensor,
    char_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Teacher-forced cross-entropy for the seq2seq character decoder.

    Args:
        logits:    (B, L, vocab_size) — decoder output logits.
        char_ids:  (B, L) — ground-truth character IDs padded with BLANK_ID.
        char_lens: (B,) — actual unpadded target lengths.
    """
    B, L, V = logits.shape
    pos = torch.arange(L, device=logits.device).unsqueeze(0)   # (1, L)
    mask = pos < char_lens.unsqueeze(1).to(logits.device)       # (B, L)
    targets = char_ids.to(logits.device).clone()
    targets[~mask] = -100
    return F.cross_entropy(
        logits.reshape(-1, V).float(),
        targets.reshape(-1),
        ignore_index=-100,
    )


def _seq2seq_cer_batch(
    model,
    resampler_out: torch.Tensor,
    char_ids: torch.Tensor,
    char_lens: torch.Tensor,
) -> tuple[int, int]:
    """
    Greedy seq2seq decode + edit distance per sample.
    Returns (total_ed, total_L) for aggregation.
    """
    max_L = int(char_lens.max().item())
    pred_ids = model.char_seq2seq_head.greedy_decode(resampler_out, max_len=max_L)
    # pred_ids: (B, max_L) on the same device; move to CPU for decode
    pred_ids = pred_ids.cpu()
    ed_sum = 0
    L_sum = 0
    for i in range(len(char_lens)):
        L = int(char_lens[i].item())
        tgt_ids = char_ids[i, :L].tolist()
        pred_str = char_vocab.decode(pred_ids[i].tolist())
        tgt_str = char_vocab.decode(tgt_ids)
        ed_sum += _edit_distance(pred_str, tgt_str)
        L_sum += max(len(tgt_str), 1)
    return ed_sum, L_sum


def _compute_pos_weight(dataset, field: str = "keystroke_active") -> torch.Tensor:
    """
    Estimate `(1 - p) / p` from the training set, where `p` is the fraction
    of active (or onset) frames. Computed once before training so the loss
    has roughly balanced positive/negative gradient mass.

    Reads directly from `dataset.samples` (all in memory after torch.load) to
    avoid spinning up DataLoader workers and burning a full shuffled epoch of
    disk IO before training even starts.
    """
    pos = 0.0
    total = 0.0
    for s in getattr(dataset, "samples", []):
        if not isinstance(s, dict):
            continue
        t = s.get(field)
        if t is None:
            continue
        pos += float(t.sum())
        total += float(t.numel())
    if total == 0 or pos == 0:
        return torch.tensor(1.0)
    p = pos / total
    return torch.tensor(max(1.0, (1.0 - p) / p), dtype=torch.float32)


def _compute_positive_rate(dataset, field: str) -> float:
    pos = 0.0
    total = 0.0
    for s in getattr(dataset, "samples", []):
        if not isinstance(s, dict):
            continue
        t = s.get(field)
        if t is None:
            continue
        pos += float(t.sum())
        total += float(t.numel())
    return pos / total if total > 0 else 0.0


def _frame_prf(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[int, int, int]:
    """
    Returns (true_positives, false_positives, false_negatives) for frame-level
    binary detection at threshold 0.5 over masked positions only. Counts
    accumulate across batches for a single F1 number per validation pass.
    """
    S = min(logits.size(1), targets.size(1), mask.size(1))
    pred = (torch.sigmoid(logits[:, :S].float()) >= 0.5) & mask[:, :S]
    gt = (targets[:, :S] >= 0.5) & mask[:, :S]
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return tp, fp, fn


def _onset_prf(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    tolerance: int = 2,
) -> tuple[int, int, int]:
    """
    Onset-level PRF: a "rising edge" (frame i where active=1, active[i-1]=0)
    in the prediction is matched to the closest GT onset within `tolerance`
    frames. Each GT onset can match at most one predicted onset.
    """
    S = min(logits.size(1), targets.size(1), mask.size(1))
    pred = (torch.sigmoid(logits[:, :S].float()) >= 0.5) & mask[:, :S]
    gt = (targets[:, :S] >= 0.5) & mask[:, :S]
    pred_n = pred.cpu().numpy()
    gt_n = gt.cpu().numpy()
    mask_n = mask[:, :S].cpu().numpy()

    import numpy as _np
    tp = fp = fn = 0
    for b in range(pred_n.shape[0]):
        valid_len = int(mask_n[b].sum())
        if valid_len == 0:
            continue
        pv = pred_n[b, :valid_len].astype(_np.int8)
        gv = gt_n[b, :valid_len].astype(_np.int8)
        pred_onsets = _np.where(_np.diff(_np.concatenate([[0], pv])) == 1)[0]
        gt_onsets = _np.where(_np.diff(_np.concatenate([[0], gv])) == 1)[0]

        matched = _np.zeros(len(gt_onsets), dtype=bool)
        for po in pred_onsets:
            if len(gt_onsets) == 0:
                fp += 1
                continue
            # Find the closest *unmatched* GT onset within tolerance, so an
            # earlier prediction doesn't starve a later one that was the
            # only candidate for a distinct GT event.
            unmatched_idx = _np.where(~matched)[0]
            if len(unmatched_idx) == 0:
                fp += 1
                continue
            d = _np.abs(gt_onsets[unmatched_idx] - po)
            j = int(d.argmin())
            if d[j] <= tolerance:
                matched[unmatched_idx[j]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # (tp, fp, fn) == (0, 0, 0) means neither predictions nor GT positives
    # were ever observed — an F1 of 0 would be misleading (indistinguishable
    # from a model that predicts and misses everything), so surface NaN and
    # let the caller's isnan filter hide the metric from logs.
    if tp == 0 and fp == 0 and fn == 0:
        return float("nan"), float("nan"), float("nan")
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #

def _onset_head_prf(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    tolerance: int = 2,
) -> tuple[int, int, int]:
    """
    PRF for a dedicated onset-logit head.

    Unlike `_onset_prf` (which derives rising edges from the activity logits),
    each predicted positive frame is treated as a candidate onset directly.
    Matched to the closest unmatched GT onset within `tolerance` frames.
    """
    S = min(logits.size(1), targets.size(1), mask.size(1))
    pred = (torch.sigmoid(logits[:, :S].float()) >= 0.5) & mask[:, :S]
    gt = (targets[:, :S] >= 0.5) & mask[:, :S]
    pred_n = pred.cpu().numpy()
    gt_n = gt.cpu().numpy()
    mask_n = mask[:, :S].cpu().numpy()

    import numpy as _np
    tp = fp = fn = 0
    for b in range(pred_n.shape[0]):
        valid_len = int(mask_n[b].sum())
        if valid_len == 0:
            continue
        pred_onsets = _np.where(pred_n[b, :valid_len])[0]
        gt_onsets = _np.where(gt_n[b, :valid_len])[0]
        matched = _np.zeros(len(gt_onsets), dtype=bool)
        for po in pred_onsets:
            if len(gt_onsets) == 0:
                fp += 1
                continue
            unmatched_idx = _np.where(~matched)[0]
            if len(unmatched_idx) == 0:
                fp += 1
                continue
            d = _np.abs(gt_onsets[unmatched_idx] - po)
            j = int(d.argmin())
            if d[j] <= tolerance:
                matched[unmatched_idx[j]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())
    return tp, fp, fn


@torch.no_grad()
def validate(
    model: RingToText,
    val_loader: DataLoader,
    device: torch.device,
    n_generate: int = 3,
    logger: logging.Logger = None,
    pos_weight: torch.Tensor | None = None,
    keystroke_weight: float = 1.0,
    onset_weight: float = 0.0,
    onset_focal_gamma: float = 2.0,
    onset_focal_alpha: float = 0.75,
    ctc_upsample_factor: int = 1,
    char_ctc_weight: float = 0.0,
    ctc_on_resampler: bool = False,
    char_seq2seq_weight: float = 0.0,
) -> dict:
    log = logger.info if logger else print

    model.adapter.eval()
    if model.keystroke_head is not None:
        model.keystroke_head.eval()
    if model.char_ctc_head is not None:
        model.char_ctc_head.eval()
    if model.char_seq2seq_head is not None:
        model.char_seq2seq_head.eval()
    if model.lora_enabled:
        model.llm.eval()

    total_lm_loss = 0.0
    total_ks_loss = 0.0
    total_on_loss = 0.0
    total_ctc_loss = 0.0
    total_seq2seq_loss = 0.0
    n_batches = 0
    n_ks_batches = 0  # batches that actually contributed keystroke metrics
    n_on_batches = 0
    n_ctc_batches = 0
    n_seq2seq_batches = 0
    ctc_ed_sum = 0
    ctc_L_sum = 0
    seq2seq_ed_sum = 0
    seq2seq_L_sum = 0
    frame_tp = frame_fp = frame_fn = 0
    onset_tp = onset_fp = onset_fn = 0  # from dedicated onset head (if active)
    onset_rising_tp = onset_rising_fp = onset_rising_fn = 0  # rising edges of activity

    pw = pos_weight.to(device) if pos_weight is not None else None

    for batch in val_loader:
        chronos_embeds = batch["chronos_embeds"].to(device)
        chronos_mask = batch["chronos_mask"].to(device)
        target_ids = batch["target_ids"].to(device) if "target_ids" in batch else None
        labels = batch["target_labels"].to(device) if "target_labels" in batch else None
        ks_targets = batch["keystroke_targets"].to(device) if "keystroke_targets" in batch else None
        ks_mask = batch["keystroke_mask"].to(device) if "keystroke_mask" in batch else None
        on_targets = batch["onset_targets"].to(device) if "onset_targets" in batch else None

        char_ids_v = batch["char_ids"].to(device) if "char_ids" in batch else None
        char_lens_v = batch["char_lens"].to(device) if "char_lens" in batch else None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(chronos_embeds, target_ids, chronos_mask, labels, char_ids_v)

        if "loss" in out:
            total_lm_loss += out["loss"].item()

        if "keystroke_logits" in out and ks_targets is not None and ks_mask is not None:
            ks_logits = out["keystroke_logits"].float()
            ks_loss = _compute_keystroke_loss(ks_logits, ks_targets, ks_mask, pw)
            total_ks_loss += ks_loss.item()

            tp, fp, fn = _frame_prf(ks_logits, ks_targets, ks_mask)
            frame_tp += tp; frame_fp += fp; frame_fn += fn
            tp, fp, fn = _onset_prf(ks_logits, ks_targets, ks_mask)
            onset_rising_tp += tp; onset_rising_fp += fp; onset_rising_fn += fn
            n_ks_batches += 1

        if (
            onset_weight > 0
            and "onset_logits" in out
            and on_targets is not None
            and ks_mask is not None
        ):
            on_logits = out["onset_logits"].float()
            on_loss = _compute_onset_loss(
                on_logits, on_targets, ks_mask,
                gamma=onset_focal_gamma, alpha=onset_focal_alpha,
            )
            total_on_loss += on_loss.item()
            tp, fp, fn = _onset_head_prf(on_logits, on_targets, ks_mask)
            onset_tp += tp; onset_fp += fp; onset_fn += fn
            n_on_batches += 1

        if (
            "char_logits" in out
            and char_ids_v is not None
        ):
            embed_lens = batch["embed_lens"].to(device)
            if ctc_on_resampler:
                _el = torch.full_like(embed_lens, model.adapter.n_soft_tokens)
                _uf = 1
            else:
                _el = embed_lens
                _uf = ctc_upsample_factor
            ctc_l = _compute_char_ctc_loss(
                out["char_logits"], char_ids_v, char_lens_v, _el, _uf,
            )
            total_ctc_loss += ctc_l.item()
            ed, L = _ctc_cer_batch(
                out["char_logits"].float(),
                char_ids_v.cpu(),
                char_lens_v.cpu(),
            )
            ctc_ed_sum += ed
            ctc_L_sum += L
            n_ctc_batches += 1

        if (
            "char_seq2seq_logits" in out
            and char_ids_v is not None
        ):
            seq2seq_l = _compute_char_seq2seq_loss(
                out["char_seq2seq_logits"], char_ids_v, char_lens_v,
            )
            total_seq2seq_loss += seq2seq_l.item()
            ed, L = _seq2seq_cer_batch(
                model,
                out["resampler_out"].float(),
                char_ids_v.cpu(),
                char_lens_v.cpu(),
            )
            seq2seq_ed_sum += ed
            seq2seq_L_sum += L
            n_seq2seq_batches += 1

        n_batches += 1

    avg_lm = total_lm_loss / max(n_batches, 1)
    # When the keystroke head is off, avg_ks stays 0 but we surface NaN for the
    # F1 metrics so downstream logging can skip them via math.isnan().
    avg_ks = (total_ks_loss / n_ks_batches) if n_ks_batches > 0 else 0.0
    avg_on = (total_on_loss / n_on_batches) if n_on_batches > 0 else 0.0
    avg_ctc = (total_ctc_loss / n_ctc_batches) if n_ctc_batches > 0 else 0.0
    ctc_cer = (ctc_ed_sum / ctc_L_sum) if ctc_L_sum > 0 else float("nan")
    avg_seq2seq = (total_seq2seq_loss / n_seq2seq_batches) if n_seq2seq_batches > 0 else 0.0
    seq2seq_cer = (seq2seq_ed_sum / seq2seq_L_sum) if seq2seq_L_sum > 0 else float("nan")
    if n_ks_batches > 0:
        f_p, f_r, f_f1 = _f1(frame_tp, frame_fp, frame_fn)
        ro_p, ro_r, ro_f1 = _f1(onset_rising_tp, onset_rising_fp, onset_rising_fn)
    else:
        f_f1 = ro_f1 = float("nan")
        f_p = f_r = ro_p = ro_r = float("nan")
    if n_on_batches > 0:
        o_p, o_r, o_f1 = _f1(onset_tp, onset_fp, onset_fn)
    else:
        o_p = o_r = o_f1 = float("nan")

    # Primary val_loss for early stopping: LM if present else
    # weighted sum of the active Phase-A losses (activity + onset + ctc).
    if not model.skip_llm:
        val_loss = avg_lm
    else:
        # Match the training-loss composition so the validation signal tracks
        # what the optimiser actually sees. CTC uses the peak weight — the
        # warmup schedule applies only to training, so val metrics reflect
        # the eventual-objective balance.
        val_loss = (
            keystroke_weight * avg_ks
            + onset_weight * avg_on
            + char_ctc_weight * avg_ctc
            + char_seq2seq_weight * avg_seq2seq
        )

    if n_ks_batches > 0:
        log(f"    frame_F1 {f_f1:.3f} (P {f_p:.3f} R {f_r:.3f}) | "
            f"onset_rising_F1 {ro_f1:.3f} (P {ro_p:.3f} R {ro_r:.3f})")
    if n_on_batches > 0:
        log(f"    onset_head_F1 {o_f1:.3f} (P {o_p:.3f} R {o_r:.3f}) | "
            f"onset_loss {avg_on:.4f}")
    if n_ctc_batches > 0:
        log(f"    char_ctc_loss {avg_ctc:.4f} | ctc_CER {ctc_cer:.3f}")
    if n_seq2seq_batches > 0:
        log(f"    char_seq2seq_loss {avg_seq2seq:.4f} | seq2seq_CER {seq2seq_cer:.3f}")

    # LLM generation (only if LLM is present)
    if n_generate > 0 and not model.skip_llm and model.tokenizer is not None:
        sample_batch = next(iter(val_loader))
        embeds = sample_batch["chronos_embeds"][:n_generate].to(device)
        mask = sample_batch["chronos_mask"][:n_generate].to(device)
        targets = sample_batch["target_ids"][:n_generate]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model.generate(embeds, mask, max_new_tokens=64)

        for i in range(min(n_generate, len(predictions))):
            gt = model.tokenizer.decode(targets[i], skip_special_tokens=True)
            pred = predictions[i]
            log(f"    [LLM] sample {i+1}")
            log(f"      GT  ({len(gt):3d}): {gt!r}")
            log(f"      PRED({len(pred):3d}): {pred!r}")
            log(f"      LLM-CER: {cer(pred, gt):.3f}")

    model.adapter.train()
    if model.keystroke_head is not None:
        model.keystroke_head.train()
    if model.char_ctc_head is not None:
        model.char_ctc_head.train()
    if model.char_seq2seq_head is not None:
        model.char_seq2seq_head.train()
    if model.lora_enabled:
        model.llm.train()

    return {
        "val_loss": val_loss,
        "val_lm_loss": avg_lm,
        "val_ks_loss": avg_ks,
        "val_on_loss": avg_on,
        "val_ctc_loss": avg_ctc,
        "val_ctc_cer": ctc_cer,
        "val_seq2seq_loss": avg_seq2seq,
        "val_seq2seq_cer": seq2seq_cer,
        "frame_f1": f_f1,
        "onset_rising_f1": ro_f1,
        "onset_head_f1": o_f1,
    }


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
    logger.info(f"Loading model: {args.llm if not args.skip_llm else '(LLM skipped — Phase A)'}")
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
        use_keystroke=not args.no_keystroke,
        skip_llm=args.skip_llm,
        use_char_ctc=(args.char_ctc_weight > 0),
        ctc_upsample_factor=args.ctc_upsample_factor,
        ctc_on_resampler=args.ctc_on_resampler,
        ctc_no_cross_attn=args.ctc_no_cross_attn,
        freeze_resampler=args.freeze_resampler,
        use_char_seq2seq=(args.char_seq2seq_weight > 0),
    ).to(device)

    if args.adapter_init is not None:
        logger.info(f"Initialising adapter from: {args.adapter_init}")
        model.load_adapter(args.adapter_init, map_location=device, weights_only=False)

    total = model.total_parameters()
    trainable = model.trainable_parameters()
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    logger.info(f"LoRA enabled:     {model.lora_enabled}")
    if model.lora_enabled:
        lora_params = sum(
            p.numel() for p in model.llm.parameters() if p.requires_grad
        )
        adapter_params = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
        logger.info(f"  Adapter params: {adapter_params:,}")
        logger.info(f"  LoRA params:    {lora_params:,}")
    if torch.cuda.is_available():
        logger.info(f"After model load: {gpu_mem_str()}")
    logger.info("-" * 70)

    # ---- Data ----
    # Phase A (skip_llm) has no tokenizer; dataset falls back to char-only targets.
    ds_tokenizer = model.tokenizer if not args.skip_llm else None
    train_ds = IMUTextDataset(
        data_dir=args.data_dir,
        data_file=args.data_file,
        tokenizer=ds_tokenizer,
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
            tokenizer=ds_tokenizer,
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

    # Optimizer steps per epoch = floor(batches / grad_accum) — the tail
    # of each epoch is dropped because optimizer.step() is gated on
    # (batch_idx + 1) % grad_accum == 0 and counters reset each epoch.
    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    logger.info(f"Total steps: {total_steps:,}  |  warmup: {args.warmup_steps}")

    # Compute pos_weight once for the keystroke BCE — sparse-positive class
    # rebalancing. Skipped if the head is disabled.
    pos_weight = None
    if model.keystroke_head is not None and (args.keystroke_weight > 0 or args.onset_weight > 0):
        # Legacy embeddings (pre-keystroke iteration) silently omit the
        # activity mask, which would leave Phase A with no active loss
        # (collate drops keystroke_targets → BCE never runs). Fail fast
        # so `loss.backward()` on a zero scalar can't masquerade as success.
        samples = getattr(train_ds, "samples", [])
        src = args.data_file or args.data_dir
        if len(samples) == 0:
            raise RuntimeError(f"Training set {src} is empty — nothing to train on.")
        n_with_ka = sum(1 for s in samples if isinstance(s, dict) and "keystroke_active" in s)
        if n_with_ka == 0 and args.skip_llm:
            raise RuntimeError(
                f"No sample in {src} has a 'keystroke_active' field — "
                f"rerun preprocess.py. Phase A has no other supervision so "
                f"training would be a no-op."
            )
        if n_with_ka < len(samples):
            logger.warning(
                f"Only {n_with_ka}/{len(samples)} train samples carry "
                f"keystroke_active — mixed legacy + new embeddings; "
                f"those batches will silently omit the BCE loss."
            )
        pos_weight = _compute_pos_weight(train_ds, field="keystroke_active")
        logger.info(f"Keystroke BCE pos_weight: {pos_weight.item():.3f}")
        if args.onset_weight > 0:
            onset_rate = _compute_positive_rate(train_ds, field="keystroke_onset")
            logger.info(
                f"Onset positive rate: {onset_rate:.4f}  "
                f"(focal gamma={args.onset_focal_gamma}, alpha={args.onset_focal_alpha})"
            )
    logger.info("=" * 70)

    # ---- Training loop ----
    global_step = 0
    best_val_loss = float("inf")
    no_improve_count = 0
    best_ckpt_path = save_dir / "adapter_best.pt"
    train_start = time.time()

    model.adapter.train()
    if model.keystroke_head is not None:
        model.keystroke_head.train()
    if model.char_ctc_head is not None:
        model.char_ctc_head.train()
    if model.char_seq2seq_head is not None:
        model.char_seq2seq_head.train()
    if model.lora_enabled:
        model.llm.train()

    stop_training = False
    for epoch in range(args.epochs):
        if stop_training:
            break

        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_ks_loss = 0.0
        epoch_on_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_seq2seq_loss = 0.0
        epoch_div_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_contrast_loss = 0.0
        epoch_grad_norm = 0.0
        n_epoch_steps = 0
        t0 = time.time()

        pw_dev = pos_weight.to(device) if pos_weight is not None else None

        for batch_idx, batch in enumerate(train_loader):
            chronos_embeds = batch["chronos_embeds"].to(device)
            chronos_mask = batch["chronos_mask"].to(device)
            target_ids = batch["target_ids"].to(device) if "target_ids" in batch else None
            labels = batch["target_labels"].to(device) if "target_labels" in batch else None
            ks_targets = batch["keystroke_targets"].to(device) if "keystroke_targets" in batch else None
            ks_mask = batch["keystroke_mask"].to(device) if "keystroke_mask" in batch else None
            on_targets = batch["onset_targets"].to(device) if "onset_targets" in batch else None

            # LR schedule
            lr = cosine_lr(global_step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            char_ids_t = batch["char_ids"].to(device) if "char_ids" in batch else None
            char_lens_t = batch["char_lens"].to(device) if "char_lens" in batch else None

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(chronos_embeds, target_ids, chronos_mask, labels, char_ids_t)
                B = chronos_embeds.size(0)

                # LM cross-entropy (only when LLM is present)
                lm_loss = out.get("loss")
                loss = lm_loss if lm_loss is not None else torch.zeros((), device=device)

                # Keystroke BCE: per-frame binary activity classifier over the
                # adapter's projected features. Primary content-grounding signal.
                ks_loss = None
                if (
                    args.keystroke_weight > 0
                    and "keystroke_logits" in out
                    and ks_targets is not None
                    and ks_mask is not None
                ):
                    ks_loss = _compute_keystroke_loss(
                        out["keystroke_logits"], ks_targets, ks_mask, pw_dev,
                    )
                    loss = loss + args.keystroke_weight * ks_loss

                # Onset focal loss: dedicated rising-edge head. Sparse positive
                # target (per-event onset impulses) paired with focal loss so
                # the optimiser pushes on hard frames near keystroke boundaries
                # instead of drowning in dense negatives.
                on_loss = None
                if (
                    args.onset_weight > 0
                    and "onset_logits" in out
                    and on_targets is not None
                    and ks_mask is not None
                ):
                    on_loss = _compute_onset_loss(
                        out["onset_logits"], on_targets, ks_mask,
                        gamma=args.onset_focal_gamma,
                        alpha=args.onset_focal_alpha,
                    )
                    loss = loss + args.onset_weight * on_loss

                # Character-level CTC loss on per-frame adapter features
                # (optionally upsampled). Dense content supervision for
                # Phase A; typically disabled in Phase B so LoRA drives LM.
                ctc_loss = None
                ctc_w_eff = 0.0
                if (
                    args.char_ctc_weight > 0
                    and "char_logits" in out
                    and "char_ids" in batch
                ):
                    ctc_w_eff = _char_ctc_weight(
                        global_step,
                        args.char_ctc_start,
                        args.char_ctc_weight,
                        args.char_ctc_warmup_steps,
                    )
                    _el = (
                        torch.full(
                            (batch["embed_lens"].size(0),),
                            model.adapter.n_soft_tokens,
                            dtype=torch.long, device=device,
                        )
                        if args.ctc_on_resampler
                        else batch["embed_lens"].to(device)
                    )
                    _uf = 1 if args.ctc_on_resampler else args.ctc_upsample_factor
                    ctc_loss = _compute_char_ctc_loss(
                        out["char_logits"],
                        batch["char_ids"].to(device),
                        batch["char_lens"].to(device),
                        _el,
                        _uf,
                    )
                    loss = loss + ctc_w_eff * ctc_loss

                # Seq2seq character decoder loss: autoregressive teacher-forced
                # cross-entropy on the 32 resampler soft tokens. No T≥L
                # constraint — the decoder treats the tokens as a memory.
                seq2seq_loss = None
                if (
                    args.char_seq2seq_weight > 0
                    and "char_seq2seq_logits" in out
                    and char_ids_t is not None
                ):
                    seq2seq_loss = _compute_char_seq2seq_loss(
                        out["char_seq2seq_logits"], char_ids_t, char_lens_t,
                    )
                    loss = loss + args.char_seq2seq_weight * seq2seq_loss

                # Diversity loss: penalise high cosine similarity between soft
                # tokens of different samples in the batch, preventing the adapter
                # from collapsing to a constant output regardless of input.
                if args.div_weight > 0 and not args.skip_llm:
                    soft = out["soft_tokens"]                        # (B, n_soft, d_llm)
                    soft_flat = soft.reshape(B, -1)                  # (B, n_soft * d_llm)
                    soft_norm = F.normalize(soft_flat.float(), dim=1)
                    sim = soft_norm @ soft_norm.T                    # (B, B)
                    off_diag = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]
                    div_loss = off_diag.mean()
                    loss = loss + args.div_weight * div_loss

                # Reconstruction loss: soft token mean (adapter_dim) should
                # reconstruct the mean of the projected input (adapter_dim).
                if args.recon_weight > 0 and not args.skip_llm:
                    pred = out["resampler_out"].mean(dim=1).float()  # (B, adapter_dim)
                    tgt = out["projected_mean"].float().detach()     # (B, adapter_dim)
                    pred_n = F.normalize(pred, dim=-1)
                    tgt_n = F.normalize(tgt, dim=-1)
                    recon_loss = F.mse_loss(pred_n, tgt_n)
                    loss = loss + args.recon_weight * recon_loss

                # Contrastive loss (InfoNCE): soft-token mean vs LLM text-embedding mean.
                if (
                    args.contrast_weight > 0
                    and not args.skip_llm
                    and target_ids is not None
                ):
                    with torch.no_grad():
                        text_emb = model._embed_layer(target_ids).float()   # (B, S, d_llm)
                        pad_mask = (labels != -100).unsqueeze(-1).float()   # (B, S, 1)
                        text_proto = (text_emb * pad_mask).sum(1) / pad_mask.sum(1).clamp(min=1)
                    soft_proto = out["soft_tokens"].mean(dim=1).float()     # (B, d_llm)
                    s_norm = F.normalize(soft_proto, dim=-1)
                    t_norm = F.normalize(text_proto, dim=-1)
                    sim_ct = s_norm @ t_norm.T / args.contrast_temp         # (B, B)
                    ct_labels = torch.arange(B, device=device)
                    contrast_loss = F.cross_entropy(sim_ct, ct_labels)
                    loss = loss + args.contrast_weight * contrast_loss

                loss = loss / args.grad_accum

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

            # Track the "primary" loss scalar (LM when available, else keystroke).
            primary = lm_loss if lm_loss is not None else ks_loss
            if primary is not None:
                epoch_loss += primary.item()
            if lm_loss is not None:
                epoch_lm_loss += lm_loss.item()
            if ks_loss is not None:
                epoch_ks_loss += ks_loss.item()
            if on_loss is not None:
                epoch_on_loss += on_loss.item()
            if ctc_loss is not None:
                epoch_ctc_loss += ctc_loss.item()
            if seq2seq_loss is not None:
                epoch_seq2seq_loss += seq2seq_loss.item()
            if args.div_weight > 0 and not args.skip_llm:
                epoch_div_loss += div_loss.item()
            if args.recon_weight > 0 and not args.skip_llm:
                epoch_recon_loss += recon_loss.item()
            if args.contrast_weight > 0 and not args.skip_llm:
                epoch_contrast_loss += contrast_loss.item()

            # Step log
            if global_step % args.log_every == 0 and global_step > 0:
                n_batches = batch_idx + 1
                avg_loss = epoch_loss / n_batches
                avg_gnorm = epoch_grad_norm / max(n_epoch_steps, 1)
                elapsed = time.time() - t0
                wall = time.time() - train_start
                mem = gpu_mem_str()
                lm_str = (f" | lm {epoch_lm_loss / n_batches:.4f}"
                          if not args.skip_llm and epoch_lm_loss > 0 else "")
                ks_str = (f" | ks {epoch_ks_loss / n_batches:.4f}"
                          if args.keystroke_weight > 0 and not args.no_keystroke else "")
                on_str = (f" | on {epoch_on_loss / n_batches:.4f}"
                          if args.onset_weight > 0 and not args.no_keystroke else "")
                ctc_str = (f" | ctc {epoch_ctc_loss / n_batches:.3f} (w={ctc_w_eff:.2f})"
                           if args.char_ctc_weight > 0 else "")
                seq2seq_str = (f" | s2s {epoch_seq2seq_loss / n_batches:.4f}"
                               if args.char_seq2seq_weight > 0 else "")
                div_str = (f" | div {epoch_div_loss / n_batches:.4f}"
                           if args.div_weight > 0 and not args.skip_llm else "")
                recon_str = (f" | recon {epoch_recon_loss / n_batches:.4f}"
                             if args.recon_weight > 0 and not args.skip_llm else "")
                contrast_str = (f" | ct {epoch_contrast_loss / n_batches:.4f}"
                                if args.contrast_weight > 0 and not args.skip_llm else "")
                logger.info(
                    f"[ep {epoch+1:02d}/{args.epochs}] "
                    f"step {global_step:5d} | "
                    f"loss {avg_loss:.4f} | "
                    f"grad_norm {avg_gnorm:.3f} | "
                    f"lr {lr:.2e} | "
                    f"epoch_t {elapsed:.1f}s | "
                    f"wall {wall/60:.1f}min"
                    + lm_str
                    + ks_str
                    + on_str
                    + ctc_str
                    + seq2seq_str
                    + div_str
                    + recon_str
                    + contrast_str
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
                    model, val_loader, device, args.n_generate_samples, logger,
                    pos_weight=pos_weight,
                    keystroke_weight=args.keystroke_weight,
                    onset_weight=args.onset_weight,
                    onset_focal_gamma=args.onset_focal_gamma,
                    onset_focal_alpha=args.onset_focal_alpha,
                    ctc_upsample_factor=args.ctc_upsample_factor,
                    char_ctc_weight=args.char_ctc_weight,
                    ctc_on_resampler=args.ctc_on_resampler,
                    char_seq2seq_weight=args.char_seq2seq_weight,
                )
                val_loss = val_result["val_loss"]
                # A NaN val_loss (e.g. bf16 overflow) must NOT silently short-
                # circuit the best-checkpoint path: `NaN < x` is always False,
                # so the run would early-stop on an untouched `best_val_loss`
                # with no surviving checkpoint. Treat NaN as no-improvement.
                improved = (not math.isnan(val_loss)) and val_loss < best_val_loss

                frame_f1 = val_result.get("frame_f1", float("nan"))
                onset_rising_f1 = val_result.get("onset_rising_f1", float("nan"))
                onset_head_f1 = val_result.get("onset_head_f1", float("nan"))
                f1_bits = []
                if not math.isnan(frame_f1):
                    f1_bits.append(f"frame_F1 {frame_f1:.3f}")
                if not math.isnan(onset_head_f1):
                    # Prefer the dedicated-head F1 for the top-line number.
                    f1_bits.append(f"onset_F1 {onset_head_f1:.3f}")
                elif not math.isnan(onset_rising_f1):
                    # Fallback to the activity-rising-edge metric if onset
                    # head was disabled.
                    f1_bits.append(f"onset_F1 {onset_rising_f1:.3f}")
                cer_str = f"  {' | '.join(f1_bits)}" if f1_bits else ""
                parts = []
                if val_result.get("val_lm_loss", 0) > 0 and not args.skip_llm:
                    parts.append(f"lm {val_result['val_lm_loss']:.4f}")
                if val_result.get("val_ks_loss", 0) > 0:
                    parts.append(f"ks {val_result['val_ks_loss']:.4f}")
                if val_result.get("val_on_loss", 0) > 0:
                    parts.append(f"on {val_result['val_on_loss']:.4f}")
                if val_result.get("val_ctc_loss", 0) > 0:
                    parts.append(f"ctc {val_result['val_ctc_loss']:.3f}")
                ctc_cer_val = val_result.get("val_ctc_cer", float("nan"))
                if not math.isnan(ctc_cer_val):
                    parts.append(f"ctc_CER {ctc_cer_val:.3f}")
                if val_result.get("val_seq2seq_loss", 0) > 0:
                    parts.append(f"s2s {val_result['val_seq2seq_loss']:.4f}")
                seq2seq_cer_val = val_result.get("val_seq2seq_cer", float("nan"))
                if not math.isnan(seq2seq_cer_val):
                    parts.append(f"s2s_CER {seq2seq_cer_val:.3f}")
                comp_str = f" ({' | '.join(parts)})" if parts else ""

                if improved:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    model.save_adapter(str(best_ckpt_path))
                    logger.info(
                        f"  val_loss: {val_loss:.4f}{comp_str}{cer_str}  [BEST] saved {best_ckpt_path.name}"
                    )
                else:
                    no_improve_count += 1
                    logger.info(
                        f"  val_loss: {val_loss:.4f}{comp_str}{cer_str}  "
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

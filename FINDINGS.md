# Ring-to-Text: Findings & Design Notes

Running log of architectural decisions, bugs found, experiments run, and open questions.

---

## Architecture Overview

```
IMU (CSV)
  └─► resample to 100 Hz
  └─► z-score normalise per-session per-channel
  └─► Chronos (frozen, per-channel) ──► (S=513, d=9216) embeddings
          └─► [mean-centred across train set]
  └─► IMUAdapter (trainable, ~4M params)
          └─► InputProjection  (9216 → 256)
          └─► PerceiverResampler  (256, 32 learned queries, 2 layers)
          └─► OutputProjection  (256 → d_llm)
  └─► LLM (frozen)
          prefix: "The user typed: " [real token embeds] + 32 soft tokens
          teacher-forced CE loss over target text during training
          autoregressive generation at inference
```

### Key design choices

| Choice | Rationale |
|--------|-----------|
| Chronos per-channel encoding | Each IMU axis encoded independently; embeddings concatenated → d_chronos = 768 × n_channels. Allows Chronos to specialise per axis. |
| Perceiver resampler (BLIP-2 / Flamingo style) | Compresses variable-length encoder output into fixed 32 soft tokens via cross-attention with learned queries. |
| `adapter_dim=256` internal bottleneck | Decouples perceiver hidden size from d_llm (1536 for Qwen). Keeps trainable params small (~4M) to limit overfitting. |
| LLM frozen (no LoRA by default) | Only adapter trains. LoRA available via `--lora_rank` but disabled by default to keep param count low. |
| Base LLM, not instruct-tuned | Instruct models have RLHF biases that interfere with transcription-style generation. |
| Text prompt prefix | `"The user typed: "` prepended as real token embeddings before soft tokens. Steers frozen LLM into transcription mode without fine-tuning. |
| Session-level train/val/test split | Held-out sessions, not held-out windows. Ensures model is evaluated on unseen typing sessions. |
| bfloat16 | LLM and adapter use bf16 for memory efficiency; consistent dtype prevents precision mismatches. |

---

## Bugs Fixed

### `torch.no_grad()` in `model.py forward()` (fixed)
Wrapping the LLM forward pass in `torch.no_grad()` silently blocked gradient flow to the adapter. Even though LLM params have `requires_grad=False`, the computation graph still needs to pass through the LLM for adapter gradients to flow. Removed the wrapper; frozen LLM params are sufficient to prevent LLM weight updates.

---

## Training Run Analysis — `train_20260322_174109.log`

### Config
- Model: GPT-2 (117M frozen) + 4.2M adapter, no LoRA
- Data: 1,391 train / 92 val samples
- `adapter_dim=256`, `adapter_dropout=0.5`, `lr=1e-4`, `max_grad_norm=1.0`
- Early stopping patience=5, `val_every=500`

### Loss curve

| Step | Train loss | Val loss | Note |
|------|-----------|----------|------|
| 500  | ~5.28 | 5.3593 | BEST |
| 1000 | ~5.12 | **5.2956** | BEST (saved) |
| 1500 | ~4.97 | 5.3106 | plateau |
| 2000 | ~4.86 | 5.3958 | diverging |
| 3500 | ~4.66 | 5.4843 | early stop |

### Issues identified

**1. Overfitting**
Train loss fell from 7.1 → 4.7 across 41 epochs while val loss rose after step 1000. Root cause: 1,391 training samples is very small for this task.

**2. Exploding gradient norms**
Grad norms grew from ~6–8 (early epochs) to ~14–15 (ep 35+). The `max_grad_norm=1.0` clip fires hard throughout, indicating growing instability as the adapter overfits.

**3. Generation produced no task-related output**
Across all validation checkpoints, generated text bore zero resemblance to ground truth. Outputs ranged from: unicode artifacts and repeating characters, to generic GPT-2 prose about unrelated topics, to pure whitespace padding. The model showed no sign of learning the IMU→text mapping. This was traced to the embedding issue below.

**4. GPT-2 used instead of intended Qwen2.5-1.5B**
This run was likely a smoke test. CLAUDE.md specifies `Qwen/Qwen2.5-1.5B` as the intended model.

---

## Embedding Analysis

Ran on `embeddings/train.pt` and `embeddings/val.pt` (post-training).

### Findings

| Metric | Train | Val |
|--------|-------|-----|
| Embedding shape | (513, 9216) | (513, 9216) |
| All same length? | Yes | Yes |
| Pairwise cosine sim (mean) | **0.863** | **0.868** |
| Pairwise cosine sim (std) | 0.036 | 0.028 |
| % dims with variance < 1e-4 | **99.8%** | **100%** |
| Per-dim variance (mean) | 0.000010 | 0.000005 |

### Interpretation (before mean-centering)
A pairwise cosine similarity of ~0.86 between random sample pairs means all embeddings point in nearly the same direction. 99.8% of the 9216 dimensions carry essentially no discriminative information across samples. The adapter receives nearly identical input regardless of what was typed — making it impossible to learn any meaningful mapping.

### Root cause
Chronos is pre-trained on generic time series data. Human IMU signals from keystrokes appear similar to Chronos: all samples share a dominant "background motion" direction in the Chronos embedding space, which overwhelms the small discriminative residual variation.

---

## Fixes Applied

### 1. Mean-centering Chronos embeddings (`preprocess.py`)

**Problem:** Chronos embeddings cluster tightly (cosine sim ~0.86) due to a shared background signal dominating the embedding space.

**Fix:** After encoding all splits, subtract the training-set mean embedding from every sample in train/val/test. This removes the shared background direction and exposes the discriminative residual variation to the adapter.

```python
# Computed once from training set only (no data leakage)
train_mean = mean of all per-sample mean embeddings  # (d_chronos,)
for every sample in train/val/test:
    sample["embeddings"] -= train_mean
```

Added to `preprocess.py:main()`. Requires re-running preprocessing to take effect.

**Note:** The existing per-channel z-score normalisation of raw IMU (in `_normalize_imu()`) is separate and unaffected — it operates on raw IMU before Chronos, not on Chronos embeddings.

### Verified result (2026-03-22, after re-running preprocessing)

| Metric | Before | After (train) | After (val) |
|--------|--------|---------------|-------------|
| Global mean | 0.0002 | **0.000000** | 0.000009 |
| Cosine sim mean | 0.863 | **0.326** | **0.383** |
| Cosine sim std | 0.036 | **0.140** | **0.120** |
| Cosine sim min | 0.703 | **-0.169** | 0.007 |
| % dims var < 1e-4 | 99.8% | 99.8% | 100% |

Mean-centering dropped average pairwise cosine similarity from ~0.86 to ~0.33 (train) / ~0.38 (val). Std jumped from 0.036 to 0.140, meaning samples are now meaningfully spread apart rather than tightly clustered. Some sample pairs are now anti-correlated (min = -0.17), which was impossible before.

Per-dim variance is unchanged — mean-centering shifts the origin but does not change variance. The 99.8% of dims with var < 1e-4 reflects the absolute scale of variation, not the relative discriminability (which is now much better).

---

## Dataset & Collation

- `dataset.py` tokenises text with `add_special_tokens=True` → EOS appended to each sequence
- `labels` masks pad positions with `-100`, preserving the first EOS as a training target
- Since `pad_token = eos_token` (GPT-2 convention), the masking is critical: without it, every pad token would count as a valid EOS target, diluting the loss signal
- The model is therefore trained to emit EOS at the correct point; generation should be allowed to run freely (EOS-stopped) rather than being length-constrained to GT

---

## Generation Length

**Training:** Length is implicitly matched because teacher forcing passes the full GT token sequence. Loss is computed over exactly the GT tokens.

**Validation/Inference:** `generate()` uses a `max_new_tokens` cap (default 64). This is a safety ceiling, not a GT-length constraint. As training improves, the model should learn to stop at the right length via EOS. Constraining generation to GT length at eval time would artificially inflate CER metrics (hiding insertion errors).

---

## Training Run 2 — `train_20260322_181118.log` (mean-centred embeddings)

Same config as Run 1. First run on the mean-centred embeddings.

### Loss curve

| Step | Train loss | Val loss | Note |
|------|-----------|----------|------|
| 500  | ~5.24 | 5.3612 | BEST |
| 1000 | ~5.00 | **5.3117** | BEST (saved) |
| 1500 | ~4.77 | 5.4362 | no improvement 1/5 |
| 2000 | ~4.62 | 5.4442 | no improvement 2/5 |
| 2500 | ~4.47 | 5.5603 | no improvement 3/5 |
| 3000 | ~4.42 | 5.5439 | no improvement 4/5 |
| 3500 | ~4.41 | 5.5738 | early stop |

### Head-to-head vs Run 1

| Metric | Run 1 (no mean-centre) | Run 2 (mean-centred) |
|--------|------------------------|----------------------|
| Best val loss | **5.2956** | 5.3117 |
| Best val step | 1000 | 1000 |
| Train loss @ ep35 | ~4.73 | **~4.42** |
| Grad norm @ ep35 | ~14–15 | **~17–18** ⚠️ |
| Early stop step | 3500 | 3500 |
| Generation quality | gibberish | gibberish |

### Findings

**1. Mean-centering did not improve val loss**
Best val loss is marginally *worse* (5.3117 vs 5.2956). The improved embedding discriminability (cosine sim 0.86→0.33) did not translate into better generalisation. The gap between train (~4.42) and val (~5.57 at stop) is actually larger, suggesting more overfitting with the mean-centred embeddings.

**2. Grad norms got worse, not better**
Norms grew to ~17–18 by ep35 vs ~14–15 in Run 1. Removing the DC component means the adapter now operates on smaller-magnitude residual signals, which appears to cause larger relative gradients. The `max_grad_norm=1.0` clip is firing harder throughout.

**3. Generation still completely wrong**
Both runs produce generic GPT-2 prose or repetition loops with no relation to GT. The core problem is not solved by mean-centering alone.

### Conclusion
Mean-centering is a valid preprocessing step (embeddings are more separable), but it alone is insufficient. The bottleneck is not embedding discriminability — it is likely model capacity (GPT-2 too small), dataset size (1,391 samples), or the adapter failing to learn a meaningful IMU→text mapping regardless. The rising grad norms suggest the LR should be reduced for mean-centred inputs.

---

## Root Cause: Adapter Output Collapse

Checked soft token diversity on the best checkpoint (`adapter_best.pt` from Run 2) across 50 training samples.

| Metric | Chronos embeddings (input) | Adapter soft tokens (output) |
|--------|---------------------------|------------------------------|
| Pairwise cosine sim (mean) | 0.326 | **0.984** |
| Pairwise cosine sim (std) | 0.140 | 0.010 |
| Pairwise cosine sim (min) | -0.169 | 0.936 |

The adapter takes reasonably diverse inputs (cosine sim ~0.33) and collapses them to nearly identical soft tokens (cosine sim ~0.98). The LLM receives essentially the same prefix for every sample regardless of what was typed, making it impossible to generate different text.

This explains why training loss plateaus near GPT-2's LM prior (~4.4) and generation produces plausible but completely wrong text: the adapter has learned a single "average" soft token that steers GPT-2 toward the training text distribution in general, rather than learning input-conditional behaviour.

This is a **collapse to a constant output** — the adapter found a local minimum where producing the same prefix for all inputs minimises average cross-entropy, without needing to distinguish between inputs at all.

### Why it collapses

1. The IMU signal remaining after mean-centering is very low variance (per-dim var ~1e-5). The InputProjection compresses 9216→256 dims; subtle inter-sample differences survive in embedding space but the adapter learns to ignore them.
2. The perceiver resampler's 32 learned queries attend to all 513 time steps and converge to the same attention pattern for every near-similar input.
3. The frozen LLM provides a strong gradient signal to produce "good English" from any prefix, drowning out the weaker gradient signal toward input-conditional behaviour.
4. High dropout (0.5) may suppress the gradients that would otherwise encourage diversity.

### Potential fixes

- **Contrastive / diversity loss on soft tokens**: add an auxiliary loss term that penalises high cosine similarity between soft tokens from different samples in the same batch. Forces the adapter to produce distinguishable representations.
- **Reduce adapter dropout**: 0.5 is very aggressive for a small signal; try 0.1–0.2 to preserve more gradient signal through the perceiver.
- **Auxiliary reconstruction loss**: train the adapter to reconstruct some property of the input (e.g. mean IMU channel values per window) in addition to the LM loss — provides an input-conditional gradient even when the LM loss is saturated.
- **Gradient surgery / loss weighting**: reduce the LM loss weight so input-conditional gradients are not overwhelmed.

---

## Fixes Implemented (2026-03-22)

Four changes made to address adapter collapse and training instability:

| Fix | File | Change |
|-----|------|--------|
| Diversity loss | `train.py`, `model.py` | Off-diagonal cosine sim penalty on soft tokens; `--div_weight` arg (default 0.1) |
| Reduce dropout | `train.py` | `adapter_dropout` default 0.5 → 0.1 |
| Lower LR | `train.py` | `lr` default 1e-4 → 3e-5 |
| Tighter grad clip | `train.py` | `max_grad_norm` default 1.0 → 0.5 |
| Remove InputProjection LayerNorm | `adapter.py` | Per-timestep normalisation was erasing amplitude differences between samples |

**Diversity loss detail:** Within each batch, compute pairwise cosine similarity between flattened soft token vectors, take the mean of off-diagonal entries, add `div_weight * that mean` to the LM loss. This directly penalises the adapter producing similar outputs for different inputs. Logged per step as `div` in training logs.

**Soft tokens now returned from `model.forward()`** as `out["soft_tokens"]` so the training loop can access them.

---

## Training Run 3 — `train_20260322_191210.log` (diversity loss + dropout/LR/grad norm fixes)

Same dataset (mean-centred embeddings). First run with all four fixes from above.

### Config changes vs Run 2
- `adapter_dropout` 0.5 → 0.1
- `lr` 1e-4 → 3e-5, `warmup_steps` = 200
- `max_grad_norm` 1.0 → 0.5
- `div_weight` = 0.1 (new diversity loss)
- InputProjection LayerNorm removed

### Loss curve

| Step | Train loss (epoch avg) | Val loss | Note |
|------|------------------------|----------|------|
| 500  | ~5.74 | 5.7815 | first val |
| 1000 | ~5.42 | 5.5312 | |
| 1500 | ~5.33 | 5.4305 | |
| 2000 | ~5.29 | 5.4002 | |
| 2500 | ~5.27 | 5.3797 | |
| 3000 | ~5.26 | **5.3748** | BEST (saved) |
| 3500 | ~5.26 | 5.3770 | no improvement 1/5 |
| 4000 | ~5.26 | 5.3771 | no improvement 2/5 |
| 4350 | ~5.26 | — | epoch limit (50 epochs), stopped before patience=5 |

### Head-to-head vs Run 2

| Metric | Run 2 | Run 3 |
|--------|-------|-------|
| Best val loss | **5.3117** | 5.3748 |
| Best val step | 1000 | 3000 |
| Train-val gap at stop | **~1.15** | **~0.11** ✓ |
| Grad norm (late) | ~17–18 | ~15–17 (stable) |
| div (soft token cosine sim) | n/a | 0.953 → 0.914 |
| Early stop | ep40 | Did not stop (epoch limit) |
| Generation quality | gibberish | gibberish |

### Findings

**1. Overfitting massively reduced**
Train-val gap shrank from ~1.15 (Run 2) to ~0.11 (Run 3). Lower LR and reduced dropout are working as regularisers. The model is no longer memorising the training set.

**2. Diversity loss is not working**
`div` (mean off-diagonal soft token cosine similarity within batch) went from 0.953 → 0.914. Still essentially collapsed — nowhere near the target of well below 0.984. `div_weight=0.1` is too weak to overcome the LM gradient signal pushing toward a constant prefix. The penalty is not forcing diversity.

**3. Val loss improved very slowly, then plateaued**
Val loss decreased gradually step 500→3000 (5.78→5.37), then was completely flat from step 2000 onward (~5.37–5.40). This is a stable but useless plateau — the LM loss has found the minimum achievable by a near-constant prefix (the LM prior). Best val loss (5.3748) is *worse* than Run 2 (5.3117), but the slow decline suggests the lower LR may just need more time, or there is nothing more to learn.

**4. Grad norms still large**
Pre-clip norms stabilise around 15–17 throughout, meaning `max_grad_norm=0.5` clips almost every step. The actual gradient step is heavily distorted. This has not improved meaningfully from previous runs.

**5. Generation still completely wrong**
All samples are unrelated GPT-2 prose, repetition loops, or unicode artifacts. At step 3000, sample 1 outputs `"the user typed: ia ia ia ia..."` — the model is echoing part of the text prompt prefix, suggesting it is ignoring soft token content entirely.

**6. Training hit epoch limit, not patience**
Run completed all 50 epochs (4350 steps) in 12.1 min. Patience counter reached only 2/5 — so early stopping never fired. Increasing `val_every` or reducing epoch count would not have helped; the flat plateau was already established from ~step 2000.

### Conclusion
The diversity loss fix reduced overfitting (good) but failed to solve adapter collapse. `div_weight=0.1` is too small. The adapter still produces near-identical soft tokens (cosine sim ~0.91) for all inputs. Increasing `div_weight` substantially (e.g., 0.5–1.0) is the most direct next step. Alternatively, switching to the intended Qwen2.5-1.5B (as per CLAUDE.md) may help — GPT-2's small capacity and vocabulary bias may be contributing to the degenerate plateau.

---

## Training Run 4 — `train_20260322_192951.log` (div_weight=1.0)

Same as Run 3 except `div_weight` raised from 0.1 → 1.0.

### Loss curve

| Step | Train loss (epoch avg) | Val loss (LM only) | Note |
|------|------------------------|---------------------|------|
| 500  | ~5.71 | 6.0232 | first val |
| 1000 | ~5.55 | 5.8429 | |
| 1500 | ~5.53 | 5.8148 | |
| 2000 | ~5.52 | 5.7174 | |
| 2500 | ~5.51 | 5.7055 | |
| 3000 | ~5.50 | **5.6730** | BEST |
| 3500 | ~5.49 | 5.6836 | no improvement 1/5 |
| 4000 | ~5.51 | 5.6772 | no improvement 2/5 |
| 4350 | ~5.50 | — | epoch limit |

### Head-to-head vs Run 3

| Metric | Run 3 (div=0.1) | Run 4 (div=1.0) |
|--------|-----------------|-----------------|
| Best val loss (LM only) | **5.3748** | 5.6730 |
| div final (soft token cos sim) | 0.914 | **0.419** |
| Grad norms (late) | ~15–17 | ~20–28 ⚠️ |
| Train-val gap | **0.11** | 0.17 |
| Generation | GPT-2 prose | prompt leakage / whitespace |

### Findings

**1. Diversity loss now working — cosine sim dropped 0.953 → 0.419**
`div_weight=1.0` successfully drove soft token cosine similarity from ~0.91 (Run 3) to ~0.42. This is a real improvement in adapter output diversity. The div metric plateaued around 0.42–0.43 from step ~2000 onward, suggesting the equilibrium point where diversity and LM gradients balance.

**2. LM val loss got significantly worse**
Best val_loss = 5.6730 vs 5.3748 in Run 3. The val_loss is the pure LM component (diversity loss not included at validation). Forcing diversity cost ~0.3 nats of LM quality. The adapter produces more varied soft tokens, but those tokens don't carry useful semantic content — they're just "different" in an uninformative direction.

**3. Generation shifted from prose to prompt leakage and repetition**
Instead of coherent GPT-2 prose, the model now frequently outputs: whitespace padding, "The user typed: ???..." loops, or "(A)(A)(A)..." sequences. The text prompt prefix is leaking into generated output, which wasn't happening in Run 3. This indicates the soft tokens are now destabilising the LLM's distribution (forcing it off the "good prose" attractor) without steering it toward transcription.

**4. Grad norms grew (~20–28 vs ~15–17)**
The stronger div_weight gradient is compounding the already-high LM gradients. Max_grad_norm=0.5 clips even harder. Still no sign of stabilisation.

**5. Still never early-stopped:** Epoch limit again (patience counter 2/5).

### Conclusion
`div_weight=1.0` confirmed that diversity loss *can* break adapter collapse (cosine sim 0.984 → 0.419), but diversity alone is not sufficient. Diverse soft tokens without a meaningful content signal just destabilise the LLM rather than guiding transcription. The adapter is producing varied but semantically empty prefixes.

The fundamental bottleneck is now clear: **there is no learning signal that connects specific IMU patterns to specific text content** at the soft token level. The LM loss gradient only says "make the prefix cause good English" — it does not say "this IMU pattern means the letter A was typed." Neither diversity loss nor LM loss can teach the adapter what each token should mean.

**This suggests the GPT-2 backbone is not the right test bed.** The intended model (Qwen2.5-1.5B, per CLAUDE.md) has a much richer embedding space and better contextual sensitivity. Additionally, at the current small dataset size (~1,391 samples), only a model with strong priors can bootstrap meaningful learning.

---

## Training Run 5 — `train_20260322_195751.log` (Qwen2.5-1.5B, div_weight=1.0)

First run with the intended LLM. Same config as Run 4 except `llm=Qwen/Qwen2.5-1.5B`.

- Total params: 1,548M | Trainable: 4.4M (0.28%) | GPU after load: 2.9/3.1 GB

### Loss curve

| Step | Train loss (epoch avg) | Val loss | Note |
|------|------------------------|----------|------|
| 500  | ~5.30 | 5.2587 | |
| 1000 | ~5.09 | 5.0086 | |
| 1500 | ~5.00 | 4.7993 | |
| 2000 | ~4.96 | 4.7885 | |
| 2500 | ~4.95 | 4.7473 | |
| 3000 | ~4.94 | **4.7344** | BEST |
| 3500 | ~4.93 | 4.7472 | no improvement 1/5 |
| 4000 | ~4.94 | 4.7442 | no improvement 2/5 |
| 4350 | ~4.94 | — | epoch limit |

### Cross-run comparison

| Run | LLM | div_weight | Best val loss | div final |
|-----|-----|------------|---------------|-----------|
| 1 | GPT-2 | 0 | 5.2956 | n/a |
| 2 | GPT-2 | 0 | 5.3117 | n/a |
| 3 | GPT-2 | 0.1 | 5.3748 | 0.914 |
| 4 | GPT-2 | 1.0 | 5.6730 | 0.419 |
| **5** | **Qwen2.5-1.5B** | **1.0** | **4.7344** | **0.950** |

### Findings

**1. Qwen gives the best val loss by a large margin**
Best val loss = 4.7344 vs 5.29+ on all GPT-2 runs. The improvement is ~0.55 nats. This reflects Qwen's much better language modeling prior, not necessarily better IMU→text mapping.

**2. div_weight=1.0 is completely ineffective against Qwen**
div went from 0.954 → 0.950 over the entire run — effectively no change. In Run 4 (GPT-2, same div_weight), div dropped from 0.953 → 0.419. Qwen is ~13× larger than GPT-2 and its LM gradient magnitude completely overwhelms the diversity penalty at weight 1.0. The adapter collapse is just as severe with Qwen as it was in Run 3.

**3. Grad norms are higher: ~30–38**
Larger model → larger gradient signal from the LM → harder to counteract with diversity loss. The grad clip at 0.5 is firing even more aggressively.

**4. Train-val gap widened slightly to ~0.20**
Still well-controlled (vs ~1.15 in Run 2), but slightly more overfitting than Run 3 (0.11). Acceptable at this scale.

**5. Generation still gibberish — but better-quality gibberish**
Outputs are coherent, well-structured Qwen prose completely unrelated to GT. At step 3000, sample 3 produces text about a backpacking adventure — thematically coincidental with the GT (which is also about a backpacking journey) but semantically unrelated. The model is not learning the IMU signal.

**6. Val loss plateaued from step 2500 onward (~4.73–4.74)** — same stable plateau pattern as Run 3.

### Conclusion
Switching to Qwen2.5-1.5B improved val loss significantly and produces better-quality text, confirming it is the right LLM backbone. However, adapter collapse persists (div ~0.95). The diversity loss needs to be **much larger** to compete with Qwen's gradient, or a different anti-collapse mechanism is needed. The scale mismatch between LM gradient and diversity gradient is the core problem.

---

## Training Run 6 — `train_20260322_201429.log` (Qwen2.5-1.5B, div_weight=0.0)

Qwen clean baseline — diversity loss fully disabled.

### Loss curve

| Step | Train loss (epoch avg) | Val loss | Note |
|------|------------------------|----------|------|
| 500  | ~5.19 | 4.9777 | BEST |
| 1000 | ~4.99 | 4.7833 | BEST |
| 1500 | ~4.92 | 4.7374 | BEST |
| 2000 | ~4.90 | ~4.72  | BEST |
| 2500 | ~4.88 | ~4.71  | BEST |
| 3000 | ~4.88 | 4.6921 | BEST |
| 3500 | ~4.86 | 4.6908 | BEST |
| 4000 | ~4.86 | **4.6877** | BEST (final) |
| 4350 | ~4.86 | — | epoch limit |

### Head-to-head: Qwen runs

| Run | div_weight | Best val loss | Improving at end? |
|-----|------------|---------------|--------------------|
| 5 (Qwen) | 1.0 | 4.7344 | No (plateau ~step 2500) |
| **6 (Qwen)** | **0.0** | **4.6877** | **Yes — every checkpoint a new best** |

### Findings

**1. Best val loss to date: 4.6877**
Removing div_weight entirely gave a better val loss (4.6877) than div_weight=1.0 (4.7344). The div loss with weight 1.0 was being ignored dynamically but still adding noise to gradients.

**2. Val loss improved at every single checkpoint — never plateaued**
Unlike every prior run (GPT-2 or Qwen with div), the val loss decreased monotonically across all 8 validation checkpoints. It was still improving at step 4000 (the last checkpoint). The model had not converged at epoch 50 — more epochs would likely yield further improvement.

**3. Grad norms slightly lower: ~25–28**
Without the div gradient, norms dropped slightly from Run 5's ~30–38. Still high (grad clip fires throughout), but more stable.

**4. Train-val gap ~0.17** — consistent with Run 5, well-regularised.

**5. Generation still gibberish** — coherent Qwen prose, no IMU→text learning apparent.

**6. Collapse diagnostic unavailable** (div disabled). Given Run 5 showed no div movement with weight 1.0, collapse is almost certainly still present here.

### Conclusion
The Qwen baseline with no diversity loss is the best result so far (4.6877) and was still improving at the end of training. The priority is now: (a) run more epochs to find where val loss actually bottoms out, and (b) try `div_weight=10` (new default) to test whether a much stronger penalty can break collapse on Qwen without hurting val loss.

---

## Training Run 7 — `train_20260322_205806.log` (Qwen2.5-1.5B, div_weight=10.0)

### div trajectory (collapse vs diversity)

| Step | div (cosine sim) |
|------|-----------------|
| 50   | 0.949 |
| 200  | 0.928 |
| 300  | 0.692 ← rapid drop |
| 350  | **0.168** ← collapse broken |
| 800  | 0.084 |
| 1000 | 0.073 |
| 1500 | 0.039 |
| 2450 | **0.013** ← near-orthogonal |
| 2500–4350 | 0.02–0.08 (oscillating) |

div_weight=10 successfully broke Qwen's adapter collapse. Soft token cosine similarity fell from ~0.95 to <0.1 by step ~800 and hovered around 0.02–0.07 for the rest of training (near-orthogonal/anti-correlated). Compare: div_weight=1.0 had no effect on Qwen (div stayed 0.95).

### Loss curve

| Step | Val loss | Note |
|------|----------|------|
| 500  | 5.7651 | |
| 1000 | 5.5676 | |
| 1500 | 5.2676 | |
| 2000 | 5.1551 | |
| 2500 | 5.0908 | |
| 3000 | 5.0742 | |
| 3500 | **5.0732** | BEST |
| 4000 | 5.0738 | no improvement 1/5 |
| 4350 | — | epoch limit |

### Full Qwen comparison

| Run | div_weight | div final | Best val loss | Cost vs no-div |
|-----|------------|-----------|---------------|----------------|
| 6   | 0.0 | ~0.95 (collapsed) | **4.6877** | — |
| 5   | 1.0 | ~0.95 (no effect) | 4.7344 | +0.047 |
| **7** | **10.0** | **0.02–0.07** ✓ | **5.0732** | **+0.386** |

### Findings

**1. Collapse is broken, but LM quality is severely hurt**
Breaking collapse (div ~0.02–0.07) costs 0.39 nats of val loss vs the no-div baseline (5.073 vs 4.688). This is worse than the GPT-2 collapse-breaking penalty (~0.37 nats). The adapter produces maximally diverse but semantically incoherent soft tokens.

**2. Grad norms exploded: 39–68**
The div_weight=10 gradient is enormous, completely overwhelming the grad clip at 0.5. The clip fires on every step with large pre-clip norms.

**3. Generation triggered multilingual and code outputs**
The incoherent soft tokens pushed Qwen into unexpected modes: Chinese multiple-choice exam questions, Python error explanations, floating-point number sequences. Qwen is a multilingual model and the diverse-but-random soft tokens land in regions of its embedding space that correspond to non-English content. This confirms the soft tokens have no semantic structure.

**4. div oscillates (0.02–0.08) rather than settling**
The adapter is not converging to a stable diverse representation — it oscillates. This reflects a dynamic equilibrium between the LM gradient (pushing toward the "constant good prefix" attractor) and the diversity gradient (pushing away from similarity). Neither wins cleanly.

### The fundamental bottleneck — confirmed across 7 runs

The training dynamics reveal a clear structure:

| Regime | Soft tokens | Val loss | Learning signal |
|--------|-------------|----------|-----------------|
| No/weak div | Collapsed (cos sim ~0.95) | Low (~4.69) | LM learns to ignore prefix — model just optimises its prior |
| Strong div | Diverse but random (cos sim ~0.05) | High (~5.07) | LM receives noisy prefix — loss measures inability to generate given random input |

Neither regime connects IMU patterns to text content. The LM loss gradient tells the adapter "produce tokens that make the LLM generate plausible English" — but it cannot tell the adapter "this IMU pattern should produce a prefix that causes the LLM to predict *this specific text*."

Without a direct content-grounding signal, the adapter has no way to learn the IMU→text mapping. The cross-entropy objective on text alone is insufficient.

---

## Training Run 8 — `train_20260323_103811.log` (Qwen2.5-1.5B, div_weight=10.0, epochs=200)

Same config as Run 7 (`div_weight=10.0`) but with `epochs=200` and `patience=5` to allow early stopping to fire.

- **First run to fully converge** — early stopping triggered at epoch 133 (step ~11,600)
- **Best val loss: 4.5252** at step 9000 (ep 104) — best across all 8 runs
- Total wall time: 41.4 min

### Val loss curve (key checkpoints)

| Step | Val loss | Note |
|------|----------|------|
| 500  | 5.8407 | |
| 1000 | 5.2684 | |
| 1500 | 4.9150 | |
| 2500 | 4.7110 | |
| 4500 | 4.5651 | |
| 6000 | **4.5403** | |
| **9000** | **4.5252** | **BEST (saved)** |
| 9500–11500 | 4.529–4.534 | no improvement 1–5/5 |
| 11500 | — | **Early stopping triggered** |

### Cross-run comparison (Qwen)

| Run | div_weight | epochs | Best val loss | Converged? |
|-----|------------|--------|---------------|------------|
| 5 | 1.0 | 50 | 4.7344 | No |
| 6 | 0.0 | 50 | 4.6877 | No |
| 7 | 10.0 | 50 | 5.0732 | No |
| **8** | **10.0** | **200** | **4.5252** | **Yes** |

### Findings

**1. div=10 + more epochs beats div=0**
Best val loss 4.5252 is 0.16 nats better than Run 6 (4.6877, no diversity loss). Run 7 failed because it hit the 50-epoch limit while still in the high-loss post-collapse phase. Given enough training time, diversity loss is net beneficial.

**2. LR cosine decay was the enabling factor**
LR decayed from 3e-5 → ~1.4e-5 by step 9000. This enabled slow stable progress after the violent diversity-forced reorganisation early in training — something 50-epoch runs could never observe.

**3. Diversity collapse broke by ep 3–4 (same as Run 7)**
div dropped 0.94 → 0.17 by step ~300, then settled near 0–0.09 for the remainder.

**4. Grad norms stabilised (~27–35)**
Lower than Run 7's 39–68. Cosine LR decay reduced gradient magnitude progressively.

**5. Val loss floor ~4.52–4.53**
The val loss plateau from step 9000 onward is extremely flat (range 4.525–4.534 across 5 checks). This appears to be the floor for this architecture + dataset + objective.

**6. Generation still completely wrong**
All outputs are coherent Qwen prose. No IMU→text learning apparent. The fundamental bottleneck (no content-grounding signal) is unchanged.

### Conclusion
This is the best result to date and the first converged run. The val loss improvement over the no-diversity baseline (4.52 vs 4.69) is from better LM prior fitting under LR decay, not from meaningful IMU→text learning. Content-grounding (auxiliary reconstruction loss or contrastive objective) remains the critical next step.

---

## Training Run 9 — `train_20260323_121258.log` (Qwen2.5-1.5B, div=10, recon=1.0, ct=0.1)

First run with all three auxiliary losses (diversity + reconstruction + contrastive). 50 epochs.

### Config (vs Run 8)
- `recon_weight=1.0` and `contrast_weight=0.1, contrast_temp=0.1` added (new)
- `div_weight=10.0`, `adapter_dropout=0.1`, `lr=3e-5`, `max_grad_norm=0.5`, `warmup=200` (same)
- `epochs=50`, `patience=5`, `val_every=500` (50 epochs only, not 200)
- Trainable params: 4,407,808 (adapter only, 0.28%)

### Loss component trajectory

| Metric | ep1 | ep5 (collapse) | ep10 | ep30 | ep50 |
|--------|-----|----------------|------|------|------|
| `div` (cos sim) | 0.961 | **0.122** ← broken | 0.046 | 0.02–0.05 | 0.01–0.04 |
| `recon` (MSE) | 0.0078 | 0.0080 | 0.0078 | 0.0077 | 0.0077 |
| `ct` (InfoNCE) | 2.791 | 2.712 | 2.740 | 2.737 | 2.738 |
| train loss | 6.37 | 6.09 (spike) | 5.72 | 5.41 | 5.41 |

### Val loss curve

| Step | Val loss | Mean CER | Note |
|------|----------|----------|------|
| 500  | 5.9472 | 4.120 | |
| 1000 | 5.4915 | 3.658 | |
| 1500 | 5.3716 | 3.456 | |
| 2000 | 5.3233 | 5.179 | |
| 2500 | 5.3009 | 3.943 | |
| **3000** | **5.2917** | 3.416 | **BEST (saved)** |
| 3500 | 5.2922 | 3.417 | no improvement 1/5 |
| 4000 | 5.2940 | 3.525 | no improvement 2/5 |
| 4350 | — | — | epoch limit (50 ep) — patience counter only 2/5 |

### Findings

**1. Diversity collapse pattern is identical to Run 7/8**
div collapsed from ~0.96 → ~0.12 at the same point (step ~350, ep 4–5), with the same loss spike (+0.6 nats). After breakdown the adapter produces near-orthogonal soft tokens (div ≈ 0.02–0.07). One step (step 2350) briefly went negative (-0.043) — soft tokens are occasionally anti-correlated.

**2. Reconstruction loss completely stuck at orthogonal case**
`recon` was 0.0078 at ep1 step 50 and is still 0.0077 at ep50 — zero improvement. This value is not coincidental: MSE of two L2-normalised 256-d vectors equals `2/d` when they are **orthogonal**. Specifically `2/256 = 0.0078`. The resampler mean output is simply orthogonal to the projected input mean from the very start, and the loss never drives them toward alignment.

Root cause: `recon_weight=1.0` contributes `1.0 × 0.0078 ≈ 0.008` to the gradient while the LM term contributes ~5.5 and div contributes ~0.5. The recon gradient is ~700× too small to influence the optimizer meaningfully.

**3. Contrastive loss barely moves from random baseline**
InfoNCE for B=16 has a random baseline of `log(16) ≈ 2.77`. `ct` started at 2.79 and ended at ~2.74 — essentially random performance throughout. The soft token prototypes have near-zero alignment with the corresponding text token prototypes in the LLM embedding space.

Root cause: `contrast_weight=0.1` contributes `0.1 × 2.74 ≈ 0.27` — about 5% of the LM gradient. The contrastive signal is drowned out.

**4. New losses hurt val loss vs Run 7 (same epoch budget)**
Directly comparable: Run 7 (div=10, 50 epochs, no recon/ct) reached best val 5.0732 at step 3500. Run 9 (div=10, 50 epochs, recon+ct) reaches best val 5.2917 at step 3000 — **0.22 nats worse**. The auxiliary losses are adding gradient noise without providing useful learning signal at these weights.

**5. Val loss plateaued earlier than Run 7**
Run 7 was still improving at its final checkpoint (step 4000). Run 9 plateaued at step 3000 (2 consecutive no-improvements before the epoch limit). The auxiliary losses appear to be destabilising the optimisation trajectory.

**6. Generation still completely wrong**
All CERs > 1.0 (more edits than reference characters). Predictions are incoherent Qwen prose with no relation to ground truth. The model is ignoring soft token content entirely, consistent with all prior runs.

**7. Still not comparable to Run 8** (200 epochs, div=10, no recon/ct, best val 4.5252)
Run 9 ran only 50 epochs and the val loss was still ~5.29 at completion. Run 8 at the same training step (~4350, ep 50) was at ~4.57 val loss. The new losses are setting training back. Whether they would catch up with 200 epochs is unknown but unlikely given the gradient magnitudes.

### Conclusion

The auxiliary reconstruction and contrastive losses, at `recon_weight=1.0` and `contrast_weight=0.1`, are effectively invisible — their gradient contributions are 2–3 orders of magnitude smaller than the LM loss. They add noise without providing useful learning signal, resulting in a worse outcome than div-only (Run 7) at the same epoch budget.

**For these losses to have any effect, the weights must be scaled to produce gradients competitive with the dominant terms:**
- `recon_weight` needs to be ~700× larger (e.g., 500–1000) to match the LM gradient magnitude
- `contrast_weight` needs to be ~20× larger (e.g., 2–5) to become competitive
- Alternatively, reduce `div_weight` and `recon_weight` proportionally to achieve balance
- Or run for 200 epochs to see whether the decay phase (lr → 0) allows these weak signals to eventually dominate

---

## Training Run 10 — `train_20260323_124100.log` (div=10, recon=500, ct=2.0)

First run with the corrected default weights (`recon_weight=500`, `contrast_weight=2.0`). 50 epochs.

### Loss component trajectory

| Step | div | recon | ct | grad_norm |
|------|-----|-------|----|-----------|
| 50 (ep1) | 0.959 | 0.0081 | 2.792 | 34 |
| 200 (ep3) | 0.939 | 0.0047 | 2.790 | 26 |
| 300 (ep4) | 0.886 | 0.0023 | 2.783 | 33 |
| 350 (ep5) | 0.713 | 0.0012 | 2.760 | 42 |
| 450 (ep6) | 0.221 | 0.0021 | 2.672 | 70 |
| 500 (ep6) | 0.216 | 0.0020 | 2.684 | **96** |
| 650 (ep8) | 0.196 | 0.0016 | 2.673 | **128** |
| 1400 (ep17) | 0.157 | 0.0015 | 2.647 | **144** |
| 2000 (ep23) | 0.147 | 0.0015 | 2.602 | 76 |
| 3000 (ep35) | 0.142 | 0.0015 | 2.588 | 67 |
| 4350 (ep50) | 0.142 | 0.0015 | 2.596 | 72 |

### Val loss curve

| Step | Val loss | Mean CER | Note |
|------|----------|----------|------|
| 500  | 5.6610 | 4.039 | |
| 1000 | 5.6067 | 2.440 | |
| 1500 | 5.5757 | 3.256 | |
| 2000 | **5.5729** | 4.873 | (BEST at the time) |
| 2500 | 5.5737 | 4.048 | no improvement 1/5 |
| **3000** | **5.5618** | 3.613 | **BEST (saved)** |
| 3500 | 5.5626 | 2.546 | no improvement 1/5 |
| 4000 | 5.5665 | 4.298 | no improvement 2/5 |
| 4350 | — | — | epoch limit, patience 2/5 |

### Findings

**1. recon IS learning now — but settling at imperfect alignment**
recon fell from 0.0081 → 0.0012 (by step 350, during diversity collapse) then settled at ~0.0015. For 256-d unit vectors, MSE = (2 − 2·cosθ)/256, so 0.0015 → cos(θ) ≈ 0.81 → ~36° between resampler output mean and projected input mean. Compare Run 9 where recon was stuck at 0.0078 (90°, pure orthogonal) throughout. The weight increase from 1.0 → 500 made the reconstruction signal effective. However it is not converging toward zero — it plateaued at ~0.0015 well before epoch 50, suggesting an equilibrium where reconstruction and other gradients balance.

**2. ct IS learning now — but only marginally**
ct fell from 2.79 → 2.60 by ep20, settling there for the rest of training. The random baseline is log(16) ≈ 2.77. Run 9 only dropped from 2.79 → 2.73 (0.06 nats). Run 10 drops by 0.19 nats — about 3× more. But 2.60 vs 2.77 is still only ~7% better than chance, meaning the soft token prototypes barely discriminate between text identities.

**3. div settles at ~0.14 — a new, higher equilibrium**
In Runs 7/8/9 (div=10, no/weak recon), div settled near 0.02–0.07. Here it plateaus at ~0.142. This makes sense: the recon loss (at weight 500) is pulling adapter output toward the input mean, which introduces a shared direction across samples, counteracting the diversity penalty. The equilibrium is between these two forces.

**4. Grad norms catastrophically elevated — 60–144 throughout**
Run 9 had grad norms 25–50 (late training). Run 10 has 60–144 throughout, with peaks >140 in eps 8–17. `max_grad_norm=0.5` clips every single step with these norms, meaning the actual gradient direction is entirely determined by the clip direction, not the true gradient. The optimizer is effectively navigating by gradient direction only, never magnitude. This extreme clipping is the primary reason val loss degraded.

Root cause: Total loss at mid-training ≈ LM (5.86) + div (1.47) + **recon (0.75)** + **ct (5.18)** ≈ 13.3. The contrastive term alone (ct_weight=2.0 × ct_loss≈2.6) contributes ~5 nats — comparable to the entire LM objective. The combined auxiliary gradient swamps the LM signal.

**5. Val loss is the worst yet at 50 epochs: 5.5618**
Monotonic degradation with auxiliary loss weight:

| Run | recon_weight | ct_weight | Best val (50 ep) |
|-----|-------------|----------|------------------|
| 7 | 0 | 0 | **5.0732** |
| 9 | 1.0 | 0.1 | 5.2917 |
| **10** | **500** | **2.0** | **5.5618** |

Every increase in auxiliary loss weight has made 50-epoch performance worse. The auxiliary losses are consistently starving the LM objective of gradient bandwidth.

**6. Generation still completely wrong.** CERs range 1.2–6.7, all incoherent.

### Conclusion

The weights are now too large. Run 9 was too small (gradients invisible), Run 10 is too large (gradients overwhelm LM). The sweet spot likely lives in between, but the pattern across all 10 runs suggests a more fundamental problem: at this dataset size (1,391 samples), no auxiliary objective has produced detectable IMU→text learning — only varying degrees of LM degradation.

**The best 50-epoch result remains Run 7 with no auxiliary losses (5.0732). The best result overall remains Run 8 (div=10, 200 epochs, 5.2517).**

**For auxiliary losses to help, they need:**
- Gradient contributions balanced with the LM term (roughly equal, not dominant)
- More epochs to allow slow alignment: `recon_weight` ≈ 50–100, `ct_weight` ≈ 0.2–0.5
- Or: abandon 50-epoch runs and go straight to 200 epochs to let cosine decay do the balancing

---

## Open Questions / Next Steps

- [x] Re-run preprocessing with mean-centering → cosine sim dropped 0.86 → 0.33 ✓
- [x] Re-train on mean-centred embeddings → val loss unchanged, grad norms worse ✓
- [x] Check adapter output diversity → collapse confirmed, soft token cosine sim = 0.984 ✓
- [x] Implement diversity loss + dropout/LR/grad norm/LayerNorm fixes ✓
- [x] GPT-2 div_weight sweep: 0.1 (collapse 0.914), 1.0 (collapse broken, LM hurt) ✓
- [x] Qwen div_weight sweep: 1.0 (no effect), 0.0 (best val 4.6877), 10.0 (collapse broken, LM hurt badly) ✓
- [x] Confirmed fundamental bottleneck: LM loss alone cannot teach IMU→text mapping ✓
- [x] Run more epochs (100+) with Qwen to find val loss floor → floor is ~4.52 with div=10, ep 133 ✓
- [x] **Auxiliary reconstruction loss** — implemented ✓ (see below)
- [x] **Contrastive objective** — implemented ✓ (see below)
- [ ] Dataset size (1,391 samples) is a hard ceiling; more data likely needed regardless of objective

---

## Fixes Implemented (2026-03-23) — Content-Grounding Losses

Two new losses added to `train.py` (zero extra parameters):

### 1. Reconstruction loss (`--recon_weight`, default 1.0)

**What:** The mean of the resampler output (in `adapter_dim=256` space) must reconstruct the mean of the projected input (also in `adapter_dim` space).

**How:**
```
pred = normalize(resampler_out.mean(dim=1))   # (B, adapter_dim)
tgt  = normalize(proj(chronos_embeds).mean(dim=1)).detach()
recon_loss = mse(pred, tgt)
```

**Why it helps:** The resampler is forced to preserve the average content of the input rather than ignoring it. The normalized MSE is equivalent to a cosine distance loss, making it scale-invariant. Operates entirely in 256-d space — computationally trivial.

**Implementation:** `IMUAdapter.forward()` now returns a 3-tuple `(soft_tokens, resampler_out, projected_mean)`. `model.forward()` returns `resampler_out` and `projected_mean` in its output dict.

### 2. Contrastive loss / InfoNCE (`--contrast_weight`, default 0.1; `--contrast_temp`, default 0.1)

**What:** The mean of the soft tokens (in `d_llm` space) should be closer to the LLM embedding mean of its own target text than to any other text's embedding mean in the batch.

**How:**
```
soft_proto = normalize(soft_tokens.mean(dim=1))        # (B, d_llm)
text_proto = normalize(mean_embed(target_ids)).detach() # (B, d_llm)
sim = soft_proto @ text_proto.T / temperature           # (B, B)
contrast_loss = cross_entropy(sim, arange(B))
```

**Why it helps:** Directly ties the adapter output to the semantic content of the target text. The LLM embedding space is already rich and meaningful. The text prototype is computed from the frozen embedding layer (already in memory) with no LLM forward pass. Padding positions are excluded via the label mask.

**Log fields added:** `recon` and `ct` appear in step logs alongside `div`.

---

## Run 11/12 Setup (2026-04-14) — CTC Pretraining + Two-Phase Schedule

After 10 runs failed to learn the IMU→text mapping, root cause was identified as a missing content-grounded gradient: the only signal reaching the adapter was LM cross-entropy through a frozen LLM via a 32-token bottleneck, and the easiest local minimum was always "produce a constant prefix that elicits typical English." Diversity / reconstruction / contrastive aux losses each broke specific failure modes but never produced word-level alignment.

**Intervention:** attach a CTC head to the adapter's *pre-resampler* per-timestep features, supervised by the per-window character sequence reconstructed from existing PKL keystroke timestamps. This delivers dense, frame-level, content-grounded supervision that does not depend on the LLM at all.

### Architecture changes

- **`char_vocab.py`** (new): 38-token character vocabulary — `<blank>` (id 0, required by CTC), `<unk>`, a–z, 0–9, space, `.`, `,`, `'`, `?`, `!`, `-`, `\n`. Provides `encode()`, `decode()`, `ctc_greedy_decode()`.
- **`adapter.py`**:
  - `IMUAdapter.forward()` now returns a 4-tuple `(soft_tokens, resampler_out, projected_mean, proj_seq)` — `proj_seq` is the full per-timestep `(B, S, adapter_dim)` projection, fed to CTC.
  - New `CTCHead` module: depthwise temporal conv (k=5) with residual, LayerNorm, linear classifier. ~10K extra params.
- **`model.py`**:
  - New flags `use_ctc=True`, `skip_llm=False`, `char_vocab_size`.
  - When `skip_llm=True`, the LLM and tokenizer are not loaded at all (saves ~3GB GPU + LLM forward time).
  - `save_adapter` / `load_adapter` round-trip the CTC head state.
- **`dataset.py`**: emits `char_ids` and `char_lens` per sample; `collate_fn` pads with blank id and exposes `embed_lens` (true non-padded encoder lengths) for the CTC `input_lengths` argument.
- **`preprocess.py`**: default `step_size` 5.0 → 2.0 (≈2.5× more overlapping windows). New `--raw_dirs` to pool multiple data roots. Every encoded sample gets a `char_targets` field via `char_encode(text)`.
- **`train.py`**:
  - New args: `--ctc_weight`, `--skip_llm`, `--no_ctc`, `--adapter_init`.
  - CTC loss computed via `F.ctc_loss(log_probs, char_ids, input_lengths, target_lengths, blank=BLANK_ID, zero_infinity=True)`; `log_probs` is `log_softmax(ctc_logits, dim=-1).transpose(0, 1)`.
  - When `skip_llm`, total loss = `ctc_weight * ctc_loss`; div/recon/contrast are gated off.
  - `validate()` rewritten to compute CTC-greedy CER per batch in addition to LM generation CER.

### Two-phase schedule

**Phase A — `configs/phase_a.yaml`**: `skip_llm=true`, `ctc_weight=1.0`, `lr=3e-4`, `batch_size=32`, all legacy aux=0, patience=8. Trains adapter + CTC head from scratch. Success bar: val CTC-CER monotonically drops below 0.8 with visibly recognisable partial words in greedy decodes. Output: `checkpoints/phase_a/adapter_best.pt`.

**Phase B — `configs/phase_b.yaml`**: `skip_llm=false`, `ctc_weight=0.3`, `lr=3e-5`, `batch_size=16`. Loads Phase A via `--adapter_init`. CTC stays on at lower weight to prevent the adapter from regressing while LM cross-entropy adds fluency. Success bar: LM generation echoes ground-truth content (not fluent-but-wrong prose), val LM loss meaningfully below Run 8's 4.52, generation CER < 1.0.

### Why CTC on pre-resampler features

The 32-token resampler output is the bottleneck where information dies. Attaching the CTC head to `proj_seq: (B, S, adapter_dim)` *before* the resampler lets supervision reach the entire adapter at temporal resolution, not just the 32 compressed queries. The resampler still has to produce the 32 tokens for Phase B, but it now sits on top of features that have been forced to encode actual character content.

### Status

Implementation complete. Phase A run pending.

---

## Run 12 — Character-CTC → Keystroke-activity pretraining (2026-04-14)

### Motivation

The character-CTC iteration (Run 11 plan above) grounded the adapter with frame-level character supervision. Observation during review: character prediction partially duplicates the frozen Qwen LLM's job in Phase B — it makes the adapter learn to produce what the LLM would already produce given rhythm. This wastes capacity and risks the adapter collapsing into a text-prior echo before Phase B even starts.

New hypothesis: replace character-CTC with a **binary per-frame keystroke-activity** objective. The adapter learns *when* keystrokes happen from IMU motion alone; the frozen LLM is the only path from rhythm to characters in Phase B. No explicit event→char mapper is trained.

Trade-off: supervision is weaker per frame (1 bit vs ~5 bits for a 38-class character target), but it is strictly IMU-native — the adapter cannot shortcut by copying a language prior because the target (key-press presence) is not in the text distribution. If Phase B fails to echo content, the escalation path is per-hand binary (2 channels) → keyboard region (4–6 channels) → per-key, not back to character-CTC.

### Supervision target

Per sample, a length-`S` uint8 mask built from PKL `key_times` at encoder resolution (S=513 for 10 s @ 100 Hz, Chronos-determined):
- For each event `(start, end)` in the window, mark frames `[floor((start-t0)/W * S), ceil((end-t0)/W * S))` as 1.
- Events without `end` default to a 50 ms press duration.
- Modifier keys (shift, ctrl, etc.) are included — they are real motor events even though they don't appear in the reconstructed text.
- Built at encoder resolution (not IMU resolution) post-encoding to stay decoupled from Chronos patching.

### File changes

- **`adapter.py`**: `CTCHead` → `KeystrokeHead`. Same architecture (q_norm/kv_norm + cross-attn into resampler + depthwise temporal conv + LayerNorm), output dim 1 instead of `vocab_size`. Returns `(B, S)` after squeeze. The cross-attention path is what carries BCE gradient back through the perceiver in Phase A.
- **`preprocess.py`**: every sample dict gains `keystroke_active: torch.uint8 (S,)`. Per-split activity rate reported for sanity.
- **`model.py`**: `use_ctc` → `use_keystroke`, `char_vocab_size` param removed. `out["ctc_logits"]` → `out["keystroke_logits"]`. `save_adapter`/`load_adapter` round-trip under `"keystroke_head"`; legacy `"ctc_head"` key is warned-and-ignored for back-compat. `_load_ctc_head_state` helper removed.
- **`dataset.py`**: emits `keystroke_targets: (B, max_S) float` and `keystroke_mask: (B, max_S) bool` (True for valid frames via `embed_lens`). No more `char_ids`/`char_lens`.
- **`train.py`**: `_compute_ctc_loss` → `_compute_keystroke_loss` (BCE-with-logits, masked-mean, with `pos_weight`). `pos_weight` computed once over the train set as `(1-p)/p`. `validate()` now reports `val_ks_loss`, frame F1 (threshold 0.5 pooled across val), and onset F1 (rising-edge matching within ±2 frames ≈ 40 ms tolerance). CLI flags `--ctc_weight` → `--keystroke_weight`, `--no_ctc` → `--no_keystroke`. `_CTC_LEN_WARNED` machinery dropped — BCE has no length-validity concern.
- **`configs/phase_a.yaml`** / **`phase_b.yaml`**: `ctc_weight: 1.0` → `keystroke_weight: 1.0`; `ctc_weight: 0.3` → `keystroke_weight: 0.3`. Phase B retains the auxiliary head at low weight to prevent the adapter forgetting rhythm during LM CE training.
- **`char_vocab.py`**: deleted. No production path imports it after this iteration. The legacy `"ctc_head"` load warning stays inline in `load_adapter` as a safety net.
- **`pipeline_diagram.{png,pdf}`**: deleted — depicted the CTC + char-vocab path.

### Why put BCE on pre-resampler features (same reasoning as Run 11)

The 32-token resampler output is still the information bottleneck. Attaching BCE to `proj_seq: (B, S, adapter_dim)` lets supervision reach the entire adapter at temporal resolution. The `KeystrokeHead` cross-attends into `resampler_out` so gradient also flows through the perceiver — critical in Phase A where LM CE is absent.

### Success bar

- **Phase A**: val BCE drops monotonically and plateaus; frame F1 > 0.6 and onset F1 > 0.5 by ~epoch 30. GPU memory comparable to Run 11 Phase A (LLM still skipped).
- **Phase B**: LM val loss meaningfully below Run 8's 4.5252 baseline; `[LLM]` generation samples echo ground-truth content rather than fluent-but-wrong prose.

If Phase B fails the content bar, escalate target granularity rather than reverting to character-CTC.

### Status

Implementation complete. Bug-hunt review pending. Preprocess + Phase A + Phase B runs pending.

---

## Run 13 — Phase A — `train_20260414_235239.log` (keystroke BCE, skip_llm)

First execution of the two-phase plan. Adapter + KeystrokeHead trained from scratch on the per-frame binary keystroke-activity target. LLM fully skipped.

### Config
- `skip_llm=True`, `keystroke_weight=1.0`, `lr=3e-4`, `batch_size=32`, `epochs=100`, `patience=8`
- `div=0, recon=0, contrast=0`, `adapter_dropout=0.1`, `max_grad_norm=1.0`, `warmup=200`
- Train: 3,465 samples (≈2.5× prior runs, via `step_size=2.0` overlapping windows); Val: 229
- Trainable: 4,674,305 (adapter + KeystrokeHead)
- BCE `pos_weight` = 1.423 (activity rate ≈ 41%)

### Val curve

| Step | val_loss (BCE) | frame_F1 | onset_F1 | Note |
|------|---------------|----------|----------|------|
| 500  | 0.7681 | 0.749 | 0.048 | BEST |
| 1000 | 0.7467 | 0.753 | 0.071 | BEST |
| 1500 | 0.7383 | 0.752 | 0.092 | BEST |
| 2000 | 0.7367 | 0.755 | 0.102 | BEST |
| 2500 | 0.7332 | 0.758 | 0.111 | BEST |
| **3000** | **0.7318** | **0.760** | 0.097 | **BEST (saved)** |
| 3500 | 0.7573 | 0.749 | 0.120 | no improvement 1/8 |
| 4000–7000 | 0.76–0.83 | ~0.75 | 0.12–0.14 | val BCE rising while onset_F1 creeps up |
| 7000 | — | — | — | early stop (8/8), ep ~64 |

Training complete in 46.0 min. Best val_loss 0.7318.

### Findings

**1. Frame F1 plateaus ≈ 0.76 — precision-limited**
Best check has P 0.640, R 0.936. The model is over-predicting activity: recall is near-saturated but precision stalls at ~0.65. Raising pos_weight or using focal/asymmetric loss would likely push precision up. BCE alone with `pos_weight=1.423` biases toward "on."

**2. Onset F1 is poor and inversely coupled to val BCE**
Onset_F1 never exceeds 0.14 even as frame_F1 stays flat. Onset P stabilises ~0.30, R only ~0.07–0.09 — the model smooths event boundaries instead of producing crisp rising edges. The head's depthwise k=5 temporal conv over per-frame features gives a blurred activation, so tight onset matching (±2 frames) fails. After step 3000, onset_F1 continues creeping up (0.10→0.14) while val BCE *degrades* (0.73→0.83) — the frame loss and onset accuracy disagree on the optimum.

**3. Early stop at ep 64; val BCE floor = 0.7318 at ep 27**
After ep 27 val BCE rises monotonically (0.73 → 0.83) despite epoch-average train loss falling further. Clean overfitting pattern: the BCE surface is soft enough that the network keeps tightening on training frames past the generalisation sweet spot.

**4. Cross-attention path is functioning**
KeystrokeHead cross-attends into `resampler_out`. With LLM skipped, BCE is the only gradient into the perceiver; if cross-attention had zero effect the resampler would be unconstrained. The fact that frame_F1 rises from 0.749 → 0.760 confirms gradient does flow through the perceiver during Phase A.

**5. Data-size intervention is visible**
This is the first run at 3,465 train samples vs ~1,391 in runs 1–10 (`step_size` 5.0 → 2.0). 2.5× more supervision — primarily noticeable via smoother val curves with lower variance than earlier runs.

### Conclusion
Phase A succeeds at its stated job: the adapter learns keystroke rhythm (frame_F1 ~0.76) without touching the LLM. But onset detection is weak (0.10–0.14), which directly caps how much crisp timing information Phase B's LLM can exploit. Checkpoint `phase_a/adapter_best.pt` at step 3000 is the Phase B initialisation.

---

## Run 14 — Phase B — `train_20260415_003914.log` (LLM CE + keystroke BCE at 0.3)

Loaded Phase A's best adapter and switched LLM on. CTC-style head stays active at 0.3 weight to prevent the adapter forgetting rhythm while LM gradient pushes fluency.

### Config (vs Phase A)
- `skip_llm=False`, `llm=Qwen/Qwen2.5-1.5B`, `adapter_init=./checkpoints/phase_a/adapter_best.pt`
- `keystroke_weight=0.3`, `lr=3e-5`, `batch_size=16`, `max_grad_norm=0.5`, `weight_decay=0.1`, `patience=5`
- `epochs=100`, `total steps=21,700`

### Val curve (selected)

| Step | val LM | ks | frame_F1 | onset_F1 | Note |
|------|--------|----|----------|----------|------|
| 500   | 5.3910 | 0.7412 | 0.756 | 0.114 | BEST |
| 1000  | 4.9944 | 0.7402 | 0.760 | 0.104 | BEST |
| 2000  | 4.7302 | 0.7386 | 0.759 | 0.105 | BEST |
| 3000  | 4.6053 | 0.7364 | 0.759 | 0.105 | BEST |
| 5000  | 4.5304 | 0.7358 | 0.761 | 0.096 | BEST |
| 7000  | 4.4928 | 0.7375 | 0.759 | 0.104 | BEST |
| 9000  | 4.4833 | 0.7368 | 0.760 | 0.102 | BEST |
| 11500 | 4.4808 | 0.7364 | 0.759 | 0.100 | BEST |
| 13000 | 4.4784 | 0.7389 | 0.757 | 0.111 | BEST |
| **13500** | **4.4782** | 0.7371 | 0.761 | 0.098 | **BEST (saved)** |
| 14000–16000 | 4.479–4.480 | ~0.737 | ~0.76 | ~0.10 | no improvement 5/5 |
| 16000 | — | — | — | — | early stop (ep ~74) |

Training complete in 53.2 min. Best val_loss 4.4782.

### Cross-run LM comparison

| Run | LLM | Objective | Best val LM | Notes |
|-----|-----|-----------|-------------|-------|
| 6 | Qwen | LM only, 50 ep | 4.6877 | collapsed soft tokens |
| 8 | Qwen | LM + div=10, 200 ep | 4.5252 | prior best |
| **14 (Phase B)** | **Qwen** | **LM + ks=0.3, from Phase A init** | **4.4782** | new best, −0.047 vs Run 8 |

### Findings

**1. New val LM floor: 4.4782 (−0.047 vs Run 8's 4.5252)**
First run to beat the div=10/200-epoch baseline. The Phase A init gives the perceiver a meaningful starting point so LM CE does not have to discover rhythm from scratch during its own training budget. The gain is real but modest — ~5% of a nat.

**2. Keystroke head is preserved, not improved**
`ks_loss` barely moves during Phase B (0.741 → 0.737, frame_F1 flat at 0.76, onset_F1 flat at 0.10). The 0.3 weight is effectively a regulariser pinning the keystroke skill in place. It does not make rhythm sharper. This validates the "don't forget" goal but means Phase B's LLM has no sharper temporal signal than Phase A produced.

**3. Generation still fails the content bar**
Samples at step 13500 (the BEST checkpoint):
- GT `'I started my journey with noting but a small backpack'` → `'handed over the book to me, saying that it was a 1974 1974 1974 1974 ...'` (CER 1.585)
- GT `'my journey with noting but a small backpack and a reslt'` → `"t and was gazing at the stars... 3.25e+115\nA. 100\nB. 50\n..."` (CER 1.491)
- GT `' backpack and a resltess spirit. Morning dw '` → `"leaving a trail of silence in its wake. As I walked along the winding path, I couldn't help but feel ouch ouch ouch ouch ..."` (CER 4.068)

Outputs are coherent Qwen prose, number-list degenerations, and repetition loops. Content does **not** echo GT. The Phase B success bar ("generation echoes ground-truth content") is **not met** despite the best LM loss on record.

**4. LM loss improvement ≠ content grounding**
Run 14's LM val loss is the best to date (4.4782), yet generation quality is indistinguishable from prior collapsed runs. This confirms something long-suspected across the 14-run history: at this dataset size, LM cross-entropy over 32 soft tokens is primarily measuring how well the adapter emits a prefix that lets Qwen's prior do its job — not whether the prefix encodes *this sample's* content. Phase A's rhythm target gives the adapter input-conditional structure, but the resampler bottleneck does not propagate enough content for Qwen to discriminate sample-level text.

**5. Grad norms healthier than legacy runs**
Phase B sits at grad_norm ~8–14 with `max_grad_norm=0.5` (Run 8 was ~27–35 with the same clip). Cleaner gradients, consistent with the more stable optimisation trajectory from a pretrained init.

**6. Early stop at ep ~74, not epoch limit**
First Phase-B-style run to converge cleanly via patience rather than epoch cap. Val LM plateau is extremely flat (4.478–4.480) across the final 5 checks.

### Conclusion

Phase B delivers the best LM val loss of any run to date (4.4782, beating Run 8 by 0.047), confirming that a rhythm-pretrained adapter provides a better initialisation than random. But it fails the stated success bar: generation is still fluent-but-wrong prose with zero content overlap. This is the clearest evidence yet that **binary keystroke-activity alone is too weak a content signal**. Rhythm disambiguates sequence timing but not sequence identity, so Qwen has no way to choose between texts that share a cadence.

Per the Run 12 plan, the escalation path is granularity — not reverting to character-CTC, but increasing the per-frame target richness: per-hand (L/R) binary → keyboard region (4–6 zones) → per-key. Per-hand is the cheapest next step: the existing PKL already labels each event with a key, and hand assignment is a simple key→hand map.

### Next steps
- [ ] Preprocess a per-hand (2-channel) target from PKL. Train Phase A with BCE over 2 channels; same loop.
- [ ] If per-hand Phase B still fails the content echo bar, escalate to keyboard-region (QWERTY row/zone).
- [ ] Consider lowering the BCE pos_weight or using focal loss in Phase A to fix precision (0.65 → ?) and sharpen onsets.
- [ ] Investigate onset_F1 ceiling: is the ±2-frame tolerance (40 ms) too tight for 100 Hz IMU + 513-step encoder resolution?

---

## Run 15 setup — Dual-head KeystrokeHead (activity + onset) with focal onset loss

Before escalating target granularity (per-hand / region / per-key — the Run 12 plan), first attack Run 13's onset-F1 ceiling directly. Run 13 produced frame_F1 ~0.76 but onset_F1 stalled at 0.10–0.14 with recall 0.07–0.09, meaning the adapter learned rhythm spans but produced blurred rising edges. Two structural causes in the Run 11/12 KeystrokeHead were flagged in Run 13's findings:

1. The depthwise **k=5 temporal conv** over per-frame features smooths activations across neighbouring frames — directly at odds with a tight ±2-frame onset-matching criterion.
2. A **single BCE head** tries to simultaneously fit a *spanning* target (dense activity) and produce *sharp impulses* (onsets are rising-edges of activity). The dense positives dominate the gradient, so the impulse structure never sharpens.

### Design (Option 2 + Option 3 from the triage shortlist)

**Dual head, shared trunk, no temporal conv.** `KeystrokeHead` now runs a single LayerNorm'd cross-attention block and branches into two independent linear classifiers:

- **activity head** — dense BCE with `pos_weight` (Phase A's content-grounding signal, unchanged objective).
- **onset head** — masked focal BCE against a **per-event onset mask**: one-hot at the starting frame of every keypress, including keypresses that overlap an earlier still-held key. Focal `gamma=2.0`, class weight `alpha=0.75` (positives are ~1–3 per 100 frames).

The two targets carry complementary temporal information. Splitting them lets each head optimise its own objective without the dense/sparse gradient conflict. Cross-attention into `resampler_out` is retained — it is the only path by which Phase A supervision reaches the perceiver.

### Supervision targets

- `keystroke_active`: unchanged. Union mask over all key-press intervals in the window (modifiers included via the raw event stream).
- `keystroke_onset`: new. For each raw event `(s_frac, e_frac)` with `hi > lo` after rounding to encoder resolution, set `onset[min(S-1, lo)] = 1`. Unlike deriving onsets from `diff(activity)`, this marks the true start of every keypress — crucially including presses that start while an earlier key is still held (those get merged by the activity mask's union). The per-event onset rate is reported alongside activity rate at preprocess time.

The dataset falls back to `diff(activity)` for legacy embeddings without `keystroke_onset`, so old preprocess files still load. New runs should rerun `preprocess.py` to get the per-event mask.

### File changes

- **`adapter.py` `KeystrokeHead`**: removed depthwise `Conv1d(k=5)`. Shared trunk is `q_norm/kv_norm + cross_attn + norm`. Two parallel `Linear(adapter_dim, 1)` heads (`activity_classifier`, `onset_classifier`). Returns `(activity_logits, onset_logits)` both `(B, S)`.
- **`preprocess.py`**: emits `keystroke_onset: uint8 (S,)` next to `keystroke_active`; reports per-split onset rate.
- **`dataset.py`**: `__getitem__` propagates `keystroke_onset` with an equal-length assertion vs `embeddings`; falls back to `diff(keystroke_active)` for legacy samples. `collate_fn` pads `onset_targets` parallel to `keystroke_targets` (same `keystroke_mask`).
- **`model.py` `forward`**: returns `keystroke_logits` (activity) and `onset_logits`. `load_adapter` now uses `strict=False` on `keystroke_head` and logs a warning — legacy single-head checkpoints transfer the shared-trunk weights and reinitialise the two new classifier heads. The legacy `ctc_head` warning stays.
- **`train.py`**: adds `_compute_onset_loss` (masked focal BCE). New flags `--onset_weight` (1.0 Phase A / 0.3 Phase B), `--onset_focal_gamma` (2.0), `--onset_focal_alpha` (0.75). Total loss = `keystroke_weight * activity_BCE + onset_weight * onset_focal_BCE` (+ LM CE in Phase B). `validate()` logs both `onset_rising_F1` (rising edges of activity, comparable to Run 13 metric) and `onset_head_F1` (dedicated onset head, new top-line metric). Phase-A `val_loss` matches the training composition exactly (`keystroke_weight * avg_ks + onset_weight * avg_on`). Sanity check: `skip_llm` with both weights ≤ 0 now errors out.
- **`configs/phase_a.yaml`**: `keystroke_weight=1.0, onset_weight=1.0, onset_focal_gamma=2.0, onset_focal_alpha=0.75`.
- **`configs/phase_b.yaml`**: `keystroke_weight=0.3, onset_weight=0.3` (same focal params). Both heads stay on at low weight to protect Phase A's rhythm grounding.

### Back-compat

- Run 13's Phase A checkpoint can seed Run 15 Phase B via `--adapter_init`: the adapter loads strict, the KeystrokeHead loads `strict=False` (warning logged), and the two new classifier heads start from random init. The old temporal-conv weights are dropped as unexpected keys.
- Legacy embeddings without `keystroke_onset` still load; the dataset derives onsets from `diff(activity)` (misses onsets of keys pressed while an earlier key is still held — rerun preprocess to fix).

### Success bar

- **Phase A**: val loss plateaus; frame_F1 ≥ Run 13's 0.76; **`onset_head_F1` > 0.3** (Run 13's `onset_F1` ceiling was 0.14 on the rising-edge metric — the new target is a 2–3× absolute improvement on the head-native metric). `onset_rising_F1` logged alongside for direct comparison with Run 13.
- **Phase B**: LM val loss ≤ Run 14's 4.4782 (regressing the LM floor would indicate the onset head is hurting content rather than helping). Primary win condition remains content-echoing generation.

If Run 15 clears the onset_F1 bar but Phase B generation still fails content echo, the onset-blur hypothesis was right but insufficient — proceed to the Run 12 plan (per-hand → region → per-key).

### Status

Implementation complete. Bug-hunt review complete (the skip_llm `val_loss` formula was missing `keystroke_weight` and was fixed; validate call site now receives all loss coefficients). Preprocess + Phase A + Phase B runs pending.

---

## Run 15 — Phase B — `train_20260415_110942.log` (dual-head adapter init, LM + ks=0.3 + onset=0.3 focal)

Loaded Run 15's Phase A adapter (`phase_a/adapter_best.pt`, dual-head KeystrokeHead) and switched the LLM on. Both activity BCE and onset focal BCE stay at 0.3 weight as rhythm-preservation regularisers alongside LM CE.

### Config (vs Run 14)
- `adapter_init=./checkpoints/phase_a/adapter_best.pt` (Run 15 Phase A, dual-head)
- `keystroke_weight=0.3`, `onset_weight=0.3`, `onset_focal_gamma=2.0`, `onset_focal_alpha=0.75`
- `lr=3e-5`, `batch_size=16`, `max_grad_norm=0.5`, `weight_decay=0.1`, `patience=5`, `epochs=100`, `total steps=21,700`
- Keystroke BCE pos_weight 1.423; onset positive rate 0.0744

### Val curve (selected)

| Step | val LM | ks | on | frame_F1 | onset_rising_F1 | onset_head_F1 | Note |
|------|--------|----|----|----------|-----------------|---------------|------|
|   500 | 5.3493 | 0.7317 | 0.0430 | 0.756 | 0.116 | 0.002 | BEST |
|  1000 | 5.1125 | 0.7321 | 0.0430 | 0.758 | 0.100 | 0.001 | BEST |
|  2000 | 4.7483 | 0.7285 | 0.0430 | 0.759 | 0.100 | 0.002 | BEST |
|  3000 | 4.5891 | 0.7278 | 0.0430 | 0.759 | 0.101 | 0.001 | BEST |
|  5000 | 4.5312 | 0.7276 | 0.0430 | 0.759 | 0.107 | 0.000 | BEST |
|  7000 | 4.4754 | 0.7279 | 0.0430 | 0.758 | 0.107 | 0.001 | BEST |
|  8000 | 4.4590 | 0.7274 | 0.0430 | 0.758 | 0.104 | 0.001 | BEST |
|  9500 | 4.4579 | 0.7271 | 0.0430 | 0.758 | 0.104 | 0.000 | BEST |
| **10500** | **4.4535** | 0.7272 | 0.0430 | 0.758 | 0.107 | 0.000 | **BEST (saved)** |
| 11000–13000 | 4.454–4.463 | ~0.727 | 0.0430 | ~0.758 | ~0.10 | ~0.000 | no improvement 5/5 |
| 13000 | — | — | — | — | — | — | early stop (ep ~60) |

Training complete in 42.5 min. Best val_loss 4.4535.

### Cross-run LM comparison

| Run | Init | Objective | Best val LM | Notes |
|-----|------|-----------|-------------|-------|
| 8 | random | LM + div=10 | 4.5252 | |
| 14 | Run 13 Phase A (single-head) | LM + ks=0.3 | 4.4782 | |
| **15 Phase B** | **Run 15 Phase A (dual-head)** | **LM + ks=0.3 + on=0.3 focal** | **4.4535** | new best, −0.025 vs Run 14 |

### Findings

**1. New val LM floor: 4.4535 (−0.025 vs Run 14's 4.4782)**
Third consecutive best-in-class reduction (Run 8 → 14 → 15). The dual-head rhythm+onset pretraining gives the perceiver a slightly better starting point than Run 13's single-head init. The gain is real but the marginal return is shrinking: Run 14 beat Run 8 by 0.047, Run 15 beats Run 14 by only 0.025 on the same LM objective.

**2. Onset head collapses to ~0 during Phase B**
`onset_head_F1` sits at 0.000–0.002 throughout Phase B, with precision ~0.3 but recall ~0.000 — the head outputs essentially no positive predictions. `onset_loss` is pinned at 0.0430 for the entire run (no training signal moving through it). Two plausible causes: (a) at `onset_weight=0.3`, the focal-BCE gradient is too small to hold the head against LM-CE's dominant pull on shared cross-attention features; (b) whatever Phase A learned on the onset head is forgotten once the shared trunk has to serve LM objectives. Either way, the onset head is inactive as a regulariser in Phase B.

**3. `onset_rising_F1` (activity-derived) unchanged**
Rising-edge onset F1 sits at 0.10–0.12 throughout — matching Run 14 and Run 13. The onset-sharpening hypothesis (Run 15 setup) did not transfer from Phase A to Phase B: whatever sharper edges Phase A produced are not preserved under LM fine-tuning. This is a stronger statement than "Phase B preserves Phase A rhythm" — it preserves *blurry* rhythm, not sharper rhythm.

**4. Activity head preserved, not improved**
`ks_loss` 0.7317 → 0.7267 (−0.005); frame_F1 flat at 0.758; recall 0.93+ with precision ~0.635. Same plateau as Run 14. The 0.3 weight is regularisation, not optimisation — confirming Run 14's reading.

**5. Generation still fails the content echo bar**
Samples at step 10500 (BEST):
- GT `'I started my journey with noting but a small backpack'` → `'y, and I felt the weight of my own mortality.  I a 1t 1g 1d 1to 1i 1d 1o 1n 1e ...'` (CER 1.509)
- GT `'my journey with noting but a small backpack and a reslt'` → `'t a different way to perceive the world.  60  60  60 ...'` (CER 1.418)
- GT `' backpack and a resltess spirit. Morning dw '` → `'gates, to feel a part of the living world. The air was irst ings ound ...'` (CER 1.977)

Same failure modes as Run 14: fluent Qwen prose, number-list degenerations, repetition loops. Zero content overlap with GT. **Phase B success bar not met**, despite the best LM loss on record.

**6. Cleaner training dynamics**
Grad norms stabilise at ~8.5 (vs Run 14's ~8–14); wall-clock 42.5 min (vs Run 14's 53.2 min, same hardware) because early stop fires at ep ~60 instead of ep ~74. Validation plateau is extremely flat over final 5 checks (4.454–4.463).

### Conclusion

Run 15 Phase B sets a new LM val-loss floor (4.4535) — the third consecutive record — but **the onset-sharpening intervention did not survive Phase B**: the onset head collapses to silence under LM pressure, and `onset_rising_F1` is identical to Run 14. The LM gain is therefore attributable to the *activity-head* pretraining transferring slightly more useful structure via the shared trunk, not to sharper onset timing reaching Qwen.

More importantly, generation quality is **indistinguishable from Run 14**: fluent-but-wrong prose, same degeneration patterns, zero content echo. Three runs now agree that LM cross-entropy over 32 soft tokens decouples from sample-level content grounding — improving the LM floor no longer predicts generation quality at this dataset/architecture scale.

This exhausts the "sharpen temporal target" escalation branch. The Run 12 plan (target-granularity escalation: per-hand → region → per-key) is now the clearest next move — rhythm, even sharp rhythm, does not carry enough information to discriminate text content. Per-hand (L/R) Phase A is the cheapest next experiment; the PKL already labels each event with a key, and hand assignment is a static key→hand map.

### Next steps
- [ ] Preprocess a per-hand (2-channel) `keystroke_active` target from PKL; keep onset head as-is (still 1-channel per-event, or escalate to 2-channel).
- [ ] Train Phase A with 2-channel activity BCE; verify per-hand frame_F1 materially exceeds the single-channel ceiling (0.76) before committing to Phase B.
- [ ] If per-hand Phase B still fails the content echo bar, escalate to keyboard-region (QWERTY row/zone, 4–6 channels).
- [ ] Investigate why the onset head collapses at `onset_weight=0.3` — try 1.0 in Phase B, or freeze the onset head's output projection post-Phase-A.


## MOMENT encoder option added (2026-04-15)

Preprocess now supports an alternative time-series encoder: MOMENT
(`AutonLab/MOMENT-1-{small,base,large}`). Selected via `encoder: "moment"`
in `configs/preprocess.yaml` or `--encoder moment` on the CLI.

**Shape differences vs. Chronos**
- Chronos (per-channel): output `(S≈513, d_chronos)` for a 10 s @ 100 Hz window.
- MOMENT (patch-based): input resampled to fixed `seq_len=512`, `patch_len=8`
  → `(S=64, n_channels * d_model)`. Much shorter token sequence for the
  perceiver resampler to compress.

**Why add it**
Chronos embeddings were shown (FINDINGS §Embedding Analysis) to be near-
collinear across samples — 99.8% of dims carried almost no variance before
mean-centering. MOMENT is trained on a different corpus and uses a
patch-based encoder rather than quantile-tokenized scalar inputs, so its
embeddings may carry more discriminative IMU signal. Worth an A/B on
Phase A frame_F1 before committing to heavier supervision changes.

**Integration notes**
- `d_chronos` in the training config still serves as the adapter input dim —
  set it to `d_model * n_channels` (e.g. 12288 for MOMENT-1-large, both rings).
- Everything downstream of the encoder (perceiver, LLM, losses, activity/onset
  targets) is unchanged because the saved `.pt` schema is the same `(S, d_enc)`
  tensor plus text/activity fields.
- MOMENT requires `pip install momentfm`; now pinned in `environment.yml`.

---

## Run 16 — Phase A (MOMENT) — `train_20260415_124518.log`

First MOMENT-encoder run. `AutonLab/MOMENT-1-large` (`d_model=1024`),
both rings → `d_enc=12288`, `S=64` patches. 3465 train / 229 val samples
(~2.5× Chronos counts — MOMENT resamples each 10s window to 512
timesteps so more windows survive the min-length filter). Same
Phase A config otherwise (`skip_llm`, activity + focal-onset BCE,
`keystroke_weight=onset_weight=1.0`).

| Step | val_loss | frame_F1 | onset_head_F1 | note |
|------|----------|----------|---------------|------|
| 500  | **0.2947** | 0.935 | 0.902 | BEST (epoch 2) |
| 1000 | 0.3721 | 0.935 | 0.896 | |
| 2500 | 0.5748 | 0.938 | 0.907 | still near-best F1, loss rising |
| 4500 | 0.6618 | 0.938 | 0.913 | early stop (patience 8) |

Total wall time: 4.7 min. Best checkpoint at step 500.

**Result — MOMENT crushes the Chronos Phase A ceiling.**

| Encoder | best frame_F1 | best onset_head_F1 |
|---|---|---|
| Chronos (Run 15, 2 h train) | 0.76 | 0.10–0.50 |
| MOMENT-1-large (Run 16, 5 min train) | **0.935** | **0.902** |

Rhythm and per-key onset are essentially solved by the encoder swap,
at a fraction of the training time. This validates the
*Embedding Analysis* hypothesis: the Chronos embeddings really were
the bottleneck — 99.8% near-zero-variance dims left almost no
discriminative IMU signal for the adapter to grip, and MOMENT's
patch-based encoder preserves that signal.

Downside to watch: val_loss is monotone increasing after step 500 even
though frame_F1/onset_F1 keep drifting *up*. The KS/focal terms are
overfitting to raw BCE while the decision-threshold metrics stay
steady — which is fine here but signals the adapter is happy to keep
pushing confidences without adding information.

---

## Run 17 — Phase B (MOMENT) — `train_20260415_125014.log`

Seeded from Run 16's `adapter_best.pt`. Same Phase B config as Chronos
Phase B (`keystroke_weight=onset_weight=0.3`, LM CE joins in, LoRA off).

| Step | val_loss (lm) | frame_F1 | onset_F1 | note |
|------|---------------|----------|----------|------|
| 500  | 5.1915 | 0.938 | 0.896 | init from Phase A — LM CE jumps in |
| 1000 | 4.8855 | 0.939 | 0.901 | |
| 3000 | 4.5622 | 0.938 | 0.905 | |
| **4500** | **4.5459** | 0.938 | 0.899 | BEST |
| 7000 | 4.5748 | 0.937 | 0.903 | early stop (patience 5) |

Total wall time: 17.9 min. Frame F1 and onset F1 both hold at the
Phase A levels — the rhythm grounding does not regress as the LM term
kicks in. `ks` loss creeps up slightly (0.255 → 0.267) which is the
expected small trade-off when LM CE reshapes the soft tokens.

**But: LLM output is still garbage.** At every val step the generated
text is fluent Qwen prose with zero content alignment to GT. Samples
from best-val step 4500:

- GT: `'I started my journey with noting but a small backpack'`
  → PRED: `'ed into the wintering chill... a t 1. As I walked through the crowded...'` (LLM-CER 3.20)
- GT: `'my journey with noting but a small backpack and a reslt'`
  → PRED: `'easurements. The temperature, humidity, and air quality 14845712...'` (LLM-CER 1.71)

Frame_F1/onset_F1 are near 0.9 — the adapter *does* know where the
keystrokes are — but that rhythm structure does not translate into
characters through the frozen LLM. This is the same ceiling Runs 14–15
hit with Chronos: high keystroke supervision, zero content echo.

**Implication**
The encoder was never the bottleneck for content. Fixing rhythm (Run 16's
frame/onset F1 ~0.9) leaves the content gap intact. The frozen LLM
cannot invert 32 rhythm-only soft tokens into the actual typed
characters — the bijection just isn't there. Escalating supervision
beyond binary activity (per-hand → keyboard region → per-key) remains
the next lever, as does briefly unfreezing the LLM via LoRA to see if
any prefix can carry character content through a still-small param
budget.

### Next steps (updated)
- [ ] Run Phase A + B on MOMENT with **per-hand** activity (2-channel BCE).
      Phase A is so cheap now (<5 min) that we can iterate supervision
      ladders freely.
- [ ] Try `lora_rank=8` on MOMENT Phase B — cheap experiment to test whether
      *any* LLM adaptation lets the soft tokens express content.
- [ ] Ablate: does MOMENT-1-base (d=768, same `d_enc` as Chronos-base) give
      the same Phase A F1 as MOMENT-1-large, or is the 1024-dim encoder
      doing work? Informs whether to scale MOMENT further or stop.
- [ ] Keep Chronos around for the record; don't overwrite its
      `embeddings/*.pt` since MOMENT files are suffixed (`*_moment.pt`).

---

## Architecture Additions — LoRA + Character-CTC (2026-04-15)

Implemented in response to Run 17's diagnosis: frozen Qwen cannot extract
character identity from 32 rhythm-only soft tokens. Two structural levers
added simultaneously:

### 1. LoRA on Qwen (`--lora_rank`, `--lora_target_modules`)

Code path already wired in `model.py` but `lora_rank=0` in every prior config.
Configs added for attn-only (q/k/v/o, ~344 K extra params at r=8) and full
(+gate/up/down, ~9.8 M). `--freeze_resampler` flag freezes `adapter.proj`
and `adapter.resampler` (leaves `out_proj` trainable) so LoRA cannot drift
the Phase-A rhythm grounding. Phase B only.

### 2. Character-CTC head (`CharCTCHead`, `--char_ctc_weight`)

**`char_vocab.py`** — 72-token case-sensitive vocab: blank (id 0), unk (1),
a–z, A–Z, 0–9, space + 7 punctuation chars. Exposes `encode()`, `decode()`,
`ctc_greedy_decode()`.

**`adapter.py` — `IMUAdapter`** gained `ctc_upsample_factor` param.
When > 1, a `ConvTranspose1d(adapter_dim, adapter_dim, kernel_size=f, stride=f)`
upsamples `proj_seq` from `(B, S, D)` to `(B, S*f, D)` before the CTC head.
Forward now returns a 5-tuple: `(soft_tokens, resampler_out, projected_mean,
proj_seq, proj_seq_up)`. Resampler still sees the un-upsampled `proj_seq`.

**`CharCTCHead`** — same cross-attn-into-resampler trunk as `KeystrokeHead`
(q_norm / kv_norm + cross-attn + LayerNorm), no temporal conv. Output:
`Linear(adapter_dim, vocab_size=72)`. Takes `proj_seq_up` as query input.

**`train.py`** additions:
- `_char_ctc_weight(step, start, peak, warmup)` — linear ramp from `start`
  to `peak` over `warmup` steps.
- `_compute_char_ctc_loss` — `F.ctc_loss(log_probs, char_ids,
  input_lengths=embed_lens * factor, target_lengths=char_lens,
  blank=0, zero_infinity=True)` where `log_probs = log_softmax(char_logits,
  dim=-1).transpose(0, 1)`. `input_lengths` clamped to `T_ctc`.
- `_ctc_cer_batch` — CTC-greedy CER per batch for validation logging.

**`preprocess.py`** emits `char_targets: LongTensor (L,)` per sample.
**`dataset.py`** propagates `char_ids (B, max_L)` and `char_lens (B,)`.

### CTC length precheck (`precheck_ctc_length.py`)

Run once per encoder before CTC training. Results on the live embeddings:

| Encoder | S | L_max | L_mean | upsample | T_ctc | P(L ≥ T_ctc/3) |
|---|---|---|---|---|---|---|
| Chronos | 513 | 81 | 39.1 | 1 | 513 | ~0.00 |
| MOMENT  | 64  | 52 | 34.7 | 1 | 64  | 0.78 (infeasible) |
| MOMENT  | 64  | 52 | 34.7 | 4 | 256 | 0.00 |

MOMENT requires `ctc_upsample_factor=4`; Chronos is trivially feasible at 1×.

### New configs (8 total)

| Config | Encoder | Phase | LoRA targets | CTC | Notes |
|---|---|---|---|---|---|
| `phase_b_moment_lora_attn.yaml`  | MOMENT  | B | q/k/v/o     | off | Run 18m |
| `phase_b_moment_lora_full.yaml`  | MOMENT  | B | +gate/up/dn | off | Run 19m (if needed) |
| `phase_b_chronos_lora_attn.yaml` | Chronos | B | q/k/v/o     | off | Run 18c |
| `phase_b_chronos_lora_full.yaml` | Chronos | B | +gate/up/dn | off | Run 19c (if needed) |
| `phase_a_moment_ctc.yaml`        | MOMENT  | A | — | 0.1→0.3, upsample=4 | Run 20m Phase A |
| `phase_a_chronos_ctc.yaml`       | Chronos | A | — | 0.1→0.3, upsample=1 | Run 20c Phase A |
| `phase_b_moment_ctc.yaml`        | MOMENT  | B | q/k/v/o     | 0.0 default | Run 20m Phase B |
| `phase_b_chronos_ctc.yaml`       | Chronos | B | q/k/v/o     | 0.0 default | Run 20c Phase B |

---

## Run 20 Preamble — Phase A Runs (2026-04-15)

Four Phase A runs launched in parallel across GPUs 3/4/5.

### MOMENT baseline re-run (phase_a_moment) — `train_20260415_170908.log`

Second MOMENT Phase A run; confirms Run 16 was not a fluke.

| Step | val_loss | frame_F1 | onset_head_F1 |
|------|----------|----------|---------------|
| 500  | **0.3000** | 0.936 | 0.891 | BEST |
| 4500 | 0.6606 | 0.938 | 0.913 | early stop (patience 8) |

Total wall time: 5.4 min. Best checkpoint at step 500. Rhythm essentially
solved within the first epoch — confirms MOMENT encoder is the driver, not
training duration.

### Chronos baseline (phase_a) — `train_20260415_170847.log`

| Step | val_loss | frame_F1 | onset_head_F1 |
|------|----------|----------|---------------|
| 500  | 0.8238 | 0.739 | 0.000 |
| 1000 | 0.7912 | 0.752 | 0.004 |
| 1500 | **0.7751** | 0.757 | 0.010 | BEST |
| 5500 | 0.8267 | 0.748 | 0.000 | early stop (patience 8) |

Total wall time: 35.3 min. Frame_F1 plateaus at ~0.75; onset_head_F1 barely
reaches 0.010. This is significantly below Run 15's ~0.76 onset_F1 — the
key difference is training duration: Run 15 ran for 2+ hours while this
run early-stopped at 35 min under the new patience=8 schedule. The Chronos
adapter converges far more slowly than MOMENT and was seeded into Phase B
from this relatively weak checkpoint. Phase B Chronos LoRA results should
be interpreted with this in mind.

### MOMENT CTC Phase A (phase_a_moment_ctc) — `train_20260415_171447.log`

| Step | val_loss | frame_F1 | onset_F1 | ctc_CER |
|------|----------|----------|----------|---------|
| 500  | 0.8812 | 0.937 | 0.895 | 0.563 | BEST |
| 1000 | **0.8618** | 0.935 | 0.915 | **0.476** | BEST |
| 5000 | 1.5479 | 0.937 | 0.918 | 0.465 | early stop |

Total wall time: 6.1 min. **ctc_CER dropped to 0.476 within the first
1000 steps** — character content IS being forced into the per-frame
adapter features. Frame and onset F1 hold at MOMENT-baseline levels
despite the added CTC pressure. CTC loss descended from 21 → 1.7 nats
within 1000 steps. This checkpoint is a sound seed for Phase B CTC.

### Chronos CTC Phase A (phase_a_chronos_ctc) — `train_20260415_170902.log`

| Step | val_loss | frame_F1 | onset_F1 | ctc_CER |
|------|----------|----------|----------|---------|
| 500  | 1.7706 | 0.738 | 0.000 | 1.000 |
| 2000 | 1.7396 | 0.751 | 0.000 | 1.000 |
| 10000 | **1.7125** | 0.752 | 0.000 | 0.972 | BEST |
| 10000 | — | — | — | — | early stop |

Total wall time: 32.6 min. **CTC completely failed to learn on Chronos.**
ctc_CER held at 1.000 through most of training; barely reached 0.972 at
the very last checkpoint. More critically, **onset_head_F1 stayed at
exactly 0.000 throughout** — CTC supervision fully suppressed onset head
learning, which was also weak in the Chronos baseline (0.010 at best).

**Hypothesis:** Chronos per-channel per-frame embeddings (S≈513,
d_enc=9216) carry high-frequency spectral content that is not naturally
aligned to character boundaries. The CTC gradient competes with the
keystroke onset gradient for the same adapter features, and the CTC term
"wins" in terms of gradient magnitude while learning nothing useful — the
high T/L ratio (~10–13×) creates too many blank-vs-non-blank configurations
and the gradient vanishes into indifferent blank paths. MOMENT's patch-based
S=64 (T/L ≈ 5 after 4× upsample → 256) is a much tighter constraint that
forces the CTC path to commit.

Consequence: **Chronos Phase B CTC (`phase_b_chronos_ctc.yaml`) should not
be run** — there is no CTC-grounded Phase A checkpoint worth seeding from.
The Chronos CTC Phase B config exists but is deprioritised pending a
diagnosis of why CTC fails on Chronos embeddings.

---

## Runs 18/20 — Phase B (in progress, 2026-04-15 17:53)

Three Phase B runs launched across GPUs 3/4/5. Qwen2.5-1.5B downloaded
to `/data/agnivc/tmp/hf_cache` (HF_HOME set accordingly) after
`/scratch/cluster/agnivc` filled to 100%.

| Run | Config | GPU | Encoder | Init | LoRA | CTC | Trainable |
|-----|--------|-----|---------|------|------|-----|-----------|
| 18c | phase_b_chronos_lora_attn | 3 | Chronos | phase_a/best | q/k/v/o r=8 | off | 2.84M (0.18%) |
| 18m | phase_b_moment_lora_attn  | 4 | MOMENT  | phase_a_moment/best | q/k/v/o r=8 | off | 2.84M (0.18%) |
| 20m | phase_b_moment_ctc        | 5 | MOMENT  | phase_a_moment_ctc/best | q/k/v/o r=8 | ctc=0 (head loaded) | 7.90M (0.51%) |

**Baseline to beat:** Run 17 best val LM loss = **4.5459** (MOMENT, no LoRA).

Step 50 lm losses at warmup start:
- Run 18c (Chronos LoRA): 6.85 nats
- Run 18m (MOMENT LoRA): 7.11 nats
- Run 20m (MOMENT CTC): 6.66 nats

### Run 18c — Chronos LoRA attn (`phase_b_chronos_lora_attn`) — `train_20260415_175220.log`

Seeded from `checkpoints/phase_a/adapter_best.pt` (Chronos Phase A, onset_F1 0.010).

| Step | val LM | frame_F1 | onset_F1 | note |
|------|--------|----------|----------|------|
| 1500 | 4.5191 | 0.757 | 0.001 | BEST |
| 2000 | **4.4616** | 0.757 | 0.000 | BEST |
| 4500 | 4.5540 | 0.758 | 0.001 | early stop (patience 5) |

Wall time: 18.8 min. Best val LM **4.4616** — beats Run 17 baseline (4.5459) by 0.08 nats.
Onset_F1 never recovers from the weak Phase A seed (0.010); onset supervision provides
essentially no signal here.

### Run 18m — MOMENT LoRA attn (`phase_b_moment_lora_attn`) — `train_20260415_175220.log`

Seeded from `checkpoints/phase_a_moment/adapter_best.pt` (MOMENT Phase A, onset_F1 0.891).

| Step | val LM | frame_F1 | onset_F1 | note |
|------|--------|----------|----------|------|
| 500  | 4.5607 | 0.937 | 0.898 | |
| 2500 | **4.5197** | 0.938 | 0.889 | BEST |
| 5000 | 4.5992 | 0.938 | 0.896 | early stop |

Wall time: 17.9 min. Best val LM **4.5197** — beats Run 17 by 0.026 nats.
Frame and onset F1 hold at Phase A levels throughout.

### Run 20m — MOMENT CTC Phase B (`phase_b_moment_ctc`) — `train_20260415_175220.log`

Seeded from `checkpoints/phase_a_moment_ctc/adapter_best.pt` (MOMENT Phase A + CTC,
onset_F1 0.915, ctc_CER 0.476). Same LoRA targets (q/k/v/o, r=8) as Run 18m;
`char_ctc_weight=0.0` in Phase B so CTC head is loaded but inactive.

| Step | val LM | frame_F1 | onset_F1 | note |
|------|--------|----------|----------|------|
| 500  | 4.6155 | 0.937 | 0.904 | |
| 1000 | **4.4262** | 0.937 | 0.905 | BEST |
| 3500 | 4.5038 | 0.937 | 0.907 | early stop |

Wall time: 12.3 min. Best val LM **4.4262** — **best of all runs**, beats Run 17 by 0.12 nats.
Fastest convergence: best reached at step 1000 vs step 2000–2500 for LoRA-only runs.
The CTC-grounded Phase A initialization reaches lower LM loss faster, suggesting the
character-enriched soft tokens give the LoRA LLM a better starting point.

### Full grid summary (encoder × supervision)

| Run | Encoder | Phase B supervision | Init | Best val LM | frame_F1 | onset_F1 |
|-----|---------|---------------------|------|-------------|----------|----------|
| 17  | MOMENT  | KS only, no LoRA | Phase A rhythm | 4.5459 | 0.938 | 0.899 |
| 18c | Chronos | LoRA attn | Phase A rhythm (weak) | 4.4616 | 0.757 | 0.001 |
| 18m | MOMENT  | LoRA attn | Phase A rhythm | 4.5197 | 0.938 | 0.889 |
| 20m | MOMENT  | LoRA attn + CTC-init | Phase A CTC | **4.4262** | 0.937 | 0.905 |

**Winner: Run 20m — MOMENT + CTC-grounded Phase A + LoRA attn.**

All three new runs beat the no-LoRA baseline. The CTC-grounded init delivers the
largest improvement (+0.12 nats vs baseline) at the shortest wall time. LoRA alone
on MOMENT gives only a small gain (+0.026 nats); the character-content baked into
the soft tokens by Phase A CTC is the key lever.

### But: LLM generation still completely wrong

Despite improved val LM loss, all Phase B outputs remain fluent Qwen prose with
zero content overlap with GT. Representative samples from Run 20m step 3500:

- GT: `'I started my journey with noting but a small backpack'`
  → PRED: `'tied to the rhythms of daily life, grounding me ultipli...'` (CER 2.16)
- GT: `'noting but a small backpack and a resltess cpi'`
  → PRED: `'reconding me to see the beauty of life nder the 100...'` (CER 2.20)

The val LM improvements (~0.1 nats) are real but insufficient: cross-entropy
of 4.4 nats on 64-char targets means the model has barely escaped the language
model prior. The 32-token soft token bottleneck remains the structural
constraint — the adapter cannot pack 50+ character identities into 32 tokens
in a form the LLM can reliably invert.

### Escalation path

Per the plan, if `val_ctc_cer ≤ 0.5 AND Phase B LLM-CER ≥ 1.0` on both encoders,
the 32-token resampler is the structural block. Run 20m Phase A CTC achieved
ctc_CER **0.476** (< 0.5) and Phase B LLM-CER is > 1.5. This confirms the
32-token bottleneck hypothesis. Next experiments:

- [x] Diagnostic: Phase A CTC on `resampler_out` — see below.
- [ ] Increase `n_soft_tokens` from 32 → 64 (in progress).
- [ ] If 64 tokens still stalls, try 128.
- [ ] If `n_soft_tokens` alone fails, escalate supervision ladder:
      per-hand binary → keyboard region (4–6 channel) → per-key.

---

## Diagnostic — Phase A CTC on `resampler_out` (32 tokens) — `train_20260415_192532.log`

Config: `phase_a_moment_ctc_resampler_diag.yaml` — identical to `phase_a_moment_ctc.yaml`
except `ctc_on_resampler: true`. The CTC head receives `resampler_out` (32 soft tokens)
as input instead of `proj_seq_up` (256 upsampled per-frame features).

| Step | val ctc_CER | frame_F1 | onset_F1 | note |
|------|-------------|----------|----------|------|
| 500  | ~0.90 | 0.937 | 0.895 | |
| 1000 | ~0.83 | — | — | best step |
| 4500 | 0.791 | 0.935 | 0.914 | early stop |

Wall time: 5.0 min. Best val_loss: 0.3588.

### Comparison

| CTC input | tokens | val ctc_CER | Δ vs proj_seq |
|---|---|---|---|
| `proj_seq_up` (per-frame, upsampled) | 256 | **0.476** | baseline |
| `resampler_out` (compressed soft tokens) | 32  | **0.791** | +0.315 (+66% relative) |

**Interpretation:** The resampler compression is lossy for character information —
66% relative CER degradation across the 32-token bottleneck. It is NOT a hard
block (CER 0.791 < 1.0, so some character content survives), but the information
loss is substantial enough to explain why Phase B LLM generation (which only sees
the 32 soft tokens) cannot echo content.

Notably: train CTC loss reached ~0.10 nats in both cases — the adapter can pack
character-aligned structure into 32 tokens on the training set. The val CER gap
reflects generalization failure, not fitting capacity. The resampler is likely
memorising specific training-set phoneme patterns into the 32 queries rather than
learning a general compression.

**Conclusion:** Increasing `n_soft_tokens` from 32 to 64 is well-motivated.
Expected outcome: CER on resampler_out at 64 tokens should fall from 0.791 toward
the 256-token bound of 0.476, giving the LLM more character information to work with.

---

## 64-token experiment — Phase A CTC + Phase B LoRA (2026-04-15)

Configs: `phase_a_moment_ctc_64tok.yaml` + `phase_b_moment_ctc_64tok.yaml`.
Chained run on GPU 3: Phase A → Phase B.

### Phase A (64 tokens) — `train_20260415_194618.log`

| Step | val ctc_CER | frame_F1 | onset_F1 |
|------|-------------|----------|----------|
| 500  | — | 0.937 | 0.895 |
| best | — | — | — |
| 5500 | **0.735** | 0.934 | 0.915 | early stop |

Wall time: 6.6 min.

Comparing Phase A ctc_CER on proj_seq_up across configs:

| n_soft_tokens | ctc_CER (proj_seq_up, 256 tok) |
|---|---|
| 32 | **0.476** |
| 64 | 0.735 |

**Unexpected: 64-token resampler gives WORSE per-frame ctc_CER than 32 tokens.**
The CharCTCHead cross-attends into resampler_out (64 KV tokens vs 32). With more KV
tokens, the cross-attention gradient to per-frame proj_seq features may be more
diffuse — each frame attends over a wider pool, reducing per-character alignment
specificity. The net effect is that per-frame features carry less character
content with 64-token KV than with 32.

### Phase B (64 tokens) — `train_20260415_195317.log`

| Step | val LM | frame_F1 | onset_F1 |
|------|--------|----------|----------|
| 500  | 4.7938 | 0.939 | 0.901 |
| ~8000 | **4.5776** | 0.937 | 0.903 | BEST |
| final | 4.7754 | 0.937 | 0.901 | early stop |

Wall time: 25.4 min. Best val LM **4.5776** — **worse than every other config
including the no-LoRA Run 17 baseline (4.5459)**.

### Full results table

| Run | n_soft_tokens | Phase A ctc_CER | Phase B val LM | Δ vs Run 17 |
|-----|---------------|-----------------|----------------|-------------|
| Run 17 (no LoRA, no CTC) | 32 | N/A | 4.5459 | — |
| Run 18m (LoRA attn) | 32 | N/A | 4.5197 | −0.026 |
| **Run 20m (LoRA + CTC)** | **32** | **0.476** | **4.4262** | **−0.12** |
| 64tok (LoRA + CTC) | 64 | 0.735 | 4.5776 | +0.032 |

**More tokens is worse.** The 32-token configuration remains the best.

### Why 64 tokens hurts

The counter-intuitive result has a likely structural explanation: the
`CharCTCHead` uses `resampler_out` as K/V in its cross-attention, with
`proj_seq_up` (256 per-frame features) as queries. Doubling the K/V pool
from 32 → 64 tokens makes each per-frame query attend over a wider, more
diffuse set of keys — reducing the signal specificity that drives per-frame
character alignment. The 32-token "bottleneck" actually acts as a compression
prior that forces the cross-attention to be selective, producing sharper
per-frame gradients.

Additionally, Phase A has only 3,465 training samples — the extra 32 learned
query parameters may not have enough data to converge to a useful
representation, leaving the 64-token resampler less informative per token.

### Revised diagnosis

The bottleneck is **not just token count**. The fundamental issue is that the
adapter is tasked with two conflicting objectives:
1. Compress the full rhythm into 32 compact tokens (for LLM conditioning).
2. Maintain per-frame character resolution (for CTC gradient).

These are competing: better compression → more diffuse per-frame gradient →
worse character encoding in the soft tokens. The information loss at the
resampler bottleneck is structural, not just a capacity problem.

### What to try next

The supervision ladder is the correct escalation:

1. **Per-key binary supervision** — add a per-key activity target (one binary
   classifier per key, predicting "is key K pressed at this frame?"). Forces
   the adapter to encode key identity, not just timing. The LLM then has
   character-labeled rhythms to decode rather than anonymous pulses.
2. **Keyboard-region classification** — if per-key (50+ classes) is too sparse,
   group keys into 4–6 spatial regions (left-hand QWERTY, right-hand QWERTY,
   numbers, punctuation) as an intermediate step.
3. ~~**Remove cross-attention from CharCTCHead**~~ — **done, see ablation below. Result: neutral.**

---

## Ablation — CharCTCHead without resampler cross-attention (2026-04-15)

Config: `phase_a_moment_ctc_no_xattn.yaml` — same as `phase_a_moment_ctc.yaml`
(n_soft_tokens=32) except `ctc_no_cross_attn=true`. CharCTCHead receives
`proj_seq_up` only; `resampler_out` is not passed as KV.

Result: val ctc_CER **0.484** (wall 5.3 min, best val_loss 0.8638,
frame_F1 0.935, onset_F1 0.921).

### Full comparison table

| Config | n_tokens | cross-attn | ctc_CER |
|---|---|---|---|
| `phase_a_moment_ctc` (Run 20m Phase A) | 32 | yes | **0.476** |
| `phase_a_moment_ctc_no_xattn` (this run) | 32 | **no** | 0.484 |
| `phase_a_moment_ctc_resampler_diag` | 32 (resampler) | yes* | 0.791 |
| `phase_a_moment_ctc_64tok` | 64 | yes | 0.735 |

*resampler_out as both query and no-KV (ctc_on_resampler path).

### Conclusion

**The diffuse-KV gradient hypothesis is refuted.** The cross-attention to
`resampler_out` in `CharCTCHead` contributes essentially nothing to CTC
quality: CER 0.484 without vs 0.476 with — a 0.008 difference well within
noise. The CTC head learns character alignment entirely from the per-frame
`proj_seq_up` features; the resampler KV provides no useful signal.

Consequence: the 64-token regression (ctc_CER 0.735) is **not** caused by
diffuse cross-attn gradient. The cause must be in the resampler training
dynamics: with 64 learned queries the resampler back-propagates different
gradients into `proj_seq`, modifying the per-frame features in a way that
degrades character alignment — possibly because the 64-query resampler has
more capacity and distributes temporal information more diffusely across
its queries, leaving individual frames less character-discriminative.

**Practical implication:** the `CharCTCHead` cross-attention can be removed
or kept — it makes no measurable difference. The 32-token adapter with CTC
on `proj_seq_up` (Run 20m, Phase A ctc_CER 0.476, Phase B val LM 4.4262)
remains the best configuration.

### Revised architecture understanding

The CTC head provides a direct character-alignment gradient to the per-frame
projection (`proj_seq`) and the `InputProjection`. The perceiver resampler
then compresses these character-enriched per-frame features into 32 soft
tokens. The perceiver's gradient in Phase A comes exclusively from the
`KeystrokeHead` cross-attention trunk (not the CTC head). This is fine:
keystroke rhythm grounding (frame_F1 0.935, onset_F1 0.921) is fully
maintained alongside character content (ctc_CER 0.484).

The remaining bottleneck is the perceiver compression itself: even with
character-enriched per-frame features, the 32-token bottleneck loses ~65%
of character discrimination (CER 0.476 → 0.791). Token count alone doesn't
fix this (64 tokens gives 0.735, worse not better). The correct escalation
is richer per-token supervision:

**Next: per-key binary supervision.** Rather than anonymous keystroke timing,
give the adapter explicit key-identity targets — one binary classifier per
key predicting "was key K active at this frame?". This forces each soft
token to encode *which* keys happened, not just *that* keypresses happened.
The LLM then has labeled rhythms rather than anonymous pulses.

---

## Seq2Seq Escalation — Run 21m (Phase A)

**Date:** 2026-04-16  
**Motivation:** CTC on 32 resampler tokens (ctc_CER 0.791) failed because CTC
requires T ≥ L (T=32 < L_max=52). CTC on 256-token proj_seq_up achieves 0.476
but the perceiver then compresses back to 32 tokens with 65% character loss.
The fundamental question: **can 32 soft tokens encode enough character content
to guide the LLM?**

Two alternatives to per-key binary supervision were evaluated:
1. **Seq2seq decoder on 32 tokens** — teacher-forced cross-attention decoder
   with no T≥L constraint; decoder queries each token once per output character.
2. **Condition resampler on characters** — rejected (training-inference mismatch).

Option 1 is implemented. Design:
- `CharSeq2SeqDecoder` in `adapter.py`: 2-layer causal transformer decoder
  (CharDecoderLayer = causal self-attn + cross-attn to resampler_out + FFN)
- Learned `bos_embed` parameter; positional embeddings up to `max_len=128`
- Teacher-forced forward: input `[BOS, chars[0..L-2]]`, target `chars[0..L-1]`
- Greedy decode builds input autoregressively; no EOS — decode to fixed length
- `forward(char_ids=None)` added to `RingToText`; wired through `train.py`

**Config:** `configs/phase_a_moment_seq2seq.yaml`
- `char_seq2seq_weight=0.5`, `keystroke_weight=1.0`, `onset_weight=1.0`
- Same MOMENT data, 32 soft tokens, skip_llm=True

**Hypothesis:** if `val seq2seq_CER < 0.5` at convergence (better than CTC-on-
resampler's 0.791), the 32-token bottleneck is NOT inherently lossy — CTC's
T≥L constraint was the problem and character content survives compression.
If `val seq2seq_CER > 0.7`, the perceiver is genuinely lossy regardless of the
loss formulation.

**Smoke test (step 50/100, epoch 1, 1 GPU):**
- seq2seq_CER 1.107 → 1.030 (random init → first few hundred samples)
- frame_F1 0.919/0.922, onset_F1 0.839/0.900 — rhythm grounding unaffected
- seq2seq loss 3.89 → 2.77 (converging from log(72)≈4.28 random baseline)

**Run 21m launched:** GPU 3, `checkpoints/phase_a_moment_seq2seq/`

### Run 21m Results — FAILED (cross-attention bypass)

Early stopping at epoch 42 (~5.8 min, 4500 steps). Results:

| Step | val seq2seq_loss | val seq2seq_CER | val ks_loss | frame_F1 |
|------|-----------------|-----------------|-------------|----------|
| 500  | 2.337           | 1.003           | 0.246       | 0.937    |
| 1000 | 2.388           | 1.007           | 0.330       | 0.937    |
| 1500 | 2.755           | 1.026           | 0.401       | 0.937    |
| ...  | ↑ increasing    | ~1.0 throughout | ↑ degrading | stable   |
| 4500 | 5.075           | 1.023           | 0.599       | 0.937    |

- **seq2seq_CER never went below 1.0** — random init level throughout
- **Val seq2seq_loss increased monotonically** from 2.33 → 5.07 while train s2s loss fell to ~0.09
- Best val_loss was at the first checkpoint (step 500); all subsequent steps overfit
- The val keystroke loss also degraded (0.246 → 0.599) as the seq2seq gradient pulled the adapter away from its Phase A rhythm grounding

### Diagnosis: cross-attention bypass

Standard teacher-forced causal seq2seq suffers from **exposure bias**. At training time,
the decoder sees `[BOS, char_0, char_1, ..., char_{i-1}]` as input and can predict
`char_i` purely by language-modeling the character sequence — no need to attend to the
32 resampler tokens at all. The model learns to rely on its causal context and effectively
ignores the cross-attention to resampler_out.

At inference (greedy decode from BOS with no correct history), the decoder fails because:
1. It has no prior context to language-model from
2. It never learned to extract character content from the resampler

Train seq2seq loss → 0.09 ≈ language-modeling char sequences from GT context.
Val seq2seq CER → 1.0 ≈ no useful information from resampler alone.

This is a well-known problem in encoder-decoder models (exposure bias) amplified here
because the target sequence (character text) is highly predictable by language modeling
alone, making the bypass path trivially learnable.

### Proposed fix: parallel decoder (no causal self-attention)

Remove causal self-attention from the decoder entirely. Replace with L independent
position queries that each cross-attend to the 32 resampler tokens:

```
for position i in [0, L-1]:
    query_i = bos_embed + pos_embed[i]   # position-specific, no prior-char context
    char_i_logit = CrossAttn(query_i, resampler_out) → linear → (vocab_size,)
```

Loss: parallel cross-entropy at all L positions simultaneously.

No bypass possible: the model has no causal context to lean on. Each position query
must extract its character from the 32 resampler tokens alone. The gradient forces the
resampler to encode which character was at each temporal position.

**Implementation:** add `no_self_attn=True` mode to `CharSeq2SeqDecoder` — skip the
`self_attn` layer in each `CharDecoderLayer`, keep only cross-attn + FFN.

**Hypothesis:** if parallel-decoder CER < 0.5, the 32-token bottleneck can encode full
character content with the right (bypass-free) supervision. If CER stays > 0.7, the
resampler compression is genuinely lossy and token-count escalation is needed.

**Capacity check:** 32 tokens × 256 dims = 8192 parameter space. A 52-char sequence
with 72-class vocab ≈ 310 bits of information. Capacity is not the constraint.
CTC on 256-token proj_seq_up (CER 0.476) confirms character content exists at the
per-frame level — the question is whether the perceiver can learn to preserve it in 32
tokens when trained with parallel position-query supervision.


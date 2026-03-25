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

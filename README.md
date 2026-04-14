## Stage 1 Chronos Workflow

This branch implements the full Stage 1 pipeline for keystroke classification using Chronos-2 embeddings plus a configurable classifier head.

The workflow has three executable phases:

1. Export dataset splits + vocabulary (`export_stage1_data.py`)
2. Train classifier head (and optional encoder fine-tuning) (`train_stage1_chronos.py`)
3. Evaluate trained checkpoints and export class diagnostics (`evaluate_stage1_chronos.py`)

## End-to-End Workflow

### 1) Export Stage 1 data

`export_stage1_data.py` generates:

- `vocab.json`
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `manifest.json`

Key export controls:

- Context window: `left_ms`, `right_ms`, `target_rate_hz`
- Split strategy: `session_random` or `session_holdout`
- Optional coarse labels (`--coarse-labels`) and case merge (`--merge-letter-case`)
- Session-level controls for explicit holdout / train-only sessions

Example:

```bash
python export_stage1_data.py \
  --out-dir stage1_export_coarse_rare_all_p \
  --data-dir data \
  --left-ms 700 \
  --right-ms 150 \
  --target-rate-hz 100 \
  --coarse-labels \
  --merge-letter-case
```

### 2) Train Stage 1 model

`train_stage1_chronos.py` supports:

- Frozen Chronos encoder + head-only training (default)
- Last-`N` block Chronos fine-tuning (`encoder_finetune_last_n > 0`)
- Multiple heads: `linear`, `mlp`, `attention_pool`, `conv1d`, `lstm`
- Class-imbalance-aware losses (`weighted_ce`, `focal`, `weighted_focal`)
- Early stopping by `val_loss` or `val_top1`
- Embedding cache (RAM or memmap; memmap is recommended for larger runs)

Example:

```bash
python train_stage1_chronos.py \
  --config configs/experiments/exp_conv1d_finetune_last2.yaml
```

### 3) Evaluate and export diagnostics

`evaluate_stage1_chronos.py` evaluates a checkpoint and writes:

- `metrics.json` (loss, top-k accuracy, micro/macro/weighted F1)
- `confusion_matrix.npy` / `confusion_matrix.csv`
- `per_class_f1.csv`

Example:

```bash
python evaluate_stage1_chronos.py \
  --head-ckpt checkpoints/exp_stage1_conv1d_finetune_last2/head.pt \
  --meta-path checkpoints/exp_stage1_conv1d_finetune_last2/train_meta.json \
  --norm-stats checkpoints/exp_stage1_conv1d_finetune_last2/norm_stats.json \
  --split test
```

## YAML Configuration Setup

Training is configured via YAML (`--config`) and merged with internal defaults in `TRAINING_DEFAULTS`:

- Base defaults file: `configs/stage1_chronos.defaults.yaml`
- Experiment overrides: `configs/experiments/*.yaml`
- CLI flags always override YAML
- Grouped YAML sections (`paths`, `model`, `training`, `early_stopping`, etc.) are flattened into known training keys

Common experiment configs in this branch:

- `exp_attention_pool.yaml`
- `exp_conv1d.yaml`
- `exp_conv1d_regularized.yaml`
- `exp_conv1d_finetune_last2.yaml`
- `exp_conv1d_regularized_trim_tokens.yaml`

## System Architecture (Modeling + Training)

### Modeling stack

- Input: multivariate IMU window (`B x V x T`, usually `V=12`)
- Encoder: `autogluon/chronos-2-small` Chronos-2 encoder
- Encoder output: token tensor (`B x V x L x D`, typically `D=512`)
- Optional token trimming before the head:
  - Drop Chronos special tail tokens
  - Optionally trim leading patches that correspond to left padding
- Classifier head on encoder tokens:
  - Mean/MLP baselines
  - Attention pooling over `V x L`
  - Conv1D temporal head (best-performing in this branch)
  - BiLSTM head

### Training system design

- Normalization is computed from train split only and saved to `norm_stats.json`
- Embeddings can be cached for frozen-encoder training:
  - `memmap` caches reduce RAM pressure and speed repeated experiments
- Weighted loss functions mitigate class imbalance
- Early stopping stores `head_best.pt`, final export stores `head.pt`
- `train_meta.json` captures reproducibility metadata:
  - resolved config
  - patch info
  - loss/head setup
  - cache metadata
  - best epoch/metric

### Fine-tuning path

When `encoder_finetune_last_n > 0`:

- Last `N` Chronos encoder blocks + final layer norm are unfrozen
- Two optimizer groups are used:
  - head learning rate (`lr`)
  - encoder learning rate (`lr_encoder`)
- Forward path switches to differentiable encoder calls (no embedding cache)

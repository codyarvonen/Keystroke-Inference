# Stage-1 Chronos experiment configs

From the repo root:

```bash
python train_stage1_chronos.py --config configs/experiments/<file>.yaml
```

## Clarification: “current loss” vs “weighted CE”

In `configs/stage1_chronos.defaults.yaml`, **“current” training loss is already weighted CE**:

- `loss_type: weighted_ce`
- `class_weight_mode: inverse_sqrt`

So the three configs **`exp_attention_pool.yaml`**, **`exp_conv1d.yaml`**, and **`exp_lstm.yaml`** all use **that same weighted CE recipe** (plus the usual epochs, lr, early stopping, etc.). They only differ by `head_type` and `out_dir`.

A **second** sweep that is *again* only “weighted CE” with the same hyperparameters would be the same experiment. To get **six meaningfully different runs**, this folder includes:

| Config | Head | Loss |
|--------|------|------|
| `exp_attention_pool.yaml` | attention_pool | weighted CE (`inverse_sqrt`) — **matches default recipe** |
| `exp_conv1d.yaml` | conv1d | same |
| `exp_lstm.yaml` | lstm | same |
| `exp_attention_pool_plain_ce.yaml` | attention_pool | **plain CE** (unweighted) |
| `exp_conv1d_plain_ce.yaml` | conv1d | plain CE |
| `exp_lstm_plain_ce.yaml` | lstm | plain CE |

Use the first row for “loss as in defaults” and the second row to compare **unweighted vs weighted** for each architecture.

## Outputs

Each file sets its own `paths.out_dir` under `checkpoints/` so runs do not overwrite.

All configs here use **`paths.export_dir: stage1_export_coarse32`** (32-class coarse labels). Produce that export with `export_stage1_data.py --coarse-labels --out-dir stage1_export_coarse32` (or change the path consistently).

Edit `paths.data_dir` and `model.device` to match your machine.

## Evaluation output directory

`evaluate_stage1_chronos.py` defaults to **`<parent-of-head.pt>/eval_<split>/`** so each checkpoint run gets its own eval folder. Override with `--out-dir` if needed.

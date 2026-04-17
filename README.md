# Ring-to-Text: Keystroke Inference from Wearable IMU Rings

Infers typed text from IMU sensor data recorded by smart rings worn on both hands. A frozen time-series foundation model (Chronos or MOMENT) and a frozen Qwen2.5-1.5B language model are bridged by a small trainable Perceiver resampler adapter.

## Architecture

```
IMU (CSV) ──► TS encoder (frozen) ──► IMUAdapter (trainable) ──┬──► KeystrokeHead ──► activity logits  (dense)
             Chronos | MOMENT           perceiver resampler    │       (dual-head)     onset logits     (sparse)
             d_enc = d_model×n_ch       32 soft tokens         ├──► CharCTCHead  ──► char logits       (optional)
                                                               └──► LLM/Qwen2.5-1.5B (frozen, optional LoRA) ──► text  (Phase B)
```

- **Time-series encoder** (configurable, frozen):
  - **Chronos** (default) — encodes each IMU axis independently; per-channel embeddings concatenated → `d_chronos = 768 × n_channels` for `chronos-t5-base` (9216 for both rings, 4608 for one). Produces a variable-length token sequence (S≈513 for a 10 s @ 100 Hz window).
  - **MOMENT** — patch-based T5 encoder; input resampled to a fixed 512 timesteps then tokenised into 64 patches of length 8. Channels handled internally via channel-independence and concatenated along the feature dim → `d_enc = d_model × n_channels` (12288 for `MOMENT-1-large`, both rings).
  - Embeddings are mean-centred (train-set mean subtracted) to remove the shared background direction.
- **IMUAdapter** (~4M params, only trainable component) projects Chronos features into `adapter_dim=256` and compresses them into 32 soft tokens via a Perceiver resampler.
- **KeystrokeHead** is a dual-head detector with a shared trunk. Per-frame projected features cross-attend into the resampler output (which is what gives the perceiver gradient in Phase A), pass through a LayerNorm, and branch into two independent linear classifiers:
  - **activity head** — dense *"is a key being pressed at this frame?"* logit. Supervised with masked BCE + positive-class weight against the activity mask derived from the PKL `key_times` intervals.
  - **onset head** — sparse *"is a keypress starting at this frame?"* logit. Supervised with masked focal BCE (`gamma=2.0`, `alpha=0.75`) against a per-event onset mask that marks each keypress's start frame — including keys pressed while another key is still held. The focal term prevents the dense negatives from drowning the sparse positives.

  The two targets carry complementary information (spanning mask vs rising-edge impulse). Splitting them into independent classifiers lets the onset head optimise a sharp-impulse objective that a single activity head was blurring. The previous depthwise `k=5` temporal conv has been removed — it smoothed neighbouring frames together and was the structural cause of the Run-13 onset-F1 ceiling (~0.10).
- **CharCTCHead** (optional, Phase A) is a per-frame character classifier trained with CTC on the adapter's projected features (optionally upsampled 4× for MOMENT, whose `S=64` is otherwise too short vs target character lengths `L` up to ~60). Shares the cross-attend-into-resampler trunk with `KeystrokeHead` so the perceiver gets gradient from a character signal. Used to bake character identity into the soft tokens before Phase B; typically switched off during LM fine-tuning.
- **LLM** (`Qwen/Qwen2.5-1.5B`, base, frozen) generates text conditioned on the soft tokens prepended with the prompt *"The user typed: "* — used in Phase B for fluency and rhythm→text mapping. Optionally LoRA-adapted (`lora_rank > 0`) so the LLM can re-route attention over the soft-token prefix; default LoRA targets are attention-only (`q/k/v/o`), full-list (`+gate/up/down`) available for higher capacity.

## Setup

```bash
conda env create -f environment.yml
conda activate ring2text
```

## Data format (`data/`)

```
{subject}_{session}_DIBS-L_corrected.csv   # Left ring IMU (Accel x/y/z, Gyro x/y/z)
{subject}_{session}_DIBS-R_corrected.csv   # Right ring IMU
{subject}_{session}_Macbook.pkl            # Keystroke timestamps: key_times dict
```

## Pipeline

### 1. Preprocess: raw data → encoder embeddings + keystroke targets

```bash
# Using the config file (recommended; default encoder = chronos)
python preprocess.py --config configs/preprocess.yaml

# MOMENT encoder (writes *_moment.pt so it doesn't clobber chronos files)
python preprocess.py --config configs/preprocess.yaml \
    --encoder moment --moment_model AutonLab/MOMENT-1-large

# Or with individual flags
python preprocess.py \
    --raw_dir ./data \
    --output_dir ./embeddings \
    --ring both \
    --window_size 10.0 \
    --step_size 5.0
```

Produces `embeddings/train.pt`, `val.pt`, `test.pt` for Chronos (unsuffixed, default) or `train_moment.pt`, `val_moment.pt`, `test_moment.pt` for MOMENT. Each sample stores the encoder embeddings, the reconstructed typed text, a `keystroke_active` uint8 mask (1 on any frame where a key is pressed, including modifiers), and a `keystroke_onset` uint8 mask (1 on the starting frame of each keypress — including keypresses that overlap an earlier still-held key). Applies per-channel z-score normalisation on raw IMU, then mean-centres encoder embeddings using the training-set mean.

Key options: `--encoder {chronos,moment}`, `--chronos_model`, `--moment_model`, `--ring {L,R,both}`, `--window_size`, `--step_size`, `--min_text_len`, `--target_hz`, `--no_normalize`, `--val_split`, `--test_split`

> **Switching encoder or model size:** edit `encoder` / `chronos_model` / `moment_model` in `configs/preprocess.yaml` and update `d_chronos` (the adapter input dim) in the training configs to match `d_model × n_channels`. See the sizing tables in `configs/default.yaml` and `configs/preprocess.yaml`. MOMENT requires `pip install momentfm` (not yet in `environment.yml`).

### 2. Train: two-phase schedule

Earlier runs showed LM cross-entropy alone could not push the adapter to actually use the IMU signal — predictions stayed fluent-but-wrong. A prior iteration grounded the adapter with character-CTC, but that partially fights the frozen LLM's job. The current design replaces character-CTC with a **binary keystroke-activity** objective: Phase A teaches the adapter *when* keystrokes happen from IMU motion; Phase B lets the frozen Qwen LLM recover characters from the rhythm structure of the soft tokens.

**Phase A — Keystroke-activity adapter pretraining** (no LLM in the loop, fast):

```bash
python train.py --config configs/phase_a.yaml
```

Trains adapter + dual-head KeystrokeHead from scratch. The head reads the adapter's per-timestep projected features, cross-attends into the resampler output (so supervision flows to the whole adapter including the perceiver), and produces activity + onset logits. Loss is `keystroke_weight * activity_BCE + onset_weight * onset_focal_BCE`. Activity BCE uses a positive-class weight computed once over the training set; onset focal uses `gamma=2.0`, `alpha=0.75` (set in `configs/phase_a.yaml`). Saves `checkpoints/phase_a/adapter_best.pt` (adapter + both classifier heads).

Phase A success bar: val loss plateaus, frame F1 > 0.6, and **`onset_head_F1` > 0.3** on held-out sessions. `onset_head_F1` is the top-line onset metric (from the dedicated head); `onset_rising_F1` (rising edges of the activity logits) is logged alongside for comparison with the old single-head design.

**Phase B — Keystroke + LM multi-task fine-tuning** (loads Phase A as init):

```bash
python train.py --config configs/phase_b.yaml \
                --adapter_init ./checkpoints/phase_a/adapter_best.pt
```

Adds the frozen Qwen LLM back via the soft-token prefix. Both keystroke heads stay on at lower weight (`keystroke_weight: 0.3`, `onset_weight: 0.3`) so the adapter cannot regress and forget rhythm grounding while the LM term adds content. Saves `checkpoints/phase_b/adapter_best.pt`.

CLI flags always override config file values.

Key args:
- `--skip_llm` — Phase A toggle: skips LLM forward entirely. Loss is pure keystroke (activity + onset).
- `--keystroke_weight` — weight on the activity BCE loss (1.0 in Phase A, 0.3 in Phase B).
- `--onset_weight` — weight on the onset focal loss (1.0 in Phase A, 0.3 in Phase B; 0 disables the onset head entirely).
- `--onset_focal_gamma`, `--onset_focal_alpha` — focusing exponent and positive-class weight for the onset focal loss.
- `--no_keystroke` — disable both keystroke heads (legacy LM-only training).
- `--adapter_init <path>` — load adapter (and keystroke head, and LoRA if present) from a checkpoint before training. Legacy single-head checkpoints load with `strict=False` — the shared trunk weights transfer and the two fresh classifiers start from random init.

Legacy anti-collapse losses (`--div_weight`, `--recon_weight`, `--contrast_weight`) default to 0 in both phase configs — Phase A's keystroke supervision makes them redundant. Re-enable cautiously if soft-token collapse reappears (watch the cosine-sim diagnostic in logs).

**Optional LoRA fine-tuning of the LLM** (`--lora_rank N`, default 0 = frozen). Closes the rhythm→content gap when the frozen Qwen cannot invent character identity from rhythm alone. Config presets:

| Config | Encoder | LoRA targets | ~params |
|---|---|---|---|
| `phase_b_{moment,chronos}_lora_attn.yaml` | MOMENT / Chronos | `q/k/v/o` | 344 K |
| `phase_b_{moment,chronos}_lora_full.yaml` | MOMENT / Chronos | `q/k/v/o + gate/up/down` | 9.8 M |

All four freeze `adapter.proj` and `adapter.resampler` (leaving `out_proj` trainable) via `--freeze_resampler` so LoRA can't drift the Phase-A grounded rhythm.

**Character CTC supervision** (`--char_ctc_weight`, default 0). Adds dense per-frame character supervision via a `CharCTCHead` on the adapter's per-timestep features, forcing character identity into the soft tokens. Linearly warmed from `--char_ctc_start` over `--char_ctc_warmup_steps`. Config presets:

| Config | Encoder | `ctc_upsample_factor` | Notes |
|---|---|---|---|
| `phase_a_{moment,chronos}_ctc.yaml` | MOMENT / Chronos | 4 / 1 | CTC active; `skip_llm=true` |
| `phase_b_{moment,chronos}_ctc.yaml` | MOMENT / Chronos | 4 / 1 | CTC weight 0 by default; seeds LoRA from CTC-grounded Phase A |

MOMENT uses `ctc_upsample_factor=4` (ConvTranspose1d on `proj_seq`) because its `S=64` patches are too short vs target character lengths. Chronos keeps `factor=1` (S≈513 has plenty of headroom). Validation logs `val_ctc_cer` (CTC-greedy CER against ground-truth text) as an adapter-only content metric independent of the LLM.

If Phase B fails to echo content even with LoRA+CTC, the natural escalation path is per-hand binary (2 channels) → keyboard region (4–6 channels) → per-key.

### 3. Inference: generate text

Use a Phase B checkpoint for full text generation (Phase A checkpoints have no LLM pathway — they only emit per-frame keystroke-activity logits).

```bash
# From pre-computed embeddings
python generate.py \
    --adapter_path ./checkpoints/phase_b/adapter_best.pt \
    --input_file ./embeddings/test.pt

# Directly from raw data (runs Chronos on-the-fly; MOMENT on-the-fly not supported yet)
python generate.py \
    --adapter_path ./checkpoints/phase_b/adapter_best.pt \
    --raw_dir ./data

# Integration test with random embeddings
python generate.py \
    --adapter_path ./checkpoints/phase_b/adapter_best.pt \
    --demo
```

## Utility scripts

| Script | Purpose |
|---|---|
| `explore_data.py` | IMU/keystroke statistics and optional visualizations (`--visualize`, `--save-plots`) |
| `visualize_phase_a.py` | Plot Phase A activity + onset probabilities vs GT masks per sample (`--adapter_path`, `--input_file`, `--output_dir`) |
| `regenerate_text.py` | Reconstruct text from keystroke PKL files (`--data-dir`, `--session`) |
| `validate_data_loader.py` | Test the `data_loader/` PyTorch pipeline and report dataset statistics |
| `gpu_test.py` | Check CUDA availability |
| `stress_test.py` | GPU memory stress test (forward + backward pass) |

## Repository layout

```
├── configs/
│   ├── preprocess.yaml                         # Preprocessing (encoder, model, windowing, splits)
│   ├── phase_a.yaml / phase_a_moment.yaml      # Phase A: keystroke pretraining (Chronos / MOMENT)
│   ├── phase_b.yaml / phase_b_moment.yaml      # Phase B: keystroke + LM multi-task
│   ├── phase_b_{moment,chronos}_lora_{attn,full}.yaml  # Phase B with LoRA on Qwen
│   ├── phase_{a,b}_{moment,chronos}_ctc.yaml   # Phase A/B with CharCTCHead
│   └── default.yaml                            # Legacy single-phase config (LM-only)
├── preprocess.py          # IMU → encoder (Chronos or MOMENT) embeddings + keystroke_active/onset + char_targets
├── train.py               # Adapter training loop (keystroke BCE + onset focal + optional CTC + optional LM/LoRA)
├── generate.py            # Inference / text generation
├── model.py               # RingToText: full forward + generation
├── adapter.py             # IMUAdapter (perceiver + optional CTC upsample) + KeystrokeHead + CharCTCHead
├── char_vocab.py          # Case-sensitive character vocab + CTC greedy decode
├── precheck_ctc_length.py # CTC length-feasibility audit per encoder
├── dataset.py             # IMUTextDataset for training
├── data_loader/           # PyTorch DataLoader pipeline (no pre-computed embeddings)
│   ├── config.py
│   ├── dataset.py
│   ├── sessions.py
│   ├── splits.py
│   ├── windows.py
│   └── labels.py
└── utils/                 # Shared utilities
    ├── constants.py       # IMU_COLS, CHRONOS_DEFAULT_MODEL, MOMENT_DEFAULT_MODEL, ENCODER_CHOICES, IMU_EPS
    ├── filename.py        # parse_filename()
    ├── imu_io.py          # get_time_column(), load_imu_csv()
    ├── keystroke.py       # keystroke parsing + text reconstruction
    ├── chronos_encode.py  # encode_with_chronos()
    └── moment_encode.py   # encode_with_moment()
```

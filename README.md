# Ring-to-Text: Keystroke Inference from Wearable IMU Rings

Infers typed text from IMU sensor data recorded by smart rings worn on both hands. A frozen Chronos time-series encoder and a frozen GPT-2 language model are bridged by a small trainable Perceiver resampler adapter.

## Architecture

```
IMU (CSV) ──► Chronos (frozen) ──► IMUAdapter (trainable) ──► LLM/gpt2 (frozen) ──► text
             per-channel encode      perceiver resampler
             d_chronos = 768×12      32 soft tokens
```

- **Chronos** encodes each IMU axis independently; embeddings are concatenated → `d_chronos = 768 × n_channels` (9216 for both rings, 4608 for one)
- **IMUAdapter** (~4M params, only trainable component) compresses variable-length Chronos output into 32 fixed soft tokens via a Perceiver resampler
- **LLM** (base, frozen) generates text conditioned on the soft tokens prepended with the prompt *"The user typed: "*; default is `gpt2`, alternatives include `gpt2-medium`, `gpt2-large`, `Qwen/Qwen2.5-1.5B`

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

### 1. Preprocess: raw data → Chronos embeddings

```bash
python preprocess.py \
    --raw_dir ./data \
    --output_dir ./embeddings \
    --ring both \
    --window_size 10.0 \
    --step_size 5.0
```

Produces `embeddings/train.pt`, `val.pt`, `test.pt`.

Key options: `--ring {L,R,both}`, `--window_size`, `--step_size`, `--min_text_len`, `--target_hz`, `--no_normalize`

### 2. Train: fit the adapter

```bash
python train.py --data_file ./embeddings/train.pt --val_data_file ./embeddings/val.pt
```

Saves `checkpoints/adapter_best.pt` and `checkpoints/adapter_final.pt`.

### 3. Inference: generate text

```bash
# From pre-computed embeddings
python generate.py \
    --adapter_path ./checkpoints/adapter_final.pt \
    --input_file ./embeddings/test.pt

# Directly from raw data (runs Chronos on-the-fly)
python generate.py \
    --adapter_path ./checkpoints/adapter_final.pt \
    --raw_dir ./data

# Integration test with random embeddings
python generate.py \
    --adapter_path ./checkpoints/adapter_final.pt \
    --demo
```

## Utility scripts

| Script | Purpose |
|---|---|
| `explore_data.py` | IMU/keystroke statistics and optional visualizations (`--visualize`, `--save-plots`) |
| `regenerate_text.py` | Reconstruct text from keystroke PKL files (`--data-dir`, `--session`) |
| `validate_data_loader.py` | Test the `data_loader/` PyTorch pipeline and report dataset statistics |
| `gpu_test.py` | Check CUDA availability |
| `stress_test.py` | GPU memory stress test (forward + backward pass) |

## Repository layout

```
├── preprocess.py          # IMU → Chronos embeddings (offline)
├── train.py               # Adapter training loop
├── generate.py            # Inference / text generation
├── model.py               # RingToText: full forward + generation
├── adapter.py             # IMUAdapter: perceiver resampler
├── dataset.py             # IMUTextDataset for training
├── data_loader/           # PyTorch DataLoader pipeline (no pre-computed embeddings)
│   ├── config.py
│   ├── dataset.py
│   ├── sessions.py
│   ├── splits.py
│   ├── windows.py
│   └── labels.py
└── utils/                 # Shared utilities
    ├── constants.py       # IMU_COLS, CHRONOS_DEFAULT_MODEL, IMU_EPS
    ├── filename.py        # parse_filename()
    ├── imu_io.py          # get_time_column(), load_imu_csv()
    ├── keystroke.py       # keystroke parsing + text reconstruction
    └── chronos_encode.py  # encode_with_chronos()
```

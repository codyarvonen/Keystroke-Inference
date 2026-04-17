"""MOMENT-based IMU encoding utilities.

MOMENT (https://github.com/moment-timeseries-foundation-model/moment) is a
T5-backbone time-series foundation model. Unlike Chronos, its encoder is
patch-based and expects a fixed sequence length of 512 timesteps. It natively
handles multi-channel input via channel-independence, so all IMU channels are
passed through in a single forward call.

With `reduction='none'`, `model.embed()` returns patch-level embeddings of
shape (B, n_channels, n_patches, d_model). We transpose to (B, n_patches,
n_channels * d_model) so downstream adapter code sees the same (S, d_enc)
layout as the Chronos path.

Model identifiers and embedding dims (d_model):
    AutonLab/MOMENT-1-small   d_model=512
    AutonLab/MOMENT-1-base    d_model=768
    AutonLab/MOMENT-1-large   d_model=1024

With default patch_len=8, seq_len=512 → n_patches=64.
"""

import torch
import torch.nn.functional as F
import numpy as np


MOMENT_SEQ_LEN = 512


def _prep_imu_for_moment(imu_data: np.ndarray) -> torch.Tensor:
    """Convert (n_timesteps, n_channels) IMU to (1, n_channels, 512) tensor.

    MOMENT expects a fixed sequence length of 512. We linearly resample along
    time to match, preserving the channel-independent channel axis.
    """
    x = torch.tensor(imu_data, dtype=torch.float32).transpose(0, 1)  # (C, T)
    if x.shape[-1] != MOMENT_SEQ_LEN:
        x = F.interpolate(
            x.unsqueeze(0),
            size=MOMENT_SEQ_LEN,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    return x.unsqueeze(0)  # (1, C, 512)


def encode_with_moment(
    samples: list[dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Run IMU data through MOMENT and return per-patch embeddings.

    All channels are encoded in a single forward pass (MOMENT is channel-
    independent internally) and concatenated along the feature dim:
        output shape: (n_patches, n_channels * d_model)   ← d_encoder

    Args:
        samples:    list of {'imu': np.ndarray, 'text': str, ...}
        model_name: HuggingFace model ID (e.g. 'AutonLab/MOMENT-1-large')
        batch_size: number of samples per forward pass
        device:     device string ('cuda' or 'cpu')

    Returns:
        list of {'embeddings': torch.Tensor (cpu), 'text': str}
    """
    from momentfm import MOMENTPipeline

    print(f"Loading MOMENT model: {model_name}")
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device).eval()

    results = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            x = torch.cat([_prep_imu_for_moment(s["imu"]) for s in batch], dim=0)
            x = x.to(device)
            out = model.embed(x_enc=x, reduction="none")
            emb = out.embeddings  # (B, C, P, D)
            B, C, P, D = emb.shape
            emb = emb.permute(0, 2, 1, 3).reshape(B, P, C * D).contiguous().cpu()
            for j, sample in enumerate(batch):
                results.append({"embeddings": emb[j], "text": sample["text"]})
            print(f"  encoded {min(i + batch_size, len(samples))}/{len(samples)}")

    return results

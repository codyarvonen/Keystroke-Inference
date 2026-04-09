"""Chronos-based IMU encoding utilities."""

import time

import torch
import numpy as np


def _encode_multichannel(pipeline, imu_data: np.ndarray) -> torch.Tensor:
    """Encode multi-channel IMU with Chronos using a per-channel strategy.

    Each IMU channel is encoded independently and the resulting embeddings
    are concatenated along the feature dimension.

    Args:
        pipeline: ChronosPipeline instance
        imu_data: (n_timesteps, n_channels)

    Returns:
        (S, embed_dim * n_channels) — concatenated per-channel Chronos embeddings
    """
    n_channels = imu_data.shape[1]
    channel_embeddings = []
    for ch in range(n_channels):
        series = torch.tensor(imu_data[:, ch], dtype=torch.float32).unsqueeze(0)
        embedding, _ = pipeline.embed(series)
        channel_embeddings.append(embedding.squeeze(0))
    return torch.cat(channel_embeddings, dim=-1)


def _load_or_create_pipeline(model_name: str, device: str):
    """Load Chronos pipeline once (cached as module-level singleton)."""
    global _chronos_pipeline, _chronos_pipeline_name
    if "_chronos_pipeline" not in globals() or _chronos_pipeline_name != model_name:
        from chronos import ChronosPipeline
        print(f"Loading Chronos model: {model_name}")
        model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _chronos_pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, dtype=model_dtype,
        )
        globals()["_chronos_pipeline_name"] = model_name
        print("Chronos model loaded.")
    return _chronos_pipeline


def encode_with_chronos(
    samples: list[dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Run IMU data through Chronos and return embeddings.

    Each IMU channel is encoded independently; embeddings are concatenated
    along the feature dimension:
        output shape: (S, embed_dim * n_channels)   ← this is d_chronos

    Embeddings are stored in float16 to reduce memory usage (~3x savings).
    They are cast back to the working dtype during training.

    Args:
        samples:    list of {'imu': np.ndarray, 'text': str}
        model_name: HuggingFace model ID for Chronos
        batch_size: number of samples to process at once (controls print frequency)
        device:     device string passed to ChronosPipeline.from_pretrained

    Returns:
        list of {'embeddings': torch.Tensor (cpu, float16), 'text': str}
    """
    pipeline = _load_or_create_pipeline(model_name, device)
    print(f"Encoding {len(samples)} samples...")

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        emb = _encode_multichannel(pipeline, sample["imu"])
        results.append({"embeddings": emb.cpu().half(), "text": sample["text"]})
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  encoded {i+1}/{len(samples)}  ({rate:.1f} samples/s, ETA {eta:.0f}s)",
                  flush=True)

    return results

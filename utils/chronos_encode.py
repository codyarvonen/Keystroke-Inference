"""Chronos-based IMU encoding utilities."""

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
        (S, 768 * n_channels) — concatenated per-channel Chronos embeddings
    """
    n_channels = imu_data.shape[1]
    channel_embeddings = []
    for ch in range(n_channels):
        series = torch.tensor(imu_data[:, ch], dtype=torch.float32).unsqueeze(0)
        embedding, _ = pipeline.embed(series)  # pipeline moves to GPU via device_map
        channel_embeddings.append(embedding.squeeze(0))  # (S, 768)
    return torch.cat(channel_embeddings, dim=-1)  # (S, 768 * n_channels)


def encode_with_chronos(
    samples: list[dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Run IMU data through Chronos and return embeddings.

    Each IMU channel is encoded independently; embeddings are concatenated
    along the feature dimension:
        output shape: (S, 768 * n_channels)   ← this is d_chronos

    Args:
        samples:    list of {'imu': np.ndarray, 'text': str}
        model_name: HuggingFace model ID for Chronos
        batch_size: number of samples to process at once (controls print frequency)
        device:     device string passed to ChronosPipeline.from_pretrained

    Returns:
        list of {'embeddings': torch.Tensor (cpu), 'text': str}
    """
    from chronos import ChronosPipeline

    print(f"Loading Chronos model: {model_name}")
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
    )
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        for sample in batch:
            emb = _encode_multichannel(pipeline, sample["imu"])
            results.append({"embeddings": emb.cpu(), "text": sample["text"]})
        print(f"  encoded {min(i + batch_size, len(samples))}/{len(samples)}")

    return results

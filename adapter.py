"""
Adapter module: bridges Chronos IMU embeddings to a frozen LLM's input space.

This is the ONLY trainable component in the pipeline. It uses a perceiver-style
cross-attention bottleneck to compress variable-length Chronos output into a
fixed number of soft tokens that the LLM treats as a learned prefix.

Architecture:
    Chronos embeddings (B, S, d_chronos)
        → Linear projection (d_chronos → adapter_dim)
        → Cross-attention perceiver resampler (adapter_dim)
        → Fixed-length soft tokens (B, n_tokens, adapter_dim)
        → Output projection (adapter_dim → d_llm)

Key design: adapter_dim decouples the perceiver's internal hidden size from
d_llm. Keeping adapter_dim small (e.g. 256) dramatically reduces parameter
count and overfitting risk when d_llm is large (e.g. 1536 for Qwen2.5-1.5B).
"""

import torch
import torch.nn as nn
import math


class InputProjection(nn.Module):
    """Projects Chronos embedding dim to adapter_dim."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResampler(nn.Module):
    """
    Compresses variable-length encoder output into a fixed set of soft tokens
    via cross-attention with learned queries.

    This is the same mechanism used in BLIP-2's Q-Former and Flamingo's
    Perceiver Resampler. The learned queries specialize to attend to different
    aspects of the IMU signal.
    """

    def __init__(
        self,
        d_model: int,
        n_queries: int = 32,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"adapter_dim ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.learned_queries = nn.Parameter(
            torch.randn(1, n_queries, d_model) * (1.0 / math.sqrt(d_model))
        )
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_out:  (B, S_enc, d_model) — projected Chronos output
            encoder_mask: (B, S_enc) — True for padded positions to ignore

        Returns:
            (B, n_queries, d_model) — fixed-length soft tokens
        """
        B = encoder_out.size(0)
        queries = self.learned_queries.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, encoder_out, encoder_mask)

        return self.norm(queries)


class PerceiverResamplerLayer(nn.Module):
    """Single layer: cross-attention → FFN, both with pre-norm residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention with pre-norm
        q = self.cross_norm(queries)
        k = v = self.kv_norm(kv)
        attn_out, _ = self.cross_attn(
            query=q, key=k, value=v, key_padding_mask=kv_mask
        )
        queries = queries + attn_out

        # FFN with pre-norm
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class IMUAdapter(nn.Module):
    """
    Full adapter: projection → perceiver resampler → output projection.

    Takes raw Chronos embeddings and produces soft tokens ready to be
    concatenated with the LLM's text embeddings.

    Args:
        d_chronos:          Chronos output embedding dim (e.g. 9216)
        d_llm:              LLM hidden dim — output size of the adapter
        n_soft_tokens:      Number of fixed-length soft tokens to produce
        adapter_dim:        Internal hidden dim of the perceiver (default 256).
                            Keep this small to avoid overfitting. Must be
                            divisible by n_heads.
        n_heads:            Attention heads in the perceiver (default 8)
        n_resampler_layers: Depth of the perceiver resampler (default 2)
        dropout:            Dropout rate throughout (default 0.1)
    """

    def __init__(
        self,
        d_chronos: int,
        d_llm: int,
        n_soft_tokens: int = 32,
        adapter_dim: int = 256,
        n_heads: int = 8,
        n_resampler_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = InputProjection(d_chronos, adapter_dim, dropout)
        self.resampler = PerceiverResampler(
            d_model=adapter_dim,
            n_queries=n_soft_tokens,
            n_heads=n_heads,
            n_layers=n_resampler_layers,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(adapter_dim, d_llm, bias=False)

    @property
    def n_soft_tokens(self) -> int:
        return self.resampler.learned_queries.size(1)

    def forward(
        self,
        chronos_embeds: torch.Tensor,
        chronos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            chronos_embeds: (B, S, d_chronos)
            chronos_mask:   (B, S) — True for padded positions

        Returns:
            (B, n_soft_tokens, d_llm)
        """
        x = self.proj(chronos_embeds)           # (B, S, adapter_dim)
        x = self.resampler(x, chronos_mask)     # (B, n_soft_tokens, adapter_dim)
        return self.out_proj(x)                 # (B, n_soft_tokens, d_llm)

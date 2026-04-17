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
            # LayerNorm removed: normalising per-timestep erases amplitude
            # differences between samples; the perceiver has its own norms.
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
        ctc_upsample_factor: int = 1,
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

        # Optional CTC-path upsample. MOMENT yields only S=64 patches per
        # window, which is tight vs. target character lengths (L up to ~60),
        # so we give the CTC head a longer temporal axis without touching the
        # resampler's input.
        self.ctc_upsample_factor = ctc_upsample_factor
        if ctc_upsample_factor > 1:
            self.ctc_upsample = nn.ConvTranspose1d(
                adapter_dim, adapter_dim,
                kernel_size=ctc_upsample_factor,
                stride=ctc_upsample_factor,
            )
        else:
            self.ctc_upsample = None

    @property
    def n_soft_tokens(self) -> int:
        return self.resampler.learned_queries.size(1)

    def forward(
        self,
        chronos_embeds: torch.Tensor,
        chronos_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            chronos_embeds: (B, S, d_chronos)
            chronos_mask:   (B, S) — True for padded positions

        Returns:
            soft_tokens:    (B, n_soft_tokens, d_llm)         — LLM-space prefix
            resampler_out:  (B, n_soft_tokens, adapter_dim)   — pre-projection
            projected_mean: (B, adapter_dim)                  — mean of projected input
            proj_seq:       (B, S, adapter_dim)               — per-timestep features
                            (consumed by KeystrokeHead)
            proj_seq_up:    (B, S * upsample, adapter_dim)    — per-timestep features
                            optionally upsampled for the CTC head. Equal to proj_seq
                            when ctc_upsample_factor == 1.
        """
        x = self.proj(chronos_embeds)                # (B, S, adapter_dim)
        projected_mean = x.mean(dim=1)               # (B, adapter_dim)
        resampler_out = self.resampler(x, chronos_mask)  # (B, n_soft_tokens, adapter_dim)
        soft_tokens = self.out_proj(resampler_out)   # (B, n_soft_tokens, d_llm)

        if self.ctc_upsample is not None:
            proj_seq_up = self.ctc_upsample(x.transpose(1, 2)).transpose(1, 2)
        else:
            proj_seq_up = x

        return soft_tokens, resampler_out, projected_mean, x, proj_seq_up


class KeystrokeHead(nn.Module):
    """
    Dual-head per-frame keystroke detector.

    Shared trunk reads two views of the IMU window:
      - `proj_seq` (B, S, adapter_dim): per-timestep projected features.
      - `resampler_out` (B, n_soft, adapter_dim): the perceiver's compressed
        global summary.

    For each frame we cross-attend into the resampler output; the result
    passes through a LayerNorm and branches into two independent linear
    classifiers:
      - activity_logits: "is a key being pressed at this time?"  (dense signal)
      - onset_logits:    "is a key press *starting* at this frame?"  (sparse)

    The two targets coexist without conflict: activity is a spanning mask and
    onset is a rising-edge impulse — they carry complementary temporal
    information. Splitting them lets the onset head optimise a sharp-impulse
    objective that the dense activity target was previously blurring. The
    previous depthwise k=5 temporal conv has been removed: it smoothed
    neighbouring frames together, directly harming onset precision (Run 13
    onset_F1 stalled at ~0.10–0.14 despite frame_F1 ~0.76).

    The cross-attention into `resampler_out` is what makes gradient flow back
    through the perceiver in Phase A (skip_llm=True, no LM loss). Without it
    the resampler would see zero gradient.

    Args:
        adapter_dim: Hidden width shared with the adapter.
        n_heads:     Cross-attention heads (default 4).
        dropout:     Dropout in cross-attn only.
    """

    def __init__(
        self,
        adapter_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_norm = nn.LayerNorm(adapter_dim)
        self.kv_norm = nn.LayerNorm(adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.activity_classifier = nn.Linear(adapter_dim, 1)
        self.onset_classifier = nn.Linear(adapter_dim, 1)

    def forward(
        self,
        proj_seq: torch.Tensor,
        resampler_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            proj_seq:      (B, S, adapter_dim) — frame features (queries)
            resampler_out: (B, n_soft, adapter_dim) — global summary (KV).
                           If None, the cross-attention is skipped.

        Returns:
            activity_logits: (B, S) — "key pressed now" logits per frame
            onset_logits:    (B, S) — "key press *starts* at this frame" logits
        """
        x = proj_seq
        if resampler_out is not None:
            q = self.q_norm(x)
            kv = self.kv_norm(resampler_out)
            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            x = x + attn_out

        x = self.norm(x)
        activity = self.activity_classifier(x).squeeze(-1)
        onset = self.onset_classifier(x).squeeze(-1)
        return activity, onset


class CharDecoderLayer(nn.Module):
    """
    Single transformer decoder layer: causal self-attention + cross-attention
    into the resampler soft tokens + FFN (all with pre-norm residuals).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        # Cross-attention to resampler soft tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        # FFN
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
        x: torch.Tensor,
        resampler_out: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Causal self-attention (pre-norm)
        q = self.self_norm(x)
        attn_out, _ = self.self_attn(q, q, q, attn_mask=causal_mask)
        x = x + attn_out
        # Cross-attention to resampler_out (pre-norm)
        q = self.cross_norm(x)
        kv = self.kv_norm(resampler_out)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x + attn_out
        # FFN (pre-norm)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CharSeq2SeqDecoder(nn.Module):
    """
    Autoregressive character decoder that cross-attends to the 32 resampler
    soft tokens.

    During training the forward pass is teacher-forced: input is
    [BOS, char_ids[0], ..., char_ids[L-2]] and the target is char_ids[0..L-1].
    The loss at padding positions (beyond char_lens[i]) is masked to -100 so
    it contributes zero cross-entropy.

    There is no CTC-style length constraint — the decoder treats the 32 soft
    tokens as a compressed *memory* and can read each one as many times as
    needed while generating character positions 0 through L-1.

    Args:
        adapter_dim:  Hidden width shared with the adapter (default 256).
        vocab_size:   Character vocabulary size (includes blank at id 0).
        n_layers:     Number of decoder layers (default 2).
        n_heads:      Attention heads in each layer (default 4).
        max_len:      Maximum sequence length for positional embeddings
                      and greedy decode (default 128).
        dropout:      Dropout in attention and FFN (default 0.1).
    """

    def __init__(
        self,
        adapter_dim: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, adapter_dim)
        self.bos_embed = nn.Parameter(
            torch.randn(1, 1, adapter_dim) * (1.0 / math.sqrt(adapter_dim))
        )
        self.pos_embed = nn.Embedding(max_len, adapter_dim)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

        self.layers = nn.ModuleList(
            [CharDecoderLayer(adapter_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.classifier = nn.Linear(adapter_dim, vocab_size)

    def forward(
        self,
        resampler_out: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            resampler_out: (B, n_soft, adapter_dim) — frozen perceiver output.
            char_ids:      (B, L) — ground-truth character IDs padded with
                           BLANK_ID (0) beyond char_lens[i].

        Returns:
            logits: (B, L, vocab_size) — logits for each target position.
        """
        B, L = char_ids.shape
        # Input:  [BOS, char_ids[0], ..., char_ids[L-2]]  shape (B, L)
        bos = self.bos_embed.expand(B, 1, -1)             # (B, 1, d)
        char_emb = self.embed(char_ids[:, :-1])           # (B, L-1, d)
        x = torch.cat([bos, char_emb], dim=1)             # (B, L, d)
        x = x + self.pos_embed(self.positions[:, :L])

        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, resampler_out, causal_mask)

        x = self.norm(x)
        return self.classifier(x)                         # (B, L, vocab_size)

    @torch.no_grad()
    def greedy_decode(
        self,
        resampler_out: torch.Tensor,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive greedy decode starting from BOS.

        Args:
            resampler_out: (B, n_soft, adapter_dim)
            max_len:       maximum characters to generate (default: self.max_len)

        Returns:
            token_ids: (B, max_len) — predicted character IDs (BLANK_ID may
                       appear; use char_vocab.decode to strip them).
        """
        max_len = max_len or self.max_len
        B = resampler_out.size(0)
        device = resampler_out.device
        dtype = resampler_out.dtype

        generated: list[torch.Tensor] = []   # each element: (B,) token-id tensor

        for step in range(max_len):
            L = step + 1
            # Build input sequence: [BOS, gen[0], ..., gen[step-1]]
            bos = self.bos_embed.expand(B, 1, -1).to(dtype)   # (B, 1, d)
            if step == 0:
                x = bos
            else:
                prev_ids = torch.stack(generated, dim=1)       # (B, step)
                char_emb = self.embed(prev_ids).to(dtype)      # (B, step, d)
                x = torch.cat([bos, char_emb], dim=1)          # (B, L, d)
            x = x + self.pos_embed(self.positions[:, :L]).to(dtype)

            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, resampler_out, causal_mask)
            x = self.norm(x)

            next_token = self.classifier(x[:, -1, :]).argmax(dim=-1)  # (B,)
            generated.append(next_token)

        return torch.stack(generated, dim=1)   # (B, max_len)


class CharCTCHead(nn.Module):
    """
    Per-frame character classifier trained with CTC on proj_seq (optionally
    upsampled). Same shared-trunk design as KeystrokeHead — cross-attention
    into resampler_out gives the perceiver dense gradient from a character
    signal — minus the temporal conv, which blurs alignment in a way CTC
    particularly dislikes.

    Args:
        adapter_dim: Hidden width shared with the adapter.
        vocab_size:  Output vocabulary (includes blank at id 0).
        n_heads:     Cross-attention heads (default 4).
        dropout:     Dropout in cross-attn only.
    """

    def __init__(
        self,
        adapter_dim: int,
        vocab_size: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_norm = nn.LayerNorm(adapter_dim)
        self.kv_norm = nn.LayerNorm(adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.classifier = nn.Linear(adapter_dim, vocab_size)

    def forward(
        self,
        proj_seq: torch.Tensor,
        resampler_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            proj_seq:      (B, T, adapter_dim) — per-frame features (queries).
                           T may be S or S * upsample depending on adapter config.
            resampler_out: (B, n_soft, adapter_dim) — global summary (KV).

        Returns:
            char_logits: (B, T, vocab_size)
        """
        x = proj_seq
        if resampler_out is not None:
            q = self.q_norm(x)
            kv = self.kv_norm(resampler_out)
            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            x = x + attn_out

        x = self.norm(x)
        return self.classifier(x)

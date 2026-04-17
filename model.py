"""
RingToText: IMU → Chronos (frozen) → Adapter (trained) → LLM (frozen) → text

The model prepends a text prompt ("The user typed: ") as real token embeddings
before the soft tokens, giving the frozen LLM context about the expected output.
Only the adapter parameters are updated during training.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from adapter import IMUAdapter, KeystrokeHead, CharCTCHead, CharSeq2SeqDecoder
from char_vocab import VOCAB_SIZE as CHAR_VOCAB_SIZE


class RingToText(nn.Module):
    """
    Full pipeline from Chronos IMU embeddings to generated text.

    Args:
        llm_name:             HuggingFace model ID for the frozen decoder LLM
        d_chronos:            Chronos output embedding dimension
        n_soft_tokens:        Number of soft tokens the adapter produces
        n_resampler_layers:   Depth of the perceiver resampler
        prompt:               Text prompt prepended before soft tokens to steer
                              the LLM into transcription mode
        dtype:                Dtype for the LLM weights (bf16 recommended)
    """

    DEFAULT_PROMPT = "The user typed: "

    @staticmethod
    def _default_lora_targets(llm_name: str) -> list[str]:
        name = llm_name.lower()
        if "qwen" in name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if "gpt2" in name:
            return ["c_attn", "c_proj"]
        raise ValueError(
            f"No default LoRA targets for {llm_name!r}. Pass lora_target_modules explicitly."
        )

    def __init__(
        self,
        llm_name: str = "gpt2",  # alternatives: "gpt2-medium", "gpt2-large", "Qwen/Qwen2.5-1.5B"
        d_chronos: int = 9216,  # 768 * 12 channels (both rings, 6 channels each)
        n_soft_tokens: int = 32,
        n_resampler_layers: int = 2,
        prompt: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.15,
        lora_target_modules: list[str] | None = None,
        adapter_dim: int = 256,
        adapter_dropout: float = 0.5,
        use_keystroke: bool = True,
        skip_llm: bool = False,
        use_char_ctc: bool = False,
        ctc_upsample_factor: int = 1,
        ctc_on_resampler: bool = False,
        ctc_no_cross_attn: bool = False,
        freeze_resampler: bool = False,
        use_char_seq2seq: bool = False,
    ):
        super().__init__()
        self.skip_llm = skip_llm
        self.use_keystroke = use_keystroke
        self.use_char_ctc = use_char_ctc
        self.ctc_upsample_factor = ctc_upsample_factor
        self.ctc_on_resampler = ctc_on_resampler
        self.ctc_no_cross_attn = ctc_no_cross_attn
        self.freeze_resampler = freeze_resampler
        self.use_char_seq2seq = use_char_seq2seq

        # --- LLM (skipped entirely in Phase A to save GPU memory) ---
        if not skip_llm:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, torch_dtype=dtype
            )
            for p in self.llm.parameters():
                p.requires_grad = False

            self.lora_enabled = lora_rank > 0
            if self.lora_enabled:
                targets = lora_target_modules or self._default_lora_targets(llm_name)
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=targets,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llm = get_peft_model(self.llm, lora_cfg)
                self.llm = self.llm.to(dtype)  # cast LoRA A/B to bfloat16 to match base model
            self.llm.eval()  # base dropout stays deterministic; LoRA grads still flow
            d_llm = self.llm.config.hidden_size
        else:
            self.tokenizer = None
            self.llm = None
            self.lora_enabled = False
            # Even with the LLM skipped, pull d_llm from the target model's
            # config so Phase A's adapter.out_proj matches Phase B's shape.
            # Using a placeholder (e.g. 256) would make the Phase B
            # strict-load of the Phase A checkpoint crash on the out_proj
            # weight shape mismatch.
            #
            # Fallback table lets Phase A run fully offline (no HF cache hit)
            # for known model families — the previous placeholder behaviour
            # was offline-friendly and we don't want to regress that.
            _KNOWN_HIDDEN = {
                "qwen/qwen2.5-1.5b": 1536,
                "qwen/qwen2.5-0.5b": 896,
                "qwen/qwen2.5-3b": 2048,
                "gpt2": 768,
                "gpt2-medium": 1024,
                "gpt2-large": 1280,
            }
            key = llm_name.lower()
            if key in _KNOWN_HIDDEN:
                d_llm = _KNOWN_HIDDEN[key]
            else:
                from transformers import AutoConfig
                d_llm = AutoConfig.from_pretrained(llm_name).hidden_size

        # --- Trainable adapter ---
        self.adapter = IMUAdapter(
            d_chronos=d_chronos,
            d_llm=d_llm,
            n_soft_tokens=n_soft_tokens,
            adapter_dim=adapter_dim,
            n_resampler_layers=n_resampler_layers,
            dropout=adapter_dropout,
            ctc_upsample_factor=ctc_upsample_factor,
        )
        # Cast adapter to same dtype as LLM for mixed-precision consistency
        self.adapter = self.adapter.to(dtype)

        # Phase B LoRA runs may want to hold the Phase-A grounded adapter
        # rhythm in place while the LLM adjusts. We leave out_proj trainable
        # because LoRA-induced distribution shift may need small rescaling
        # into LLM space.
        if freeze_resampler:
            for p in self.adapter.proj.parameters():
                p.requires_grad = False
            for p in self.adapter.resampler.parameters():
                p.requires_grad = False

        # --- Keystroke head (binary frame-level activity detector) ---
        # Kept in fp32: this is the only path delivering dense gradient to
        # the adapter in Phase A, so we want full-precision master weights
        # even though autocast still runs the forward in bf16.
        self.keystroke_head = (
            KeystrokeHead(adapter_dim, dropout=adapter_dropout)
            if use_keystroke
            else None
        )

        # --- Character CTC head (dense content supervision) ---
        self.char_ctc_head = (
            CharCTCHead(adapter_dim, vocab_size=CHAR_VOCAB_SIZE, dropout=adapter_dropout)
            if use_char_ctc
            else None
        )

        # --- Character seq2seq head (cross-attn decoder on 32 soft tokens) ---
        # Unlike CTC, has no T≥L length constraint: the 32 tokens are treated
        # as compressed memory and can be queried at each decode step.
        self.char_seq2seq_head = (
            CharSeq2SeqDecoder(
                adapter_dim=adapter_dim,
                vocab_size=CHAR_VOCAB_SIZE,
                dropout=adapter_dropout,
            )
            if use_char_seq2seq
            else None
        )

        # --- Prompt prefix ---
        self.prompt_text = prompt or self.DEFAULT_PROMPT
        if not skip_llm:
            self._register_prompt_embeds()

    # --------------------------------------------------------------------- #
    #  Prompt embedding cache
    # --------------------------------------------------------------------- #

    def _register_prompt_embeds(self):
        """Tokenize the text prompt and cache its embeddings (frozen)."""
        ids = self.tokenizer.encode(self.prompt_text, add_special_tokens=False)
        self.register_buffer(
            "prompt_ids", torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        )

    @property
    def _embed_layer(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()

    def _get_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        """(B, S_prompt, d_llm) — frozen text prompt embeddings."""
        embeds = self._embed_layer(self.prompt_ids)  # (1, S_prompt, d)
        return embeds.expand(batch_size, -1, -1)

    # --------------------------------------------------------------------- #
    #  Build the combined input: [prompt | soft tokens | text tokens]
    # --------------------------------------------------------------------- #

    def _build_inputs(
        self,
        soft_tokens: torch.Tensor,
        text_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Returns:
            combined_embeds: (B, S_prompt + S_soft + S_text, d_llm)
            n_prefix:        length of the non-text prefix (prompt + soft tokens)
        """
        B = soft_tokens.size(0)
        parts = [self._get_prompt_embeds(B), soft_tokens]
        n_prefix = parts[0].size(1) + parts[1].size(1)

        if text_ids is not None:
            text_embeds = self._embed_layer(text_ids)
            parts.append(text_embeds)

        return torch.cat(parts, dim=1), n_prefix

    # --------------------------------------------------------------------- #
    #  Forward (training with teacher forcing)
    # --------------------------------------------------------------------- #

    def forward(
        self,
        chronos_embeds: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        chronos_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        With `skip_llm=True` (Phase A) only the adapter + keystroke head run —
        the LLM is not loaded and there is no LM loss. With `skip_llm=False`
        (Phase B) both the LM cross-entropy and the keystroke logits are
        produced.

        Args:
            chronos_embeds: (B, S_enc, d_chronos) — frozen Chronos output
            target_ids:     (B, S_text)           — LLM token IDs (ignored if skip_llm)
            chronos_mask:   (B, S_enc)            — True for padded encoder positions
            labels:         (B, S_text)           — LM loss targets; pad positions = -100

        Returns:
            dict with any of: 'loss', 'logits', 'soft_tokens', 'resampler_out',
            'projected_mean', 'keystroke_logits'
        """
        soft_tokens, resampler_out, projected_mean, proj_seq, proj_seq_up = self.adapter(
            chronos_embeds, chronos_mask
        )

        out: dict[str, torch.Tensor] = {
            "soft_tokens": soft_tokens,
            "resampler_out": resampler_out,
            "projected_mean": projected_mean,
        }

        if self.keystroke_head is not None:
            activity_logits, onset_logits = self.keystroke_head(proj_seq, resampler_out)
            out["keystroke_logits"] = activity_logits
            out["onset_logits"] = onset_logits

        if self.char_ctc_head is not None:
            if self.ctc_on_resampler:
                # Diagnostic: run CTC on the 32 compressed soft tokens to
                # measure how much character info survives resampler bottleneck.
                out["char_logits"] = self.char_ctc_head(resampler_out, None)
            elif self.ctc_no_cross_attn:
                # Ablation: CTC on proj_seq_up with no cross-attn to resampler.
                # Tests whether the diffuse-KV hypothesis explains the 64-token
                # regression (more KV tokens → worse per-frame character gradient).
                out["char_logits"] = self.char_ctc_head(proj_seq_up, None)
            else:
                out["char_logits"] = self.char_ctc_head(proj_seq_up, resampler_out)

        if self.char_seq2seq_head is not None and char_ids is not None:
            out["char_seq2seq_logits"] = self.char_seq2seq_head(resampler_out, char_ids)

        if self.skip_llm or target_ids is None:
            return out

        combined, n_prefix = self._build_inputs(soft_tokens, target_ids)

        # NOTE: do NOT wrap in torch.no_grad() here — gradients must flow
        # back through the LLM to update the adapter. LLM parameters are
        # already frozen (requires_grad=False) so they won't accumulate grads.
        outputs = self.llm(inputs_embeds=combined)

        # Slice logits to only the text region
        # The model predicts next token, so logits at position i predict token i+1
        text_logits = outputs.logits[:, n_prefix - 1 : -1, :]  # (B, S_text, V)

        loss_targets = (labels if labels is not None else target_ids).reshape(-1)
        loss = nn.functional.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            loss_targets,
            ignore_index=-100,
        )

        out["loss"] = loss
        out["logits"] = text_logits
        return out

    # --------------------------------------------------------------------- #
    #  Generation (inference)
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        chronos_embeds: torch.Tensor,
        chronos_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_token: str | None = None,
    ) -> list[str]:
        """
        Autoregressively generate text conditioned on IMU embeddings.

        Returns a list of decoded strings (one per batch element).
        """
        soft_tokens, _, _, _, _ = self.adapter(chronos_embeds, chronos_mask)
        combined, _ = self._build_inputs(soft_tokens)

        # Prepare stop token
        stop_id = None
        if stop_token is not None:
            stop_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
            stop_id = stop_ids[0] if stop_ids else None

        B = combined.size(0)
        device = combined.device
        generated_ids: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B

        # KV cache bootstrap: run the prefix through the model once
        past = None
        out = self.llm(inputs_embeds=combined, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]  # (B, V)

        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                # Top-p (nucleus) filtering
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                mask = cumulative - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(
                    1, torch.multinomial(sorted_probs, 1)
                ).squeeze(-1)
            else:
                next_token = next_logits.argmax(dim=-1)

            # Append to generated sequences
            for i in range(B):
                if not finished[i]:
                    tid = next_token[i].item()
                    generated_ids[i].append(tid)
                    if tid == self.tokenizer.eos_token_id or tid == stop_id:
                        finished[i] = True

            if all(finished):
                break

            # Next step with KV cache
            next_embeds = self._embed_layer(next_token.unsqueeze(1))
            out = self.llm(inputs_embeds=next_embeds, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # --------------------------------------------------------------------- #
    #  Utilities
    # --------------------------------------------------------------------- #

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_adapter(self, path: str):
        ckpt = {"adapter": self.adapter.state_dict()}
        if self.keystroke_head is not None:
            ckpt["keystroke_head"] = self.keystroke_head.state_dict()
        if self.char_ctc_head is not None:
            ckpt["char_ctc_head"] = self.char_ctc_head.state_dict()
        if self.char_seq2seq_head is not None:
            ckpt["char_seq2seq_head"] = self.char_seq2seq_head.state_dict()
        if self.lora_enabled:
            ckpt["lora"] = get_peft_model_state_dict(self.llm)
        torch.save(ckpt, path)

    def load_adapter(self, path: str, **kwargs):
        """
        Load adapter weights (and optional keystroke head / LoRA) from disk.

        Checkpoints from the legacy CTC iteration carry a `ctc_head` key with
        a different output dim; we ignore it with a warning so that the
        adapter weights still transfer cleanly.
        """
        import logging
        log = logging.getLogger("ring2text")

        raw = torch.load(path, **kwargs)
        if isinstance(raw, dict) and "adapter" in raw:
            # strict=False because ctc_upsample may or may not be present in
            # either side (Phase A no-CTC checkpoint vs. Phase B CTC model, etc.)
            missing_a, unexpected_a = self.adapter.load_state_dict(
                raw["adapter"], strict=False,
            )
            if missing_a or unexpected_a:
                log.warning(
                    "[load_adapter] adapter partially loaded from %s "
                    "(missing=%d, unexpected=%d). Typically the ctc_upsample "
                    "layer: present in one model but not the other.",
                    path, len(missing_a), len(unexpected_a),
                )
            if self.char_ctc_head is not None and "char_ctc_head" in raw:
                self.char_ctc_head.load_state_dict(
                    raw["char_ctc_head"], strict=False,
                )
            if self.char_seq2seq_head is not None and "char_seq2seq_head" in raw:
                self.char_seq2seq_head.load_state_dict(
                    raw["char_seq2seq_head"], strict=False,
                )
            if self.keystroke_head is not None and "keystroke_head" in raw:
                # Checkpoints from the single-head KeystrokeHead iteration
                # (activity only, with a depthwise temporal conv) have a
                # different state-dict shape. Use strict=False so the
                # current shared trunk loads whatever overlapping weights
                # exist (q_norm, kv_norm, cross_attn, norm) and the two
                # fresh classifier heads + removed conv are simply
                # reinitialised. This is intentional — the old architecture
                # is being replaced to fix the onset-F1 ceiling.
                missing, unexpected = self.keystroke_head.load_state_dict(
                    raw["keystroke_head"], strict=False,
                )
                if missing or unexpected:
                    log.warning(
                        "[load_adapter] keystroke_head partially loaded from %s "
                        "(missing=%d, unexpected=%d). This is expected when "
                        "moving from the legacy single-head design to the new "
                        "dual-head (activity + onset) head.",
                        path, len(missing), len(unexpected),
                    )
            if "ctc_head" in raw:
                log.warning(
                    "[load_adapter] checkpoint %s contains a legacy 'ctc_head' "
                    "from the character-CTC iteration; ignoring it. Keystroke "
                    "head (if present) will start from random init.",
                    path,
                )
            if "lora" in raw and self.lora_enabled:
                set_peft_model_state_dict(self.llm, raw["lora"])
        else:
            # Old format: plain adapter state dict (backward compat)
            self.adapter.load_state_dict(raw)
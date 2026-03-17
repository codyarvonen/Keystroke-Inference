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
from adapter import IMUAdapter


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
    ):
        super().__init__()

        # --- Frozen LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, dtype=dtype
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

        # --- Trainable adapter ---
        self.adapter = IMUAdapter(
            d_chronos=d_chronos,
            d_llm=d_llm,
            n_soft_tokens=n_soft_tokens,
            adapter_dim=adapter_dim,
            n_resampler_layers=n_resampler_layers,
            dropout=adapter_dropout,
        )
        # Cast adapter to same dtype as LLM for mixed-precision consistency
        self.adapter = self.adapter.to(dtype)

        # --- Prompt prefix ---
        self.prompt_text = prompt or self.DEFAULT_PROMPT
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
        target_ids: torch.Tensor,
        chronos_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with teacher-forced cross-entropy loss.

        Args:
            chronos_embeds: (B, S_enc, d_chronos) — frozen Chronos output
            target_ids:     (B, S_text)           — token IDs for teacher-forced input
            chronos_mask:   (B, S_enc)            — True for padded encoder positions
            labels:         (B, S_text)           — loss targets; pad positions set to
                                                    -100 so EOS is not silently masked
                                                    (falls back to target_ids if None)

        Returns:
            dict with 'loss' and 'logits'
        """
        soft_tokens = self.adapter(chronos_embeds, chronos_mask)
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

        return {"loss": loss, "logits": text_logits}

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
        soft_tokens = self.adapter(chronos_embeds, chronos_mask)
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
        if self.lora_enabled:
            ckpt["lora"] = get_peft_model_state_dict(self.llm)
        torch.save(ckpt, path)

    def load_adapter(self, path: str, **kwargs):
        raw = torch.load(path, **kwargs)
        if isinstance(raw, dict) and "adapter" in raw:
            # New combined format
            self.adapter.load_state_dict(raw["adapter"])
            if "lora" in raw and self.lora_enabled:
                set_peft_model_state_dict(self.llm, raw["lora"])
        else:
            # Old format: plain adapter state dict (backward compat)
            self.adapter.load_state_dict(raw)
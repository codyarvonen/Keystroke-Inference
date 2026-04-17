"""
Dataset for pairing pre-computed Chronos embeddings with ground-truth text.

Expected data layout on disk:
    data_dir/
        sample_0000.pt    # {"embeddings": (S, d_chronos), "text": "hello world"}
        sample_0001.pt
        ...

Or provide a single file:
    dataset.pt            # list of {"embeddings": Tensor, "text": str}

You can also subclass and override __getitem__ to load from your own format
(e.g. HDF5, numpy, database).
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class IMUTextDataset(Dataset):
    """
    Loads pre-computed Chronos embeddings paired with ground-truth text.

    Each sample is a dict with:
        - embeddings: (S_enc, d_chronos) float tensor
        - text:       str — the ground-truth typed text
    """

    def __init__(
        self,
        data_dir: str | None = None,
        data_file: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        max_text_len: int = 128,
    ):
        super().__init__()
        assert data_dir or data_file, "Provide either data_dir or data_file"

        if data_file:
            self.samples = torch.load(data_file, weights_only=False)
        else:
            # Each .pt on disk may itself be a list (preprocess saves a whole
            # split as a single list-of-dicts); flatten so self.samples is
            # always a flat list of sample dicts.
            data_path = Path(data_dir)
            files = sorted(data_path.glob("*.pt"))
            self.samples = []
            for f in files:
                loaded = torch.load(f, weights_only=False)
                if isinstance(loaded, list):
                    self.samples.extend(loaded)
                else:
                    self.samples.append(loaded)

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        # Coerce to tensor so legacy numpy-format embeddings don't trip the
        # `.float()` / `.size(0)` calls below.
        emb = torch.as_tensor(sample["embeddings"])
        item = {"embeddings": emb.float()}
        text = sample["text"]
        item["text"] = text

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,  # adds EOS at end of text
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            # Labels: -100 at pad positions so pad_token == eos_token doesn't
            # suppress the EOS signal in the cross-entropy loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            item["input_ids"] = input_ids
            item["attention_mask"] = attention_mask
            item["labels"] = labels

        # Per-frame binary keystroke-activity target (length matches the
        # encoder output S of this sample's `embeddings`). Built in
        # preprocess.py from the PKL key_times intervals; missing only on
        # legacy embeddings files preprocessed before the keystroke iteration.
        if "keystroke_active" in sample:
            ka = torch.as_tensor(sample["keystroke_active"])
            # Per-sample encoder length and mask length must match; padding in
            # collate_fn is tied to embeddings.size(0), so a stale/mis-sized
            # mask would supervise real frames with zero-padded targets.
            assert ka.size(0) == emb.size(0), (
                f"keystroke_active length {ka.size(0)} != embeddings length "
                f"{emb.size(0)} — rerun preprocess."
            )
            item["keystroke_active"] = ka.float()

            # Per-frame onset target. Prefer the per-event onset mask emitted
            # by preprocess.py (marks each keypress's start frame, including
            # keys that overlap another key's press). Fall back to the diff
            # of the activity mask for legacy embeddings — the fallback misses
            # onsets of keys pressed while an earlier key is still held.
            if "keystroke_onset" in sample:
                ko = torch.as_tensor(sample["keystroke_onset"])
                assert ko.size(0) == emb.size(0), (
                    f"keystroke_onset length {ko.size(0)} != embeddings length "
                    f"{emb.size(0)} — rerun preprocess."
                )
                item["keystroke_onset"] = ko.float()
            else:
                ka_bool = ka > 0.5
                onset = torch.zeros_like(ka, dtype=torch.float32)
                onset[0] = ka_bool[0].float()
                onset[1:] = (ka_bool[1:] & ~ka_bool[:-1]).float()
                item["keystroke_onset"] = onset

        # Per-sample character-level CTC targets. Emitted by preprocess.py via
        # char_vocab.encode(text) — missing only on legacy embeddings.
        if "char_targets" in sample:
            item["char_targets"] = torch.as_tensor(
                sample["char_targets"], dtype=torch.long
            )
        return item


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collates variable-length Chronos embeddings with padding,
    and stacks tokenized text.
    """
    # Pad Chronos embeddings to the longest in the batch
    embeds = [b["embeddings"] for b in batch]
    max_len = max(e.size(0) for e in embeds)
    d = embeds[0].size(1)

    padded_embeds = torch.zeros(len(batch), max_len, d)
    embed_mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padded

    for i, e in enumerate(embeds):
        L = e.size(0)
        padded_embeds[i, :L] = e
        embed_mask[i, :L] = False

    # Encoder input lengths (frames that are NOT padded). Reused as the
    # valid-frame mask for the per-frame keystroke BCE loss.
    embed_lens = torch.tensor(
        [e.size(0) for e in embeds], dtype=torch.long
    )

    result = {
        "chronos_embeds": padded_embeds,
        "chronos_mask": embed_mask,
        "embed_lens": embed_lens,
        "texts": [b["text"] for b in batch],
    }

    if "input_ids" in batch[0]:
        result["target_ids"] = torch.stack([b["input_ids"] for b in batch])
        result["target_mask"] = torch.stack([b["attention_mask"] for b in batch])
        result["target_labels"] = torch.stack([b["labels"] for b in batch])

    # Emit keystroke targets only if every sample has them — mixed batches
    # (legacy + new embeddings) would KeyError mid-epoch otherwise.
    if all("keystroke_active" in b for b in batch):
        # Pad keystroke targets to longest in batch with zeros (no activity);
        # the loss masks beyond `embed_lens` so padded positions don't count.
        padded_targets = torch.zeros(len(batch), max_len, dtype=torch.float32)
        padded_onsets = torch.zeros(len(batch), max_len, dtype=torch.float32)
        for i, b in enumerate(batch):
            L = b["keystroke_active"].size(0)
            padded_targets[i, :L] = b["keystroke_active"]
            padded_onsets[i, :L] = b["keystroke_onset"]
        keystroke_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, L in enumerate(embed_lens.tolist()):
            keystroke_mask[i, :L] = True
        result["keystroke_targets"] = padded_targets
        result["onset_targets"] = padded_onsets
        result["keystroke_mask"] = keystroke_mask

    if all("char_targets" in b for b in batch):
        char_lens = torch.tensor(
            [b["char_targets"].size(0) for b in batch], dtype=torch.long
        )
        max_L = int(char_lens.max().item()) if len(batch) > 0 else 0
        # Right-pad with 0 (BLANK_ID). CTC ignores positions beyond char_lens.
        char_ids = torch.zeros(len(batch), max_L, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["char_targets"].size(0)
            char_ids[i, :L] = b["char_targets"]
        result["char_ids"] = char_ids
        result["char_lens"] = char_lens

    return result
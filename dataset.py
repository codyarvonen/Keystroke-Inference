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

import os
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
            data_path = Path(data_dir)
            files = sorted(data_path.glob("*.pt"))
            self.samples = [torch.load(f, weights_only=False) for f in files]

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        item = {"embeddings": sample["embeddings"].float()}

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                sample["text"],
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
        else:
            item["text"] = sample["text"]

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

    result = {
        "chronos_embeds": padded_embeds,
        "chronos_mask": embed_mask,
    }

    if "input_ids" in batch[0]:
        result["target_ids"] = torch.stack([b["input_ids"] for b in batch])
        result["target_mask"] = torch.stack([b["attention_mask"] for b in batch])
        result["target_labels"] = torch.stack([b["labels"] for b in batch])

    return result
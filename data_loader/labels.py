from typing import Any, Dict, List

import numpy as np
from transformers import GPT2TokenizerFast

from .config import DataConfig
from .windows import WindowRecord
from utils.keystroke import (
    get_keystroke_events,
    parse_key_name,
    translate_to_text,
    post_process_text,
)


def slice_events_to_window(
    events: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> List[Dict[str, Any]]:
    """Return events whose timestamp falls within [start_time, end_time)."""
    return [e for e in events if start_time <= e["timestamp"] < end_time]


def attach_labels_to_windows(
    windows: List[WindowRecord],
    keystroke_data: Dict[str, Any],
    cfg: DataConfig,
    tokenizer: GPT2TokenizerFast,
) -> List[WindowRecord]:
    """
    Populate raw_key_sequence / clean_text / token_ids on each window.
    Skips windows that end up with no tokens (token_ids remains None).
    """
    events = get_keystroke_events(keystroke_data)

    for w in windows:
        window_events = slice_events_to_window(events, w.start_time, w.end_time)
        if not window_events:
            continue

        raw_seq = [parse_key_name(ev["key"]) for ev in window_events]
        w.raw_key_sequence = raw_seq

        clean = translate_to_text(raw_seq)
        clean = post_process_text(clean)
        w.clean_text = clean

        if not clean:
            w.token_ids = None
            w.token_length = 0
            continue

        tokens = tokenizer.encode(clean, add_special_tokens=False)

        if not tokens:
            w.token_ids = None
            w.token_length = 0
            continue

        max_len = cfg.max_tokens
        token_ids = tokens[:max_len]
        token_length = len(token_ids)

        if token_length < max_len:
            pad_id = tokenizer.eos_token_id
            token_ids = token_ids + [pad_id] * (max_len - token_length)

        w.token_ids = np.array(token_ids, dtype=np.int64)
        w.token_length = token_length

    return windows


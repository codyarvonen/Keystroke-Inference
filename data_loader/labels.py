from typing import Any, Dict, List

import numpy as np
from transformers import GPT2TokenizerFast

from .config import DataConfig
from .windows import WindowRecord


def regenerate_key_sequence_for_session(keystroke_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a keystroke_data dict (from Macbook.pkl) into a cleaned list of
    event dicts with fields: timestamp, key, end.
    """
    if "key_times" not in keystroke_data:
        return []

    events: List[Dict[str, Any]] = []
    for key, times_list in keystroke_data["key_times"].items():
        for time_dict in times_list:
            if "start" in time_dict and time_dict["start"] is not None:
                events.append(
                    {
                        "timestamp": time_dict["start"],
                        "key": key,
                        "end": time_dict.get("end"),
                    }
                )

    events.sort(key=lambda x: x["timestamp"])

    events = remove_sync_artifacts(events)
    events = detect_command_sequences(events)
    return events


def parse_key_name(key_str: str) -> str:
    special_keys = {
        "Key.space": " ",
        "Key.backspace": "<BACKSPACE>",
        "Key.enter": "\n",
        "Key.tab": "\t",
        "Key.shift": "<SHIFT>",
        "Key.shift_r": "<SHIFT>",
        "Key.ctrl": "<CTRL>",
        "Key.control": "<CTRL>",
        "Key.alt": "<ALT>",
        "Key.cmd": "<CMD>",
        "Key.cmd_r": "<CMD>",
        "Key.esc": "<ESC>",
    }

    if key_str in special_keys:
        return special_keys[key_str]

    if len(key_str) == 1:
        return key_str

    return f"<{key_str}>"


def remove_sync_artifacts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(events) < 10:
        return events

    # --- Handle sync patterns at the very start ---
    start_pattern: List[str] = []
    for i in range(min(20, len(events))):
        key = events[i]["key"]
        if key in ("l", "r"):
            start_pattern.append(key)
        else:
            break

    if len(start_pattern) >= 4 and all(k in ("l", "r") for k in start_pattern):
        events = events[len(start_pattern) :]

    # --- Handle sync patterns at the very end (with optional Ctrl+C) ---
    control_c_start = None
    for i in range(len(events) - 1, max(len(events) - 10, -1), -1):
        key = events[i]["key"]
        if key in ("Key.ctrl", "Key.control"):
            if i + 1 < len(events) and events[i + 1]["key"] == "c":
                control_c_start = i
                break
        elif key == "c" and i > 0 and events[i - 1]["key"] in ("Key.ctrl", "Key.control"):
            control_c_start = i - 1
            break

    search_start = control_c_start if control_c_start is not None else len(events)
    end_pattern: List[str] = []
    pattern_start_idx = search_start

    for i in range(search_start - 1, max(search_start - 30, -1), -1):
        key = events[i]["key"]
        if key in ("l", "r"):
            end_pattern.insert(0, key)
            pattern_start_idx = i
        else:
            break

    if len(end_pattern) >= 4 and all(k in ("l", "r") for k in end_pattern):
        if control_c_start is not None:
            events = events[:pattern_start_idx]
    elif control_c_start is not None:
        events = events[:control_c_start]

    # --- Handle sync-like l/r bursts in the middle of the sequence ---
    # We look for contiguous runs of l/r of length >= 4 anywhere in the event list
    # (e.g., "llllrrrr" or "rrrllll"). These are treated as sync artifacts and removed.
    cleaned: List[Dict[str, Any]] = []
    i = 0
    n = len(events)
    while i < n:
        if events[i]["key"] in ("l", "r"):
            j = i
            while j < n and events[j]["key"] in ("l", "r"):
                j += 1
            segment_len = j - i
            if segment_len >= 4:
                # Skip this entire l/r burst
                i = j
                continue
        cleaned.append(events[i])
        i += 1

    events = cleaned

    return events


def detect_command_sequences(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cmd_keys = ["Key.cmd", "Key.cmd_r", "Key.ctrl", "Key.control"]
    to_remove: set[int] = set()

    i = 0
    while i < len(events):
        event = events[i]
        if event["key"] in cmd_keys:
            cmd_start = event["timestamp"]
            cmd_end = event.get("end")
            if cmd_end is None:
                cmd_end = cmd_start + 0.3

            j = i + 1
            found_overlap = False
            while j < len(events) and events[j]["timestamp"] < cmd_end + 0.2:
                char_event = events[j]
                char_key = char_event["key"]
                char_start = char_event["timestamp"]

                if len(char_key) == 1 and char_key.isalnum():
                    char_end = char_event.get("end") or (char_start + 0.1)

                    if cmd_start - 0.05 <= char_start <= cmd_end + 0.1:
                        to_remove.add(i)
                        to_remove.add(j)
                        found_overlap = True

                        k = j + 1
                        while k < len(events) and events[k]["timestamp"] < char_end + 0.2:
                            if events[k]["key"] in cmd_keys:
                                to_remove.add(k)
                            k += 1
                        break
                j += 1

            if not found_overlap:
                j = i + 1
                consecutive_cmds = 0
                while j < len(events) and events[j]["timestamp"] < cmd_end + 0.3:
                    if events[j]["key"] in cmd_keys:
                        consecutive_cmds += 1
                        to_remove.add(j)
                    j += 1

                if consecutive_cmds > 0:
                    to_remove.add(i)

        i += 1

    if to_remove:
        events = [e for idx, e in enumerate(events) if idx not in to_remove]

    return events


def translate_to_text(key_sequence: List[str]) -> str:
    text: List[str] = []

    ignore_keys = {
        "<SHIFT>",
        "<CTRL>",
        "<CMD>",
        "<ALT>",
        "<Key.left>",
        "<Key.right>",
        "<Key.up>",
        "<Key.down>",
        "<Key.home>",
        "<Key.end>",
        "<Key.page_up>",
        "<Key.page_down>",
    }

    i = 0
    while i < len(key_sequence):
        key = key_sequence[i]

        if key in ignore_keys or (key.startswith("<Key.") and key.endswith(">")):
            i += 1
            continue

        if key == "<BACKSPACE>":
            if text:
                text.pop()
            i += 1
            continue

        if key == "<ENTER>":
            text.append("\n")
            i += 1
            continue

        if key == "<TAB>":
            text.append("\t")
            i += 1
            continue

        if key.startswith("<") and key.endswith(">"):
            i += 1
            continue

        if len(key) == 1:
            text.append(key)
            i += 1
        else:
            i += 1

    return "".join(text)


def post_process_text(text: str) -> str:
    import re

    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])(?=\s*[A-Za-z])", r"\1 \2", text)
    text = re.sub(r"(\d+)\)\s*([A-Z])", r"\1) \2", text)
    text = re.sub(r"(\d+\.)\s*([A-Z])", r"\1 \2", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r" {3,}", " ", text)
    return text


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
    events = regenerate_key_sequence_for_session(keystroke_data)

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


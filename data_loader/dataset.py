from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

from .config import DataConfig
from .labels import attach_labels_to_windows
from .sessions import discover_sessions, load_session_raw
from .splits import make_splits
from .windows import WindowRecord, align_rings_to_grid, build_imu_windows


class KeystrokeIMUDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[WindowRecord],
        indices: Sequence[int],
        cfg: DataConfig,
    ) -> None:
        self.windows = windows
        self.indices = list(indices)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        w = self.windows[self.indices[idx]]

        imu_L = torch.from_numpy(w.imu_L) if w.imu_L is not None else None
        imu_R = torch.from_numpy(w.imu_R) if w.imu_R is not None else None

        # Ensure IMU tensors exist for both sides according to rings_used
        # (zeroing out the unused side preserves shapes for the model).

        if self.cfg.rings_used == "L":
            imu_R = torch.zeros_like(imu_L) if imu_L is not None else None
        elif self.cfg.rings_used == "R":
            imu_L = torch.zeros_like(imu_R) if imu_R is not None else None

        # Prepare all label variants
        raw_keys = w.raw_key_sequence
        clean_text = w.clean_text
        token_ids_tensor = (
            torch.from_numpy(w.token_ids) if w.token_ids is not None else None
        )

        # Canonical target alias based on config
        if self.cfg.target_variant == "raw_keystrokes":
            target = raw_keys
        elif self.cfg.target_variant == "clean_text":
            target = clean_text
        else:  # "clean_tokens"
            target = token_ids_tensor

        sample: Dict[str, Any] = {
            "imu_l": imu_L,
            "imu_r": imu_R,
            # All label views
            "raw_key_sequence": raw_keys,
            "clean_text": clean_text,
            "token_ids": token_ids_tensor,
            "token_length": w.token_length,
            # Canonical target (driven by cfg.target_variant)
            "target": target,
            # Metadata
            "subject": w.subject,
            "session": w.session,
            "window_start": w.start_time,
            "window_end": w.end_time,
        }

        return sample


def build_session_windows(
    subject: str,
    session: str,
    files: Dict[str, Any],
    cfg: DataConfig,
    tokenizer: GPT2TokenizerFast,
    is_test: bool,
) -> List[WindowRecord]:
    raw = load_session_raw(files)

    imu_L = raw.get("imu_L")
    imu_R = raw.get("imu_R")
    keystrokes = raw["keystrokes"]

    if imu_L is None or imu_R is None:
        return []

    gaps_L = raw.get("gaps_L") or []
    gaps_R = raw.get("gaps_R") or []
    # Any large gap in either ring should mark that time region as low-quality.
    bad_intervals = list(gaps_L) + list(gaps_R)

    time_grid, imu_L_grid, imu_R_grid = align_rings_to_grid(imu_L, imu_R)

    windows = build_imu_windows(
        time_grid=time_grid,
        imu_L_grid=imu_L_grid,
        imu_R_grid=imu_R_grid,
        subject=subject,
        session=session,
        cfg=cfg,
        is_test=is_test,
        bad_intervals=bad_intervals,
    )

    windows = attach_labels_to_windows(
        windows=windows,
        keystroke_data=keystrokes,
        cfg=cfg,
        tokenizer=tokenizer,
    )

    windows = [w for w in windows if w.token_ids is not None]
    return windows


def build_all_windows(
    cfg: DataConfig,
    tokenizer: GPT2TokenizerFast,
) -> List[WindowRecord]:
    session_files = discover_sessions(cfg)
    session_keys = sorted(session_files.keys())

    all_windows: List[WindowRecord] = []

    for key in session_keys:
        subject, session = key.split("_", 1)

        is_test = False
        if cfg.split_strategy == "LOSO" and cfg.test_session is not None:
            is_test = key == cfg.test_session
        elif cfg.split_strategy == "LOPO" and cfg.test_subject is not None:
            is_test = subject == cfg.test_subject

        ws = build_session_windows(
            subject=subject,
            session=session,
            files=session_files[key],
            cfg=cfg,
            tokenizer=tokenizer,
            is_test=is_test,
        )
        all_windows.extend(ws)

    return all_windows


def make_dataloaders(
    cfg: DataConfig,
    tokenizer: GPT2TokenizerFast,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[WindowRecord]]:
    """
    Build all windows, create splits, and return
    (train_loader, val_loader, test_loader, windows).
    """
    all_windows = build_all_windows(cfg, tokenizer)
    train_ids, val_ids, test_ids = make_splits(all_windows, cfg)

    train_ds = KeystrokeIMUDataset(all_windows, train_ids, cfg)
    val_ds = KeystrokeIMUDataset(all_windows, val_ids, cfg)
    test_ds = KeystrokeIMUDataset(all_windows, test_ids, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=keystroke_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=keystroke_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=keystroke_collate_fn,
    )

    return train_loader, val_loader, test_loader, all_windows


def keystroke_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that stacks tensor fields and leaves
    variable-length / non-tensor fields as Python lists.
    """
    if not batch:
        return {}

    collated: Dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        # If all values are None, keep None
        if all(v is None for v in values):
            collated[key] = None
            continue

        # If all non-None values are tensors and there are no Nones, stack them
        if all(isinstance(v, torch.Tensor) for v in values):
            collated[key] = torch.stack(values, dim=0)
            continue

        # Otherwise, keep as a list (e.g., strings, lists of strings, etc.)
        collated[key] = values

    return collated



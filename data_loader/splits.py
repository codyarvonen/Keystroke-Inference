from typing import List, Sequence, Tuple

import numpy as np

from .config import DataConfig
from .windows import WindowRecord


def make_splits(
    windows: Sequence[WindowRecord],
    cfg: DataConfig,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (train_ids, val_ids, test_ids) as index lists into `windows`.
    """
    n = len(windows)
    all_indices = np.arange(n)

    test_mask = np.zeros(n, dtype=bool)

    if cfg.split_strategy == "LOSO" and cfg.test_session is not None:
        for i, w in enumerate(windows):
            sess_key = f"{w.subject}_{w.session}"
            if sess_key == cfg.test_session:
                test_mask[i] = True

    elif cfg.split_strategy == "LOPO" and cfg.test_subject is not None:
        for i, w in enumerate(windows):
            if w.subject == cfg.test_subject:
                test_mask[i] = True

    test_ids = all_indices[test_mask].tolist()

    trainval_ids = all_indices[~test_mask]

    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(trainval_ids)

    n_val = int(len(trainval_ids) * cfg.val_ratio)
    val_ids = trainval_ids[:n_val].tolist()
    train_ids = trainval_ids[n_val:].tolist()

    return train_ids, val_ids, test_ids


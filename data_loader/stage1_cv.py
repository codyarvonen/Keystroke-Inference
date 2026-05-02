"""
Leave-one-participant-out (LOPO) and leave-one-session-out (LOSO) fold definitions for Stage-1 pool exports.

Expects JSONL rows with ``subject`` and ``session_key`` (as written by export_stage1_to_dir).
Train/validation splits are random row subsets of the non-test pool (default 80/20, seeded).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np

CVMode = Literal["lopo", "loso_all", "loso_personalized"]


def train_val_split_indices(
    trainval_indices: List[int],
    val_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    """Shuffle ``trainval_indices`` and take the first ``n_val`` as validation."""
    idxs = list(trainval_indices)
    if not idxs:
        return [], []
    rng.shuffle(idxs)
    n = len(idxs)
    n_val = int(round(n * float(val_ratio)))
    if n > 1:
        n_val = min(n_val, n - 1)
    else:
        n_val = 0
    val_i = idxs[:n_val]
    train_i = idxs[n_val:]
    return train_i, val_i


def build_stage1_cv_folds(
    pool_rows: List[Dict[str, Any]],
    mode: CVMode,
    *,
    val_ratio: float,
    split_seed: int,
) -> List[Dict[str, Any]]:
    """
    Build fold specs with global row indices into ``pool_rows`` (order preserved).

    Each fold has:
      - fold_index, cv_mode, train_indices, val_indices, test_indices
      - test_subject and/or test_session, personalized_subject (mode-dependent)
    """
    n = len(pool_rows)
    if n == 0:
        return []

    rng = np.random.default_rng(int(split_seed))
    folds: List[Dict[str, Any]] = []
    fi = 0

    if mode == "lopo":
        subjects = sorted({str(r["subject"]) for r in pool_rows})
        if len(subjects) < 2:
            raise ValueError("LOPO requires at least 2 distinct subjects in the pool.")
        for subj in subjects:
            test_idx = [i for i in range(n) if str(pool_rows[i]["subject"]) == subj]
            trainval_idx = [i for i in range(n) if str(pool_rows[i]["subject"]) != subj]
            tr, va = train_val_split_indices(trainval_idx, val_ratio, rng)
            folds.append(
                {
                    "fold_index": fi,
                    "cv_mode": mode,
                    "test_subject": subj,
                    "test_session": None,
                    "personalized_subject": None,
                    "train_indices": tr,
                    "val_indices": va,
                    "test_indices": test_idx,
                }
            )
            fi += 1
        return folds

    if mode == "loso_all":
        sessions = sorted({str(r["session_key"]) for r in pool_rows})
        if len(sessions) < 2:
            raise ValueError("LOSO (all) requires at least 2 distinct sessions in the pool.")
        for sk in sessions:
            test_idx = [i for i in range(n) if str(pool_rows[i]["session_key"]) == sk]
            trainval_idx = [i for i in range(n) if str(pool_rows[i]["session_key"]) != sk]
            tr, va = train_val_split_indices(trainval_idx, val_ratio, rng)
            folds.append(
                {
                    "fold_index": fi,
                    "cv_mode": mode,
                    "test_subject": None,
                    "test_session": sk,
                    "personalized_subject": None,
                    "train_indices": tr,
                    "val_indices": va,
                    "test_indices": test_idx,
                }
            )
            fi += 1
        return folds

    if mode == "loso_personalized":
        subjects = sorted({str(r["subject"]) for r in pool_rows})
        for subj in subjects:
            p_idx = [i for i in range(n) if str(pool_rows[i]["subject"]) == subj]
            sess_keys = sorted({str(pool_rows[i]["session_key"]) for i in p_idx})
            if len(sess_keys) < 2:
                continue
            for sk in sess_keys:
                test_idx = [i for i in p_idx if str(pool_rows[i]["session_key"]) == sk]
                trainval_idx = [i for i in p_idx if str(pool_rows[i]["session_key"]) != sk]
                if not trainval_idx:
                    continue
                tr, va = train_val_split_indices(trainval_idx, val_ratio, rng)
                folds.append(
                    {
                        "fold_index": fi,
                        "cv_mode": mode,
                        "test_subject": None,
                        "test_session": sk,
                        "personalized_subject": subj,
                        "train_indices": tr,
                        "val_indices": va,
                        "test_indices": test_idx,
                    }
                )
                fi += 1
        if not folds:
            raise ValueError(
                "LOSO (personalized): no folds (each participant needs at least 2 sessions)."
            )
        return folds

    raise ValueError(f"Unknown CV mode: {mode!r}")

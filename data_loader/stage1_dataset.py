from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import DataConfig, Stage1ExportConfig
from .sessions import discover_sessions, load_session_raw
from .windows import align_rings_to_grid

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


class Stage1IMUKeyDataset(Dataset):  # type: ignore[misc]
    """
    Per-key IMU windows with raw keystroke class ids (from export JSONL).

    Expects rows produced by export_stage1_to_dir (label_id, imu_idx_start/end).
    Lazily caches aligned L/R IMU grids per session.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        cfg: Stage1ExportConfig,
        rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("Stage1IMUKeyDataset requires PyTorch (torch).")

        self.cfg = cfg
        self._path = Path(jsonl_path)
        if rows is not None:
            self._rows: List[Mapping[str, Any]] = list(rows)
        else:
            from .stage1 import load_stage1_jsonl_rows

            self._rows = load_stage1_jsonl_rows(self._path)

        base = DataConfig(
            data_dir=cfg.data_dir,
            include_sessions=cfg.include_sessions,
            exclude_sessions=cfg.exclude_sessions,
        )
        self._session_files = discover_sessions(base)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def _get_grids(self, session_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if session_key not in self._cache:
            if session_key not in self._session_files:
                raise KeyError(f"Unknown session_key (not in data_dir): {session_key}")
            files = self._session_files[session_key]
            raw = load_session_raw(files)
            imu_L = raw.get("imu_L")
            imu_R = raw.get("imu_R")
            if imu_L is None or imu_R is None:
                raise ValueError(f"Session {session_key} missing IMU L/R")
            _, L, R = align_rings_to_grid(
                imu_L,
                imu_R,
                target_rate_hz=self.cfg.target_rate_hz,
            )
            self._cache[session_key] = (L, R)
        return self._cache[session_key]  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._rows[idx]
        session_key = str(row["session_key"])
        L, R = self._get_grids(session_key)
        i0 = int(row["imu_idx_start"])
        i1 = int(row["imu_idx_end"])
        imu_l = np.ascontiguousarray(L[i0:i1], dtype=np.float32)
        imu_r = np.ascontiguousarray(R[i0:i1], dtype=np.float32)

        if self.cfg.rings_used == "L":
            imu_r = np.zeros_like(imu_l)
        elif self.cfg.rings_used == "R":
            imu_l = np.zeros_like(imu_r)

        label_id = int(row["label_id"])
        sample: Dict[str, Any] = {
            "imu_l": torch.from_numpy(imu_l),
            "imu_r": torch.from_numpy(imu_r),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "session_key": session_key,
            "split": row.get("split"),
            "key_press_ts": float(row["key_press_ts"]),
            "key_token": str(row["key_token"]),
            "is_unk": bool(row.get("is_unk", False)),
        }
        return sample

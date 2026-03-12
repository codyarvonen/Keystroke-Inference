from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig
from .sessions import load_session_raw


def get_time_column(df: pd.DataFrame) -> str:
    if "Effective Timestamp" in df.columns:
        return "Effective Timestamp"
    if "Time Stamp" in df.columns:
        return "Time Stamp"
    raise ValueError("No timestamp column found.")


def align_rings_to_grid(
    imu_L: pd.DataFrame,
    imu_R: pd.DataFrame,
    target_rate_hz: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (time_grid, imu_L_grid, imu_R_grid)
    where imu_*_grid have shape (T, 6) for the 6 IMU channels.
    """
    t_col_L = get_time_column(imu_L)
    t_col_R = get_time_column(imu_R)

    tL = imu_L[t_col_L].to_numpy()
    tR = imu_R[t_col_R].to_numpy()

    t_start = max(tL.min(), tR.min())
    t_end = min(tL.max(), tR.max())

    dt = 1.0 / target_rate_hz
    time_grid = np.arange(t_start, t_end, dt, dtype=np.float64)

    imu_cols = ["Accel-x", "Accel-y", "Accel-z", "Gyro-x", "Gyro-y", "Gyro-z"]
    L_vals = imu_L[imu_cols].to_numpy(dtype=np.float32)
    R_vals = imu_R[imu_cols].to_numpy(dtype=np.float32)

    imu_L_grid = np.empty((time_grid.shape[0], len(imu_cols)), dtype=np.float32)
    imu_R_grid = np.empty_like(imu_L_grid)
    for i in range(len(imu_cols)):
        imu_L_grid[:, i] = np.interp(time_grid, tL, L_vals[:, i])
        imu_R_grid[:, i] = np.interp(time_grid, tR, R_vals[:, i])

    return time_grid, imu_L_grid, imu_R_grid


@dataclass
class WindowRecord:
    subject: str
    session: str
    start_time: float
    end_time: float
    imu_L: Optional[np.ndarray]  # shape (T_window, 6)
    imu_R: Optional[np.ndarray]  # shape (T_window, 6)

    # Labels populated later
    raw_key_sequence: Optional[List[str]] = None
    clean_text: Optional[str] = None
    token_ids: Optional[np.ndarray] = None  # shape (max_tokens,)
    token_length: Optional[int] = None


def build_imu_windows(
    time_grid: np.ndarray,
    imu_L_grid: np.ndarray,
    imu_R_grid: np.ndarray,
    subject: str,
    session: str,
    cfg: DataConfig,
    is_test: bool,
    bad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[WindowRecord]:
    windows: List[WindowRecord] = []

    window_size = cfg.window_size_s
    stride = cfg.test_stride_s if is_test else cfg.train_stride_s

    t_start = float(time_grid[0])
    t_end = float(time_grid[-1])

    current_start = t_start

    def window_overlaps_bad_interval(
        w_start: float,
        w_end: float,
        intervals: Sequence[Tuple[float, float]],
    ) -> bool:
        for g_start, g_end in intervals:
            if w_start < g_end and w_end > g_start:
                return True
        return False
    while current_start + window_size <= t_end:
        current_end = current_start + window_size

        mask = (time_grid >= current_start) & (time_grid < current_end)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            current_start += stride
            continue

        if bad_intervals:
            if window_overlaps_bad_interval(current_start, current_end, bad_intervals):
                current_start += stride
                continue

        imu_L_win = imu_L_grid[idx] if cfg.rings_used in ("L", "both") else None
        imu_R_win = imu_R_grid[idx] if cfg.rings_used in ("R", "both") else None

        windows.append(
            WindowRecord(
                subject=subject,
                session=session,
                start_time=current_start,
                end_time=current_end,
                imu_L=imu_L_win,
                imu_R=imu_R_win,
            )
        )

        current_start += stride

    return windows



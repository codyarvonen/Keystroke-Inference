from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd

from .config import DataConfig, make_session_key, parse_subject_session


def discover_sessions(cfg: DataConfig) -> Dict[str, Dict[str, Path]]:
    """
    Discover available sessions under cfg.data_dir.

    Returns a mapping:
        session_key -> {
            "imu_L": Path,
            "imu_R": Path,
            "keystrokes": Path,
        }
    where session_key is of the form "<subject>_<session>".
    """
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    sessions: Dict[str, Dict[str, Path]] = {}

    # IMU CSVs
    for csv_path in data_dir.glob("*.csv"):
        subject, session = parse_subject_session(csv_path.name)
        session_key = make_session_key(subject, session)

        # Apply include/exclude filters if configured
        if cfg.include_sessions is not None and session_key not in cfg.include_sessions:
            continue
        if cfg.exclude_sessions is not None and session_key in cfg.exclude_sessions:
            continue

        entry = sessions.setdefault(session_key, {})

        # Decide whether this is left or right ring
        name = csv_path.name
        if "DIBS-L" in name:
            entry["imu_L"] = csv_path
        elif "DIBS-R" in name:
            entry["imu_R"] = csv_path

    # Keystroke PKLs
    for pkl_path in data_dir.glob("*_Macbook.pkl"):
        subject, session = parse_subject_session(pkl_path.name)
        session_key = make_session_key(subject, session)

        if cfg.include_sessions is not None and session_key not in cfg.include_sessions:
            continue
        if cfg.exclude_sessions is not None and session_key in cfg.exclude_sessions:
            continue

        entry = sessions.setdefault(session_key, {})
        entry["keystrokes"] = pkl_path

    # Filter out incomplete sessions that are missing keystrokes or IMU files
    complete_sessions: Dict[str, Dict[str, Path]] = {}
    for key, files in sessions.items():
        if "keystrokes" not in files:
            continue
        # We require at least one IMU side; typically both L and R are present.
        if "imu_L" not in files and "imu_R" not in files:
            continue
        complete_sessions[key] = files

    if not complete_sessions:
        raise RuntimeError(f"No complete sessions found under {data_dir}")

    return complete_sessions


def _get_time_column(df: pd.DataFrame) -> str:
    if "Effective Timestamp" in df.columns:
        return "Effective Timestamp"
    if "Time Stamp" in df.columns:
        return "Time Stamp"
    raise ValueError("No timestamp column found in IMU dataframe.")


def _compute_large_gaps(
    df: pd.DataFrame,
    time_col: str,
    factor: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Identify large gaps in the time series where the timestamp difference
    between consecutive samples exceeds `factor * median_dt`.

    Returns:
        List of (gap_start, gap_end) in seconds.
    """
    if df.empty:
        return []

    t = df[time_col].to_numpy()
    if t.size < 2:
        return []

    dt = t[1:] - t[:-1]
    median_dt = float(pd.Series(dt).median())
    if median_dt <= 0.0:
        return []

    threshold = factor * median_dt
    gaps: List[Tuple[float, float]] = []
    for i, delta in enumerate(dt):
        if delta > threshold:
            gaps.append((float(t[i]), float(t[i + 1])))

    return gaps


def load_session_raw(files: Mapping[str, Path]) -> Dict[str, object]:
    """
    Load raw IMU (L/R) and keystroke data for one session.

    Returns:
        {
            "imu_L": pd.DataFrame | None,
            "imu_R": pd.DataFrame | None,
            "keystrokes": dict,
        }
    """
    result: Dict[str, object] = {}

    imu_cols = ["Accel-x", "Accel-y", "Accel-z", "Gyro-x", "Gyro-y", "Gyro-z"]

    if "imu_L" in files:
        df_L = pd.read_csv(files["imu_L"])
        # Drop rows containing NaNs in IMU sensor columns
        df_L = df_L.dropna(subset=[c for c in imu_cols if c in df_L.columns])
        t_col_L = _get_time_column(df_L)
        result["gaps_L"] = _compute_large_gaps(df_L, t_col_L)
        result["imu_L"] = df_L

    if "imu_R" in files:
        df_R = pd.read_csv(files["imu_R"])
        df_R = df_R.dropna(subset=[c for c in imu_cols if c in df_R.columns])
        t_col_R = _get_time_column(df_R)
        result["gaps_R"] = _compute_large_gaps(df_R, t_col_R)
        result["imu_R"] = df_R

    import pickle

    with open(files["keystrokes"], "rb") as f:
        keystrokes = pickle.load(f)
    result["keystrokes"] = keystrokes

    return result


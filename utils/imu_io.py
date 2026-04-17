"""IMU CSV loading and timestamp-column utilities."""

import numpy as np
import pandas as pd

from utils.constants import IMU_COLS


def get_time_column(df: pd.DataFrame) -> str:
    """Return the name of the timestamp column in an IMU DataFrame.

    Raises ValueError if neither expected column is present.
    """
    if "Effective Timestamp" in df.columns:
        return "Effective Timestamp"
    if "Time Stamp" in df.columns:
        return "Time Stamp"
    raise ValueError("No timestamp column found in IMU dataframe.")


def load_imu_csv(csv_path) -> tuple[np.ndarray, np.ndarray]:
    """Load an IMU CSV file and return (timestamps, imu_array).

    Returns:
        timestamps: (N,) float64 array
        imu_array:  (N, n_channels) float32 array for available IMU_COLS
    """
    df = pd.read_csv(csv_path)
    time_col = get_time_column(df)
    available = [c for c in IMU_COLS if c in df.columns]
    df = df[[time_col] + available].dropna().sort_values(time_col)
    return df[time_col].values.astype(np.float64), df[available].values.astype(np.float32)

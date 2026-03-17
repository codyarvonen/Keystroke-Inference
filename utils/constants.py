"""Project-wide constants shared across the pipeline."""

IMU_COLS = ["Accel-x", "Accel-y", "Accel-z", "Gyro-x", "Gyro-y", "Gyro-z"]

CHRONOS_DEFAULT_MODEL = "amazon/chronos-t5-base"

# Guard against zero std during per-channel z-score normalization
IMU_EPS = 1e-8

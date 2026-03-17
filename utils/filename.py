"""Canonical filename parsing for data files."""

from pathlib import Path


def parse_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    """Return (subject, session, ring_side) from a data filename.

    Expected formats:
      - '003_005_DIBS-L_corrected.csv'
      - '003_005_DIBS-R_corrected.csv'
      - '003_005_Macbook.pkl'

    ring_side is 'L', 'R', or None for non-IMU files.
    Returns (None, None, None) if the filename cannot be parsed.
    """
    stem = (Path(filename).name
            .replace('_corrected', '')
            .replace('.csv', '')
            .replace('.pkl', ''))
    parts = stem.split('_')
    if len(parts) < 2:
        return None, None, None
    subject, session = parts[0], parts[1]
    device = '_'.join(parts[2:])
    ring_side = 'L' if 'DIBS-L' in device else ('R' if 'DIBS-R' in device else None)
    return subject, session, ring_side

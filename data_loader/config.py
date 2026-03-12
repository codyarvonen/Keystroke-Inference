from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple


SplitStrategy = Literal["LOSO", "LOPO", "random"]
TargetVariant = Literal["raw_keystrokes", "clean_text", "clean_tokens"]
RingsUsed = Literal["L", "R", "both"]


@dataclass
class DataConfig:
    data_dir: str = "data"

    # Windowing
    window_size_s: float = 5.0
    train_stride_s: float = 2.5
    test_stride_s: float = 5.0

    # Text length cap
    max_tokens: int = 50

    # Label variant
    target_variant: TargetVariant = "clean_tokens"

    # Rings / channels
    rings_used: RingsUsed = "both"

    # Cleaning profile (reserved for future extension)
    cleaning_profile: str = "full"  # e.g., "full", "minimal"

    # Splitting
    split_strategy: SplitStrategy = "LOSO"
    test_session: Optional[str] = None   # e.g., "003_005"
    test_subject: Optional[str] = None   # e.g., "003"
    val_ratio: float = 0.2
    split_seed: int = 42

    # Optional restrictions for experiments
    include_sessions: Optional[Sequence[str]] = None
    exclude_sessions: Optional[Sequence[str]] = None


def parse_subject_session(filename: str) -> Tuple[str, str]:
    """
    Parse a data filename to extract subject and session identifiers.

    Expected formats (examples):
      - '003_005_DIBS-L_corrected.csv'
      - '003_005_Macbook.pkl'

    Returns:
      (subject, session) as strings.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse subject/session from filename: {filename}")
    subject = parts[0]
    session = parts[1]
    return subject, session


def make_session_key(subject: str, session: str) -> str:
    """Return a canonical session key like '003_005'."""
    return f"{subject}_{session}"


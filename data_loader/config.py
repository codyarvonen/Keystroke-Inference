from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple


SplitStrategy = Literal["LOSO", "LOPO", "random"]
TargetVariant = Literal["raw_keystrokes", "clean_text", "clean_tokens"]
RingsUsed = Literal["L", "R", "both"]
SessionSplitStrategy = Literal[
    "session_random",
    "session_holdout",
    "session_holdout_random_train_val",
]


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


@dataclass
class Stage1ExportConfig:
    """
    Stage-1 preprocessing/export config for per-key IMU windows.
    """

    data_dir: str = "data"
    rings_used: RingsUsed = "both"
    target_rate_hz: float = 100.0

    # Causal-biased key-centered window: [press - left_ms, press + right_ms]
    left_context_ms: int = 700
    right_context_ms: int = 150

    # Session-based split control (explicit strategy)
    session_split_strategy: SessionSplitStrategy = "session_random"
    test_sessions: Sequence[str] = field(default_factory=tuple)
    val_sessions: Sequence[str] = field(default_factory=tuple)
    val_ratio: float = 0.2
    # If set (session_random only): fraction of *pool* sessions (excluding train_only) for test.
    # If None, legacy rule: n_test = max(1, n_val) with n_val from val_ratio only.
    test_ratio: Optional[float] = None
    split_seed: int = 42
    # session_random: assign val/test sessions to balance approximate row counts (extra pass over data).
    balance_val_test_by_session_rows: bool = False

    # Sessions forced to train only (never val/test). Use canonical keys, e.g. "003_014".
    train_only_sessions: Sequence[str] = field(default_factory=tuple)

    # Optional session filters for experiments
    include_sessions: Optional[Sequence[str]] = None
    exclude_sessions: Optional[Sequence[str]] = None

    # If True, single-letter alphabetic key tokens are lowercased so e.g. 'A' and 'a' share one class.
    merge_letter_case: bool = False

    # If True, collapse labels to 32 classes: a–z, space, backspace, shift, punctuation, digits, other.
    coarse_labels: bool = False


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


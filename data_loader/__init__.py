from .config import DataConfig, Stage1ExportConfig
from .splits import make_splits
from .stage1 import (
    Stage1KeyWindowRow,
    build_stage1_rows,
    build_stage1_vocab,
    build_stage1_vocab_from_train,
    export_stage1_to_dir,
    load_stage1_vocab,
    stage1_export_config_from_manifest,
)
from .windows import WindowRecord

try:
    from .stage1_dataset import Stage1IMUKeyDataset
except ModuleNotFoundError:
    Stage1IMUKeyDataset = None  # type: ignore[assignment]

try:
    from .dataset import KeystrokeIMUDataset, build_all_windows, make_dataloaders
except ModuleNotFoundError:
    # Allow using non-torch utilities (e.g., stage-1 preprocessing) without torch.
    KeystrokeIMUDataset = None  # type: ignore[assignment]
    build_all_windows = None  # type: ignore[assignment]
    make_dataloaders = None  # type: ignore[assignment]

__all__ = [
    "DataConfig",
    "Stage1ExportConfig",
    "KeystrokeIMUDataset",
    "WindowRecord",
    "Stage1KeyWindowRow",
    "Stage1IMUKeyDataset",
    "build_all_windows",
    "build_stage1_rows",
    "build_stage1_vocab",
    "build_stage1_vocab_from_train",
    "export_stage1_to_dir",
    "load_stage1_vocab",
    "stage1_export_config_from_manifest",
    "make_dataloaders",
    "make_splits",
]


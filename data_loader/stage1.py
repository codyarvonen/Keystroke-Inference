import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DataConfig, Stage1ExportConfig
from .coarse_labels import build_fixed_coarse_32_vocab, map_key_token_to_coarse_32
from .labels import normalize_key_token_case, parse_key_name, regenerate_key_sequence_for_session
from .sessions import discover_sessions, load_session_raw
from .windows import align_rings_to_grid

STAGE1_PAD_TOKEN = "<PAD>"
STAGE1_UNK_TOKEN = "<UNK>"


@dataclass
class Stage1KeyWindowRow:
    """
    Canonical row schema for Stage-1 supervision:
      - subject/session identifiers
      - key press timestamp anchor
      - key token label
      - IMU window metadata
      - optional dwell duration
    """

    subject: str
    session: str
    session_key: str
    split: str

    key_press_ts: float
    key_release_ts: Optional[float]
    key_duration_s: Optional[float]

    key_raw: str
    key_token: str

    window_start_ts: float
    window_end_ts: float
    imu_idx_start: int
    imu_idx_end: int  # exclusive slice on aligned IMU grid
    n_imu_samples: int
    imu_target_rate_hz: float
    rings_used: str


def _event_duration_s(event: Dict[str, Any]) -> Optional[float]:
    end = event.get("end")
    start = event.get("timestamp")
    if end is None or start is None:
        return None
    dur = float(end) - float(start)
    if dur < 0:
        return None
    return dur


def _window_indices(
    time_grid: np.ndarray,
    center_ts: float,
    left_context_s: float,
    right_context_s: float,
) -> np.ndarray:
    start = center_ts - left_context_s
    end = center_ts + right_context_s
    return np.nonzero((time_grid >= start) & (time_grid <= end))[0]


def _count_kept_keys_for_session(files: Mapping[str, Any], cfg: Stage1ExportConfig) -> int:
    """Same keep/drop rules as build_stage1_rows, without building rows (for split balancing)."""
    raw = load_session_raw(files)
    imu_L = raw.get("imu_L")
    imu_R = raw.get("imu_R")
    if imu_L is None or imu_R is None:
        return 0
    time_grid, _, _ = align_rings_to_grid(
        imu_L=imu_L,
        imu_R=imu_R,
        target_rate_hz=cfg.target_rate_hz,
    )
    if time_grid.size == 0:
        return 0
    imu_start = float(time_grid[0])
    imu_end = float(time_grid[-1])
    left_context_s = cfg.left_context_ms / 1000.0
    right_context_s = cfg.right_context_ms / 1000.0
    events = regenerate_key_sequence_for_session(raw["keystrokes"])
    n = 0
    for event in events:
        key_press_ts = float(event["timestamp"])
        key_raw = str(event["key"])
        key_token = normalize_key_token_case(parse_key_name(key_raw), cfg.merge_letter_case)
        if cfg.coarse_labels:
            key_token = map_key_token_to_coarse_32(key_token)
        if key_press_ts < imu_start or key_press_ts > imu_end:
            continue
        idx = _window_indices(
            time_grid=time_grid,
            center_ts=key_press_ts,
            left_context_s=left_context_s,
            right_context_s=right_context_s,
        )
        if idx.size == 0:
            continue
        n += 1
    return n


def _clamp_n_test_val(n_test: int, n_val: int, pool_size: int) -> Tuple[int, int]:
    """Ensure val/test session counts fit in pool and train gets the remainder."""
    n_test = max(0, min(n_test, pool_size))
    n_val = max(0, min(n_val, pool_size))
    while n_test + n_val > pool_size and (n_test > 0 or n_val > 0):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            n_test = max(0, n_test - 1)
    return n_test, n_val


def _greedy_balance_test_val(
    pool: Sequence[str],
    counts: Mapping[str, int],
    n_test: int,
    n_val: int,
) -> Tuple[set, set]:
    """Pick n_test / n_val sessions so approximate row totals stay close (large sessions split across splits)."""
    ordered = sorted(pool, key=lambda s: counts.get(s, 0), reverse=True)
    test_set: set = set()
    val_set: set = set()
    sum_t = 0
    sum_v = 0
    for s in ordered:
        if len(test_set) >= n_test and len(val_set) >= n_val:
            break
        c = int(counts.get(s, 0))
        need_t = len(test_set) < n_test
        need_v = len(val_set) < n_val
        if need_t and not need_v:
            test_set.add(s)
            sum_t += c
        elif need_v and not need_t:
            val_set.add(s)
            sum_v += c
        elif need_t and need_v:
            if sum_t <= sum_v:
                test_set.add(s)
                sum_t += c
            else:
                val_set.add(s)
                sum_v += c
        else:
            break
    return test_set, val_set


def _assign_session_splits(
    session_keys: Sequence[str],
    cfg: Stage1ExportConfig,
    *,
    session_row_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    session_keys = sorted(session_keys)
    split_map: Dict[str, str] = {}
    train_only = frozenset(cfg.train_only_sessions)
    unknown_train_only = train_only.difference(session_keys)
    if unknown_train_only:
        import warnings

        warnings.warn(
            "train_only_sessions not found among discovered sessions (ignored): "
            f"{sorted(unknown_train_only)}",
            stacklevel=2,
        )

    if cfg.session_split_strategy == "session_holdout":
        test_sessions = set(cfg.test_sessions)
        val_sessions = set(cfg.val_sessions)
        overlap = test_sessions.intersection(val_sessions)
        if overlap:
            raise ValueError(f"Sessions cannot be both val and test: {sorted(overlap)}")
        bad = (test_sessions | val_sessions).intersection(train_only)
        if bad:
            raise ValueError(
                "train_only_sessions cannot be listed in val_sessions or test_sessions: "
                f"{sorted(bad)}"
            )
        for sess in session_keys:
            if sess in test_sessions:
                split_map[sess] = "test"
            elif sess in val_sessions:
                split_map[sess] = "val"
            else:
                split_map[sess] = "train"
        return split_map

    # session_random: split by whole sessions (no within-session leakage).
    # train_only_sessions are excluded from the val/test pool and always go to train.
    pool = [k for k in session_keys if k not in train_only]
    pool_size = len(pool)

    if cfg.test_ratio is not None:
        if cfg.test_ratio + cfg.val_ratio >= 1.0:
            raise ValueError(
                f"test_ratio + val_ratio must be < 1.0 (fractions of pool sessions), "
                f"got test_ratio={cfg.test_ratio} val_ratio={cfg.val_ratio}"
            )
        n_test = int(round(pool_size * cfg.test_ratio))
        n_val = int(round(pool_size * cfg.val_ratio))
        n_test, n_val = _clamp_n_test_val(n_test, n_val, pool_size)
    else:
        n_val = int(pool_size * cfg.val_ratio)
        n_test = max(1, n_val) if pool_size >= 3 else 0
        n_test, n_val = _clamp_n_test_val(n_test, n_val, pool_size)

    use_balance = (
        cfg.balance_val_test_by_session_rows
        and session_row_counts is not None
        and pool_size > 0
        and (n_test > 0 or n_val > 0)
    )
    if use_balance:
        pool_counts = {k: int(session_row_counts.get(k, 0)) for k in pool}
        test_set, val_set = _greedy_balance_test_val(pool, pool_counts, n_test, n_val)
    else:
        keys_pool = np.array(pool)
        rng = np.random.default_rng(cfg.split_seed)
        rng.shuffle(keys_pool)
        test_set = set(keys_pool[:n_test].tolist()) if n_test > 0 else set()
        val_set = set(keys_pool[n_test : n_test + n_val].tolist()) if n_val > 0 else set()

    for sess in session_keys:
        if sess in test_set:
            split_map[sess] = "test"
        elif sess in val_set:
            split_map[sess] = "val"
        else:
            split_map[sess] = "train"

    return split_map


def build_stage1_rows(
    cfg: Stage1ExportConfig,
) -> Tuple[List[Stage1KeyWindowRow], Dict[str, Any]]:
    base_cfg = DataConfig(
        data_dir=cfg.data_dir,
        include_sessions=cfg.include_sessions,
        exclude_sessions=cfg.exclude_sessions,
    )

    session_files = discover_sessions(base_cfg)  # reuses existing discovery logic
    session_keys_sorted = sorted(session_files.keys())
    session_row_counts: Optional[Dict[str, int]] = None
    if cfg.session_split_strategy == "session_random" and cfg.balance_val_test_by_session_rows:
        session_row_counts = {
            k: _count_kept_keys_for_session(session_files[k], cfg) for k in session_keys_sorted
        }
    split_map = _assign_session_splits(
        session_keys_sorted,
        cfg,
        session_row_counts=session_row_counts,
    )

    rows: List[Stage1KeyWindowRow] = []
    stats: Dict[str, Any] = {
        "keys_total": 0,
        "keys_kept": 0,
        "dropped_no_imu_window": 0,
        "dropped_outside_imu_range": 0,
        "label_distribution": Counter(),
        "by_split": Counter(),
        "by_session": {},
    }

    left_context_s = cfg.left_context_ms / 1000.0
    right_context_s = cfg.right_context_ms / 1000.0

    for session_key in sorted(session_files.keys()):
        subject, session = session_key.split("_", 1)
        files = session_files[session_key]
        split = split_map[session_key]
        raw = load_session_raw(files)
        imu_L = raw.get("imu_L")
        imu_R = raw.get("imu_R")
        if imu_L is None or imu_R is None:
            continue

        time_grid, _, _ = align_rings_to_grid(
            imu_L=imu_L,
            imu_R=imu_R,
            target_rate_hz=cfg.target_rate_hz,
        )
        if time_grid.size == 0:
            continue

        imu_start = float(time_grid[0])
        imu_end = float(time_grid[-1])

        events = regenerate_key_sequence_for_session(raw["keystrokes"])
        sess_stats = Counter()

        for event in events:
            key_press_ts = float(event["timestamp"])
            key_release_ts = event.get("end")
            key_raw = str(event["key"])
            key_token = normalize_key_token_case(parse_key_name(key_raw), cfg.merge_letter_case)
            if cfg.coarse_labels:
                key_token = map_key_token_to_coarse_32(key_token)

            stats["keys_total"] += 1
            sess_stats["keys_total"] += 1

            # Press timestamp is the v1 anchor by design.
            if key_press_ts < imu_start or key_press_ts > imu_end:
                stats["dropped_outside_imu_range"] += 1
                sess_stats["dropped_outside_imu_range"] += 1
                continue

            idx = _window_indices(
                time_grid=time_grid,
                center_ts=key_press_ts,
                left_context_s=left_context_s,
                right_context_s=right_context_s,
            )
            if idx.size == 0:
                stats["dropped_no_imu_window"] += 1
                sess_stats["dropped_no_imu_window"] += 1
                continue

            i0 = int(idx[0])
            i1 = int(idx[-1]) + 1
            row = Stage1KeyWindowRow(
                subject=subject,
                session=session,
                session_key=session_key,
                split=split,
                key_press_ts=key_press_ts,
                key_release_ts=float(key_release_ts) if key_release_ts is not None else None,
                key_duration_s=_event_duration_s(event),
                key_raw=key_raw,
                key_token=key_token,
                window_start_ts=float(time_grid[idx[0]]),
                window_end_ts=float(time_grid[idx[-1]]),
                imu_idx_start=i0,
                imu_idx_end=i1,
                n_imu_samples=int(idx.size),
                imu_target_rate_hz=float(cfg.target_rate_hz),
                rings_used=cfg.rings_used,
            )
            rows.append(row)

            stats["keys_kept"] += 1
            stats["label_distribution"][key_token] += 1
            stats["by_split"][split] += 1
            sess_stats["keys_kept"] += 1

        stats["by_session"][session_key] = dict(sess_stats)

    sessions_by_split: Dict[str, List[str]] = {s: [] for s in ("train", "val", "test")}
    for session_key, sp in split_map.items():
        if sp in sessions_by_split:
            sessions_by_split[sp].append(session_key)
    for sp in sessions_by_split:
        sessions_by_split[sp].sort()
    stats["sessions_by_split"] = sessions_by_split

    stats["label_distribution"] = dict(stats["label_distribution"])
    stats["by_split"] = dict(stats["by_split"])
    return rows, stats


def stage1_rows_to_dicts(rows: Sequence[Stage1KeyWindowRow]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in rows]


def build_stage1_vocab_from_train(
    rows: Sequence[Stage1KeyWindowRow],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build token_to_id from train-split rows only (no val/test leakage).
    IDs: 0 = <PAD>, 1 = <UNK>, then unique train key_token strings (stable order).
    """
    train_tokens = sorted({r.key_token for r in rows if r.split == "train"})
    id_to_token: List[str] = [STAGE1_PAD_TOKEN, STAGE1_UNK_TOKEN]
    token_to_id: Dict[str, int] = {STAGE1_PAD_TOKEN: 0, STAGE1_UNK_TOKEN: 1}
    for t in train_tokens:
        if t not in token_to_id:
            token_to_id[t] = len(id_to_token)
            id_to_token.append(t)
    return token_to_id, id_to_token


def build_stage1_vocab(
    rows: Sequence[Stage1KeyWindowRow],
    cfg: Stage1ExportConfig,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Full-vocab mode: train-only tokens. Coarse mode: fixed 32 classes (+ PAD/UNK).
    """
    if cfg.coarse_labels:
        return build_fixed_coarse_32_vocab()
    return build_stage1_vocab_from_train(rows)


def encode_key_token(key_token: str, token_to_id: Mapping[str, int]) -> int:
    return int(token_to_id.get(key_token, token_to_id[STAGE1_UNK_TOKEN]))


def vocab_to_serializable(
    token_to_id: Mapping[str, int],
    id_to_token: Sequence[str],
    label_scheme: str | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "token_to_id": dict(token_to_id),
        "id_to_token": list(id_to_token),
        "special_tokens": [STAGE1_PAD_TOKEN, STAGE1_UNK_TOKEN],
    }
    if label_scheme is not None:
        out["label_scheme"] = label_scheme
    return out


def save_stage1_vocab(
    path: Path,
    token_to_id: Mapping[str, int],
    id_to_token: Sequence[str],
    label_scheme: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = vocab_to_serializable(token_to_id, id_to_token, label_scheme=label_scheme)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_stage1_vocab(path: Path) -> Tuple[Dict[str, int], List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
    id_to_token = [str(x) for x in payload["id_to_token"]]
    return token_to_id, id_to_token


def row_to_export_record(
    row: Stage1KeyWindowRow,
    token_to_id: Mapping[str, int],
) -> Dict[str, Any]:
    label_id = encode_key_token(row.key_token, token_to_id)
    unk_id = token_to_id[STAGE1_UNK_TOKEN]
    d = asdict(row)
    d["label_id"] = label_id
    d["is_unk"] = label_id == unk_id
    return d


def stage1_export_config_from_manifest(
    manifest: Mapping[str, Any],
    *,
    data_dir: Optional[str] = None,
) -> Stage1ExportConfig:
    """
    Build Stage1ExportConfig from manifest.json written by export_stage1_to_dir.
    Use this so validation/eval use the same settings as the export (including merge_letter_case).
    """
    c = manifest.get("config", {})
    tr = c.get("test_ratio")
    test_ratio: Optional[float] = None if tr is None else float(tr)
    return Stage1ExportConfig(
        data_dir=data_dir if data_dir is not None else str(c.get("data_dir", "data")),
        rings_used=c.get("rings_used", "both"),  # type: ignore[arg-type]
        target_rate_hz=float(c.get("target_rate_hz", 100.0)),
        left_context_ms=int(c.get("left_context_ms", 700)),
        right_context_ms=int(c.get("right_context_ms", 150)),
        session_split_strategy=c.get("session_split_strategy", "session_random"),  # type: ignore[arg-type]
        test_sessions=tuple(c.get("test_sessions") or ()),
        val_sessions=tuple(c.get("val_sessions") or ()),
        val_ratio=float(c.get("val_ratio", 0.2)),
        test_ratio=test_ratio,
        split_seed=int(c.get("split_seed", 42)),
        balance_val_test_by_session_rows=bool(c.get("balance_val_test_by_session_rows", False)),
        merge_letter_case=bool(c.get("merge_letter_case", False)),
        coarse_labels=bool(c.get("coarse_labels", False)),
        train_only_sessions=tuple(c.get("train_only_sessions") or ()),
    )


def export_stage1_to_dir(
    cfg: Stage1ExportConfig,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """
    Build rows, train-only vocab, write vocab.json and train/val/test.jsonl.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, stats = build_stage1_rows(cfg)
    token_to_id, id_to_token = build_stage1_vocab(rows, cfg)
    label_scheme = "coarse_32" if cfg.coarse_labels else None
    save_stage1_vocab(out / "vocab.json", token_to_id, id_to_token, label_scheme=label_scheme)

    by_split: Dict[str, List[Stage1KeyWindowRow]] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r.split, []).append(r)

    unk_per_split: Dict[str, int] = {}
    written_per_split: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_rows = by_split.get(split_name, [])
        path = out / f"{split_name}.jsonl"
        unk_count = 0
        with path.open("w", encoding="utf-8") as f:
            for r in split_rows:
                rec = row_to_export_record(r, token_to_id)
                if rec["is_unk"]:
                    unk_count += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        unk_per_split[split_name] = unk_count
        written_per_split[split_name] = len(split_rows)

    manifest: Dict[str, Any] = {
        "stage1_export_version": 1,
        "config": {
            "data_dir": cfg.data_dir,
            "left_context_ms": cfg.left_context_ms,
            "right_context_ms": cfg.right_context_ms,
            "target_rate_hz": cfg.target_rate_hz,
            "rings_used": cfg.rings_used,
            "session_split_strategy": cfg.session_split_strategy,
            "test_sessions": list(cfg.test_sessions),
            "val_sessions": list(cfg.val_sessions),
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "split_seed": cfg.split_seed,
            "balance_val_test_by_session_rows": cfg.balance_val_test_by_session_rows,
            "merge_letter_case": cfg.merge_letter_case,
            "coarse_labels": cfg.coarse_labels,
            "train_only_sessions": list(cfg.train_only_sessions),
        },
        "build_stats": stats,
        "vocab_size": len(token_to_id),
        "rows_written": written_per_split,
        "unk_label_count_by_split": unk_per_split,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def load_stage1_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_out.append(json.loads(line))
    return rows_out


def dict_to_stage1_row(d: Mapping[str, Any]) -> Stage1KeyWindowRow:
    """Rehydrate Stage1KeyWindowRow from an export dict (ignores label_id, is_unk)."""
    field_names = {f.name for f in fields(Stage1KeyWindowRow)}
    kwargs = {k: d[k] for k in field_names if k in d}
    return Stage1KeyWindowRow(**kwargs)

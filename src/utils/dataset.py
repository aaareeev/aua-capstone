"""
Shared utilities for the C2C Behavioral Dynamics capstone project.

Functions and classes here are used across multiple notebooks
(03_NFM_training, 06_UMAG) to avoid code duplication.
"""

import hashlib
import math
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
PAD = 0  # padding token index for sequence models


# ── Hashing ───────────────────────────────────────────────────────────────────

def stable_hash(x, buckets: int) -> int:
    """
    Deterministic hash of a string/int to a bucket index.

    Uses blake2b for stability across Python runs and machines,
    unlike the built-in hash() which is randomised per-process.

    Args:
        x:       Value to hash. None is treated as empty string.
        buckets: Number of hash buckets (output range: [0, buckets)).

    Returns:
        Integer in [0, buckets).
    """
    if x is None:
        x = ""
    if not isinstance(x, str):
        x = str(x)
    h = hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % buckets


# ── Sequence utilities ────────────────────────────────────────────────────────

def pad_or_trim(seq: list, length: int) -> list:
    """
    Ensure a sequence has exactly `length` elements.

    - If len(seq) >= length: keep the MOST RECENT items (right-trim).
    - If len(seq) <  length: left-pad with PAD tokens.

    Left-padding preserves recency at the right end, which is the
    convention used by most session-based recommender models.

    Args:
        seq:    List of item indices (ints).
        length: Target sequence length.

    Returns:
        List of exactly `length` ints.
    """
    if len(seq) >= length:
        return seq[-length:]
    return [PAD] * (length - len(seq)) + seq


# ── Dataset paths ─────────────────────────────────────────────────────────────

def get_dataset_path(env_var: str = "DATASET_ROOT", default: str = "/Volumes/T5 EVO") -> Path:
    """
    Resolve the dataset root path from an environment variable.

    Args:
        env_var: Environment variable name holding the root path.
        default: Fallback path if the env var is not set.

    Returns:
        Path to the MerRec parquet directory.
    """
    root = Path(os.environ.get(env_var, default))
    return root / "hf" / "merrec"


# ── Streaming dataset (NFM) ───────────────────────────────────────────────────

class MerRecStreamingCTR(IterableDataset):
    """
    Streams MerRec parquet files and yields (user_idx, item_idx, label) batches.

    Designed for the NFM model. Supports:
    - File-level shuffling per epoch for variance
    - Row-level limit (max_total_rows) for quick experiments
    - Stable hashing of user/item IDs to bucket indices

    Args:
        dataset_path: Path to the parquet directory.
        cfg:          Configuration dict (see notebooks/03_NFM_training.ipynb).
        epoch:        Current epoch number (used for deterministic file shuffle).
    """

    def __init__(self, dataset_path: Path, cfg: dict, epoch: int = 0):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.cfg = cfg
        self.epoch = epoch

        self.files = sorted(glob(
            str(self.dataset_path / "**" / "*.parquet"), recursive=True
        ))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {self.dataset_path}")

    def _iter_files(self):
        files = self.files[:]
        if self.cfg.get("shuffle_files", True):
            g = torch.Generator()
            g.manual_seed(12345 + self.epoch)
            idx = torch.randperm(len(files), generator=g).tolist()
            files = [files[i] for i in idx]
        return files

    def __iter__(self):
        import pyarrow.dataset as pads

        bs          = self.cfg["batch_size"]
        ub          = self.cfg["user_buckets"]
        ib          = self.cfg["item_buckets"]
        col_u       = self.cfg["col_user"]
        col_i       = self.cfg["col_item"]
        col_a       = self.cfg["col_action"]
        pos_action  = self.cfg["positive_action"]
        max_rows    = self.cfg.get("max_total_rows", None)
        max_per_file = self.cfg.get("max_rows_per_file", None)

        batch_u, batch_i, batch_y = [], [], []
        total_rows = 0

        for fp in self._iter_files():
            try:
                dataset = pads.dataset(fp, format="parquet")
                scanner = dataset.scanner(
                    columns=[col_u, col_i, col_a], batch_size=65536
                )
                seen_rows = 0

                for rb in scanner.to_batches():
                    if max_per_file is not None and seen_rows >= max_per_file:
                        break

                    for u, it, act in zip(
                        rb.column(col_u).to_pylist(),
                        rb.column(col_i).to_pylist(),
                        rb.column(col_a).to_pylist(),
                    ):
                        if max_rows is not None and total_rows >= max_rows:
                            if batch_y:
                                yield _make_ctr_batch(batch_u, batch_i, batch_y)
                            return

                        total_rows += 1
                        if u is None or it is None:
                            continue

                        batch_u.append(stable_hash(u, ub))
                        batch_i.append(stable_hash(it, ib))
                        batch_y.append(1.0 if act == pos_action else 0.0)

                        if len(batch_y) >= bs:
                            yield _make_ctr_batch(batch_u, batch_i, batch_y)
                            batch_u, batch_i, batch_y = [], [], []

                    seen_rows += rb.num_rows

            except Exception as e:
                import warnings
                warnings.warn(f"Failed reading {fp}: {e}")
                continue

        if batch_y:
            yield _make_ctr_batch(batch_u, batch_i, batch_y)


def _make_ctr_batch(batch_u, batch_i, batch_y):
    return (
        torch.tensor(batch_u, dtype=torch.long),
        torch.tensor(batch_i, dtype=torch.long),
        torch.tensor(batch_y, dtype=torch.float32),
    )


# ── History builder (UMAG) ────────────────────────────────────────────────────

class HistoryBuilder:
    """
    Single-pass streaming builder for per-user sorted interaction histories.

    Reads user_id, item_id, stime from parquet and builds:
        history[user_id] = [(unix_timestamp, item_hash), ...]
    sorted ascending by time.

    Used by UMAG to construct multi-scale temporal context vectors.

    Args:
        dataset_path: Path to the parquet directory.
        cfg:          Configuration dict (see notebooks/06_UMAG.ipynb).
    """

    def __init__(self, dataset_path: Path, cfg: dict):
        import pyarrow.dataset as ds
        self.dataset_path = str(dataset_path)
        self.cfg = cfg
        self.dataset = ds.dataset(self.dataset_path, format="parquet")

    def build(self) -> dict:
        col_u    = self.cfg["col_user"]
        col_i    = self.cfg["col_item"]
        col_t    = self.cfg["col_time"]
        ib       = self.cfg["item_buckets"]
        max_rows = self.cfg.get("max_total_rows", None)

        history = defaultdict(list)
        total   = 0

        scanner = self.dataset.scanner(
            columns=[col_u, col_i, col_t], batch_size=65536
        )

        for rb in scanner.to_batches():
            if max_rows and total >= max_rows:
                break

            for u, it, t in zip(
                rb.column(col_u).to_pylist(),
                rb.column(col_i).to_pylist(),
                rb.column(col_t).to_pylist(),
            ):
                if max_rows and total >= max_rows:
                    break
                total += 1
                if u is None or it is None:
                    continue

                if hasattr(t, "as_py"):
                    t = t.as_py()
                t_int = int(t.timestamp()) if hasattr(t, "timestamp") else int(t or 0)

                history[u].append((t_int, stable_hash(it, ib)))

        for u in history:
            history[u].sort(key=lambda x: x[0])

        print(f"History built: {len(history):,} users, {total:,} events")
        return history

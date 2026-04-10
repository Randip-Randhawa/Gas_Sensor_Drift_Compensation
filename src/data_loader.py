"""Data loading utilities for UCI Gas Sensor Array Drift dataset in LibSVM format."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file


@dataclass(slots=True)
class LoadedData:
    """Container for loaded dataset and metadata."""

    df: pd.DataFrame
    feature_cols: List[str]
    batch_ids: List[int]


_BATCH_RE = re.compile(r"batch(\d+)\.dat$")


def _batch_index(file_path: Path) -> int:
    match = _BATCH_RE.search(file_path.name)
    if not match:
        raise ValueError(f"Unexpected batch filename: {file_path.name}")
    return int(match.group(1))


def load_batches(dataset_dir: Path, pattern: str, n_features: int) -> LoadedData:
    """Load all batch files and return one time-ordered dataframe."""
    files = sorted(dataset_dir.glob(pattern), key=_batch_index)
    if not files:
        raise FileNotFoundError(f"No files found in {dataset_dir} with pattern {pattern}")

    feature_cols = [f"f{i}" for i in range(1, n_features + 1)]
    chunks: list[pd.DataFrame] = []
    gidx = 0

    for fp in files:
        bid = _batch_index(fp)
        X_sparse, y = load_svmlight_file(str(fp), n_features=n_features)
        X = X_sparse.toarray()

        chunk = pd.DataFrame(X, columns=feature_cols)
        chunk["label"] = y.astype(int)
        chunk["batch_id"] = bid
        chunk["sample_in_batch"] = np.arange(len(chunk), dtype=int)
        chunk["global_index"] = np.arange(gidx, gidx + len(chunk), dtype=int)
        gidx += len(chunk)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["batch_id", "sample_in_batch"]).reset_index(drop=True)
    batch_ids = sorted(df["batch_id"].unique().tolist())

    return LoadedData(df=df, feature_cols=feature_cols, batch_ids=batch_ids)

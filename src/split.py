"""Time-ordered split helpers for drift-focused experiments."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def split_static_train_test(
    df: pd.DataFrame,
    train_batches: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Split by batch index: early batches for train, later for test."""
    batch_ids = sorted(df["batch_id"].unique().tolist())
    if train_batches <= 0 or train_batches >= len(batch_ids):
        raise ValueError("train_batches must be between 1 and n_batches-1")

    train_ids = batch_ids[:train_batches]
    test_ids = batch_ids[train_batches:]

    train_df = df[df["batch_id"].isin(train_ids)].copy()
    test_df = df[df["batch_id"].isin(test_ids)].copy()

    return train_df, test_df, train_ids, test_ids


def get_batch_slices(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return per-batch dataframe map preserving order."""
    out: dict[int, pd.DataFrame] = {}
    for bid in sorted(df["batch_id"].unique().tolist()):
        out[bid] = df[df["batch_id"] == bid].copy()
    return out

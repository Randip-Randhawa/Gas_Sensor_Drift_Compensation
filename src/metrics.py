"""Metrics helpers for overall and per-batch evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Compute accuracy as float."""
    return float(accuracy_score(list(y_true), list(y_pred)))


def per_batch_accuracy(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Compute per-batch accuracy from prediction dataframe.

    Expected columns: batch_id, y_true, y_pred
    """
    grouped = (
        df_pred.assign(correct=(df_pred["y_true"] == df_pred["y_pred"]).astype(float))
        .groupby("batch_id", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    return grouped


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)

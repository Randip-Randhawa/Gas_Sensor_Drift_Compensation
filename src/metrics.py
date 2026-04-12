from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    return float(accuracy_score(list(y_true), list(y_pred)))
def precision(y_true: Iterable[int], y_pred: Iterable[int], average: str = "weighted") -> float:
    return float(precision_score(list(y_true), list(y_pred), average=average, zero_division=0))
def recall(y_true: Iterable[int], y_pred: Iterable[int], average: str = "weighted") -> float:
    return float(recall_score(list(y_true), list(y_pred), average=average, zero_division=0))
def f1(y_true: Iterable[int], y_pred: Iterable[int], average: str = "weighted") -> float:
    return float(f1_score(list(y_true), list(y_pred), average=average, zero_division=0))
def per_batch_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    results = []
    for batch_id, group in df_pred.groupby("batch_id"):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        results.append({
            "batch_id": batch_id,
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1_score": f1(y_true, y_pred),
        })
    return pd.DataFrame(results)
def per_class_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> pd.DataFrame:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    precision_vals, recall_vals, f1_vals, support_vals = precision_recall_fscore_support(
        y_true_list, y_pred_list, zero_division=0
    )
    unique_classes = np.unique(y_true_list)
    results = []
    for i, class_label in enumerate(unique_classes):
        results.append({
            "class_label": int(class_label),
            "precision": float(precision_vals[i]),
            "recall": float(recall_vals[i]),
            "f1_score": float(f1_vals[i]),
            "support": int(support_vals[i]),
        })
    return pd.DataFrame(results)

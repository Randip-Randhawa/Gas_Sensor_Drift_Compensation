from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from river.drift import ADWIN
@dataclass(slots=True)
class ADWINResult:
    drift_indices: list[int]
    error_stream: list[float]

def drift_magnitude_by_batch(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    batch_means = df.groupby("batch_id")[feature_cols].mean().sort_index()
    rows: list[dict[str, float | int]] = []
    prev_mu: np.ndarray | None = None
    prev_batch: int | None = None
    for batch_id, mu in batch_means.iterrows():
        cur_mu = mu.to_numpy(dtype=float)
        if prev_mu is not None and prev_batch is not None:
            mag = float(np.linalg.norm(cur_mu - prev_mu, ord=2))
            rows.append(
                {
                    "batch_prev": int(prev_batch),
                    "batch_curr": int(batch_id),
                    "drift_magnitude": mag,
                }
            )
        prev_mu = cur_mu
        prev_batch = int(batch_id)
    return pd.DataFrame(rows)

def detect_drift_with_adwin(error_stream: list[float], delta: float) -> ADWINResult:
    detector = ADWIN(delta=delta)
    drift_indices: list[int] = []
    for i, err in enumerate(error_stream):
        detector.update(float(err))
        if detector.drift_detected:
            drift_indices.append(i)
    return ADWINResult(drift_indices=drift_indices, error_stream=error_stream)

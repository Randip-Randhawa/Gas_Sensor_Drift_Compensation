# Preprocessing components with strict temporal safety.

from __future__ import annotations
from dataclasses import dataclass

from typing import Iterable

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, StandardScaler
@dataclass(slots=True)
class Preprocessor:
    # Fitted preprocessing pipeline artifacts.

    imputer: SimpleImputer
    scaler: StandardScaler
    label_encoder: LabelEncoder

def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_values: Iterable[int] | None = None,
) -> Preprocessor:
    """Fit preprocessing only on training slice to avoid leakage."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    le = LabelEncoder()
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_train = imputer.fit_transform(X_train)
    scaler.fit(X_train)
    if label_values is None:
        le.fit(y_train)
    else:
        le.fit(np.array(list(label_values), dtype=int))
    return Preprocessor(imputer=imputer, scaler=scaler, label_encoder=le)

def transform_df(pre: Preprocessor, df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # Transform features and labels using a fitted preprocessor.

    X = df[feature_cols].to_numpy(dtype=float)
    X = pre.imputer.transform(X)
    X = pre.scaler.transform(X)
    y = pre.label_encoder.transform(df["label"].to_numpy())
    return X, y

def encode_with_global_labels(labels: Iterable[int]) -> LabelEncoder:
    # Fit label encoder on all known labels for stable online partial_fit classes.

    le = LabelEncoder()
    le.fit(np.array(list(labels), dtype=int))
    return le

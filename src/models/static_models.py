from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_random_forest(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

def build_logistic_regression(seed: int, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=max_iter,
        random_state=seed,
    )

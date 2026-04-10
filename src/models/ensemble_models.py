"""Ensemble model builders for static training setup."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def build_voting_classifier(seed: int, n_estimators: int, max_iter: int) -> VotingClassifier:
    """Soft voting ensemble of RF + LR."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    lr = LogisticRegression(max_iter=max_iter, random_state=seed)
    return VotingClassifier(
        estimators=[("rf", rf), ("lr", lr)],
        voting="soft",
        n_jobs=-1,
    )


def build_stacking_classifier(seed: int, n_estimators: int, max_iter: int) -> StackingClassifier:
    """Stacking ensemble with logistic meta learner."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    lr = LogisticRegression(max_iter=max_iter, random_state=seed)
    final = LogisticRegression(max_iter=max_iter, random_state=seed)
    return StackingClassifier(
        estimators=[("rf", rf), ("lr", lr)],
        final_estimator=final,
        n_jobs=-1,
    )

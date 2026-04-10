"""Adaptive online models using river."""

from __future__ import annotations

from river import forest


def build_adaptive_random_forest(seed: int):
    """Adaptive Random Forest classifier from river."""
    return forest.ARFClassifier(seed=seed)

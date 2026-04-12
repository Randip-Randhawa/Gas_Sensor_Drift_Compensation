from __future__ import annotations
from river import forest

def build_adaptive_random_forest(seed: int):
    return forest.ARFClassifier(seed=seed)

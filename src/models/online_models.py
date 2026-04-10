"""Online incremental models using scikit-learn partial_fit."""

from __future__ import annotations

from sklearn.linear_model import Perceptron, SGDClassifier


def build_sgd_classifier(
    seed: int,
    max_iter: int,
    tol: float | None,
    alpha: float,
    eta0: float,
    average: bool,
    class_weight: str | dict[int, float] | None,
) -> SGDClassifier:
    """SGD classifier configured for incremental learning."""
    return SGDClassifier(
        loss="log_loss",
        random_state=seed,
        max_iter=max_iter,
        tol=tol,
        alpha=alpha,
        learning_rate="optimal",
        eta0=eta0,
        average=average,
        class_weight=class_weight,
    )


def build_perceptron(seed: int, max_iter: int) -> Perceptron:
    """Optional incremental perceptron model."""
    return Perceptron(
        random_state=seed,
        max_iter=max_iter,
        tol=None,
    )

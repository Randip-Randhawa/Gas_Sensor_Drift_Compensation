"""Experiment runners for static, sliding-window, online, and controlled drift studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import Config
from src.drift_detection import ADWINResult, detect_drift_with_adwin
from src.metrics import accuracy, per_batch_accuracy
from src.models.adaptive_models import build_adaptive_random_forest
from src.models.ensemble_models import build_stacking_classifier, build_voting_classifier
from src.models.online_models import build_perceptron, build_sgd_classifier
from src.models.static_models import build_logistic_regression, build_random_forest
from src.preprocessing import encode_with_global_labels, fit_preprocessor, transform_df
from src.split import get_batch_slices, split_static_train_test


@dataclass(slots=True)
class ExperimentOutput:
    """Container for experiment outputs used by downstream comparison/reporting."""

    metrics_rows: List[dict]
    per_batch_rows: List[dict]
    controlled_rows: List[dict]
    adwin_result: ADWINResult


def _make_pred_df(batch_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"batch_id": batch_ids, "y_true": y_true, "y_pred": y_pred})


def run_static_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict]]:
    """Train once on early batches and test on later batches."""
    train_df, test_df, train_ids, test_ids = split_static_train_test(df, cfg.static_train_batches)

    global_labels = sorted(df["label"].unique().tolist())
    pre = fit_preprocessor(train_df, feature_cols, label_values=global_labels)
    X_train, y_train = transform_df(pre, train_df, feature_cols)
    X_test, y_test = transform_df(pre, test_df, feature_cols)

    models = {
        "random_forest": build_random_forest(cfg.random_seed, cfg.rf_n_estimators),
        "logistic_regression": build_logistic_regression(cfg.random_seed, cfg.lr_max_iter),
        "voting_classifier": build_voting_classifier(cfg.random_seed, cfg.rf_n_estimators, cfg.lr_max_iter),
        "stacking_classifier": build_stacking_classifier(cfg.random_seed, cfg.rf_n_estimators, cfg.lr_max_iter),
    }

    metrics_rows: list[dict] = []
    per_batch_rows: list[dict] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        overall_acc = accuracy(y_test, pred)
        metrics_rows.append(
            {
                "experiment": "static_train_once",
                "model": name,
                "learning_type": "Static",
                "metric_level": "overall",
                "batch_id": "overall",
                "accuracy": overall_acc,
                "train_batches": ",".join(map(str, train_ids)),
                "test_batches": ",".join(map(str, test_ids)),
            }
        )

        pred_df = _make_pred_df(test_df["batch_id"].to_numpy(), y_test, pred)
        pb = per_batch_accuracy(pred_df)
        for _, row in pb.iterrows():
            per_batch_rows.append(
                {
                    "experiment": "static_train_once",
                    "model": name,
                    "learning_type": "Static",
                    "metric_level": "per_batch",
                    "batch_id": int(row["batch_id"]),
                    "accuracy": float(row["accuracy"]),
                }
            )

    return metrics_rows, per_batch_rows


def run_sliding_window_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict]]:
    """Train on recent batches only and evaluate on next batch."""
    batch_map = get_batch_slices(df)
    batch_ids = sorted(batch_map.keys())
    global_labels = sorted(df["label"].unique().tolist())

    models_builders = {
        "sliding_logistic_regression": lambda: build_logistic_regression(cfg.random_seed, cfg.lr_max_iter),
        "sliding_random_forest": lambda: build_random_forest(cfg.random_seed, cfg.rf_n_estimators),
    }

    per_batch_rows: list[dict] = []

    for i in range(cfg.sliding_window_size, len(batch_ids)):
        train_ids = batch_ids[i - cfg.sliding_window_size : i]
        test_id = batch_ids[i]

        train_df = pd.concat([batch_map[b] for b in train_ids], ignore_index=True)
        test_df = batch_map[test_id]

        pre = fit_preprocessor(train_df, feature_cols, label_values=global_labels)
        X_train, y_train = transform_df(pre, train_df, feature_cols)
        X_test, y_test = transform_df(pre, test_df, feature_cols)

        for name, builder in models_builders.items():
            model = builder()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy(y_test, pred)
            per_batch_rows.append(
                {
                    "experiment": "sliding_window",
                    "model": name,
                    "learning_type": "Adaptive",
                    "metric_level": "per_batch",
                    "batch_id": int(test_id),
                    "accuracy": acc,
                }
            )

    metrics_rows: list[dict] = []
    pb_df = pd.DataFrame(per_batch_rows)
    for model_name, g in pb_df.groupby("model"):
        metrics_rows.append(
            {
                "experiment": "sliding_window",
                "model": model_name,
                "learning_type": "Adaptive",
                "metric_level": "overall",
                "batch_id": "overall",
                "accuracy": float(g["accuracy"].mean()),
                "train_batches": f"window={cfg.sliding_window_size}",
                "test_batches": "next_batch",
            }
        )

    return metrics_rows, per_batch_rows


def _to_river_dict(row: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    return {feature_cols[i]: float(row[i]) for i in range(len(feature_cols))}


def _partial_fit_epochs(
    model,
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    epochs: int,
    classes: np.ndarray | None = None,
) -> None:
    """Run repeated partial_fit passes over a batch with shuffled sample order."""
    indices = np.arange(len(y))
    for epoch in range(max(1, epochs)):
        if len(indices) > 1:
            rng.shuffle(indices)
        X_epoch = X[indices]
        y_epoch = y[indices]
        if epoch == 0 and classes is not None:
            model.partial_fit(X_epoch, y_epoch, classes=classes)
        else:
            model.partial_fit(X_epoch, y_epoch)


def run_online_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict], ADWINResult]:
    """Incremental updates over time for SGD, ARF, and optional perceptron."""
    batch_map = get_batch_slices(df)
    batch_ids = sorted(batch_map.keys())
    rng = np.random.default_rng(cfg.random_seed)

    init_id = batch_ids[0]
    init_df = batch_map[init_id]

    global_labels = sorted(df["label"].unique().tolist())
    pre = fit_preprocessor(init_df, feature_cols, label_values=global_labels)
    X_init, y_init = transform_df(pre, init_df, feature_cols)

    label_encoder = encode_with_global_labels(df["label"].to_numpy())
    all_classes = np.arange(len(label_encoder.classes_), dtype=int)
    if cfg.sgd_class_weight is None:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=all_classes,
            y=label_encoder.transform(df["label"].to_numpy()),
        )
        sgd_class_weight = {int(cls): float(weight) for cls, weight in zip(all_classes, class_weights)}
    else:
        sgd_class_weight = cfg.sgd_class_weight

    sgd = build_sgd_classifier(
        cfg.random_seed,
        cfg.sgd_max_iter,
        cfg.sgd_tol,
        cfg.sgd_alpha,
        cfg.sgd_eta0,
        cfg.sgd_average,
        sgd_class_weight,
    )
    perc = build_perceptron(cfg.random_seed, cfg.perceptron_max_iter)
    arf = build_adaptive_random_forest(cfg.random_seed)

    _partial_fit_epochs(sgd, X_init, y_init, rng, cfg.sgd_epochs_per_batch, classes=all_classes)
    perc.partial_fit(X_init, y_init, classes=all_classes)
    for x, y in zip(X_init, y_init):
        arf.learn_one(_to_river_dict(x, feature_cols), int(y))

    per_batch_rows: list[dict] = []
    arf_error_stream: list[float] = []

    for bid in batch_ids[1:]:
        bdf = batch_map[bid]
        Xb, yb = transform_df(pre, bdf, feature_cols)

        # SGD
        pred_sgd = sgd.predict(Xb)
        acc_sgd = accuracy(yb, pred_sgd)
        per_batch_rows.append(
            {
                "experiment": "online_incremental",
                "model": "sgd_classifier",
                "learning_type": "Online",
                "metric_level": "per_batch",
                "batch_id": int(bid),
                "accuracy": acc_sgd,
            }
        )
        _partial_fit_epochs(sgd, Xb, yb, rng, cfg.sgd_epochs_per_batch)

        # Perceptron (optional but enabled)
        pred_perc = perc.predict(Xb)
        acc_perc = accuracy(yb, pred_perc)
        per_batch_rows.append(
            {
                "experiment": "online_incremental",
                "model": "perceptron",
                "learning_type": "Online",
                "metric_level": "per_batch",
                "batch_id": int(bid),
                "accuracy": acc_perc,
            }
        )
        perc.partial_fit(Xb, yb)

        # ARF (river)
        pred_arf_list: list[int] = []
        for x, y in zip(Xb, yb):
            x_dict = _to_river_dict(x, feature_cols)
            pred = arf.predict_one(x_dict)
            pred_int = int(pred) if pred is not None else 0
            pred_arf_list.append(pred_int)
            arf_error_stream.append(0.0 if pred_int == int(y) else 1.0)
            arf.learn_one(x_dict, int(y))

        acc_arf = accuracy(yb, pred_arf_list)
        per_batch_rows.append(
            {
                "experiment": "online_incremental",
                "model": "adaptive_random_forest",
                "learning_type": "Adaptive",
                "metric_level": "per_batch",
                "batch_id": int(bid),
                "accuracy": acc_arf,
            }
        )

    metrics_rows: list[dict] = []
    pb_df = pd.DataFrame(per_batch_rows)
    for model_name, g in pb_df.groupby("model"):
        ltype = g["learning_type"].iloc[0]
        metrics_rows.append(
            {
                "experiment": "online_incremental",
                "model": model_name,
                "learning_type": ltype,
                "metric_level": "overall",
                "batch_id": "overall",
                "accuracy": float(g["accuracy"].mean()),
                "train_batches": str(init_id),
                "test_batches": f"{batch_ids[1]}..{batch_ids[-1]}",
            }
        )

    adwin_result = detect_drift_with_adwin(arf_error_stream, delta=cfg.adwin_delta)
    return metrics_rows, per_batch_rows, adwin_result


def run_controlled_decay_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> list[dict]:
    """Train on batch 1 only, test sequentially on future batches."""
    batch_map = get_batch_slices(df)
    batch_ids = sorted(batch_map.keys())

    train_id = batch_ids[0]
    train_df = batch_map[train_id]
    global_labels = sorted(df["label"].unique().tolist())
    pre = fit_preprocessor(train_df, feature_cols, label_values=global_labels)
    X_train, y_train = transform_df(pre, train_df, feature_cols)

    models = {
        "controlled_rf": build_random_forest(cfg.random_seed, cfg.rf_n_estimators),
        "controlled_lr": build_logistic_regression(cfg.random_seed, cfg.lr_max_iter),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    rows: list[dict] = []
    for bid in batch_ids[1:]:
        bdf = batch_map[bid]
        Xb, yb = transform_df(pre, bdf, feature_cols)

        for name, model in models.items():
            pred = model.predict(Xb)
            rows.append(
                {
                    "experiment": "controlled_decay",
                    "model": name,
                    "learning_type": "Static",
                    "metric_level": "per_batch",
                    "batch_id": int(bid),
                    "accuracy": accuracy(yb, pred),
                }
            )

    return rows


def run_all_experiments(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> ExperimentOutput:
    """Execute all required experiments and aggregate outputs."""
    static_metrics, static_per_batch = run_static_experiment(df, feature_cols, cfg)
    sliding_metrics, sliding_per_batch = run_sliding_window_experiment(df, feature_cols, cfg)
    online_metrics, online_per_batch, adwin_result = run_online_experiment(df, feature_cols, cfg)
    controlled_rows = run_controlled_decay_experiment(df, feature_cols, cfg)

    metrics_rows = static_metrics + sliding_metrics + online_metrics
    per_batch_rows = static_per_batch + sliding_per_batch + online_per_batch + controlled_rows

    return ExperimentOutput(
        metrics_rows=metrics_rows,
        per_batch_rows=per_batch_rows,
        controlled_rows=controlled_rows,
        adwin_result=adwin_result,
    )

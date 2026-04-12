from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from config import Config
from src.drift_detection import ADWINResult, detect_drift_with_adwin
from src.metrics import accuracy, f1, per_batch_metrics, precision, recall
from src.models.adaptive_models import build_adaptive_random_forest
from src.models.static_models import build_logistic_regression, build_random_forest
from src.preprocessing import fit_preprocessor, transform_df
from src.split import get_batch_slices, split_static_train_test
@dataclass(slots=True)
class ExperimentOutput:
    metrics_rows: List[dict]
    per_batch_rows: List[dict]
    controlled_rows: List[dict]
    adwin_result: ADWINResult
def _make_pred_df(batch_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"batch_id": batch_ids, "y_true": y_true, "y_pred": y_pred})
def run_static_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict]]:
    train_df, test_df, train_ids, test_ids = split_static_train_test(df, cfg.static_train_batches)
    global_labels = sorted(df["label"].unique().tolist())
    pre = fit_preprocessor(train_df, feature_cols, label_values=global_labels)
    X_train, y_train = transform_df(pre, train_df, feature_cols)
    X_test, y_test = transform_df(pre, test_df, feature_cols)
    models = {
        "random_forest": build_random_forest(cfg.random_seed, cfg.rf_n_estimators),
        "logistic_regression": build_logistic_regression(cfg.random_seed, cfg.lr_max_iter),
    }
    metrics_rows: list[dict] = []
    per_batch_rows: list[dict] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        overall_acc = accuracy(y_test, pred)
        overall_prec = precision(y_test, pred)
        overall_rec = recall(y_test, pred)
        overall_f1 = f1(y_test, pred)
        metrics_rows.append(
            {
                "experiment": "static_train_once",
                "model": name,
                "learning_type": "Static",
                "metric_level": "overall",
                "batch_id": "overall",
                "accuracy": overall_acc,
                "precision": overall_prec,
                "recall": overall_rec,
                "f1_score": overall_f1,
                "train_batches": ",".join(map(str, train_ids)),
                "test_batches": ",".join(map(str, test_ids)),
            }
        )
        pred_df = _make_pred_df(test_df["batch_id"].to_numpy(), y_test, pred)
        pb = per_batch_metrics(pred_df)
        for _, row in pb.iterrows():
            per_batch_rows.append(
                {
                    "experiment": "static_train_once",
                    "model": name,
                    "learning_type": "Static",
                    "metric_level": "per_batch",
                    "batch_id": int(row["batch_id"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                }
            )
    return metrics_rows, per_batch_rows
def run_sliding_window_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict]]:
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
            prec = precision(y_test, pred)
            rec = recall(y_test, pred)
            f1_val = f1(y_test, pred)
            per_batch_rows.append(
                {
                    "experiment": "sliding_window",
                    "model": name,
                    "learning_type": "Adaptive",
                    "metric_level": "per_batch",
                    "batch_id": int(test_id),
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1_val,
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
                "precision": float(g["precision"].mean()),
                "recall": float(g["recall"].mean()),
                "f1_score": float(g["f1_score"].mean()),
                "train_batches": f"window={cfg.sliding_window_size}",
                "test_batches": "next_batch",
            }
        )
    return metrics_rows, per_batch_rows
def _to_river_dict(row: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    return {feature_cols[i]: float(row[i]) for i in range(len(feature_cols))}
def run_online_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> tuple[list[dict], list[dict], ADWINResult]:
    batch_map = get_batch_slices(df)
    batch_ids = sorted(batch_map.keys())
    init_id = batch_ids[0]
    init_df = batch_map[init_id]
    global_labels = sorted(df["label"].unique().tolist())
    pre = fit_preprocessor(init_df, feature_cols, label_values=global_labels)
    X_init, y_init = transform_df(pre, init_df, feature_cols)
    arf = build_adaptive_random_forest(cfg.random_seed)
    for x, y in zip(X_init, y_init):
        arf.learn_one(_to_river_dict(x, feature_cols), int(y))
    per_batch_rows: list[dict] = []
    arf_error_stream: list[float] = []
    for bid in batch_ids[1:]:
        bdf = batch_map[bid]
        Xb, yb = transform_df(pre, bdf, feature_cols)
        pred_arf_list: list[int] = []
        for x, y in zip(Xb, yb):
            x_dict = _to_river_dict(x, feature_cols)
            pred = arf.predict_one(x_dict)
            pred_int = int(pred) if pred is not None else 0
            pred_arf_list.append(pred_int)
            arf_error_stream.append(0.0 if pred_int == int(y) else 1.0)
            arf.learn_one(x_dict, int(y))
        acc_arf = accuracy(yb, pred_arf_list)
        prec_arf = precision(yb, pred_arf_list)
        rec_arf = recall(yb, pred_arf_list)
        f1_arf = f1(yb, pred_arf_list)
        per_batch_rows.append(
            {
                "experiment": "online_incremental",
                "model": "adaptive_random_forest",
                "learning_type": "Adaptive",
                "metric_level": "per_batch",
                "batch_id": int(bid),
                "accuracy": acc_arf,
                "precision": prec_arf,
                "recall": rec_arf,
                "f1_score": f1_arf,
            }
        )
    metrics_rows: list[dict] = []
    pb_df = pd.DataFrame(per_batch_rows)
    if not pb_df.empty:
        g = pb_df.groupby("model").agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1_score=("f1_score", "mean"),
        ).reset_index()
        for _, row in g.iterrows():
            metrics_rows.append(
                {
                    "experiment": "online_incremental",
                    "model": row["model"],
                    "learning_type": "Adaptive",
                    "metric_level": "overall",
                    "batch_id": "overall",
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                    "train_batches": str(init_id),
                    "test_batches": f"{batch_ids[1]}..{batch_ids[-1]}",
                }
            )
    adwin_result = detect_drift_with_adwin(arf_error_stream, delta=cfg.adwin_delta)
    return metrics_rows, per_batch_rows, adwin_result
def run_controlled_decay_experiment(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> list[dict]:
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
                    "precision": precision(yb, pred),
                    "recall": recall(yb, pred),
                    "f1_score": f1(yb, pred),
                }
            )
    return rows
def run_all_experiments(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> ExperimentOutput:
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

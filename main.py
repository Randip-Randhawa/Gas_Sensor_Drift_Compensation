from __future__ import annotations
import pandas as pd
from config import CONFIG
from src.comparison import build_summary_text, generate_comparison_table
from src.data_loader import load_batches
from src.drift_detection import drift_magnitude_by_batch
from src.eda import run_basic_eda
from src.experiments import run_all_experiments
from src.utils import ensure_dir, save_text, set_seed
from src.visualization import (
    plot_accuracy_vs_batch,
    plot_adwin_error,
    plot_drift_magnitude,
    plot_f1_vs_batch,
    plot_model_comparison_bar,
    plot_pca_by_batch,
    plot_pca_by_class,
    plot_precision_vs_batch,
    plot_recall_vs_batch,
)
def main() -> None:
    set_seed(CONFIG.random_seed)
    ensure_dir(CONFIG.data_dir)
    ensure_dir(CONFIG.plots_dir)
    ensure_dir(CONFIG.results_dir)
    loaded = load_batches(CONFIG.dataset_dir, CONFIG.batch_pattern, CONFIG.n_features)
    df = loaded.df
    feature_cols = loaded.feature_cols
    run_basic_eda(df, CONFIG.results_dir)
    drift_df = drift_magnitude_by_batch(df, feature_cols)
    drift_df.to_csv(CONFIG.results_dir / "drift_magnitude.csv", index=False)
    plot_pca_by_batch(df, feature_cols, CONFIG.plots_dir / "pca_by_batch.png")
    plot_pca_by_class(df, feature_cols, CONFIG.plots_dir / "pca_by_class.png")
    plot_drift_magnitude(drift_df, CONFIG.plots_dir / "drift_magnitude.png")
    out = run_all_experiments(df, feature_cols, CONFIG)
    metrics_df = pd.DataFrame(out.metrics_rows)
    per_batch_df = pd.DataFrame(out.per_batch_rows)
    all_metrics = pd.concat([metrics_df, per_batch_df], ignore_index=True)
    all_metrics.to_csv(
        CONFIG.results_dir / "metrics.csv",
        index=False,
        columns=["model", "batch_id", "accuracy", "precision", "recall", "f1_score"],
    )
    generate_comparison_table(metrics_df, CONFIG.results_dir / "comparison_table.csv")
    exp_lines = per_batch_df[
        per_batch_df["experiment"].isin(["static_train_once", "sliding_window", "online_incremental"])
    ]
    plot_accuracy_vs_batch(
        exp_lines,
        CONFIG.plots_dir / "accuracy_vs_batch_all_models.png",
        "Accuracy vs Batch Index (All Main Experiments)",
    )
    plot_f1_vs_batch(
        exp_lines,
        CONFIG.plots_dir / "f1_vs_batch_all_models.png",
        "F1-Score vs Batch Index (All Main Experiments)",
    )
    plot_precision_vs_batch(
        exp_lines,
        CONFIG.plots_dir / "precision_vs_batch_all_models.png",
        "Precision vs Batch Index (All Main Experiments)",
    )
    plot_recall_vs_batch(
        exp_lines,
        CONFIG.plots_dir / "recall_vs_batch_all_models.png",
        "Recall vs Batch Index (All Main Experiments)",
    )
    controlled_df = per_batch_df[per_batch_df["experiment"] == "controlled_decay"]
    plot_accuracy_vs_batch(
        controlled_df,
        CONFIG.plots_dir / "controlled_decay_curve.png",
        "Controlled Experiment: Train Batch 1, Test on Future Batches",
    )
    plot_f1_vs_batch(
        controlled_df,
        CONFIG.plots_dir / "controlled_decay_f1_curve.png",
        "Controlled Experiment: F1-Score (Train Batch 1, Test on Future Batches)",
    )
    plot_adwin_error(
        out.adwin_result.error_stream,
        out.adwin_result.drift_indices,
        CONFIG.plots_dir / "adwin_drift_detection.png",
    )
    overall_df = metrics_df[metrics_df["metric_level"] == "overall"].copy()
    plot_model_comparison_bar(overall_df, CONFIG.plots_dir / "model_comparison_bar.png")
    summary = build_summary_text(metrics_df)
    save_text(CONFIG.results_dir / "summary.txt", summary)
    print("Pipeline complete.")
    print(f"Metrics: {CONFIG.results_dir / 'metrics.csv'}")
    print(f"Comparison: {CONFIG.results_dir / 'comparison_table.csv'}")
    print(f"Summary: {CONFIG.results_dir / 'summary.txt'}")
    print(f"Plots directory: {CONFIG.plots_dir}")
if __name__ == "__main__":
    main()

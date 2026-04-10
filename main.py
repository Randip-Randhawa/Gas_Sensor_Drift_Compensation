"""Main entry point for gas sensor drift compensation project."""

from __future__ import annotations

import pandas as pd

from config import CONFIG
from src.comparison import build_comparison_table, build_summary_text
from src.data_loader import load_batches
from src.drift_detection import drift_magnitude_by_batch
from src.eda import run_basic_eda
from src.experiments import run_all_experiments
from src.utils import ensure_dir, save_json, save_text, set_seed
from src.visualization import (
    plot_accuracy_vs_batch,
    plot_adwin_error,
    plot_drift_magnitude,
    plot_model_comparison_bar,
    plot_pca_by_batch,
    plot_pca_by_class,
)


def main() -> None:
    """Run full pipeline end-to-end."""
    set_seed(CONFIG.random_seed)
    ensure_dir(CONFIG.data_dir)
    ensure_dir(CONFIG.plots_dir)
    ensure_dir(CONFIG.results_dir)

    loaded = load_batches(CONFIG.dataset_dir, CONFIG.batch_pattern, CONFIG.n_features)
    df = loaded.df
    feature_cols = loaded.feature_cols

    # EDA
    run_basic_eda(df, CONFIG.results_dir)

    # Drift analysis
    drift_df = drift_magnitude_by_batch(df, feature_cols)
    drift_df.to_csv(CONFIG.results_dir / "drift_magnitude.csv", index=False)

    # Visual drift plots
    plot_pca_by_batch(df, feature_cols, CONFIG.plots_dir / "pca_by_batch.png")
    plot_pca_by_class(df, feature_cols, CONFIG.plots_dir / "pca_by_class.png")
    plot_drift_magnitude(drift_df, CONFIG.plots_dir / "drift_magnitude.png")

    # Experiments
    out = run_all_experiments(df, feature_cols, CONFIG)
    metrics_df = pd.DataFrame(out.metrics_rows)
    per_batch_df = pd.DataFrame(out.per_batch_rows)

    all_metrics = pd.concat([metrics_df, per_batch_df], ignore_index=True)
    all_metrics.to_csv(CONFIG.results_dir / "metrics.csv", index=False)

    # Required plots
    exp_lines = per_batch_df[
        per_batch_df["experiment"].isin(["static_train_once", "sliding_window", "online_incremental"])
    ]
    plot_accuracy_vs_batch(
        exp_lines,
        CONFIG.plots_dir / "accuracy_vs_batch_all_models.png",
        "Accuracy vs Batch Index (All Main Experiments)",
    )

    controlled_df = per_batch_df[per_batch_df["experiment"] == "controlled_decay"]
    plot_accuracy_vs_batch(
        controlled_df,
        CONFIG.plots_dir / "controlled_decay_curve.png",
        "Controlled Experiment: Train Batch 1, Test on Future Batches",
    )

    plot_adwin_error(
        out.adwin_result.error_stream,
        out.adwin_result.drift_indices,
        CONFIG.plots_dir / "adwin_drift_detection.png",
    )

    overall_df = metrics_df[metrics_df["metric_level"] == "overall"].copy()
    plot_model_comparison_bar(overall_df, CONFIG.plots_dir / "model_comparison_bar.png")

    # Comparison and interpretation
    comparison_df = build_comparison_table(overall_df)
    comparison_df.to_csv(CONFIG.results_dir / "comparison_table.csv", index=False)
    save_json(
        CONFIG.results_dir / "comparison_table.json",
        comparison_df.to_dict(orient="records"),
    )

    summary = build_summary_text(
        comparison_df=comparison_df,
        per_batch_df=per_batch_df,
        drift_df=drift_df,
        adwin_drifts=out.adwin_result.drift_indices,
    )
    save_text(CONFIG.results_dir / "summary.txt", summary)

    print("Pipeline complete.")
    print(f"Metrics: {CONFIG.results_dir / 'metrics.csv'}")
    print(f"Summary: {CONFIG.results_dir / 'summary.txt'}")
    print(f"Plots directory: {CONFIG.plots_dir}")


if __name__ == "__main__":
    main()

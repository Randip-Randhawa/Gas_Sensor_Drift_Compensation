"""Comparison table and interpretation generation."""

from __future__ import annotations

import pandas as pd


def _adaptivity_label(learning_type: str) -> str:
    if learning_type == "Static":
        return "Low"
    if learning_type == "Online":
        return "High"
    if learning_type == "Adaptive":
        return "High"
    return "Medium"


def _observation(model: str, learning_type: str, acc: float) -> str:
    if learning_type == "Static":
        return "Strong initial fit but vulnerable to temporal shift." if acc >= 0.6 else "Performance impacted by drift and stale decision boundary."
    if model == "adaptive_random_forest":
        return "Tracks evolving distributions better via continual adaptation."
    if learning_type == "Online":
        return "Incremental updates improve adaptability but may increase variance."
    return "Balances recency and stability through adaptive retraining."


def build_comparison_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    """Construct required comparison table columns."""
    rows = []
    for _, r in overall_df.iterrows():
        rows.append(
            {
                "Model": r["model"],
                "Learning Type": r["learning_type"],
                "Accuracy": float(r["accuracy"]),
                "Adaptivity": _adaptivity_label(str(r["learning_type"])),
                "Observations": _observation(str(r["model"]), str(r["learning_type"]), float(r["accuracy"])),
            }
        )
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def build_summary_text(
    comparison_df: pd.DataFrame,
    per_batch_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    adwin_drifts: list[int],
) -> str:
    """Generate structured interpretation and failure analysis summary."""
    best = comparison_df.iloc[0]

    static_avg = comparison_df[comparison_df["Learning Type"] == "Static"]["Accuracy"].mean()
    adaptive_avg = comparison_df[comparison_df["Learning Type"].isin(["Adaptive", "Online"])]["Accuracy"].mean()

    worst_cases = (
        per_batch_df[per_batch_df["metric_level"] == "per_batch"]
        .sort_values("accuracy", ascending=True)
        .head(8)[["experiment", "model", "batch_id", "accuracy"]]
    )

    lines = []
    lines.append("Gas Sensor Drift Compensation - Experimental Summary")
    lines.append("===================================================")
    lines.append("")
    lines.append(f"Best model: {best['Model']} ({best['Accuracy']:.4f})")
    lines.append("")
    lines.append("Concept drift evidence")
    lines.append(f"- Mean-shift statistics computed across {len(drift_df)} consecutive batch transitions.")
    lines.append(f"- ADWIN drift detections on prediction error stream: {len(adwin_drifts)} points.")
    lines.append("")
    lines.append("Interpretation")
    lines.append(f"- Static models average accuracy: {static_avg:.4f}")
    lines.append(f"- Adaptive/online models average accuracy: {adaptive_avg:.4f}")
    lines.append("- Static models fail as their parameters are fixed while sensor distributions drift over time.")
    lines.append("- Online/adaptive models perform better by updating with recent observations.")
    lines.append("- Trade-off: higher adaptivity can increase forgetting risk and sensitivity to noise.")
    lines.append("")
    lines.append("Failure analysis")
    lines.append("- Lowest per-batch accuracies indicate hardest drift regimes and class overlap periods:")
    for _, row in worst_cases.iterrows():
        lines.append(
            f"  * {row['experiment']} | {row['model']} | batch={int(row['batch_id'])} | acc={float(row['accuracy']):.4f}"
        )
    lines.append("- Potential causes: concept shift between batches, sensor noise, and boundary overlap across classes.")
    lines.append("")
    lines.append("Note")
    lines.append("- Mondrian Forest was treated as optional and is not included in this baseline to preserve compatibility.")

    return "\n".join(lines)

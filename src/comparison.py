from __future__ import annotations
import pandas as pd
from pathlib import Path

def build_summary_text(metrics_df: pd.DataFrame) -> str:
    overall = metrics_df[metrics_df["batch_id"] == "overall"].copy()
    
    best_accuracy = overall.loc[overall["accuracy"].idxmax()]
    best_f1 = overall.loc[overall["f1_score"].idxmax()]
    best_precision = overall.loc[overall["precision"].idxmax()]
    best_recall = overall.loc[overall["recall"].idxmax()]
    
    lines = [
        f"Best Accuracy: {best_accuracy['model']} ({best_accuracy['accuracy']:.4f})",
        f"Best F1: {best_f1['model']} ({best_f1['f1_score']:.4f})",
        f"Best Precision: {best_precision['model']} ({best_precision['precision']:.4f})",
        f"Best Recall: {best_recall['model']} ({best_recall['recall']:.4f})",
    ]
    return "\n".join(lines)

def generate_comparison_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    overall = metrics_df[metrics_df["batch_id"] == "overall"].copy()
    comparison = overall.groupby("model")[["accuracy", "precision", "recall", "f1_score"]].mean().reset_index()
    comparison.to_csv(output_path, index=False)

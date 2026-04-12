from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sns.set_theme(style="whitegrid")
def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
def plot_pca_by_batch(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    X = df[feature_cols].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    plot_df = pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], "batch_id": df["batch_id"].to_numpy()})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="batch_id", palette="viridis", s=16, alpha=0.7)
    plt.title("PCA Projection Colored by Batch")
    _save(out_path)
def plot_pca_by_class(df: pd.DataFrame, feature_cols: list[str], out_path: Path) -> None:
    X = df[feature_cols].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    plot_df = pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], "label": df["label"].to_numpy()})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="label", palette="tab10", s=16, alpha=0.7)
    plt.title("PCA Projection Colored by Class")
    _save(out_path)
def plot_drift_magnitude(drift_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = drift_df["batch_curr"].to_numpy()
    y = drift_df["drift_magnitude"].to_numpy()
    sns.lineplot(x=x, y=y, marker="o", color="#005f73")
    plt.title("Drift Magnitude: ||mu_t - mu_{t-1}||")
    plt.xlabel("Current Batch")
    plt.ylabel("Magnitude")
    _save(out_path)
def plot_accuracy_vs_batch(lines_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=lines_df, x="batch_id", y="accuracy", hue="model", marker="o")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    _save(out_path)
def plot_precision_vs_batch(lines_df: pd.DataFrame, out_path: Path, title: str = "Precision vs Batch Index") -> None:
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=lines_df, x="batch_id", y="precision", hue="model", marker="s")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Precision (Weighted)")
    _save(out_path)
def plot_recall_vs_batch(lines_df: pd.DataFrame, out_path: Path, title: str = "Recall vs Batch Index") -> None:
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=lines_df, x="batch_id", y="recall", hue="model", marker="^")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Recall (Weighted)")
    _save(out_path)
def plot_f1_vs_batch(lines_df: pd.DataFrame, out_path: Path, title: str = "F1-Score vs Batch Index") -> None:
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=lines_df, x="batch_id", y="f1_score", hue="model", marker="D")
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("F1-Score (Weighted)")
    _save(out_path)
def plot_adwin_error(error_stream: list[float], drift_indices: list[int], out_path: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(error_stream, color="#ae2012", linewidth=1.0, label="Error Stream")
    for idx in drift_indices:
        plt.axvline(idx, color="#0a9396", linestyle="--", alpha=0.65)
    plt.title("ADWIN on Prediction Error Stream")
    plt.xlabel("Time Index")
    plt.ylabel("Error (0/1)")
    plt.legend()
    _save(out_path)
def plot_model_comparison_bar(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="model", y="accuracy", hue="learning_type")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.title("Overall Accuracy Comparison")
    _save(out_path)

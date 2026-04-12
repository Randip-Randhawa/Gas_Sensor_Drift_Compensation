from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils import save_json

def run_basic_eda(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_samples": int(len(df)),
        "n_features": int(len([c for c in df.columns if c.startswith("f")])),
        "n_batches": int(df["batch_id"].nunique()),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
        "missing_total": int(df.isna().sum().sum()),
    }
    save_json(output_dir / "eda_summary.json", summary)
    return summary

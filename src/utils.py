"""Utility helpers for reproducibility and file operations."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    """Write JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_text(path: Path, text: str) -> None:
    """Write text file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

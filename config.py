"""Central configuration for gas sensor drift compensation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Config:
    """Project-level configuration parameters."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    dataset_dir_name: str = "Dataset"
    data_dir_name: str = "data"
    plots_dir_name: str = "plots"
    results_dir_name: str = "results"

    batch_pattern: str = "batch*.dat"
    n_features: int = 128
    random_seed: int = 42

    # Experiments
    static_train_batches: int = 3
    sliding_window_size: int = 3

    # Online/adaptive
    adwin_delta: float = 0.002

    # Model params
    rf_n_estimators: int = 300
    lr_max_iter: int = 1000
    perceptron_max_iter: int = 1000
    sgd_max_iter: int = 1
    sgd_tol: float | None = None
    sgd_epochs_per_batch: int = 3
    sgd_alpha: float = 1e-5
    sgd_eta0: float = 0.01
    sgd_average: bool = True
    sgd_class_weight: dict[int, float] | None = None

    @property
    def dataset_dir(self) -> Path:
        return self.root_dir / self.dataset_dir_name

    @property
    def data_dir(self) -> Path:
        return self.root_dir / self.data_dir_name

    @property
    def plots_dir(self) -> Path:
        return self.root_dir / self.plots_dir_name

    @property
    def results_dir(self) -> Path:
        return self.root_dir / self.results_dir_name


CONFIG = Config()

# Gas Sensor Drift Compensation

A machine learning project addressing **concept drift** in gas sensor classification using adaptive and static models to compare their performance in handling sensor drift over time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Models & Approaches](#models--approaches)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Results & Key Findings](#results--key-findings)
9. [Configuration](#configuration)
10. [Contributing](#contributing)

---

## Project Overview

This project investigates how different machine learning models handle **concept drift** in gas sensor data—where sensor distributions change over time due to aging, environmental factors, and degradation. The project compares:

- **Static Models**: Train once and apply to all future batches
- **Adaptive Models**: Update parameters as new data arrives
- **Online Learning Models**: Learn incrementally from streaming data

The goal is to demonstrate that adaptive approaches significantly outperform static models when dealing with drifting distributions.

---

## Problem Statement

Gas sensors experience **sensor drift**—a gradual shift in sensor output distributions over time. This causes static models trained on initial data to degrade in performance. Key challenges:

1. **Concept Drift**: The underlying data distribution changes across batches
2. **Class Imbalance**: Certain gas sensor classes are underrepresented
3. **Accuracy vs. F1-Score Trade-off**: Accuracy can be misleading with imbalanced data; F1-score is more reliable

**Solution**: Use adaptive models that update with new batches to maintain performance.

---

## Dataset

- **Source**: UCI Gas Sensor Array Drift Dataset
- **Location**: `Dataset/` folder (batch files)
- **Format**: Binary `.dat` files with one batch per file
- **Features**: 128 sensor features per sample
- **Classes**: Multiple gas sensor classes
- **Batches**: Sequential temporal data with documented drift

The data is organized in 10 batches.
Each batch represents data collected at different time periods, allowing analysis of drift patterns.

---

## Models & Approaches

### Static Models
These models are trained once on initial batches and never updated:

1. **Logistic Regression** (`logistic_regression`)
   - Linear classifier baseline
   - No drift adaptation

2. **Random Forest** (`random_forest`)
   - Ensemble method with fixed trees
   - No parameter updates

### Adaptive/Online Models
These models update parameters as new data arrives:

1. **Sliding Window Logistic Regression** (`sliding_logistic_regression`)
   - Maintains a fixed-size window of recent batches
   - Retrains on the window after each new batch
   - Window size: configurable (default: 3 batches)

2. **Sliding Window Random Forest** (`sliding_random_forest`)
   - Random Forest trained on sliding window of batches
   - Adapts to recent drift patterns

3. **Adaptive Random Forest** (`adaptive_random_forest`)
   - River library's ARFClassifier
   - Online learning algorithm specifically designed for concept drift
   - Updates trees incrementally without maintaining history
   - **Best performer in this project** (95.25% accuracy)

### Controlled Experiments
Benchmark experiments to measure degradation:

1. **Controlled Random Forest** (`controlled_rf`)
   - Train on batch 1, test on all future batches
   - Pure measurement of model decay

2. **Controlled Logistic Regression** (`controlled_lr`)
   - Train on batch 1, test on all future batches
   - Baseline for linear model degradation

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone and navigate to the repository**:
   ```bash
   cd Gas_Sensor_Drift_Compensation
   ```

2. **Create a virtual environment**:

   **Option A: Using `venv` (Built-in)**
   ```bash
   python -m venv myenv
   
   source myenv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The virtual environment is now active and ready with all dependencies installed.

4. **Verify installation**:
   ```bash
   python --version
   pip list
   ```

5. **Deactivate virtual environment** (when done working):
   ```bash
   deactivate
   ```

### Required Packages
- `numpy>=1.24` - Numerical computing
- `pandas>=2.0` - Data manipulation
- `scikit-learn>=1.3` - ML models and preprocessing
- `scipy>=1.10` - Scientific computing
- `matplotlib>=3.7` - Plotting
- `seaborn>=0.12` - Statistical visualization
- `river>=0.21` - Online/streaming machine learning
- `typing_extensions>=4.8` - Type hints

---

## Usage

### Run the Full Pipeline

Execute all experiments, train models, and generate reports:

```bash
python main.py
```

This command:
1. Loads and preprocesses all batch data
2. Performs exploratory data analysis (EDA)
3. Detects drift patterns using ADWIN algorithm
4. Runs static and adaptive experiments
5. Generates visualizations
6. Produces comparison tables and summary report

**Output locations**:
- Metrics: `results/metrics.csv`
- Summary: `results/summary.txt`
- Plots: `plots/` directory
- Comparison Table: `results/comparison_table.csv` & `.json`

### Run Specific Components

```python
# Import modules
from src.data_loader import load_batches
from src.drift_detection import drift_magnitude_by_batch
from src.experiments import run_all_experiments

# Load data
loaded = load_batches(config.dataset_dir, config.batch_pattern, config.n_features)
df = loaded.df
feature_cols = loaded.feature_cols

# Detect drift
drift_df = drift_magnitude_by_batch(df, feature_cols)

# Run experiments
results = run_all_experiments(df, feature_cols, config)
```

---

## Results & Key Findings

### Overall Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Adaptive Random Forest** | 0.9525 | 0.9560 | 0.9525 | 0.9534 |
| Sliding Random Forest | 0.7795 | 0.8083 | 0.7795 | 0.7633 |
| Sliding Logistic Regression | 0.8029 | 0.8298 | 0.8029 | 0.7988 |
| Random Forest (Static) | 0.5510 | 0.5101 | 0.5510 | 0.5125 |
| Logistic Regression (Static) | 0.6214 | 0.5520 | 0.6214 | 0.5626 |

### Key Insights

1. **Adaptive Models Outperform Static Models**
   - Static average accuracy: 58.62%
   - Adaptive/online average accuracy: 84.50%
   - **45% improvement** with adaptive approaches

2. **Adaptive Random Forest is Superior**
   - Achieves 95.25% accuracy
   - Only method achieving near-perfect performance
   - Efficiently adapts without maintaining batch history

3. **Concept Drift is Real**
   - ADWIN detected 9 drift points across 9 batch transitions
   - Mean-shift statistics confirm distribution changes
   - Controlled experiment shows severe decay: 99%→24.5% accuracy

4. **Class Imbalance Matters**
   - F1-score is more reliable than accuracy
   - Static models struggle with minority classes
   - Adaptive models better handle class-specific drift

5. **Sliding Window Approach Effective**
   - Sliding Random Forest: 77.95% accuracy
   - Sliding Logistic Regression: 80.29% accuracy
   - Window-based retraining captures recent patterns

### Visualizations Generated

- **PCA Projections**: Batch-wise and class-wise dimensionality reduction
- **Drift Magnitude Curves**: Euclidean norm of batch-to-batch feature shifts
- **Accuracy & F1 vs Batch**: Per-batch performance across all models
- **Model Comparison Bar Chart**: Overall metric comparison
- **ADWIN Drift Detection**: Error stream with detected drift points

---

## Configuration

Edit `config.py` to customize experiments:

```python
class Config:
    n_features: int = 128                    # Number of sensor features
    random_seed: int = 42                    # Reproducibility
    static_train_batches: int = 3            # Batches to train on
    sliding_window_size: int = 3             # Window for sliding approaches
    adwin_delta: float = 0.002               # ADWIN sensitivity
    rf_n_estimators: int = 300               # Random forest trees
    lr_max_iter: int = 1000                  # Logistic regression iterations
```

### Key Parameters

- **`random_seed`**: Set to ensure reproducible results
- **`sliding_window_size`**: Increase for more stability, decrease for responsiveness
- **`adwin_delta`**: Lower = more sensitive to drift, higher = more conservative
- **`rf_n_estimators`**: More trees = better but slower

---

## Metrics Explained

- **Accuracy**: Percentage of correct predictions (can be misleading with imbalanced data)
- **Precision**: Of predicted positives, how many were correct (per-class)
- **Recall**: Of actual positives, how many were found (per-class)
- **F1-Score**: Harmonic mean of precision and recall (recommended for imbalanced data)

---


## References

- **ADWIN**: Bifet & Gavaldà (2007) - Learning from Time-Changing Data with Adaptive Windowing
- **Concept Drift**: Gama et al. (2013) - A Survey on Concept Drift Adaptation
- **UCI Dataset**: Vergara et al. (2012) - Chemical gas sensor array drift dataset
- **River Library**: https://riverml.xyz/ - Online machine learning in Python
- **Adaptive Random Forest**: Gomes et al. (2017) - Adaptive Random Forests with Resampling

---

## License

This is an academic project. Use freely for educational purposes.

---

## Contact

For questions or issues, refer to the project repository or contact the project maintainer.

---

## Changelog

### v1.0 (Current)
- Initial implementation
- 5 models tested (2 static, 3 adaptive)
- ADWIN drift detection
- Comprehensive visualization suite
- Full pipeline automation


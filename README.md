# Gas_Sensor_Drift_Compensation

A machine learning project that tackles **sensor drift** in chemical gas sensors using the UCI Gas Sensor Array Drift Dataset. The notebook demonstrates how sensor performance degrades over time and evaluates two compensation strategies to recover accuracy.

---

## Dataset

**UCI Gas Sensor Array Drift Dataset**

- **Format:** LibSVM sparse format (converted to DataFrame)
- **Samples:** 10 batches collected over time
- **Features:** 128 features per sample — 8 statistical descriptors (DR, NI, DN, LN, SD, M, AM, LN) across 16 sensors
- **Labels:** 6 gas classes (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/)

---

## Project Structure

```
├── Dataset/
│   ├── batch1.dat
│   ├── batch2.dat
│   └── ... (batch3–batch10.dat)
└── Gas_Sensor_Drift_Compensation.ipynb
```

---

## Requirements

```
Python 3.x
numpy
pandas
matplotlib
scikit-learn
```

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

The notebook was developed in the `MYML_env` conda/virtual environment.

---

## Workflow Overview

### Section 0 — Library Imports
Loads NumPy, Pandas, Matplotlib, and Scikit-learn components.

### Section 1 — Data Loading
Parses all 10 batch `.dat` files from LibSVM format into a unified Pandas DataFrame with 128 feature columns and a batch ID column.

### Section 2 — Exploratory Data Analysis
Examines class balance across batches, batch sizes, missing values, and gas-to-batch distributions through summary statistics and visualisations.

### Section 3 — Preprocessing
Checks for missing values (none found) and applies `StandardScaler` for feature scaling. Discusses why global scaling is the baseline approach.

### Section 4 — Time-Based Train/Test Split
Splits data temporally — **Batches 1–7 for training**, **Batches 8–10 for testing** — to simulate real-world deployment where future drift is unknown.

### Section 5 — Baseline Model
Trains a **Random Forest** (200 estimators, max depth 20) on globally scaled training data. This is the performance floor with no drift compensation.

### Section 6 — Drift Demonstration
Evaluates the baseline model batch-by-batch to confirm that accuracy degrades over time, providing direct evidence of sensor drift causing distribution mismatch.

### Section 7 — Drift Visualisation with PCA
Reduces 128 features to 2 dimensions with PCA and plots batch clusters and centroid trajectories, visually confirming the drift phenomenon.

### Section 8 — Drift Compensation Methods

Two compensation strategies are implemented and compared:

**Method A: Batch-wise Normalisation**
Each batch is normalised independently using its own mean and standard deviation, removing batch-specific bias:

$$x'_{i,b} = \frac{x_{i,b} - \mu_b}{\sigma_b}$$

**Method B: Sliding Window Training**
Only the most recent 3 training batches (B5, B6, B7) are used for training, keeping the model's distribution aligned with the test period.

### Section 9 — Comparison of All Methods
Side-by-side accuracy table, bar charts, per-batch accuracy line plots, and confusion matrices for all three approaches (Baseline, Method A, Method B).

### Section 10 — Final Conclusion
Summarises test set accuracy and improvement (Δ) for each method, and shows a full-story plot of per-batch performance across all 10 batches.

---

## Key Results

| Method | Test Accuracy |
|---|---|
| Baseline (Global Scaling) | Lower — degrades on drifted batches |
| Method A: Batch-wise Normalisation | ✅ Improved |
| Method B: Sliding Window (last 3 batches) | ✅ Improved |

Both compensation methods outperform the baseline on the drifted test batches (8–10).

---

## References

1. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). *Chemical gas sensor drift compensation using classifier ensembles.* Sensors and Actuators B: Chemical, 166, 320–329.
2. UCI Machine Learning Repository — Gas Sensor Array Drift Dataset: https://archive.ics.uci.edu/ml/datasets/

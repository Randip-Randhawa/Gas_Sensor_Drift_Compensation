# Gas Sensor Drift Compensation using Machine Learning

**Domain:** Machine Learning · Sensor Data · Concept Drift  
**Dataset:** UCI Gas Sensor Array Drift Dataset (Real Data)  
**Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn

---

## Project Overview

This project uses the real-world **UCI Gas Sensor Array Drift Dataset** — 13,910 readings collected over 36 months across 10 measurement batches from a 16-sensor array. Due to physical degradation of sensor materials, sensor responses drift progressively over time.

The challenge: a model trained on early batches must classify gases in later batches where sensor characteristics have shifted.

---

## Gas Classes

| Code | Gas | Code | Gas |
|------|-------------|------|-------------|
| 1 | Ethanol | 4 | Acetone |
| 2 | Ethylene | 5 | Acetaldehyde |
| 3 | Ammonia | 6 | Toluene |

> Some gases are absent from certain batches — this reflects real collection constraints in the original experiment (e.g. Toluene is absent from batches 3–5).

---

## Key ML Concepts

| Concept | Definition |
|---------|------------|
| **Covariate Shift** | P(X) changes over time; P(Y\|X) remains stable |
| **Concept Drift** | The relationship P(Y\|X) itself changes |
| **Temporal Leakage** | Using future data in training — inflates accuracy dishonestly |
| **Distribution Mismatch** | Train and test come from different distributions |

---

## Dataset

**Format:** LibSVM sparse format — each row:
```
<gas_label>  1:<value>  2:<value>  ...  128:<value>
```

- **128 features:** 8 statistical descriptors (DR, NI, DN, LN, SD, M, AM, LNSD) from each of 16 sensors
- **Total samples:** 13,910 across 10 batches
- **Missing values:** None

**Samples per batch:**

| Batch | Samples | Split |
|-------|---------|-------|
| 1 | 445 | Train |
| 2 | 1,244 | Train |
| 3 | 1,586 | Train |
| 4 | 161 | Train |
| 5 | 197 | Train |
| 6 | 2,300 | Train |
| 7 | 3,613 | Train |
| 8 | 294 | **Test** |
| 9 | 470 | **Test** |
| 10 | 3,600 | **Test** |

---

## Requirements

```
Python 3.x
numpy
pandas
matplotlib
scikit-learn
```

Install:
```bash
pip install numpy pandas matplotlib scikit-learn
```

Developed in the `MYML_env` environment (NumPy 2.4.3, Pandas 3.0.1).

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

## Notebook Walkthrough

### Section 0 — Library Imports
Loads NumPy, Pandas, Matplotlib, and Scikit-learn. Sets `RANDOM_STATE = 42` for reproducibility.

### Section 1 — Data Loading
Parses all 10 `.dat` files from LibSVM format into a single DataFrame with 128 feature columns (`f1`–`f128`), a `label` column, and a `batch` column preserving temporal order.

### Section 2 — Exploratory Data Analysis
Key observations from the data:
- Dataset is **imbalanced** — Acetaldehyde (21.6%) and Ethylene (21.0%) dominate; Ammonia is smallest (11.8%)
- Batch sizes vary greatly (161–3,613) — real-world irregularity
- Some gases absent from batches 3–5

### Section 3 — Preprocessing
Feature ranges vary enormously before scaling (e.g. `f1` spans −16,757 to 670,687; `f2` spans 0.09 to 1,339). `StandardScaler` is fit **only on training data** to prevent leakage.

### Section 4 — Time-Based Train/Test Split

```
Timeline ────────────────────────────────────────────────►
  Batch: [1][2][3][4][5][6][7]   |   [8][9][10]
          ◄─── TRAINING ─────►       ◄─── TEST ──►
                               ↑
                        Deployment point
```

- **Train:** Batches 1–7 → 9,546 samples (68.6%)
- **Test:** Batches 8–10 → 4,364 samples (31.4%)

A random shuffle would create temporal leakage, inflating accuracy dishonestly.

### Section 5 — Baseline Model
A **Random Forest** (200 estimators, max depth 20, min samples leaf 2) trained on globally scaled training data.

**Baseline test accuracy: 61.46%**

Per-class breakdown on test set:

| Gas | Precision | Recall | F1 |
|-----|-----------|--------|----|
| Ethanol | 0.71 | 0.48 | 0.57 |
| Ethylene | 0.56 | 0.60 | 0.58 |
| Ammonia | 0.66 | 0.63 | 0.64 |
| Acetone | 0.47 | 0.70 | 0.56 |
| Acetaldehyde | 0.76 | 0.84 | 0.80 |
| Toluene | 0.58 | 0.41 | 0.48 |

### Section 6 — Drift Demonstration
The baseline evaluated batch-by-batch reveals a dramatic drop at the train/test boundary:

| Batch | Accuracy | Split |
|-------|----------|-------|
| 1–7 | ~99.5–100% | Train |
| 8 | 86.39% ⚠️ | Test |
| 9 | 76.38% ⚠️ | Test |
| 10 | 57.47% ⚠️ | Test |

**Drift-induced drop: 26.45 percentage points** (mean train 99.87% → mean test 73.42%).

### Section 7 — Drift Visualisation with PCA
PCA reduces 128 features to 2 dimensions (PC1 = 55.15%, PC2 = 14.79%, total = 69.94% variance explained).

- **Left plot (coloured by batch):** Batches form distinct, non-overlapping clusters confirming covariate shift
- **Right plot (coloured by gas class):** Gas classes remain separable within batches — P(Y|X) is stable
- **Centroid trajectory:** Shows a progressive, systematic shift across time — classic sensor drift

**Diagnosis: Covariate shift, not concept drift. Compensation can fix this.**

### Section 8 — Drift Compensation Methods

**Method A: Batch-wise Normalisation**

Each batch is normalised independently using its own mean and standard deviation:

$$x'_{i,b} = \frac{x_{i,b} - \mu_b}{\sigma_b}$$

At test time, μ and σ are computed from the test batch itself (realistic — a small unlabelled calibration set is usually available in practice).

**Method B: Sliding Window Training**

Only the 3 most recent training batches (B5, B6, B7) are used:

```
Full training:    [B1][B2][B3][B4][B5][B6][B7]  → TEST [B8][B9][B10]
Sliding window:               [B5][B6][B7]       → TEST [B8][B9][B10]
```

6,110 samples used (vs 9,546 full). Recent batches are most similar to test data; old data can actively hurt the model by anchoring it to outdated patterns.

### Section 9 — Comparison of All Methods

| Method | Test Accuracy | Δ vs Baseline | Strategy |
|--------|--------------|---------------|----------|
| Baseline (Global Scaling) | 61.46% | — | No compensation |
| Method A: Batch-wise Normalisation | 56.99% | −4.47 pp | Remove per-batch mean/variance |
| **Method B: Sliding Window (last 3)** | **61.64%** | **+0.18 pp** | Train on recent batches only |

**Winner: Method B** with 61.64% test accuracy.

### Section 10 — Final Conclusion
Method B marginally outperforms the baseline overall, and shows notably better performance on Batch 8 (88.78% vs 86.39%). Method A underperforms the baseline on this dataset — batch-wise normalisation alone is insufficient when the distribution shift involves more than simple mean/variance changes.

---

## Viva Q&A

**Q: Why is random train-test split wrong here?**  
A: It creates temporal leakage — future data enters training, inflating accuracy. Real deployment uses only past data to predict future sensor readings.

**Q: Is this covariate shift or concept drift?**  
A: Covariate shift. P(X) drifts with sensor degradation, but P(Y|X) — the gas-to-sensor relationship — is physically stable. PCA shows classes stay separable within each batch.

**Q: Why does batch normalisation help in theory?**  
A: Drift shifts sensor baselines (mean). Per-batch standardisation removes this offset, aligning distributions across batches.

**Q: Why does sliding window help?**  
A: Recent batches are most similar to test data. Old batches anchor the model to outdated patterns that no longer reflect sensor behaviour.

**Q: What are advanced alternatives?**  
A: CORAL (covariance alignment), DANN (domain-adversarial neural networks), importance reweighting, online learning with ADWIN.

---

## Saved Plots

| File | Contents |
|------|----------|
| `eda_plots.png` | Class distribution, samples per batch, class × batch heatmap |
| `confusion_baseline.png` | Baseline confusion matrix |
| `accuracy_vs_batch.png` | Per-batch accuracy showing drift drop |
| `pca_viz.png` | PCA scatter coloured by batch and by gas class |
| `pca_trajectory.png` | Batch centroid trajectory in PCA space |
| `method_comparison.png` | Bar chart + per-batch line plot for all methods |
| `confusion_comparison.png` | Side-by-side confusion matrices for all methods |
| `final_summary.png` | Full story plot: all methods across all batches |

---

## References

1. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012). *Chemical gas sensor drift compensation using classifier ensembles.* Sensors and Actuators B: Chemical, 166, 320–329.
2. UCI Machine Learning Repository — Gas Sensor Array Drift Dataset: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
3. Gama, J., et al. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys.
4. Shimodaira, H. (2000). *Improving predictive inference under covariate shift.* Journal of Statistical Planning and Inference.


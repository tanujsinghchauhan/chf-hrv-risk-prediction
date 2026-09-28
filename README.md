# AI-Powered Heart Failure Risk Prediction from HRV

An end-to-end research prototype for patient-aware heart-failure risk prediction using heart-rate variability (HRV), patient-specific baseline drift, temporal modeling, uncertainty information, and explainable AI.

> **Research prototype — not a clinical diagnostic system.**

---

## Overview

This project investigates whether HRV patterns and their temporal changes can be used to estimate heart-failure risk from publicly available PhysioNet data.

The system combines three complementary approaches:

1. **Random Forest baseline** using engineered HRV and trend features.
2. **Personalized Random Forest** that adjusts the population-level risk using patient-specific HRV baseline drift and history.
3. **GRU temporal model** that learns from sequential HRV windows.

The project also includes:

- patient-aware validation
- uncertainty/confidence analysis
- patient-level error analysis
- model agreement analysis
- threshold analysis
- risk stratification
- SHAP explainability

---

## Project Pipeline

```text
Public PhysioNet ECG/RR data
            ↓
       RR processing
            ↓
      Quality filtering
            ↓
         Windowing
            ↓
      HRV extraction
            ↓
    HRV trend features
            ↓
    Patient-aware labels
            ↓
   ┌────────┴─────────┐
   │                  │
Random Forest        GRU
   │                  │
   ↓                  ↓
Population risk    Temporal risk
   │
   ↓
Patient HRV baseline
   │
   ↓
Drift detection
   │
   ↓
Personalized risk
   │
   └──────────┬───────┘
              ↓
       Patient-level
          evaluation
              ↓
      Risk + confidence
              ↓
        SHAP explanation
```

---

## Dataset

Two public PhysioNet CHF datasets are used.

### CHF2DB

```text
chf201 – chf229
29 patients
```

### CHFDB

```text
chf01 – chf15
15 patients
```

Total:

```text
44 patients
```

The project uses publicly available research data rather than collecting clinical training data from real patients.

The planned ESP32 + MAX30102 hardware component is separate from model training/evaluation and is intended for live demonstration.

---

## HRV Features

The current model uses 11 features:

```text
mean_rr
sdnn
rmssd
pnn50
lf_power
hf_power
lf_hf
rr_count
sdnn_trend
rmssd_trend
pnn50_trend
```

The trend features are included to capture short-term changes in HRV rather than relying only on absolute HRV measurements.

---

## Validation Strategy

Patient-level leakage is addressed using:

```text
5-fold GroupKFold
```

with:

```python
groups = ml_dataset["patient_id"]
```

All windows belonging to a patient remain within the same fold.

Patient overlap between training and validation partitions was verified as zero.

This is important because the dataset contains many windows per patient.

---

## Baseline Results

The initial GroupKFold window-level results were:

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.595620 | 0.820841 | 0.688439 | 0.694627 | 0.777169 |
| Random Forest | 0.709600 | 0.785361 | 0.853533 | 0.801782 | 0.658424 |
| XGBoost | 0.681482 | 0.796587 | 0.779920 | 0.767054 | 0.661563 |

These are window-level GroupKFold results and should not be confused with the final patient-level comparison.

---

## Final Patient-Level Comparison

The final patient-level evaluation contains:

```text
44 patients
44 unique predictions
0 duplicate patient predictions
```

Results:

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| RF Baseline | 0.704545 | 0.771429 | 0.843750 | 0.805970 | 0.554688 |
| Personalized RF | 0.727273 | 0.777778 | 0.875000 | 0.823529 | 0.557292 |
| GRU | 0.613636 | 0.894737 | 0.531250 | 0.666667 | 0.747396 |

The models exhibit different metric profiles rather than one model dominating every metric.

---

## Personalization

The personalized model uses patient-specific HRV behavior.

The personalization pipeline includes:

```text
Patient HRV history
        ↓
Patient baseline
        ↓
Robust deviation measures
        ↓
Drift signal
        ↓
Population RF probability
        ↓
Personalized probability
```

Relevant signals include:

```text
robust_drift_score
recent_drift_score
drift_acceleration
robust_drift_flag
drift_persistence
history_count
baseline_confidence
```

### Personalization improvement

| Metric | RF | Personalized RF | Change |
|---|---:|---:|---:|
| Accuracy | 0.704545 | 0.727273 | +0.022727 |
| Precision | 0.771429 | 0.777778 | +0.006349 |
| Recall | 0.843750 | 0.875000 | +0.031250 |
| F1 | 0.805970 | 0.823529 | +0.017559 |
| AUROC | 0.554688 | 0.557292 | +0.002604 |

Error count decreased from:

```text
13 → 12
```

---

## Drift Analysis

Personalization magnitude was positively associated with patient drift.

### Pearson

```text
r = 0.5264
p = 0.000242
```

### Spearman

```text
rho = 0.421
p = 0.00443
```

Low vs high drift:

| Group | Patients | Mean drift | Mean probability change |
|---|---:|---:|---:|
| Low Drift | 22 | 0.256831 | 0.013794 |
| High Drift | 22 | 0.606096 | 0.037231 |

The largest probability adjustment occurred for:

```text
chf210
RF probability          0.125104
Personalized probability 0.296196
Absolute change          0.171092
Drift                    0.977786
```

---

## Temporal Modeling

A GRU model was implemented to capture sequential HRV information.

Patient-level GRU results:

```text
Accuracy  : 0.613636
Precision : 0.894737
Recall    : 0.531250
F1        : 0.666667
AUROC     : 0.747396
```

Confusion matrix:

```text
[[10  2]
 [15 17]]
```

The GRU provides a temporal modeling perspective complementary to the feature-based RF models.

---

## Uncertainty and Confidence

Random Forest tree disagreement was used as an uncertainty signal.

Recorded values include:

```text
Mean RF probability       = 0.715
Mean tree disagreement    = 0.0239
Maximum tree disagreement = 0.325
```

Probability variability was also used to derive:

```text
probability_certainty = 1 - rf_probability_std
```

The purpose is to avoid treating every model probability as equally reliable.

---

## Explainability

SHAP was used to inspect feature contributions to individual predictions.

For an example prediction from `chf04`, the largest absolute SHAP contributions included:

```text
sdnn_trend
pnn50_trend
sdnn
lf_power
lf_hf
```

Example SHAP values:

| Feature | SHAP |
|---|---:|
| sdnn_trend | 0.194989 |
| pnn50_trend | 0.174119 |
| sdnn | 0.037881 |
| lf_power | 0.037072 |
| lf_hf | 0.031174 |

This provides a feature-level explanation of model output.

SHAP explanations describe model behavior; they are not independent clinical evidence.

---

## Error Analysis

RF:

```text
Correct          31
False Positive    8
False Negative    5
```

Personalized RF:

```text
Correct          32
False Positive    8
False Negative    4
```

The principal binary prediction change was:

```text
chf225

RF probability          0.497271
Personalized probability 0.534051

RF prediction            0
Personalized prediction  1
```

---

## Model Agreement

Agreement rates:

```text
RF vs GRU:
28 / 44 = 63.64%

Personalized RF vs GRU:
27 / 44 = 61.36%

All three models:
27 / 44 = 61.36%
```

All three models correctly classified:

```text
21 patients
```

All three models misclassified:

```text
6 patients
```

---

## Threshold Analysis

Thresholds from:

```text
0.20 → 0.80
```

were evaluated.

At threshold `0.20`:

### Personalized RF

```text
Accuracy  = 0.750000
Precision = 0.744186
Recall    = 1.000000
F1        = 0.853333
```

### GRU

```text
Accuracy  = 0.704545
Precision = 0.720930
Recall    = 0.968750
F1        = 0.826667
```

Threshold selection is experimental and should not be interpreted as a clinically validated operating threshold.

---

## Risk Stratification

Personalized probabilities were grouped into:

```text
High Risk
Moderate Risk
Low Risk
```

Current patient-level groups:

| Risk group | Patients | Mean probability | Accuracy |
|---|---:|---:|---:|
| High Risk | 25 | 0.929180 | 0.720000 |
| Moderate Risk | 11 | 0.669802 | 0.909091 |
| Low Risk | 8 | 0.272892 | 0.500000 |

These categories are model-output groups and are not clinical diagnoses.

---

## Repository Structure

A recommended repository structure is:

```text
chf-hrv-risk-prediction/
│
├── data/
│   └── raw/
│       ├── chfdb/
│       └── chf2db/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_rr_cleaning.ipynb
│   ├── 03_hrv_features.ipynb
│   ├── 04_label_mapping.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_temporal_modeling.ipynb
│   └── 07_personalization.ipynb
│
├── figures/
├── src/
│
├── requirements.txt
├── PROJECT_KNOWLEDGE.md
└── README.md
```

Adjust the directory names to match the actual repository before pushing.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Running the Project

Run the notebooks in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07
```

The notebooks progressively perform:

```text
Dataset exploration
        ↓
RR cleaning
        ↓
HRV extraction
        ↓
Label mapping
        ↓
Baseline models
        ↓
Temporal modeling
        ↓
Personalization + explainability
```

If the notebooks have already been executed and outputs are saved, there is no need to rerun the entire pipeline just to inspect the recorded results.

---

## Figures

The repository should retain the final report-quality figures in:

```text
figures/
```

Important figure categories include:

- model metric comparison
- ROC curves
- confusion matrices
- threshold analysis
- drift vs personalization change
- low-vs-high drift comparison
- RF vs personalized probability
- SHAP explanation
- patient risk stratification
- model agreement/disagreement
- temporal/GRU probability visualization

Only figures that have actually been generated should be committed.

---

## Limitations

This is a research prototype and has important limitations:

- only 44 patients are available
- many windows come from the same patient
- patient-aware validation is therefore essential
- the risk label is based on available clinical severity information
- the model is not prospectively validated
- risk thresholds are experimental
- personalization/drift relationships are observational
- results should not be interpreted as clinical validation
- hardware demonstration data is separate from model training/evaluation

---

## Research Contribution

The project focuses on the combination of:

```text
HRV features
+
HRV trends
+
patient-specific baseline
+
robust drift detection
+
uncertainty/confidence
+
temporal GRU modeling
+
SHAP explainability
```

The goal is to move beyond a generic "AI + IoT" architecture toward a reproducible, patient-aware CHF risk prediction pipeline using public data.

---

## Current Status

Completed:

```text
[x] Data acquisition
[x] Data processing
[x] RR cleaning
[x] HRV feature extraction
[x] Trend features
[x] Risk labels
[x] Baseline models
[x] Patient-aware GroupKFold
[x] Patient-level RF evaluation
[x] GRU temporal model
[x] Patient-level GRU evaluation
[x] Personalized RF
[x] Patient drift analysis
[x] Uncertainty analysis
[x] Error analysis
[x] Threshold analysis
[x] Risk stratification
[x] Model agreement analysis
[x] SHAP explainability
```

Remaining:

```text
[ ] Final figure cleanup
[ ] Final robustness / ablation experiments
[ ] Final end-to-end inference pipeline
[ ] Dashboard integration
[ ] Hardware demonstration integration
[ ] Final report
[ ] Final presentation
```

---

## Important

This repository is intended for academic/research use.

It does **not** provide medical diagnosis or clinical decision support.

Do not use the reported model probabilities or thresholds for real patient treatment decisions.


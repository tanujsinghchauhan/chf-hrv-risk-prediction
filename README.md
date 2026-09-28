# CHF-HRV Risk Prediction

Patient-aware heart failure severity prediction from heart-rate variability (HRV), using public PhysioNet data.

> **Research prototype. Not a clinical or diagnostic system.**

**Project-I (2026), School of Electronics Engineering (SENSE), VIT Chennai**
Team: Tanuj Singh Chauhan, Aryan Duhan, Abhigyan Sharma. Guide: Dr. Suhasini S.

---

## What this project does

Given long-term RR-interval recordings from congestive heart failure (CHF) patients, the pipeline:

1. cleans the RR series and splits it into 5-minute windows,
2. extracts time- and frequency-domain HRV features plus trend features,
3. maps each patient's NYHA class to a binary label (NYHA I/II = 0, NYHA III/III-IV = 1),
4. trains and compares classical baselines, a personalized model, and a GRU temporal model, always splitting **by patient**,
5. adds drift analysis, uncertainty estimates and SHAP explanations.

An ESP32 + MAX30102 live demo is planned as a separate illustration. It is never used for model training or evaluation.

## Data

| Database | Patients | NYHA | Content |
|---|---|---|---|
| CHF2DB (`chf2db`) | 29 (`chf201`-`chf229`) | I: 4, II: 8, III: 17 | Beat-annotation / RR data only, no raw ECG |
| BIDMC-CHF (`chfdb`) | 15 (`chf01`-`chf15`) | recorded as joint "III-IV" | Two-lead ECG, 250 Hz |

Total: 44 patients, 11,109 five-minute windows. Label counts: 32 high-risk (1), 12 low-risk (0).

The data is **not** stored in this repo. Notebook 01 downloads it with `wfdb` into `data/raw/`.

## Features

11 features per window: `mean_rr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf`, `rr_count`, `sdnn_trend`, `rmssd_trend`, `pnn50_trend`.

- RR intervals outside 300-2000 ms are removed, and every removal is logged in `results/*_cleaning_stats.csv`.
- Frequency-domain features: RR series resampled to a uniform 4 Hz grid, detrended, Welch PSD, then LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) band power.
- Personalization features use only past windows (`shift(1)` rolling baselines), so no future information leaks in.

## Validation

- Splits are by `patient_id` (5-fold `GroupKFold`); patient overlap between train and validation was checked and is zero.
- Window-level metrics are computed inside folds; patient-level metrics aggregate each patient's predictions into one out-of-fold prediction (44 patients, 44 predictions).

## Results

### Window-level, 5-fold GroupKFold

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.596 | 0.821 | 0.688 | 0.695 | 0.777 |
| Random Forest | 0.710 | 0.785 | 0.854 | 0.802 | 0.658 |
| XGBoost | 0.681 | 0.797 | 0.780 | 0.767 | 0.662 |

### Patient-level (44 patients)

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Always predict high risk (reference) | 0.727 | 0.727 | 1.000 | 0.842 | 0.500 |
| RF baseline | 0.705 | 0.771 | 0.844 | 0.806 | 0.555 |
| Personalized RF | 0.727 | 0.778 | 0.875 | 0.824 | 0.557 |
| GRU | 0.614 | 0.895 | 0.531 | 0.667 | 0.747 |

**How to read this:** because 73% of patients are high-risk, accuracy and F1 look decent even for a model that ignores the input. AUROC is the more informative column. The GRU is the only model with clear ranking signal; personalization changes RF AUROC by +0.003 and reduces errors from 13 to 12 (one patient), which is not distinguishable from noise at n = 44. A bootstrap on the saved GRU predictions gives an AUROC 95% interval of roughly 0.58-0.90.

Other analyses in notebook 07: patient drift vs. personalization magnitude, RF tree-disagreement uncertainty, SHAP explanations, error analysis, model agreement, threshold sweep (0.20-0.80) and risk stratification. Thresholds and risk groups are exploratory, not clinically validated.

## Known limitations

- **Database confound.** Every `chfdb` patient is NYHA III-IV and therefore label 1, while all label-0 patients come from `chf2db`. Models can partly learn recording source instead of severity. On the saved GRU predictions, AUROC falls from 0.747 (all 44) to 0.691 on `chf2db` only (n = 29). A chf2db-only ablation for all models is planned.
- **Trend features.** `*_trend` is currently one linear slope per patient, copied to every window of that patient, rather than a rolling, time-varying signal. These features dominate the SHAP output, so results that use them (RF, personalized RF, GRU input, SHAP) are provisional until this is replaced with a rolling computation.
- **Small cohort.** 44 patients, so patient-level metrics have wide uncertainty. Many windows per patient are correlated.
- **Label granularity.** Binary risk only; `chfdb` does not separate NYHA III from IV.
- **Not prospectively validated.** Personalization/drift relationships are observational.

## Repository structure

```text
chf-hrv-risk-prediction/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # download + inspect both databases
│   ├── 02_rr_cleaning.ipynb           # RR extraction, plausibility filtering, cleaning logs
│   ├── 03_hrv_features.ipynb          # 5-min windows, HRV + trend features
│   ├── 04_label_mapping.ipynb         # NYHA -> binary risk label, patient metadata
│   ├── 05_baseline_models.ipynb       # LR / RF / XGBoost, GroupKFold, patient-level RF
│   ├── 06_temporal_modeling.ipynb     # GRU (PyTorch), patient-level evaluation
│   └── 07_personalization.ipynb       # baselines, drift, personalized RF, SHAP, analyses
├── results/                           # cleaning statistics and saved predictions
├── requirements.txt
└── README.md
```

Generated at runtime and git-ignored: `data/raw/`, `data/processed/`.

## Setup and running

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order, 01 to 07. Each one reads files produced by the previous one (`data/processed/...`), and notebook 01 downloads the raw data. Notebooks are written to run from inside `notebooks/` (they use `../data/...` paths). Saved outputs are included, so you can read results without rerunning.

## Roadmap

- [x] Data acquisition, RR cleaning, windowing, HRV and trend features, labels
- [x] Baselines with patient-aware cross-validation, patient-level evaluation
- [x] GRU temporal model, personalized RF, drift, uncertainty, SHAP, error and threshold analysis
- [ ] Replace per-patient trend with rolling window-level trend, rerun notebooks 05-07
- [ ] chf2db-only ablation and database-indicator check
- [ ] Patient-level AUROC with confidence intervals for all models
- [ ] End-to-end inference pipeline, dashboard
- [ ] ESP32 + MAX30102 live demo (illustrative only)
- [ ] Final report and presentation

## References

1. Task Force of the ESC and NASPE, "Heart rate variability: standards of measurement, physiological interpretation, and clinical use," *Circulation*, 93(5):1043-1065, 1996.
2. "Detection of congestive heart failure from RR intervals during long-term electrocardiographic recordings," *Heart Rhythm O2*, 2025.
3. F. Noci et al., "Wearable technologies to predict and prevent heart failure hospitalizations: a systematic review," *Eur. Heart J. Digit. Health*, 6(5):868-877, 2025.

Data: PhysioNet CHF RR Interval Database and BIDMC Congestive Heart Failure Database (Goldberger et al., *Circulation*, 101(23):e215-e220, 2000).

## Disclaimer

For academic use only. The models, probabilities, thresholds and risk groups here must not be used for patient care or clinical decisions.

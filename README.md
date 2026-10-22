# CHF-HRV Risk Prediction

Patient-aware estimation of heart-failure severity risk from heart-rate variability (HRV), built on public PhysioNet data, with a Streamlit dashboard and an ESP32 + MAX30102 live-sensing prototype.

> **Research prototype. Not a medical device and not a diagnostic tool.** Model output is a "dataset risk class" learned from 44 patients and must not be used for medical decisions.

**Project-I (2026), School of Electronics Engineering (SENSE), VIT Chennai**
Team: Tanuj Singh Chauhan, Aryan Duhan, Abhigyan Sharma. Guide: Dr. Suhasini S.

---

## 1. What this repository contains

1. **Research notebooks (01-07):** data exploration, RR cleaning, HRV features, label mapping, baseline models, a GRU temporal model, and personalization with drift, uncertainty and SHAP analysis.
2. **Dashboard (`app.py`, `src/`):** a Streamlit app that replays test RR data or reads live records from an ESP32, computes HRV features, and shows a prediction from an exported Random Forest.
3. **Firmware (`firmware/`):** Arduino sketch for ESP32-S3 + MAX30102 that streams pulse-to-pulse intervals over USB serial.

## 2. Data

| Database            | Patients               | NYHA                       | Content                                    |
| ------------------- | ---------------------- | -------------------------- | ------------------------------------------ |
| CHF2DB (`chf2db`)   | 29 (`chf201`-`chf229`) | I: 4, II: 8, III: 17       | Beat-annotation / RR data only, no raw ECG |
| BIDMC-CHF (`chfdb`) | 15 (`chf01`-`chf15`)   | recorded as joint "III-IV" | Two-lead ECG, 250 Hz                       |

Total: 44 patients, 11,109 five-minute windows. Label: NYHA I/II = 0 (12 patients), NYHA III and III-IV = 1 (32 patients).

The PhysioNet data is **not stored in this repo**. Notebook 01 downloads it with `wfdb` into `data/raw/`, and notebooks 02-04 write `data/processed/` (both git-ignored).

**Dashboard test fixtures.** `data/hrv_healthy_profiles.csv` and `data/hrv_high_risk_profiles.csv` hold 10 participant IDs with 300 RR values each. They exist only to exercise CSV replay, feature calculation and the dashboard. They are not labelled clinical data, are never used for training or evaluation, and their scenario names do not imply a diagnosis or a particular model output.

## 3. Pipeline

```text
RR intervals -> cleaning (300-2000 ms) -> 5-min windows -> HRV features
-> baselines (LR / RF / XGBoost) -> patient-level evaluation
-> GRU on window sequences
-> personalization (patient baseline, drift, confidence) -> uncertainty + SHAP
```

- **Features (11 per window):** `mean_rr`, `sdnn`, `rmssd`, `pnn50`, `lf_power`, `hf_power`, `lf_hf`, `rr_count`, `sdnn_trend`, `rmssd_trend`, `pnn50_trend`.
- **Windows:** non-overlapping 5 min; windows with fewer than 30 RR intervals are dropped.
- **Frequency domain:** RR series linearly interpolated to 4 Hz, detrended, Welch PSD; LF 0.04-0.15 Hz, HF 0.15-0.40 Hz.
- **Validation:** all splits are by patient (5-fold `GroupKFold`), with zero patient overlap checked. Patient-level metrics collapse each patient's window predictions into one out-of-fold prediction.
- **Personalization:** uses only past windows (`shift(1)` rolling baselines), so no future windows leak into the baseline.

## 4. Results

### Patient-level (44 patients, one out-of-fold prediction each)

| Model                                | Accuracy | Precision | Recall |    F1 | AUROC |
| ------------------------------------ | -------: | --------: | -----: | ----: | ----: |
| Always predict high risk (reference) |    0.727 |     0.727 |  1.000 | 0.842 | 0.500 |
| RF baseline                          |    0.705 |     0.771 |  0.844 | 0.806 | 0.555 |
| Personalized RF                      |    0.727 |     0.778 |  0.875 | 0.824 | 0.557 |
| GRU                                  |    0.614 |     0.895 |  0.531 | 0.667 | 0.747 |

Because 73% of patients are high-risk, accuracy and F1 look reasonable even for a model that ignores its input, so **AUROC is the column to read**. Only the GRU shows clear ranking ability. Personalization changes one prediction (13 errors to 12) and moves AUROC by 0.003, which is not distinguishable from noise at n = 44. A bootstrap on the saved GRU predictions gives an AUROC 95% interval of roughly 0.58-0.90.

### Window-level, 5-fold GroupKFold (mean)

| Model               | Accuracy | Precision | Recall |    F1 | AUROC |
| ------------------- | -------: | --------: | -----: | ----: | ----: |
| Logistic Regression |    0.596 |     0.821 |  0.688 | 0.695 | 0.777 |
| Random Forest       |    0.710 |     0.785 |  0.854 | 0.802 | 0.658 |
| XGBoost             |    0.681 |     0.797 |  0.780 | 0.767 | 0.662 |

Window-level numbers are not comparable with patient-level numbers; windows from one patient are correlated.

Further analyses in notebook 07: drift vs. personalization magnitude, tree-disagreement uncertainty, SHAP, error analysis, model agreement, a threshold sweep, and exploratory risk bands. Thresholds and risk bands are not clinically validated.

## 5. Dashboard and hardware

```text
MAX30102 -> ESP32-S3 firmware -> USB serial -> src/serial_reader.py
-> last 300 valid RR intervals -> src/hrv_features.py -> src/inference.py -> Streamlit
```

- **Test mode:** replays RR intervals from a CSV in `data/`; no hardware needed.
- **Live mode:** select the serial port and baud (115200 by default). Only rows tagged `REAL`, with finger present, a detected beat and an RR interval of 300-2000 ms, enter the features. `DEMO_SIMULATED` rows are display-only. Close the Arduino Serial Monitor first, since it holds the port.
- **Model file:** the app loads `models/rf_hrv_pipeline.joblib`, a Random Forest fitted on all 44 patients, exported by the last cell of notebook 05. Use the same scikit-learn version as in `requirements.txt` when loading it.
- The sketch is committed with `DEMO_MODE = true`; simulated values are never used for features, but set it to `false` for any real data collection.
- MAX30102 pulse intervals measure **pulse rate variability**, not ECG-derived HRV. The model was trained on ECG-derived RR data from heart-failure patients, so live output is a software demonstration only.

## 6. Repository structure

```text
chf-hrv-risk-prediction/
├── app.py                          # Streamlit dashboard
├── src/
│   ├── hrv_features.py             # feature calculation (mirrors notebook 03)
│   ├── inference.py                # model loading and prediction
│   └── serial_reader.py            # ESP32 serial parsing
├── models/rf_hrv_pipeline.joblib   # exported RF (see section 5)
├── data/
│   ├── hrv_healthy_profiles.csv    # dashboard test fixture
│   └── hrv_high_risk_profiles.csv  # dashboard test fixture
├── notebooks/                      # 01-07 research notebooks
│   └── gru_patient_oof_for_analysis.csv
├── results/                        # cleaning statistics
├── requirements.txt
└── README.md
```

## 7. Setup and running

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Research notebooks:** `jupyter lab`, then run notebooks 01 to 07 in order from inside `notebooks/` (they use `../data/...` paths). Saved outputs are included.
- **Dashboard:** `streamlit run app.py` from the repository root. If the model file is missing, run the last cell of notebook 05 to create it.

## 8. Known limitations

- **Database confound.** Every `chfdb` patient is NYHA III-IV and so carries label 1, while all label-0 patients come from `chf2db`. A model can partly learn recording source instead of severity. On the saved GRU predictions, AUROC falls from 0.747 (all 44) to 0.691 on `chf2db` only (n = 29). A `chf2db`-only ablation for every model is still to do.
- **Trend features.** `*_trend` is one linear slope per patient over the whole recording, repeated on every window of that patient, not a rolling signal. These features dominate SHAP, so the RF, personalized RF, GRU input and SHAP results are provisional until this is replaced.
- **Training/serving mismatch (open).** The dashboard computes trend features from the last up to 30 overlapping live windows (0 for the first window), which is not the quantity the model was trained on. Until the trend features are redefined identically in training and serving, treat live predictions as a software demonstration, not as meaningful risk estimates.
- **Personalization is not a nested estimate.** Its strength was swept (0.00-0.30) on the same out-of-fold predictions that are reported.
- **Small cohort and coarse label.** 44 patients; binary risk only, because `chfdb` does not separate NYHA III from IV.
- **Not clinically validated.** No prospective or external validation has been done.

## References

1. Task Force of the ESC and NASPE, "Heart rate variability: standards of measurement, physiological interpretation, and clinical use," _Circulation_, 93(5):1043-1065, 1996. https://doi.org/10.1161/01.CIR.93.5.1043
2. F. Shaffer and J. P. Ginsberg, "An overview of heart rate variability metrics and norms," _Front. Public Health_, 5:258, 2017. https://doi.org/10.3389/fpubh.2017.00258
3. T. Pukkila, M. Molkkari, J. Hernesniemi, M. Kanniainen, and E. Räsänen, "Detection of congestive heart failure from RR intervals during long-term electrocardiographic recordings," _Heart Rhythm O2_, 2025. https://doi.org/10.1016/j.hroo.2025.01.014
4. Analog Devices, _MAX30102 High-Sensitivity Pulse Oximeter and Heart-Rate Sensor_ datasheet. https://www.analog.com/en/products/max30102.html
5. PhysioNet: Congestive Heart Failure RR Interval Database (`chf2db`) and BIDMC Congestive Heart Failure Database (`chfdb`). https://physionet.org

## 11. Disclaimer

For academic use only. The models, probabilities, thresholds and risk groups here must not be used for patient care or clinical decisions.

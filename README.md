# HRV-Based Risk Prediction and Monitoring System

## 1. Project Overview

This project explores heart-rate variability (HRV) features derived from beat-to-beat RR intervals and uses a trained machine-learning model to produce a risk-related prediction. The project includes a notebook-based development workflow, an ESP32 sensor data-collection path, and a Streamlit dashboard for viewing measurements, calculated features, and model output.

The system is intended as a **research and prototype project**, not a clinically validated diagnostic tool. Model predictions should not be used to make medical decisions.

## 2. What Has Been Implemented

- **Dataset exploration** in Jupyter notebooks.
- **RR-interval cleaning and preparation** for downstream analysis.
- **HRV feature extraction** from RR-interval windows.
- **Label mapping** for the target classes used in modelling.
- **Baseline model experiments** and evaluation.
- **Temporal modelling experiments** to investigate sequential/window-based information.
- **Personalization experiments** exploring individual-level handling.
- **ESP32 serial data path** for receiving sensor records in the dashboard.
- **Streamlit dashboard** with Test and Live ESP32 modes, charts, HRV feature display, prediction output, sample history/logging, and export-related functionality.
- **CSV replay in Test mode** using the two included test profiles. These files are generated test fixtures for checking the dashboard pipeline; they are **not real participant recordings**.

## 3. Repository Structure

The project is organized around the following files and folders. Some generated files (such as the trained model) may be present locally but are not included in every copy of the repository.

```text
chf-hrv-risk-prediction/
├── app.py                         # Streamlit dashboard entry point
├── data/
│   ├── hrv_healthy_profiles.csv    # Generated healthy-like test scenario
│   └── hrv_high_risk_profiles.csv  # Generated high-risk-like test scenario
├── models/
│   └── rf_hrv_pipeline.joblib      # Trained model bundle expected by inference code
├── src/
│   ├── serial_reader.py            # ESP32 serial connection and record parsing
│   ├── hrv_features.py              # HRV feature calculation and feature order
│   └── inference.py                 # Model loading and prediction helpers
├── 01_dataset_exploration.ipynb     # Dataset exploration
├── 02_rr_cleaning.ipynb             # RR-interval cleaning
├── 03_hrv_features.ipynb            # HRV feature engineering
├── 04_label_mapping.ipynb           # Target-label mapping
├── 05_baseline_models.ipynb         # Baseline model experiments
├── 06_temporal_modeling.ipynb       # Temporal modelling experiments
├── 07_personalization.ipynb          # Personalization experiments
├── requirements.txt                 # Python dependencies
└── README.md
```

## 4. Data and Test Scenarios

The dashboard's Test mode replays RR-interval values from the CSV files in `data/`:

- `hrv_healthy_profiles.csv`
- `hrv_high_risk_profiles.csv`

These are **software test fixtures**, not verified healthy/high-risk clinical examples and are human recordings collected from 10 individuals. They are useful for checking CSV loading, replay, feature calculation, and dashboard rendering. A prediction shown for a scenario is determined by the loaded model and its learned decision boundary; the scenario name does not guarantee a particular prediction.

## 5. Hardware and Live Data Flow

The intended live collection path uses an **ESP32/ESP32-S3 development board** with a **MAX30102 optical pulse sensor module**. The microcontroller reads sensor data and sends serial records to the computer, where the dashboard's serial reader parses incoming records.

High-level flow:

```text
MAX30102 sensor
      ↓
ESP32 / ESP32-S3 firmware
      ↓ USB serial
src/serial_reader.py
      ↓ valid beat-to-beat RR intervals
RR-interval window
      ↓
src/hrv_features.py
      ↓
src/inference.py + trained model bundle
      ↓
Streamlit dashboard: features, plots, and prediction
```

The sensor provides optical pulse signals; the software derives or receives beat timing and RR intervals through the device firmware/data protocol. Reliable live HRV depends on stable sensor contact, valid beat detection, and appropriate signal quality. The dashboard's live prediction may take longer than Test mode because it must accumulate enough valid RR intervals.

## 6. HRV and Machine-Learning Workflow

The notebook sequence documents the analysis workflow:

1. Explore the available dataset and its fields.
2. Clean and validate RR-interval data.
3. Calculate HRV features from RR-interval windows.
4. Map source labels to the model target classes.
5. Train and evaluate baseline models.
6. Experiment with temporal modelling.
7. Explore personalization approaches.

The application uses a shared feature definition so inference follows the feature order expected by the trained model. When retraining or changing feature calculations, ensure the model bundle and inference code remain compatible.

## 7. Setup and Run

Use a Python environment supported by the dependencies in the project's `requirements.txt`.

### Windows (PowerShell)

```powershell
# From the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

If your checkout does not contain `requirements.txt`, install the dependencies used by the project (including Streamlit, pandas, NumPy, plotting libraries, scikit-learn, joblib, and pyserial as required by the code) or restore the project's dependency file.

### Required local files

Before running the dashboard, verify that:

- `app.py` is at the repository root.
- The modules imported by `app.py` exist under `src/`.
- The model bundle is at `models/rf_hrv_pipeline.joblib`, or update the model path in the inference/application code to match the actual file.
- The two CSV test files are under `data/` if you want to use CSV replay in Test mode.
- For Live ESP32 mode, the board is connected, its serial port is selected in the dashboard, and no other application (for example, Arduino Serial Monitor) is holding that port open.

## 8. Operating Modes

### Test mode

Select one of the CSV test scenarios. The dashboard replays its RR intervals and runs the feature/prediction pipeline as the required data window becomes available. This mode is intended for software and interface testing without a live sensor.

### Live ESP32 mode

Select the serial port and baud rate matching the firmware. The dashboard reads incoming serial records and processes valid RR intervals according to the live-mode window and validation rules implemented in the current application. If no data appears, check USB connection, board firmware output, the selected COM port, baud rate, sensor contact, and whether another program has opened the port.

## 9. Important Limitations

- This is a prototype and research workflow, not a medical device.
- Generated CSV scenarios do not establish clinical validity or represent real patient measurements.
- Sensor motion, poor contact, missed beats, serial formatting, and signal artefacts can affect RR intervals and derived HRV features.
- A model prediction is only as reliable as its training data, labels, validation method, and deployment compatibility.
- Results from experiments should be reported with the dataset, split strategy, metrics, and limitations clearly stated. Avoid describing model output as a diagnosis.
- Do not claim clinical performance, real-world accuracy, or patient benefit unless supported by appropriate evaluation evidence.

## 10. References

The following references provide background for HRV methodology, the sensor, and the software stack. They are background resources; they do not by themselves validate this project's model.

1. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. “Heart rate variability: Standards of measurement, physiological interpretation and clinical use.” _Circulation_, 93(5), 1043–1065, 1996. https://doi.org/10.1161/01.CIR.93.5.1043
2. Shaffer, F., and Ginsberg, J. P. “An Overview of Heart Rate Variability Metrics and Norms.” _Frontiers in Public Health_, 5, 258, 2017. https://doi.org/10.3389/fpubh.2017.00258
3. Analog Devices. _MAX30102 High-Sensitivity Pulse Oximeter and Heart-Rate Sensor for Wearable Health_. Product information and datasheet: https://www.analog.com/en/products/max30102.html
4. Espressif Systems. _ESP32-S3 Documentation_. https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/
5. Streamlit. _Streamlit Documentation_. https://docs.streamlit.io/
6. scikit-learn. _User Guide_. https://scikit-learn.org/stable/user_guide.html
7. pySerial. _Documentation_. https://pyserial.readthedocs.io/

## 11. Reproducibility Notes

For reproducible experiments, record the dataset source/version, label definitions, preprocessing rules, feature list and order, model parameters, train/validation/test split strategy, evaluation metrics, and software versions. Keep generated test fixtures clearly distinguished from collected sensor data and from any labelled research dataset.

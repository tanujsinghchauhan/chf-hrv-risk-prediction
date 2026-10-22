from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.serial_reader import ESP32Serial
from src.hrv_features import calculate_hrv_features, FEATURE_COLUMNS
from src.inference import load_model_bundle, predict

st.set_page_config(page_title="HRV Monitoring", page_icon="💓", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HEALTHY_CSV = DATA_DIR / "hrv_healthy_profiles.csv"
HIGH_RISK_CSV = DATA_DIR / "hrv_high_risk_profiles.csv"

DEFAULTS = {
    "samples": [], "rr_values": [], "feature_history": [], "trend_windows": [],
    "latest_features": None, "latest_prediction": None, "prediction_source": None,
    "reader": None, "reader_port": None, "reader_baud": None, "test_tick": 0,
    "test_rr": [], "test_csv_df": None, "test_csv_index": 0, "test_group": None, "session_start": datetime.now().isoformat(timespec="seconds")
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

def close_reader():
    reader = st.session_state.get("reader")
    if reader is not None:
        try:
            reader.close()
        except Exception:
            pass
    st.session_state.reader = None
    st.session_state.reader_port = None
    st.session_state.reader_baud = None

def clear_session():
    close_reader()
    for key in ("samples", "rr_values", "feature_history", "trend_windows", "test_rr"):
        st.session_state[key] = []
    for key in ("latest_features", "latest_prediction", "prediction_source"):
        st.session_state[key] = None
    st.session_state.test_tick = 0
    st.session_state.test_csv_df = None
    st.session_state.test_csv_index = 0
    st.session_state.test_group = None
    st.session_state.session_start = datetime.now().isoformat(timespec="seconds")

def add_sample(row):
    st.session_state.samples.append(row)
    st.session_state.samples = st.session_state.samples[-5000:]

def process_feature_window(rr_values, source):
    features = calculate_hrv_features(rr_values, st.session_state.trend_windows)
    st.session_state.latest_features = features
    st.session_state.trend_windows.append(features.copy())
    st.session_state.trend_windows = st.session_state.trend_windows[-30:]
    st.session_state.feature_history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        **features
    })
    st.session_state.feature_history = st.session_state.feature_history[-300:]
    bundle, error = load_model_bundle()
    if bundle is None:
        st.session_state.latest_prediction = {"error": error}
        st.session_state.prediction_source = "MODEL_UNAVAILABLE"
    else:
        try:
            st.session_state.latest_prediction = predict(bundle, features)
            st.session_state.prediction_source = f"MODEL_ON_{source}_FEATURES"
        except Exception as exc:
            st.session_state.latest_prediction = {"error": str(exc)}
            st.session_state.prediction_source = "PREDICTION_ERROR"

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Monitoring mode", ["Test mode", "Live ESP32 sensor"], index=0)
    if mode == "Test mode":
        scenario = st.selectbox(
            "Test dataset",
            ["High-risk scenario", "Healthy-like scenario"],
            index=0
        )
        csv_path = HIGH_RISK_CSV if scenario == "High-risk scenario" else HEALTHY_CSV
        st.caption("Test mode replays RR intervals from the selected CSV. These files are from collected data of 10 healthy human recordings.")
    else:
        scenario = None
        csv_path = None
        port = st.text_input("Serial port", value=os.getenv("HRV_SERIAL_PORT", "COM7"))
        baud = st.selectbox("Baud rate", [115200, 9600], index=0)
    if mode == "Test mode":
        port, baud = "TEST", 115200
    refresh_ms = st.slider("Refresh interval (ms)", 500, 3000, 1000, 250)
    if st.button("Clear session data", use_container_width=True):
        clear_session()
        st.rerun()

st_autorefresh(interval=refresh_ms, key="hrv_auto_refresh")
is_test = mode == "Test mode"

if is_test:
    close_reader()

    if st.session_state.test_group != scenario:
        st.session_state.test_group = scenario
        st.session_state.test_csv_df = None
        st.session_state.test_csv_index = 0
        st.session_state.test_rr = []
        st.session_state.rr_values = []
        st.session_state.samples = []
        st.session_state.feature_history = []
        st.session_state.trend_windows = []
        st.session_state.latest_features = None
        st.session_state.latest_prediction = None
        st.session_state.prediction_source = None

    if st.session_state.test_csv_df is None:
        try:
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {csv_path}. Place the CSV in the project's data folder."
                )
            df = pd.read_csv(csv_path)
            required = {"participant_id", "sample_index", "rr_ms", "bpm_derived"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
            df["rr_ms"] = pd.to_numeric(df["rr_ms"], errors="coerce")
            df = df.dropna(subset=["rr_ms"])
            df = df[df["rr_ms"].between(300, 2000)].reset_index(drop=True)
            if df.empty:
                raise ValueError("The selected CSV contains no valid RR intervals.")
            st.session_state.test_csv_df = df
        except Exception as exc:
            st.error(f"Could not load the selected test dataset: {exc}")
            st.stop()

    df = st.session_state.test_csv_df
    idx = st.session_state.test_csv_index
    if idx >= len(df):
        idx = 0
        st.session_state.test_csv_index = 0
        st.session_state.test_rr = []
        st.session_state.rr_values = []
        st.session_state.feature_history = []
        st.session_state.trend_windows = []
        st.session_state.latest_features = None
        st.session_state.latest_prediction = None
        st.session_state.prediction_source = None

    row = df.iloc[idx]
    st.session_state.test_csv_index = idx + 1
    st.session_state.test_tick += 1
    tick = st.session_state.test_tick
    rr_now = float(row["rr_ms"])
    now = datetime.now()
    elapsed_ms = int(row["sample_index"]) * refresh_ms

    # IR/RED traces are display-only; RR intervals are read from the CSV.
    rng = np.random.default_rng(62000 + tick)
    add_sample({
        "timestamp": now.isoformat(timespec="milliseconds"),
        "time_display": now.strftime("%H:%M:%S"),
        "time_ms": elapsed_ms,
        "IR": int(49200 + 700*np.sin(tick*0.28) + rng.normal(0, 90)),
        "RED": int(43500 + 420*np.sin(tick*0.25 + 0.5) + rng.normal(0, 75)),
        "Finger": 1,
        "Beat": 1,
        "BPM": round(60000.0 / rr_now, 1),
        "RR_ms": round(rr_now, 1),
        "Source": "TEST",
        "Participant_id": str(row["participant_id"]),
        "Scenario": str(row.get("scenario", scenario))
    })

    st.session_state.test_rr.append(rr_now)
    st.session_state.test_rr = st.session_state.test_rr[-300:]
    valid_count = len(st.session_state.test_rr)

    if valid_count >= 30 and (valid_count == 30 or valid_count % 5 == 0):
        try:
            process_feature_window(st.session_state.test_rr, "TEST")
        except Exception as exc:
            st.session_state.latest_prediction = {"error": f"Feature calculation failed: {exc}"}
            st.session_state.prediction_source = "FEATURE_ERROR"
else:
    if (st.session_state.reader is None
        or st.session_state.reader_port != port
        or st.session_state.reader_baud != baud):
        close_reader()
        st.session_state.reader = ESP32Serial(port, baud)
        st.session_state.reader_port = port
        st.session_state.reader_baud = baud

    reader = st.session_state.reader
    if reader.connect():
        for row in reader.read_available():
            now = datetime.now()
            add_sample({
                "timestamp": now.isoformat(timespec="milliseconds"),
                "time_display": now.strftime("%H:%M:%S"),
                "time_ms": row.time_ms, "IR": row.ir, "RED": row.red,
                "Finger": int(row.finger), "Beat": int(row.beat),
                "BPM": row.bpm, "RR_ms": row.rr_ms, "Source": row.source
            })
            if row.source == "REAL" and row.finger and row.beat and 300 <= row.rr_ms <= 2000:
                st.session_state.rr_values.append(float(row.rr_ms))
                st.session_state.rr_values = st.session_state.rr_values[-300:]
                # Require 300 valid RR intervals before live inference.
                if len(st.session_state.rr_values) >= 300 and len(st.session_state.samples) % 5 == 0:
                    try:
                        process_feature_window(st.session_state.rr_values, "REAL")
                    except Exception as exc:
                        st.session_state.latest_prediction = {"error": f"Feature calculation failed: {exc}"}
                        st.session_state.prediction_source = "FEATURE_ERROR"
    else:
        st.error(f"Could not open {port}: {reader.last_error or 'Unknown serial error'}")
        st.info("Check Device Manager and close Arduino Serial Monitor. Only one program can use the COM port.")

st.title("Live HRV Monitoring & Research Classification")
st.caption("ESP32-S3 + MAX30102 · Test mode or live USB serial · Research prototype, not a diagnostic device")
if is_test:
    st.warning("TEST MODE — the selected CSV is replayed through the HRV feature and model pipeline. Test records are from 10 human recordings.")
elif st.session_state.samples:
    last_source = str(st.session_state.samples[-1]["Source"]).upper()
    if last_source != "REAL":
        st.warning(f"Live HRV inference accepts only genuine sensor rows tagged REAL.")

samples = pd.DataFrame(st.session_state.samples)
history = pd.DataFrame(st.session_state.feature_history)

if not samples.empty:
    latest = samples.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Finger status", "Detected" if int(latest["Finger"]) else "Not detected")
    c2.metric("IR signal", f"{int(latest['IR']):,}")
    c3.metric("RED signal", f"{int(latest['RED']):,}")
    c4.metric("Heart rate", f"{float(latest['BPM']):.1f} BPM" if float(latest["BPM"]) > 0 else "—")
    c5.metric("Data source", "TEST" if is_test else str("LIVE"))

    st.subheader("Live signal charts")
    tab1, tab2, tab3 = st.tabs(["IR and RED", "Heart rate", "RR intervals"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=samples["time_display"], y=samples["IR"], mode="lines", name="IR"))
        fig.add_trace(go.Scatter(x=samples["time_display"], y=samples["RED"], mode="lines", name="RED"))
        fig.update_layout(height=300, margin=dict(l=5,r=5,t=15,b=5), xaxis_title="Time", yaxis_title="Raw signal")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = go.Figure(go.Scatter(x=samples["time_display"], y=samples["BPM"], mode="lines", name="BPM"))
        fig.update_layout(height=300, margin=dict(l=5,r=5,t=15,b=5), xaxis_title="Time", yaxis_title="Beats per minute")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = go.Figure(go.Scatter(x=samples["time_display"], y=samples["RR_ms"], mode="lines", name="RR"))
        fig.update_layout(height=300, margin=dict(l=5,r=5,t=15,b=5), xaxis_title="Time", yaxis_title="RR interval (ms)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("HRV metrics")
    feat = st.session_state.latest_features
    if feat:
        metric_defs = [
            ("Mean RR", "mean_rr", "ms"), ("SDNN", "sdnn", "ms"),
            ("RMSSD", "rmssd", "ms"), ("pNN50", "pnn50", "%"),
            ("LF power", "lf_power", ""), ("HF power", "hf_power", ""),
            ("LF/HF", "lf_hf", ""), ("Valid RR count", "rr_count", "")
        ]
        cols = st.columns(4)
        for i, (label, key, unit) in enumerate(metric_defs):
            val = feat.get(key, np.nan)
            display = f"{val:.3f} {unit}".strip() if isinstance(val, (float, int, np.number)) and np.isfinite(val) else "Unavailable"
            cols[i % 4].metric(label, display)
        st.caption("Test-mode metrics are calculated from RR intervals; live metrics use valid sensor intervals tagged REAL.")
    else:
        st.info("Waiting for enough valid RR intervals.")

    st.subheader("HRV matrix — latest feature vector")
    if feat:
        matrix = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Value": [feat.get(col, np.nan) for col in FEATURE_COLUMNS]
        })
        matrix["Availability"] = matrix["Value"].apply(
            lambda x: "Available" if pd.notna(x) and np.isfinite(float(x)) else "Unavailable"
        )
        st.dataframe(matrix, use_container_width=True, hide_index=True)
    else:
        st.info("The HRV matrix will populate when the feature window is ready.")

    st.subheader("Model classification")
    prediction = st.session_state.latest_prediction
    if is_test:
        st.info("TEST MODEL INFERENCE: any result shown is the trained model output for the selected CSV input, if a compatible model artifact is installed.")
    if prediction and prediction.get("error"):
        st.warning(prediction["error"])
        st.caption("To enable model inference, export the fitted model from notebook 05 to models/rf_hrv_pipeline.joblib.")
    elif prediction:
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted dataset class", prediction["label_text"])
        probability = prediction.get("probability_class_1")
        p2.metric("Class 1 model probability", f"{probability:.1%}" if probability is not None else "Not provided")
        p3.metric("Inference input", "Test features" if is_test else "Real sensor features")
        st.caption("A prediction on inputs is for software integration testing only. It is not a diagnosis or a validated clinical risk score.")
    else:
        st.info("No prediction yet. A compatible model file and a valid feature vector are required.")

    st.subheader("HRV feature history")
    if not history.empty:
        shown_cols = [c for c in [
            "timestamp", "source", "mean_rr", "sdnn", "rmssd", "pnn50",
            "lf_power", "hf_power", "lf_hf", "rr_count",
            "sdnn_trend", "rmssd_trend", "pnn50_trend"
        ] if c in history.columns]
        st.dataframe(history[shown_cols].tail(100), use_container_width=True, hide_index=True)
        fig = go.Figure()
        for metric in ["sdnn", "rmssd", "pnn50"]:
            if metric in history:
                fig.add_trace(go.Scatter(x=history["timestamp"], y=history[metric], mode="lines", name=metric.upper()))
        fig.update_layout(height=320, margin=dict(l=5,r=5,t=15,b=5), xaxis_title="Feature window", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("HRV feature history will appear as feature vectors are calculated.")

    st.subheader("Sample log")
    st.dataframe(samples.tail(100), use_container_width=True, hide_index=True)
    x1, x2 = st.columns(2)
    x1.download_button(
        "Export sample CSV", samples.to_csv(index=False).encode("utf-8"),
        file_name=f"hrv_samples_{'test' if is_test else 'live'}.csv",
        mime="text/csv", use_container_width=True
    )
    if not history.empty:
        history_csv = history.to_csv(index=False).encode("utf-8")
    else:
        history_csv = pd.DataFrame(columns=["timestamp", "source"] + FEATURE_COLUMNS).to_csv(index=False).encode("utf-8")
    x2.download_button(
        "Export HRV feature history CSV", history_csv,
        file_name=f"hrv_feature_history_{'test' if is_test else 'live'}.csv",
        mime="text/csv", use_container_width=True
    )
else:
    st.info("Select Test mode to preview live-updating signals, HRV metrics, the feature matrix, feature history and model integration without connecting hardware.")

st.divider()
st.caption("Research prototype only — not a diagnostic device. Predictions on inputs are for software integration testing only.")

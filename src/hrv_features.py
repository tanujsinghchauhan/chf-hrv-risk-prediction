from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

# Ordered exactly as the baseline feature_columns in notebook 05.
FEATURE_COLUMNS = [
    "mean_rr", "sdnn", "rmssd", "pnn50", "lf_power", "hf_power",
    "lf_hf", "rr_count", "sdnn_trend", "rmssd_trend", "pnn50_trend"
]

def calculate_hrv_features(rr_ms, prior_windows=None):
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr) & (rr >= 300) & (rr <= 2000)]
    if len(rr) < 30:
        raise ValueError("At least 30 valid RR intervals are required.")
    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.mean(np.abs(diffs) > 50) * 100)

    # Frequency-domain procedure mirrors notebook 03: 4 Hz interpolation + Welch.
    time_sec = np.cumsum(rr / 1000.0)
    time_sec = time_sec - time_sec[0]
    uniform_time = np.arange(time_sec[0], time_sec[-1], 0.25)
    rr_interp = np.interp(uniform_time, time_sec, rr)
    rr_interp = signal.detrend(rr_interp)
    frequencies, psd = signal.welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))
    lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_mask = (frequencies >= 0.15) & (frequencies < 0.40)
    lf_power = float(trapezoid(psd[lf_mask], frequencies[lf_mask])) if lf_mask.sum() >= 2 else np.nan
    hf_power = float(trapezoid(psd[hf_mask], frequencies[hf_mask])) if hf_mask.sum() >= 2 else np.nan
    lf_hf = float(lf_power / hf_power) if np.isfinite(lf_power) and np.isfinite(hf_power) and hf_power > 0 else np.nan

    history = list(prior_windows or [])
    trends = {}
    for key, value in (("sdnn", sdnn), ("rmssd", rmssd), ("pnn50", pnn50)):
        series = [float(row[key]) for row in history if key in row and np.isfinite(row[key])]
        series.append(float(value))
        if len(series) >= 2:
            trends[f"{key}_trend"] = float(np.polyfit(np.arange(len(series), dtype=float), np.asarray(series), 1)[0])
        else:
            trends[f"{key}_trend"] = 0.0

    return {
        "mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50,
        "lf_power": lf_power, "hf_power": hf_power, "lf_hf": lf_hf,
        "rr_count": int(len(rr)), **trends
    }

def model_feature_frame(features, feature_columns=None):
    columns = list(feature_columns or FEATURE_COLUMNS)
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    frame = pd.DataFrame([[features[c] for c in columns]], columns=columns)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Feature vector contains undefined NaN/Inf values.")
    return frame

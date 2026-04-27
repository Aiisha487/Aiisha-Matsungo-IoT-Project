"""
acoustic_inference.py
EECE 5155 – Acoustic Monitoring System for Quiet Zones
Batch inference loop: simulates reading from InfluxDB every 5 minutes,
running the trained compliance classifier, and logging predictions.
Aiisha Matsungo | Spring 2026

Integration pattern:
    InfluxDB (buffered readings)
        └─► query_recent_window()
                └─► engineer_features()
                        └─► model.predict()
                                └─► inference_log.csv

This closes the pipeline loop described in B2: the same RandomForest
classifier trained on acoustic features is applied in a batch inference
cycle, replacing the conceptual HAR-to-acoustic bridge with actual
acoustic inference on project data.
"""

import json
import os
import time
from datetime import datetime

import joblib
import pandas as pd

from acoustic_features import engineer_features

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH      = "analysis/acoustic_model.joblib"
SOURCE_CSV      = "data/one_hour_acoustic.csv"          # stand-in for InfluxDB
INFERENCE_LOG   = "analysis/inference_log.csv"
BATCH_MINUTES   = 5      # inference window size
DRY_RUN_BATCHES = 12     # simulate 12 × 5-min windows (= 1 hour) then exit
SLEEP_SECONDS   = 0      # set > 0 for real-time simulation; 0 for dry run


# ── InfluxDB query stub ───────────────────────────────────────────────────────
def query_recent_window(full_df: pd.DataFrame,
                         window_start: pd.Timestamp,
                         window_end:   pd.Timestamp) -> pd.DataFrame:
    """
    In production this would execute a Flux query against InfluxDB:

        from(bucket: "iot-sensors")
          |> range(start: window_start, stop: window_end)
          |> filter(fn: (r) => r["_measurement"] == "sound_level")
          |> pivot(rowKey:["_time","zone_id"], columnKey:["_field"], valueColumn:["_value"])

    For the dry-run simulation we slice the CSV by timestamp window.
    """
    mask = (full_df["timestamp"] >= window_start) & (full_df["timestamp"] < window_end)
    return full_df.loc[mask].copy()


def run_inference(model_bundle: dict, window_df: pd.DataFrame) -> pd.DataFrame:
    """Feature-engineer the window and return a predictions DataFrame."""
    model       = model_bundle["model"]
    le          = model_bundle["label_encoder"]
    feature_cols = model_bundle["feature_cols"]

    features    = engineer_features(window_df)
    X           = features[feature_cols]
    y_pred_enc  = model.predict(X)
    y_pred      = le.inverse_transform(y_pred_enc)
    proba       = model.predict_proba(X).max(axis=1)

    out = features[["timestamp", "zone_id", "dba_spl", "status"]].copy()
    out["predicted_status"]    = y_pred
    out["confidence"]          = proba.round(4)
    out["correct"]             = (out["status"] == out["predicted_status"])
    out["inference_timestamp"] = datetime.utcnow().isoformat()
    return out


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run acoustic_train.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    print(f"[acoustic_inference] Model loaded from {MODEL_PATH}")
    print(f"[acoustic_inference] Classes: {list(bundle['label_encoder'].classes_)}")

    # ── Load source data (CSV stands in for live InfluxDB) ────────────────────
    full_df = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)

    t_start = full_df["timestamp"].min()
    t_end   = full_df["timestamp"].max()
    window  = pd.Timedelta(minutes=BATCH_MINUTES)

    print(f"[acoustic_inference] Data window : {t_start} → {t_end}")
    print(f"[acoustic_inference] Batch size  : {BATCH_MINUTES} minutes")
    print(f"[acoustic_inference] Dry-run     : {DRY_RUN_BATCHES} batches\n")

    all_results = []
    current     = t_start

    for batch_num in range(1, DRY_RUN_BATCHES + 1):
        w_end  = current + window
        window_df = query_recent_window(full_df, current, w_end)

        if window_df.empty:
            print(f"  Batch {batch_num:02d}: no data in window, skipping.")
            current = w_end
            continue

        results = run_inference(bundle, window_df)
        all_results.append(results)

        n_readings  = len(results)
        n_correct   = results["correct"].sum()
        n_violation = (results["predicted_status"] == "violation").sum()
        batch_acc   = n_correct / n_readings if n_readings else 0

        print(f"  Batch {batch_num:02d} | {current.strftime('%H:%M')}→{w_end.strftime('%H:%M')} "
              f"| readings={n_readings:3d} | violations={n_violation} "
              f"| accuracy={batch_acc:.3f}")

        current = w_end

        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    # ── Persist inference log ─────────────────────────────────────────────────
    if all_results:
        os.makedirs("analysis", exist_ok=True)
        log_df = pd.concat(all_results, ignore_index=True)
        log_df.to_csv(INFERENCE_LOG, index=False)

        overall_acc = log_df["correct"].mean()
        print(f"\n[acoustic_inference] Overall accuracy : {overall_acc:.4f}")
        print(f"[acoustic_inference] Inference log    → {INFERENCE_LOG}")
        print(f"[acoustic_inference] Total readings   : {len(log_df)}")

        # ── Per-zone summary ──────────────────────────────────────────────────
        summary = (
            log_df.groupby("zone_id")
            .agg(
                readings=("dba_spl", "count"),
                mean_dba=("dba_spl", "mean"),
                violations_predicted=("predicted_status",
                                      lambda x: (x == "violation").sum()),
                accuracy=("correct", "mean"),
            )
            .round(3)
        )
        print("\n[acoustic_inference] Per-zone summary:")
        print(summary.to_string())


if __name__ == "__main__":
    main()

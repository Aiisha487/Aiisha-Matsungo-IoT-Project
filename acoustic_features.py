"""
acoustic_features.py
EECE 5155 – Acoustic Monitoring System for Quiet Zones
Feature engineering from raw dBA sensor readings.
Aiisha Matsungo | Spring 2026
"""

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = "data/one_hour_acoustic.csv"
OUTPUT_CSV = "data/acoustic_features.csv"
WINDOW     = 5   # rolling window size (readings)
THRESHOLD  = 45  # dBA compliance threshold (matches Grafana alert rule)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-reading feature set from raw acoustic measurements.
    All rolling features are computed within each zone to avoid
    cross-zone leakage — zones are physically separate spaces.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    feature_frames = []

    for zone, grp in df.groupby("zone_id"):
        grp = grp.copy().reset_index(drop=True)
        spl = grp["dba_spl"]

        # ── Rolling statistics (window=5 ≈ 25 s at 30 s collection interval) ─
        grp["spl_roll_mean"]  = spl.rolling(WINDOW, min_periods=1).mean()
        grp["spl_roll_std"]   = spl.rolling(WINDOW, min_periods=1).std().fillna(0)
        grp["spl_roll_min"]   = spl.rolling(WINDOW, min_periods=1).min()
        grp["spl_roll_max"]   = spl.rolling(WINDOW, min_periods=1).max()
        grp["spl_roll_range"] = grp["spl_roll_max"] - grp["spl_roll_min"]

        # ── Rate of change ───────────────────────────────────────────────────
        grp["spl_delta"]     = spl.diff().fillna(0)
        grp["spl_delta_abs"] = grp["spl_delta"].abs()

        # ── Threshold proximity ──────────────────────────────────────────────
        grp["spl_above_thresh"] = (spl > THRESHOLD).astype(int)
        grp["spl_margin"]       = spl - THRESHOLD        # negative = compliant
        grp["spl_breach_rate"]  = (
            grp["spl_above_thresh"]
            .rolling(WINDOW, min_periods=1)
            .mean()
        )

        # ── Time-of-day (captures diurnal noise patterns seen in analysis) ───
        grp["hour"]   = grp["timestamp"].dt.hour
        grp["minute"] = grp["timestamp"].dt.minute

        # ── Zone one-hot encoding (snell_group_room = reference category) ────
        grp["zone_reading_hall"] = int(zone == "snell_reading_hall")
        grp["zone_silent_study"] = int(zone == "snell_silent_study")

        feature_frames.append(grp)

    result = (
        pd.concat(feature_frames)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"[acoustic_features] Input rows   : {len(df)}")
    print(f"[acoustic_features] Output cols  : {result.shape[1]}")
    print(f"[acoustic_features] Class balance:\n{result['status'].value_counts()}\n")
    return result


def main():
    df = pd.read_csv(INPUT_CSV)
    features = engineer_features(df)
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"[acoustic_features] Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

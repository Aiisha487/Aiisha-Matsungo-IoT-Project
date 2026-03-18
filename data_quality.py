#!/usr/bin/env python3
"""
Data Quality Pipeline: runs on the CSV dataset and produces a quality report.
Implements: outlier detection, missing data analysis, dedup, stuck sensor check.

Usage:
    python data_quality.py
    python data_quality.py --csv IOT-temp.csv
"""

import argparse
import sys

import pandas as pd

CSV_FILE = "IOT-temp.csv"


def load_data(csv_path):
    """Load and do basic cleanup of the CSV."""
    df = pd.read_csv(csv_path)
    print(f"[quality] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"[quality] Columns: {list(df.columns)}")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Parse temperature as float
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

    # Build a logical device_id from room + location
    room_col = "room_id/id" if "room_id/id" in df.columns else "room_id/area"
    if room_col in df.columns and "out/in" in df.columns:
        df["device"] = df[room_col].astype(str).str.strip() + "_" + df["out/in"].astype(str).str.strip()
    else:
        df["device"] = df["id"].astype(str).str.strip()

    print(f"[quality] Logical devices (room+location): {df['device'].nunique()}")
    print(f"[quality] Devices: {sorted(df['device'].unique())}")

    # Try to parse timestamps
    if "noted_date" in df.columns:
        df["noted_date"] = pd.to_datetime(df["noted_date"],
                                           format="mixed",
                                           dayfirst=True,
                                           errors="coerce")

    return df


def check_outliers(df, sigma=3):
    """Flag readings outside mean ± sigma*std per device."""
    df["z_score"] = df.groupby("device")["temp"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    outliers = df[df["z_score"].abs() > sigma]
    print(f"\n[quality] OUTLIER DETECTION (>{sigma}σ per device)")
    print(f"  Found {len(outliers)} outliers out of {len(df)} readings "
          f"({100*len(outliers)/len(df):.2f}%)")

    if len(outliers) > 0:
        print(f"  Outlier temp range: [{outliers['temp'].min():.1f}, "
              f"{outliers['temp'].max():.1f}]")
        print(f"  Top devices with outliers:")
        top = outliers.groupby("device").size().sort_values(ascending=False).head(5)
        for dev, cnt in top.items():
            print(f"    {dev}: {cnt} outliers")

    return outliers


def check_missing(df):
    """Analyze missing data per device."""
    print(f"\n[quality] MISSING DATA ANALYSIS")
    null_temp = df["temp"].isna().sum()
    null_date = df["noted_date"].isna().sum() if "noted_date" in df.columns else 0
    print(f"  Null temperatures: {null_temp}")
    print(f"  Null timestamps: {null_date}")

    # Per-device reading count
    counts = df.groupby("device").size()
    print(f"  Readings per device:")
    for dev, cnt in counts.sort_values(ascending=False).items():
        print(f"    {dev}: {cnt:,} readings")

    # Time gap analysis per device
    if "noted_date" in df.columns and df["noted_date"].notna().any():
        print(f"\n  Time gap analysis:")
        for dev, group in df.groupby("device"):
            dates = group["noted_date"].dropna().sort_values()
            if len(dates) > 1:
                gaps = dates.diff().dropna()
                max_gap = gaps.max()
                median_gap = gaps.median()
                print(f"    {dev}: median interval={median_gap}, max gap={max_gap}")

    return counts


def check_duplicates(df):
    """Check for duplicate rows."""
    print(f"\n[quality] DUPLICATE CHECK")
    if "noted_date" in df.columns:
        dupes = df.duplicated(subset=["device", "noted_date"], keep=False)
        n_dupes = dupes.sum()
        print(f"  Duplicate (device+timestamp) rows: {n_dupes}")
    else:
        dupes = df.duplicated(keep=False)
        n_dupes = dupes.sum()
        print(f"  Fully duplicate rows: {n_dupes}")

    return n_dupes


def check_stuck_sensor(df, window=10):
    """Flag devices where the last N readings are identical."""
    print(f"\n[quality] STUCK SENSOR CHECK (last {window} readings identical)")
    stuck_devices = []

    for device_id, group in df.groupby("device"):
        sorted_group = group.sort_values("noted_date") if "noted_date" in group.columns else group
        if len(sorted_group) < window:
            continue
        last_n = sorted_group.tail(window)["temp"].values
        if len(set(last_n)) == 1:
            stuck_devices.append((device_id, last_n[0], len(sorted_group)))

    if stuck_devices:
        print(f"  Found {len(stuck_devices)} potentially stuck sensors:")
        for dev, val, total in stuck_devices:
            print(f"    {dev}: stuck at {val}°C (last {window} of {total} readings)")
    else:
        print(f"  No stuck sensors detected")

    return stuck_devices


def quality_report(df, counts, outliers, n_dupes, stuck):
    """Print a summary quality report."""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Total readings:    {len(df)}")
    print(f"Logical devices:   {df['device'].nunique()}")
    print(f"Temp range:        [{df['temp'].min():.1f}, {df['temp'].max():.1f}]°C")
    if "noted_date" in df.columns and df["noted_date"].notna().any():
        print(f"Date range:        {df['noted_date'].min()} → {df['noted_date'].max()}")
    print(f"Outliers (3σ):     {len(outliers)} ({100*len(outliers)/len(df):.2f}%)")
    print(f"Duplicates:        {n_dupes}")
    print(f"Stuck sensors:     {len(stuck)}")
    print(f"Null temps:        {df['temp'].isna().sum()}")

    # Per-device table
    print(f"\n{'='*60}")
    print(f"PER-DEVICE SUMMARY (top 10 by reading count)")
    print(f"{'='*60}")
    print(f"{'Device':<20} {'Readings':>8} {'Mean':>7} {'Std':>6} "
          f"{'Outliers':>8} {'Stuck?':>6}")
    print("-" * 60)

    device_stats = df.groupby("device").agg(
        readings=("temp", "size"),
        mean_temp=("temp", "mean"),
        std_temp=("temp", "std"),
    ).sort_values("readings", ascending=False)

    outlier_counts = outliers.groupby("device").size()
    stuck_ids = {s[0] for s in stuck}

    for dev_id, row in device_stats.head(10).iterrows():
        n_out = outlier_counts.get(dev_id, 0)
        is_stuck = "YES" if dev_id in stuck_ids else "No"
        print(f"{str(dev_id):<20} {int(row['readings']):>8} "
              f"{row['mean_temp']:>7.1f} {row['std_temp']:>6.1f} "
              f"{n_out:>8} {is_stuck:>6}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="IoT Data Quality Pipeline")
    parser.add_argument("--csv", default=CSV_FILE, help="Path to CSV file")
    args = parser.parse_args()

    try:
        df = load_data(args.csv)
    except FileNotFoundError:
        print(f"[quality] ERROR: File not found: {args.csv}")
        print(f"[quality] Download from: "
              f"https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices")
        sys.exit(1)

    outliers = check_outliers(df)
    counts = check_missing(df)
    n_dupes = check_duplicates(df)
    stuck = check_stuck_sensor(df)
    quality_report(df, counts, outliers, n_dupes, stuck)


if __name__ == "__main__":
    main()

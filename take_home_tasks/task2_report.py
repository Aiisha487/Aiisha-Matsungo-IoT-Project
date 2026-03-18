#!/usr/bin/env python3
"""
EECE 5155 - Take-Home Task 2
Data Quality Report for IOT-temp.csv
Produces exact table format from slide 24.

Usage:
    python task2_report.py
    python task2_report.py --csv IOT-temp.csv
"""

import argparse
import sys
import pandas as pd

CSV_FILE = "IOT-temp.csv"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    room_col = "room_id/id" if "room_id/id" in df.columns else "room_id"
    df["device"] = (df[room_col].astype(str).str.strip()
                    + "_" + df["out/in"].astype(str).str.strip())
    df["noted_date"] = pd.to_datetime(df["noted_date"],
                                      format="mixed",
                                      dayfirst=True,
                                      errors="coerce")
    return df


def outlier_count(series):
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return 0
    return int(((series - mu).abs() > 3 * sigma).sum())


def duplicate_count(grp):
    return int(grp.duplicated(subset=["device", "noted_date"]).sum())


def max_gap(series):
    sorted_ts = series.dropna().sort_values()
    if len(sorted_ts) < 2:
        return "N/A"
    gap = sorted_ts.diff().dropna().max()
    total_s = int(gap.total_seconds())
    d, rem = divmod(total_s, 86400)
    h, rem = divmod(rem, 3600)
    m, _   = divmod(rem, 60)
    if d > 0:
        return f"{d}d {h}h {m}m"
    return f"{h}h {m}m"


def is_stuck(series, n=10):
    tail = series.dropna().tail(n)
    return len(tail) == n and tail.nunique() == 1


def build_report(df):
    rows = []
    for device, grp in df.groupby("device"):
        temps = grp["temp"].dropna()
        rows.append({
            "Device"     : device,
            "Readings"   : len(grp),
            "Mean"       : round(temps.mean(), 1),
            "Outliers"   : outlier_count(temps),
            "Duplicates" : duplicate_count(grp),
            "Max Gap"    : max_gap(grp["noted_date"]),
            "Stuck?"     : "YES" if is_stuck(temps) else "No",
        })
    return rows


def print_report(rows, total):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  EECE 5155 - IoT Data Quality Report")
    print(f"  Source: IOT-temp.csv  |  Total rows: {total:,}")
    print(sep)
    print(f"\n{'Device':<20} {'Readings':>9} {'Mean':>6} "
          f"{'Outliers':>9} {'Duplicates':>11} {'Max Gap':>14} {'Stuck?':>7}")
    print("-" * 72)
    for r in rows:
        print(f"{r['Device']:<20} {r['Readings']:>9,} {r['Mean']:>6.1f} "
              f"{r['Outliers']:>9} {r['Duplicates']:>11,} "
              f"{r['Max Gap']:>14} {r['Stuck?']:>7}")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_FILE)
    args = parser.parse_args()
    try:
        df = load_data(args.csv)
    except FileNotFoundError:
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)
    rows = build_report(df)
    print_report(rows, total=len(df))


if __name__ == "__main__":
    main()

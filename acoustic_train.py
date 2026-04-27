"""
acoustic_train.py
EECE 5155 – Acoustic Monitoring System for Quiet Zones
Train and evaluate a zone compliance classifier on acoustic pipeline data.
Aiisha Matsungo | Spring 2026

Model: RandomForestClassifier on engineered acoustic features
Split: time-based (first 75% train / last 25% test) to respect temporal order
       and prevent future data leaking into training — analogous to the
       subject-based split used in the HAR seminar, but appropriate for
       time-series sensor data.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV   = "data/acoustic_features.csv"
MODEL_PATH     = "analysis/acoustic_model.joblib"
METRICS_PATH   = "analysis/acoustic_metrics.json"
CM_PLOT        = "screenshots/acoustic_confusion_matrix.png"
FEAT_PLOT      = "screenshots/acoustic_feature_importance.png"
TRAIN_SPLIT    = 0.75   # fraction of timesteps used for training

FEATURE_COLS = [
    "dba_spl",
    "spl_roll_mean", "spl_roll_std", "spl_roll_min",
    "spl_roll_max",  "spl_roll_range",
    "spl_delta",     "spl_delta_abs",
    "spl_above_thresh", "spl_margin", "spl_breach_rate",
    "hour", "minute",
    "zone_reading_hall", "zone_silent_study",
]


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame):
    """
    Split by timestamp order (not random) to avoid temporal data leakage.
    Training set = first 75% of the collection window.
    Test set     = final 25% (unseen future readings).
    """
    cutoff = int(len(df) * TRAIN_SPLIT)
    train  = df.iloc[:cutoff].copy()
    test   = df.iloc[cutoff:].copy()
    print(f"[acoustic_train] Train : {len(train)} rows "
          f"({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"[acoustic_train] Test  : {len(test)} rows "
          f"({test['timestamp'].min()} → {test['timestamp'].max()})")
    return train, test


def encode_labels(train, test):
    le = LabelEncoder()
    y_train = le.fit_transform(train["status"])
    y_test  = le.transform(test["status"])
    print(f"[acoustic_train] Classes: {list(le.classes_)}")
    return y_train, y_test, le


def train_models(X_train, y_train):
    """Train RF and LR; return both for comparison (mirrors HAR seminar approach)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)

    return rf, lr


def evaluate(model, X_test, y_test, le, name="Model"):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(f"\n[acoustic_train] ── {name} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print(report)
    return acc, f1, y_pred


def plot_confusion_matrix(model, X_test, y_test, le, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=le.classes_,
        cmap="Blues", ax=ax
    )
    ax.set_title("Acoustic Zone Compliance Classifier\nConfusion Matrix (Time-Based Test Split)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[acoustic_train] Confusion matrix → {path}")


def plot_feature_importance(model, feature_cols, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest Feature Importances\nAcoustic Zone Compliance Classification")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[acoustic_train] Feature importance → {path}")


def simulate_drift(model, X_test, y_test, le):
    """
    Simulate concept drift by adding Gaussian noise to dba_spl,
    showing model degradation — consistent with KS drift analysis
    from the HAR seminar component.
    """
    print("\n[acoustic_train] ── Drift Simulation ──")
    results = []
    for sigma in [0, 2, 4, 6, 8]:
        X_noisy = X_test.copy()
        X_noisy["dba_spl"] += np.random.normal(0, sigma, len(X_noisy))
        f1 = f1_score(y_test, model.predict(X_noisy), average="macro")
        print(f"  Noise σ={sigma:2d} dBA  → Macro F1 = {f1:.4f}")
        results.append({"noise_sigma_dba": sigma, "macro_f1": round(f1, 4)})
    return results


def main():
    os.makedirs("analysis",     exist_ok=True)
    os.makedirs("screenshots",  exist_ok=True)

    df          = load_features(FEATURES_CSV)
    train, test = time_based_split(df)

    X_train = train[FEATURE_COLS]
    X_test  = test[FEATURE_COLS]
    y_train, y_test, le = encode_labels(train, test)

    rf, lr = train_models(X_train, y_train)

    rf_acc, rf_f1, _ = evaluate(rf, X_test, y_test, le, "RandomForest")
    lr_acc, lr_f1, _ = evaluate(lr, X_test, y_test, le, "LogisticRegression")

    plot_confusion_matrix(rf, X_test, y_test, le, CM_PLOT)
    plot_feature_importance(rf, FEATURE_COLS, FEAT_PLOT)

    drift_results = simulate_drift(rf, X_test, y_test, le)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "split_strategy": "time_based_75_25",
        "train_rows": len(train),
        "test_rows":  len(test),
        "classes": list(le.classes_),
        "RandomForest":       {"accuracy": round(rf_acc, 4), "macro_f1": round(rf_f1, 4)},
        "LogisticRegression": {"accuracy": round(lr_acc, 4), "macro_f1": round(lr_f1, 4)},
        "drift_simulation": drift_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[acoustic_train] Metrics saved → {METRICS_PATH}")

    # ── Save best model ───────────────────────────────────────────────────────
    joblib.dump({"model": rf, "label_encoder": le, "feature_cols": FEATURE_COLS},
                MODEL_PATH)
    print(f"[acoustic_train] Model saved   → {MODEL_PATH}")


if __name__ == "__main__":
    main()

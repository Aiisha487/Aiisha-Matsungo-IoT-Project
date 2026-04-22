"""
step4_compare_models.py
=======================
Compare three classifiers on accuracy, inference time, and model size.

This step answers: which algorithm should we deploy on the target hardware?
Accuracy alone is not enough -- a model that does not fit in flash memory
cannot be deployed, regardless of how accurate it is.

Expected output (times will vary by machine):
  k-NN (k=5)             acc=0.884  train=~0.1s  infer=~8ms   size=~16000 kB
  Random Forest          acc=0.929  train=~12s   infer=~0.05ms size=~5800 kB
  SVM (linear)           acc=0.963  train=~4s    infer=~0.01ms size=~27 kB

Question: a microcontroller has 256 kB of flash. Which algorithm fits?
"""

import time
import os
import pickle
import tempfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()

# StandardScaler: fit ONLY on training data, then apply to test
# (fitting on test would leak future information into the model)
scaler  = StandardScaler().fit(X_train)
Xtr_sc  = scaler.transform(X_train)
Xte_sc  = scaler.transform(X_test)

models = {
    "k-NN (k=5)":    (KNeighborsClassifier(n_neighbors=5, n_jobs=-1),                      True),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    "SVM (linear)":  (LinearSVC(max_iter=2000, random_state=42),                            True),
}

print(f"{'Model':<22}  {'Accuracy':>8}  {'Train':>8}  {'Infer/sample':>13}  {'Size':>8}")
print("-" * 68)

for name, (clf, use_scale) in models.items():
    Xtr = Xtr_sc if use_scale else X_train
    Xte = Xte_sc if use_scale else X_test

    t0 = time.time()
    clf.fit(Xtr, y_train)
    train_t = time.time() - t0

    t0 = time.time()
    preds = clf.predict(Xte)
    infer_ms = (time.time() - t0) / len(Xte) * 1000

    acc = accuracy_score(y_test, preds)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(clf, f)
        size_kb = os.path.getsize(f.name) / 1024
        os.unlink(f.name)

    print(f"{name:<22}  {acc:>8.3f}  {train_t:>7.1f}s  {infer_ms:>11.4f}ms  {size_kb:>6.0f} kB")

print("\nMCU constraint: 256 kB flash => only SVM (linear) fits on-device.")
print("k-NN and Random Forest require a gateway (Raspberry Pi, edge server).")

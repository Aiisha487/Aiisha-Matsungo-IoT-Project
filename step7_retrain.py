"""
step7_retrain.py
================
Retrain the SVM with a mixed dataset: original young adult data +
newly labeled samples from the drifted (elderly) population.

In a real deployment, this means going to the hospital, collecting
labeled data from elderly patients, and adding it to the training set.

Here we simulate that by taking 40% of the test set, applying the
same scale=0.75 transformation, and treating it as newly collected data.

Expected results:
  v1 on drifted (elderly): ~0.741   <-- original model fails on new population
  v2 on drifted (elderly): ~0.975   <-- retrained model recovers
  v2 on clean  (young):    ~0.977   <-- does NOT get worse on original population

Questions:
  - Why is the recovery so high in this simulation? Would it be this high
    in a real hospital deployment? (Hint: where does the simulated data come from?)
  - v2 does not regress on the clean (young adult) test set. Why does this matter?
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()

rng      = np.random.default_rng(7)
acc_cols = [c for c in X_train.columns if "Acc" in c]

# --- Simulate newly labeled data from the elderly population ---
# 40% of test set, scaled to simulate elderly gait
X_new = X_test.copy().sample(frac=0.4, random_state=1)
y_new = y_test.loc[X_new.index]
X_new[acc_cols] = X_new[acc_cols] * 0.75
X_new += rng.normal(0, 0.03 * (1.2 - 0.75), X_new.shape)

# --- Mix original training data + new elderly data ---
X_mixed = pd.concat([X_train, X_new], ignore_index=True)
y_mixed = pd.concat([y_train, y_new], ignore_index=True)

# --- Train v2 on mixed data ---
scaler_v2 = StandardScaler().fit(X_mixed)
svm_v2    = LinearSVC(max_iter=2000, random_state=42)
svm_v2.fit(scaler_v2.transform(X_mixed), y_mixed)

# --- Build drifted test set for evaluation ---
X_drifted = X_test.copy()
X_drifted[acc_cols] = X_drifted[acc_cols] * 0.75
X_drifted += rng.normal(0, 0.03 * (1.2 - 0.75), X_drifted.shape)

# --- Train v1 on original data (for fair comparison) ---
scaler_v1 = StandardScaler().fit(X_train)
svm_v1    = LinearSVC(max_iter=2000, random_state=42)
svm_v1.fit(scaler_v1.transform(X_train), y_train)

acc_v1       = accuracy_score(y_test, svm_v1.predict(scaler_v1.transform(X_drifted)))
acc_v2       = accuracy_score(y_test, svm_v2.predict(scaler_v2.transform(X_drifted)))
acc_v2_clean = accuracy_score(y_test, svm_v2.predict(scaler_v2.transform(X_test)))

print(f"v1 on drifted (elderly): {acc_v1:.3f}   <-- original model")
print(f"v2 on drifted (elderly): {acc_v2:.3f}   <-- retrained on mixed data")
print(f"v2 on clean  (young):   {acc_v2_clean:.3f}   <-- no regression on original population")
print(f"\nRecovery: +{acc_v2 - acc_v1:.3f} accuracy points on the elderly population")

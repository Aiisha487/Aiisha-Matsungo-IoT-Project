"""
step6_drift_sim.py
==================
Simulates what happens when the SVM (trained on young adults) is deployed
to elderly patients who move with lower acceleration amplitude.

The model is NOT changed. Only the input data is modified to simulate
the new population. This is a distribution mismatch -- the model was
never trained on this population, so accuracy is already lower on day one.

Simulation: multiply all acceleration features by a scale factor.
  scale=1.0  --> original young adult data  (baseline)
  scale=0.75 --> simulates elderly gait (25% lower amplitude, Menz et al. 2003)

Outputs:
  drift_plot.png  -- accuracy vs scale factor curve

Questions:
  - At scale=0.75, what is the accuracy? (compare to baseline)
  - Which activities are most affected? Which are stable? Why?
  - The model gives no error. How would you detect this in production?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()

# Train SVM on original (young adult) data
scaler = StandardScaler().fit(X_train)
svm    = LinearSVC(max_iter=2000, random_state=42)
svm.fit(scaler.transform(X_train), y_train)
baseline = accuracy_score(y_test, svm.predict(scaler.transform(X_test)))
print(f"Baseline accuracy (young adults): {baseline:.3f}")

# Acceleration features to scale
acc_cols = [c for c in X_train.columns if "Acc" in c]
print(f"Scaling {len(acc_cols)} acceleration features")

rng = np.random.default_rng(7)
results = []
print("\nSimulating drift (scale = fraction of original amplitude):")
for scale in np.arange(1.0, 0.69, -0.05):
    Xd = X_test.copy()
    Xd[acc_cols] = Xd[acc_cols] * scale
    Xd = Xd + rng.normal(0, 0.03 * (1.2 - scale), Xd.shape)
    acc = accuracy_score(y_test, svm.predict(scaler.transform(Xd)))
    results.append({"scale": round(scale, 2), "accuracy": acc})
    print(f"  scale={scale:.2f}  acc={acc:.3f}")

# Per-class recall at scale=0.75
print("\nPer-class recall at scale=0.75 (elderly simulation):")
rng2 = np.random.default_rng(7)
Xd_75 = X_test.copy()
Xd_75[acc_cols] = Xd_75[acc_cols] * 0.75
Xd_75 = Xd_75 + rng2.normal(0, 0.03 * (1.2 - 0.75), Xd_75.shape)
preds_75 = svm.predict(scaler.transform(Xd_75))
for act in sorted(y_test.unique()):
    mask = (y_test == act).values
    recall = (preds_75[mask] == act).mean()
    print(f"  {act:<30}  recall={recall:.3f}")

# Plot
df_r = pd.DataFrame(results)
plt.figure(figsize=(8, 4))
plt.plot(df_r["scale"], df_r["accuracy"], "o-r", label="Drifted accuracy")
plt.axhline(baseline, linestyle="--", color="blue", label=f"Baseline {baseline:.3f}")
plt.axhline(0.85, linestyle=":", color="orange", label="Alert threshold 0.85")
plt.gca().invert_xaxis()
plt.xlabel("Acc feature scale  (1.0 = young adults  |  0.75 = elderly gait)")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy Under Gait Distribution Mismatch")
plt.legend()
plt.tight_layout()
plt.savefig("drift_plot.png")
plt.close()
print("\nSaved: drift_plot.png")

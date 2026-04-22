"""
step2_train.py
==============
Train a Random Forest classifier on the UCI HAR dataset.
Split strategy: by subject (subjects 1-21 train, 22-30 test).
This is the CORRECT split for evaluating on new users.

Outputs:
  rf_model.pkl  -- saved model for later steps
  Console: accuracy + classification report (precision, recall, F1 per class)

Questions to answer after running:
  - What is the overall accuracy?
  - Which two activities have the lowest recall? Does that match what you saw
    in the scatter plot from step 1?
  - What does a recall of 0.89 for SITTING mean in practice for a rehab device?
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()
print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# Train Random Forest (100 trees, ~15-30 seconds)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
print(f"\nRandom Forest accuracy: {acc:.3f}")
print("\nClassification report (precision / recall / F1 per class):")
print(classification_report(y_test, rf.predict(X_test)))

pickle.dump(rf, open("rf_model.pkl", "wb"))
print("Model saved: rf_model.pkl")

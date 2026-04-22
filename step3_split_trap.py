"""
step3_split_trap.py
===================
Demonstrates the split strategy trap: random shuffle vs subject split.

Random split leaks windows from the same person into both train and test.
The model recognizes the individual -- not the activity.
This inflates accuracy by ~5 percentage points.

Question: your manager sees 0.978 and approves deployment.
In production, with new users, you get 0.929. What went wrong?
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()

# --- CORRECT split: by subject (subjects 1-21 train, 22-30 test) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc_subject = accuracy_score(y_test, rf.predict(X_test))

# --- WRONG split: random 80/20 shuffle across all subjects ---
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X_tr2, y_tr2)
acc_random = accuracy_score(y_te2, rf2.predict(X_te2))

print(f"Subject split:  {acc_subject:.3f}   <-- real deployment accuracy (new users)")
print(f"Random split:   {acc_random:.3f}   <-- inflated, do NOT report this")
print(f"\nGap: {acc_random - acc_subject:.3f} percentage points")
print("\nRule: your split strategy must match your deployment scenario.")
print("Deploying to new users? Evaluate on users not in training.")

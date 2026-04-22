"""
step1_explore.py
================
Visualize the dataset before training anything.

Outputs:
  class_balance.png  -- bar chart of samples per activity
  feature_scatter.png -- scatter of 2 key features coloured by activity

Look at the scatter plot and ask: which activities overlap?
That tells you what the model will struggle with -- before you train anything.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv("har_X_train.csv")
X_test  = pd.read_csv("har_X_test.csv")
y_train = pd.read_csv("har_y_train.csv").squeeze()
y_test  = pd.read_csv("har_y_test.csv").squeeze()

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Features: {X_train.shape[1]}  |  Classes: {y_train.nunique()}")

# 1. Class balance (train set)
y_train.value_counts().plot(kind="bar", title="Samples per activity (train)",
                             color="steelblue")
plt.tight_layout()
plt.savefig("class_balance.png")
plt.close()
print("Saved: class_balance.png")

# 2. Scatter: two features that strongly separate activities
fig, ax = plt.subplots(figsize=(9, 4))
palette = sns.color_palette("tab10", 6)
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)
for i, act in enumerate(y_all.unique()):
    mask = (y_all == act)
    ax.scatter(X_all.loc[mask, "tBodyAccMag-mean()"],
               X_all.loc[mask, "tGravityAcc-mean()-X"],
               label=act, alpha=0.3, s=10, color=palette[i])
ax.set_xlabel("Body Acc Magnitude (mean)")
ax.set_ylabel("Gravity Acc X (mean)")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig("feature_scatter.png")
plt.close()
print("Saved: feature_scatter.png")

print("\nOpen both plots.")
print("Question: which two activities overlap completely in the scatter plot? Why?")

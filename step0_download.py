"""
step0_download.py
=================
Downloads the UCI HAR dataset and saves it as four CSV files.
Run this ONCE before any other script.

Expected output:
    Train: (7352, 561)   Test: (2947, 561)
    LAYING      1407
    STANDING    1374
    ...
"""

import urllib.request
import zipfile
import io
import pandas as pd

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")

print("Downloading UCI HAR dataset (~58 MB, takes 10-30 s)...")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=120) as r:
    outer = zipfile.ZipFile(io.BytesIO(r.read()))
inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))


def read(path):
    with inner.open(path) as f:
        return f.read().decode()


# Feature names (make unique — a few duplicates exist in the original file)
raw_feat = read("UCI HAR Dataset/features.txt")
features_raw = [line.split(" ", 1)[1].strip()
                for line in raw_feat.strip().split("\n")]
seen = {}
features = []
for name in features_raw:
    if name in seen:
        seen[name] += 1
        features.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        features.append(name)

# Activity labels
label_raw = read("UCI HAR Dataset/activity_labels.txt")
activity_map = {int(l.split()[0]): l.split()[1]
                for l in label_raw.strip().split("\n")}


def load_split(split):
    X = pd.read_csv(
        io.StringIO(read(f"UCI HAR Dataset/{split}/X_{split}.txt")),
        sep=r"\s+", header=None, names=features)
    y_int = pd.read_csv(
        io.StringIO(read(f"UCI HAR Dataset/{split}/y_{split}.txt")),
        header=None)[0]
    y = y_int.map(activity_map).rename("Activity")
    return X, y


X_train, y_train = load_split("train")
X_test,  y_test  = load_split("test")

X_train.to_csv("har_X_train.csv", index=False)
X_test.to_csv( "har_X_test.csv",  index=False)
y_train.to_csv("har_y_train.csv", index=False)
y_test.to_csv( "har_y_test.csv",  index=False)

print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")
print(f"\nClass distribution (train):\n{y_train.value_counts().to_string()}")
print("\nDone. CSV files saved. Run step1_explore.py next.")

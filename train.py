
import pandas as pd
import joblib
import json
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load dataset
df = pd.read_csv("data/heart_disease_uci_cleaned.csv")

# Remove dataset version column if present
if "dataset_version" in df.columns:
    df = df.drop("dataset_version", axis=1)

X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

results = {}

# -----------------------------
# Model V1
# -----------------------------

p1 = params["model_v1"]

model_v1 = RandomForestClassifier(
    n_estimators=p1["n_estimators"],
    max_depth=p1["max_depth"],
    min_samples_split=p1["min_samples_split"],
    random_state=p1["random_state"]
)

model_v1.fit(X_train, y_train)

pred_v1 = model_v1.predict(X_test)

metrics_v1 = {
    "accuracy": accuracy_score(y_test, pred_v1),
    "precision": precision_score(
        y_test, pred_v1, average="weighted", zero_division=0
    ),
    "recall": recall_score(
        y_test, pred_v1, average="weighted", zero_division=0
    ),
    "f1_score": f1_score(
        y_test, pred_v1, average="weighted", zero_division=0
    )
}

joblib.dump(model_v1, "models/HeartDisease_RF_v1.joblib")

# -----------------------------
# Model V2
# -----------------------------

p2 = params["model_v2"]

model_v2 = RandomForestClassifier(
    n_estimators=p2["n_estimators"],
    max_depth=p2["max_depth"],
    min_samples_split=p2["min_samples_split"],
    random_state=p2["random_state"]
)

model_v2.fit(X_train, y_train)

pred_v2 = model_v2.predict(X_test)

metrics_v2 = {
    "accuracy": accuracy_score(y_test, pred_v2),
    "precision": precision_score(
        y_test, pred_v2, average="weighted", zero_division=0
    ),
    "recall": recall_score(
        y_test, pred_v2, average="weighted", zero_division=0
    ),
    "f1_score": f1_score(
        y_test, pred_v2, average="weighted", zero_division=0
    )
}

joblib.dump(model_v2, "models/HeartDisease_RF_v2.joblib")

# Save metrics
metrics = {
    "model_v1": metrics_v1,
    "model_v2": metrics_v2
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training completed.")
print("V1:", metrics_v1)
print("V2:", metrics_v2)

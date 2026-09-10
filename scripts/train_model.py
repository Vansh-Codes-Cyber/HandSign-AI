import os
from pathlib import Path

import joblib
import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "dataset" / "data.pickle"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "hand_sign_model.joblib"

# ==========================================
# LOAD DATASET
# ==========================================

if not DATASET_FILE.exists():
    raise SystemExit(
        f"Dataset not found at {DATASET_FILE}. Run scripts/create_dataset.py first."
    )

with open(DATASET_FILE, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

print(f"Total Samples : {len(X)}")
print(f"Total Classes : {len(set(y))}")

# ==========================================
# ENCODE LABELS
# ==========================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
)

print("\nTraining model...")

model.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# ==========================================
# SAVE MODEL
# ==========================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump({"model": model, "label_encoder": label_encoder}, MODEL_FILE)

print("\nModel saved successfully!")
print(f"Location : {MODEL_FILE}")

# test_model_accuracy.py

import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "realfishdataset (1).csv")
MODEL_FILE = os.path.join(BASE_DIR, "fish_species_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "label_encoder.pkl")


# ---------------------------------------------------------
# 2. Load files
# ---------------------------------------------------------
print("Loading model and data...\n")

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
label_encoder = joblib.load(ENCODER_FILE)


# ---------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------
required_columns = ["ph", "temperature", "turbidity", "fish"]

df = df[required_columns].dropna().drop_duplicates()

X = df[["ph", "temperature", "turbidity"]]
y = df["fish"]

# Encode labels
y_encoded = label_encoder.transform(y)

# Scale features
X_scaled = scaler.transform(X)


# ---------------------------------------------------------
# 4. Predict
# ---------------------------------------------------------
y_pred = model.predict(X_scaled)


# ---------------------------------------------------------
# 5. Accuracy
# ---------------------------------------------------------
accuracy = accuracy_score(y_encoded, y_pred)

print("=" * 60)
print(f"Model Accuracy on FULL Dataset: {accuracy * 100:.2f}%")
print("=" * 60)


# ---------------------------------------------------------
# 6. Detailed Report
# ---------------------------------------------------------
print("\nClassification Report:\n")
print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_encoded, y_pred))


# ---------------------------------------------------------
# 7. Per-class accuracy (extra)
# ---------------------------------------------------------
print("\nPer-Class Accuracy:\n")

for i, class_name in enumerate(label_encoder.classes_):
    class_indices = (y_encoded == i)
    class_accuracy = accuracy_score(y_encoded[class_indices], y_pred[class_indices])
    print(f"{class_name:15s} : {class_accuracy * 100:.2f}%")
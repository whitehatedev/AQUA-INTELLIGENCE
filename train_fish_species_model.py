# train_fish_species_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "realfishdataset (1).csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("Smart Aquaculture - Fish Species Model Training")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------------
required_columns = ["ph", "temperature", "turbidity", "fish"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[required_columns].copy()
df = df.dropna()
df = df.drop_duplicates()

print("\nAfter cleaning:")
print(df.shape)

print("\nClass distribution:")
print(df["fish"].value_counts())


# ---------------------------------------------------------
# 3. Features and target
# ---------------------------------------------------------
X = df[["ph", "temperature", "turbidity"]]
y = df["fish"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# 4. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0])
print("Test size :", X_test.shape[0])


# ---------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 6. Evaluate model
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ---------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------
model_file = os.path.join(BASE_DIR, "fish_species_model.pkl")
scaler_file = os.path.join(BASE_DIR, "scaler.pkl")
encoder_file = os.path.join(BASE_DIR, "label_encoder.pkl")

joblib.dump(model, model_file)
joblib.dump(scaler, scaler_file)
joblib.dump(label_encoder, encoder_file)

print("\nSaved files:")
print(f"- {model_file}")
print(f"- {scaler_file}")
print(f"- {encoder_file}")


# ---------------------------------------------------------
# 8. Try sample prediction
# ---------------------------------------------------------
sample = pd.DataFrame({
    "ph": [7.5],
    "temperature": [28.0],
    "turbidity": [4.5]
})

sample_scaled = scaler.transform(sample)
pred_encoded = model.predict(sample_scaled)
pred_label = label_encoder.inverse_transform(pred_encoded)

print("\nSample Prediction:")
print(sample)
print("Predicted Fish Species:", pred_label[0])
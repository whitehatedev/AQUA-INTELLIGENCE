import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ================================================================
# 1. LOAD THE DATASET
# ================================================================
file_path = "realfishdataset (1).csv"
df = pd.read_csv(file_path)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# ================================================================
# 2. COUNT AND LIST FISH SPECIES
# ================================================================
species_counts = df['fish'].value_counts()
total_species = len(species_counts)

print("="*50)
print(f"📊 Total number of fish species: {total_species}")
print("="*50)
print("\nSpecies and their record counts:")
for species, count in species_counts.items():
    print(f"  {species}: {count} records")
print("="*50)

# ================================================================
# 3. (OPTIONAL) CLASSIFICATION MODEL
# ================================================================
# Prepare features and target
X = df[['ph', 'temperature', 'turbidity']]
y = df['fish']

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n📈 Classification Results (Random Forest):")
print(f"  Accuracy: {accuracy:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ================================================================
# 4. (OPTIONAL) SAVE THE TRAINED MODEL FOR LATER USE
# ================================================================
import joblib
joblib.dump(model, "fish_species_model_from_csv.pkl")
joblib.dump(scaler, "scaler_from_csv.pkl")
joblib.dump(encoder, "label_encoder_from_csv.pkl")
print("\n✅ Models saved as: fish_species_model_from_csv.pkl, scaler_from_csv.pkl, label_encoder_from_csv.pkl")

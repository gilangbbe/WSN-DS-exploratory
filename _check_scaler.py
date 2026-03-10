"""Check data consistency between arm pipeline and training pipeline."""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Replicate training pipeline exactly ===
df = pd.read_csv('data/WSN-DS.csv')
df.columns = df.columns.str.strip()
df_clean = df.drop_duplicates()
print(f"After dedup: {len(df_clean)} rows")

# Replicate prepare_data_no_fe exactly
df_features = df_clean.copy()
df_features = df_features.drop(columns=['id', 'who CH'], errors='ignore')
X = df_features.drop('Attack type', axis=1)
y = df_features['Attack type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Handle numeric
X_numeric = X.select_dtypes(include=[np.number])
for col in X.columns:
    if col not in X_numeric.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')


X_values = X.values
valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
X_clean = X[valid_mask]
y_clean = y_encoded[valid_mask]
print(f"After nan/inf removal: {len(X_clean)} rows")

# Split - note: X_clean is still a DataFrame
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)
print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
print(f"X_test type: {type(X_test)}, shape: {X_test.shape}")

# Now fit fresh scaler
fresh_scaler = StandardScaler()
X_train_fresh = fresh_scaler.fit_transform(X_train)
X_test_fresh = fresh_scaler.transform(X_test)

# Load saved scaler
saved_scaler = joblib.load('mlflow_artifacts_no_fe/standard_scaler_no_fe.pkl')
X_test_saved = saved_scaler.transform(X_test)

# Compare scaler parameters
print("\n=== Scaler Comparison ===")
print(f"Fresh mean[:3]:  {fresh_scaler.mean_[:3]}")
print(f"Saved mean[:3]:  {saved_scaler.mean_[:3]}")
print(f"Fresh scale[:3]: {fresh_scaler.scale_[:3]}")
print(f"Saved scale[:3]: {saved_scaler.scale_[:3]}")
print(f"Means match: {np.allclose(fresh_scaler.mean_, saved_scaler.mean_)}")
print(f"Scales match: {np.allclose(fresh_scaler.scale_, saved_scaler.scale_)}")

# Test model with both scalers
model = joblib.load('mlruns/410992055011183175/models/m-314ff2a6346d4417b349fa640ab31867/artifacts/model.pkl')
y_pred_fresh = model.predict(X_test_fresh)
y_pred_saved = model.predict(X_test_saved)
print(f"\nHistGB with fresh scaler: acc={accuracy_score(y_test, y_pred_fresh):.4f}")
print(f"HistGB with saved scaler: acc={accuracy_score(y_test, y_pred_saved):.4f}")

# Check if transformations are the same
max_diff = np.max(np.abs(X_test_fresh - X_test_saved))
print(f"Max transform diff: {max_diff}")

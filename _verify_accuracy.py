"""Verify model accuracy with proper preprocessing."""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data exactly like wsn_mlflow_pipeline_no_fe.py
df = pd.read_csv('data/WSN-DS.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.drop(columns=['id', 'who CH'], errors='ignore')

X = df.drop('Attack type', axis=1)
y = df['Attack type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# Handle numeric/inf/nan
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X_values = X.values
valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
X = X[valid_mask]
y_encoded = y_encoded[valid_mask]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Test set: {len(X_test)} samples")

# Try with and without scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Load each model and test both ways
models = {
    'Extra_Trees': 'mlruns/410992055011183175/models/m-3e057b03d6814337877607b8b495ccc6/artifacts/model.pkl',
    'Random_Forest': 'mlruns/410992055011183175/models/m-6d14c8f09176402abac542494fd2871a/artifacts/model.pkl',
    'Gradient_Boosting': 'mlruns/410992055011183175/models/m-8b94e721d09b479e9c1ba33fc7debc6d/artifacts/model.pkl',
    'HistGradient_Boosting': 'mlruns/410992055011183175/models/m-314ff2a6346d4417b349fa640ab31867/artifacts/model.pkl',
}

for name, path in models.items():
    model = joblib.load(path)
    
    # Raw (no scaling)
    y_pred_raw = model.predict(X_test.astype(np.float32))
    acc_raw = accuracy_score(y_test, y_pred_raw)
    f1_raw = f1_score(y_test, y_pred_raw, average='macro')
    
    # Scaled
    y_pred_scaled = model.predict(X_test_s)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)
    f1_scaled = f1_score(y_test, y_pred_scaled, average='macro')
    
    print(f"\n{name}:")
    print(f"  Raw:    acc={acc_raw:.4f}  f1_macro={f1_raw:.4f}")
    print(f"  Scaled: acc={acc_scaled:.4f}  f1_macro={f1_scaled:.4f}")

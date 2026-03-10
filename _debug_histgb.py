"""Debug HistGB model - check type and test with different approaches."""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv('data/WSN-DS.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.drop(columns=['id', 'who CH'], errors='ignore')
X = df.drop('Attack type', axis=1)
y = df['Attack type']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X_values = X.values
valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
X_clean = X[valid_mask]
y_encoded = y_encoded[valid_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Load HistGB model
model = joblib.load('mlruns/410992055011183175/models/m-314ff2a6346d4417b349fa640ab31867/artifacts/model.pkl')
print(f"Type: {type(model).__name__}")
print(f"Params: max_iter={model.max_iter}, max_depth={model.max_depth}, lr={model.learning_rate}")
print(f"n_iter_: {model.n_iter_}")
print(f"classes_: {model.classes_}")

# Test with float32 vs float64
print("\n=== Float64 scaled ===")
y_pred = model.predict(X_test_s)
print(f"  acc={accuracy_score(y_test, y_pred):.4f}  f1m={f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n=== Float32 scaled ===")
y_pred = model.predict(X_test_s.astype(np.float32))
print(f"  acc={accuracy_score(y_test, y_pred):.4f}  f1m={f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n=== Float64 raw ===")
y_pred = model.predict(X_test.astype(np.float64))
print(f"  acc={accuracy_score(y_test, y_pred):.4f}  f1m={f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n=== Float32 raw ===")
y_pred = model.predict(X_test.astype(np.float32))
print(f"  acc={accuracy_score(y_test, y_pred):.4f}  f1m={f1_score(y_test, y_pred, average='macro'):.4f}")

# Try loading the saved scaler
try:
    saved_scaler = joblib.load('mlflow_artifacts_no_fe/standard_scaler_no_fe.pkl')
    X_test_saved = saved_scaler.transform(X_test)
    print("\n=== Saved scaler ===")
    y_pred = model.predict(X_test_saved)
    print(f"  acc={accuracy_score(y_test, y_pred):.4f}  f1m={f1_score(y_test, y_pred, average='macro'):.4f}")
except Exception as e:
    print(f"\nSaved scaler error: {e}")

# Also check: is the data split identical?
print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_train[0,:3]: {X_train[0,:3]}")
print(f"X_test[0,:3]: {X_test[0,:3]}")

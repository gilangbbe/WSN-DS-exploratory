"""Test float32 conversion effect on HistGB accuracy."""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/WSN-DS.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.drop(columns=['id', 'who CH'], errors='ignore')
X = df.drop('Attack type', axis=1)
y = df['Attack type']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
for col in X.columns:
    if col not in X.select_dtypes(include=[np.number]).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
X_values = X.values
valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
X = X[valid_mask]
y_encoded = y_encoded[valid_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = joblib.load('mlflow_artifacts_no_fe/standard_scaler_no_fe.pkl')
X_test_scaled = scaler.transform(X_test)

model = joblib.load('mlruns/410992055011183175/models/m-314ff2a6346d4417b349fa640ab31867/artifacts/model.pkl')

# Test float64
y_pred = model.predict(X_test_scaled)
print(f"float64: acc={accuracy_score(y_test, y_pred):.4f}")

# Test float32
y_pred32 = model.predict(X_test_scaled.astype(np.float32))
print(f"float32: acc={accuracy_score(y_test, y_pred32):.4f}")

# Small perturbation test
diff = X_test_scaled - X_test_scaled.astype(np.float32).astype(np.float64)
print(f"Max float32 truncation diff: {np.max(np.abs(diff)):.2e}")
print(f"Mean float32 truncation diff: {np.mean(np.abs(diff)):.2e}")

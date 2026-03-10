# WSN Intrusion Detection - MLflow Training Pipeline

## Overview

This training pipeline implements a comprehensive machine learning framework for Wireless Sensor Network (WSN) intrusion detection with **MLflow integration** for experiment tracking, model versioning, and model registry.

## Features

### Two Experiments

1. **Without Oversampling Strategy**
   - Uses class-weighted algorithms (Random Forest, Logistic Regression, Extra Trees)
   - Specialized ensemble methods for imbalanced data (Balanced Random Forest, Balanced Bagging)
   - Total: 8 models

2. **With Oversampling Strategy**
   - SMOTE-ENN (Hybrid approach with Edited Nearest Neighbors)
   - BorderlineSMOTE (Focus on decision boundary samples)
   - ADASYN (Adaptive Synthetic Sampling)
   - Conservative SMOTE (Novel approach - 20% majority class limit)
   - Total: 24 models (6 models × 4 strategies)

### Models Trained

| Model | Description |
|-------|-------------|
| Random Forest | With balanced class weights |
| Gradient Boosting | Standard gradient boosting |
| HistGradient Boosting | Histogram-based, faster variant |
| Extra Trees | Extremely randomized trees |
| Neural Network | Multi-layer perceptron (100, 50) |
| Logistic Regression | With balanced class weights |
| Balanced Random Forest | Specialized for imbalanced data |
| Balanced Bagging | Balanced bootstrap sampling |

### MLflow Integration

- **Experiment Tracking**: All runs logged with parameters and metrics
- **Model Registry**: Models registered with versioning
- **Artifacts**: Classification reports, confusion matrices, feature importance
- **Metrics Logged**:
  - Accuracy, Precision, Recall, F1 (weighted and macro)
  - Per-class metrics (Blackhole, Flooding, Grayhole, Normal, TDMA)
  - Training time, Log loss

## Preprocessing Pipeline

The pipeline replicates the exact preprocessing from the notebook:

1. **Data Cleaning**: Duplicate removal
2. **Feature Engineering**:
   - Distance features: `Distance_Efficiency`, `Distance_Ratio`
   - Energy features: `Energy_Per_Data`, `Energy_Efficiency`, `Energy_Rank_Ratio`
   - Communication features: `Total_Messages_Sent/Received`, `Message_Balance`, `Communication_Ratio`
   - Temporal features: `Time_Normalized`, `Time_Category_Num`
   - Network role features: `CH_Distance_Product`, `CH_Energy_Product`, `NonCH_Rank_Product`
3. **Scaling**: StandardScaler normalization
4. **Stratified Split**: 80/20 train/test with stratification

## Installation

```bash
# Install required packages
pip install -r requirements_mlflow.txt
```

## Usage

### Run the Full Pipeline

```bash
python wsn_mlflow_pipeline.py
```

### View Results in MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000 in your browser.

## Output Structure

```
TugasAkhir/
├── mlruns/                           # MLflow tracking directory
│   ├── 0/                            # Default experiment
│   ├── <experiment_id>/              # Experiment runs
│   │   └── <run_id>/
│   │       ├── artifacts/
│   │       │   ├── model/            # Logged model
│   │       │   ├── classification_report.json
│   │       │   ├── confusion_matrix.csv
│   │       │   └── feature_importance.json
│   │       ├── metrics/
│   │       ├── params/
│   │       └── tags/
├── mlflow_artifacts/                 # Preprocessing components
│   ├── standard_scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.json
├── experiment_summary.csv            # Summary of all experiments
└── wsn_mlflow_pipeline.py           # Main pipeline script
```

## Model Registry

All models are automatically registered in MLflow with naming convention:
```
WSN_IDS_{ModelName}_{SamplingStrategy}
```

Examples:
- `WSN_IDS_Gradient_Boosting_Conservative_SMOTE`
- `WSN_IDS_Balanced_Random_Forest_Ensemble_Imbalanced`

## Programmatic Access

### Load a Registered Model

```python
import mlflow

# Load model from registry
model_uri = "models:/WSN_IDS_Gradient_Boosting_Conservative_SMOTE/1"
model = mlflow.sklearn.load_model(model_uri)

# Load from run ID
run_id = "abc123..."
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

### Query Experiments

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("WSN_Intrusion_Detection_With_Oversampling")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.f1_weighted > 0.99",
    order_by=["metrics.f1_weighted DESC"]
)

# Print best models
for run in runs[:5]:
    print(f"Run: {run.info.run_id}")
    print(f"  F1: {run.data.metrics['f1_weighted']:.4f}")
    print(f"  Model: {run.data.tags['model_name']}")
```

## Expected Results

Based on the original notebook, expected performance:

| Configuration | F1-Score |
|---------------|----------|
| Gradient Boosting + Conservative SMOTE | ~99.62% |
| Random Forest + BorderlineSMOTE | ~99.57% |
| Gradient Boosting + ADASYN | ~99.57% |
| Balanced Random Forest (no oversampling) | ~98.78% |
| Balanced Bagging (no oversampling) | ~98.90% |

## Configuration

Modify the `Config` class in `wsn_mlflow_pipeline.py`:

```python
class Config:
    DATA_PATH = "data/WSN-DS.csv"
    MLFLOW_TRACKING_URI = "mlruns"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    MODEL_VERSION = "1.0.0"
```

## Authors

Machine Learning Engineering Team - WSN Intrusion Detection Project

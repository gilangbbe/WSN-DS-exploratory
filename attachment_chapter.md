# Attachment: Source Code

## A.1 Library Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from scipy import stats
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN

warnings.filterwarnings('ignore')
```

## A.2 Data Loading and Cleaning

```python
# Load dataset
df = pd.read_csv('data/WSN-DS.csv')
df.columns = df.columns.str.strip()

# Remove duplicate records
original_size = len(df)
df_clean = df.drop_duplicates()
removed_duplicates = original_size - len(df_clean)

# Remove redundant features (identified via correlation analysis)
redundant_features = ['id', 'who CH']
df_features = df_clean.drop(columns=redundant_features)

df = df_clean.copy()
```

## A.3 Label Encoding and Feature-Target Separation

```python
# Separate features and target variable
X = df_features.drop('Attack type', axis=1)
y = df_features['Attack type']

# Encode categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle non-numeric columns
for col in X.columns:
    if X[col].dtype == 'category' or X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Remove rows with infinite or NaN values
X_numeric = X.select_dtypes(include=[np.number])
inf_mask = np.isinf(X_numeric.values).any(axis=1)
nan_mask = np.isnan(X_numeric.values).any(axis=1)
problematic_rows = inf_mask | nan_mask

if problematic_rows.sum() > 0:
    X = X_numeric[~problematic_rows]
    y_encoded = y_encoded[~problematic_rows]
else:
    X = X_numeric
```

## A.4 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
```

## A.5 Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## A.6 Oversampling Methods

### A.6.1 SMOTE-ENN (Hybrid Approach)

```python
smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
X_smote_enn, y_smote_enn = smote_enn.fit_resample(X_train_scaled, y_train)
```

### A.6.2 Borderline-SMOTE

```python
borderline_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto')
X_borderline, y_borderline = borderline_smote.fit_resample(X_train_scaled, y_train)
```

### A.6.3 ADASYN (Adaptive Synthetic Sampling)

```python
adasyn = ADASYN(random_state=42, sampling_strategy='auto')
X_adasyn, y_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
```

### A.6.4 Conservative SMOTE

```python
def create_conservative_strategy(y_train):
    """Limit minority classes to 20% of majority class size."""
    counts = Counter(y_train)
    majority_class_count = max(counts.values())
    target_count = int(majority_class_count * 0.2)

    conservative_strategy = {}
    for class_label, count in counts.items():
        if count < target_count:
            conservative_strategy[class_label] = target_count
    return conservative_strategy

conservative_strategy = create_conservative_strategy(y_train)
smote_conservative = SMOTE(random_state=42,
                           sampling_strategy=conservative_strategy)
X_conservative, y_conservative = smote_conservative.fit_resample(
    X_train_scaled, y_train
)
```

### A.6.5 Oversampling Strategy Storage

```python
sampling_strategies = {
    'SMOTE-ENN':          (X_smote_enn,    y_smote_enn),
    'BorderlineSMOTE':    (X_borderline,   y_borderline),
    'ADASYN':             (X_adasyn,       y_adasyn),
    'Conservative_SMOTE': (X_conservative, y_conservative),
}
```

## A.7 Synthetic Data Quality Evaluation

### A.7.1 Kolmogorov-Smirnov (KS) Test

```python
def calculate_ks_statistics(X_original, X_oversampled, feature_names):
    """Calculate KS test statistics for each feature."""
    ks_results = {}
    for i, feature_name in enumerate(feature_names):
        original_values = X_original[:, i]
        oversampled_values = X_oversampled[:, i]

        if np.var(original_values) == 0 and np.var(oversampled_values) == 0:
            ks_stat = 0.0 if np.mean(original_values) == np.mean(oversampled_values) else 1.0
            p_value = 1.0 if ks_stat == 0.0 else 0.0
        elif np.var(original_values) == 0 or np.var(oversampled_values) == 0:
            ks_stat, p_value = 1.0, 0.0
        else:
            ks_stat, p_value = ks_2samp(original_values, oversampled_values)

        ks_results[feature_name] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
    return ks_results
```

### A.7.2 Jensen-Shannon Divergence (JSD)

```python
def calculate_jensen_shannon_divergence(X_original, X_oversampled,
                                        feature_names, n_bins=50):
    """Calculate JS divergence for each feature."""
    js_results = {}
    for i, feature_name in enumerate(feature_names):
        original_values = X_original[:, i]
        oversampled_values = X_oversampled[:, i]

        if np.var(original_values) == 0 and np.var(oversampled_values) == 0:
            js_div = 0.0 if np.mean(original_values) == np.mean(oversampled_values) else 1.0
        elif np.var(original_values) == 0 or np.var(oversampled_values) == 0:
            js_div = 1.0
        else:
            combined_min = min(original_values.min(), oversampled_values.min())
            combined_max = max(original_values.max(), oversampled_values.max())

            if combined_min == combined_max:
                js_div = 0.0
            else:
                bins = np.linspace(combined_min, combined_max, n_bins)
                orig_hist, _ = np.histogram(original_values, bins=bins, density=True)
                over_hist, _ = np.histogram(oversampled_values, bins=bins, density=True)

                epsilon = 1e-10
                orig_prob = (orig_hist + epsilon) / np.sum(orig_hist + epsilon)
                over_prob = (over_hist + epsilon) / np.sum(over_hist + epsilon)

                js_div = jensenshannon(orig_prob, over_prob)
                if np.isnan(js_div) or np.isinf(js_div):
                    js_div = 1.0

        js_results[feature_name] = {
            'js_divergence': js_div,
            'similarity_score': 1 - js_div
        }
    return js_results
```

### A.7.3 Correlation Preservation

```python
def calculate_correlation_preservation(X_original, X_oversampled,
                                       feature_names):
    """Assess preservation of correlation structure."""
    corr_original = np.corrcoef(X_original.T)
    corr_oversampled = np.corrcoef(X_oversampled.T)

    # Extract upper triangular elements (excluding diagonal)
    mask = np.triu(np.ones_like(corr_original, dtype=bool), k=1)
    orig_corr_values = corr_original[mask]
    over_corr_values = corr_oversampled[mask]

    correlation_diff = np.abs(orig_corr_values - over_corr_values)
    mean_abs_error = np.mean(correlation_diff)
    max_abs_error = np.max(correlation_diff)

    if np.var(orig_corr_values) > 0 and np.var(over_corr_values) > 0:
        correlation_correlation = np.corrcoef(
            orig_corr_values, over_corr_values
        )[0, 1]
    else:
        correlation_correlation = (
            1.0 if np.mean(orig_corr_values) == np.mean(over_corr_values) else 0.0
        )

    return {
        'mean_absolute_error': mean_abs_error,
        'max_absolute_error': max_abs_error,
        'correlation_correlation': correlation_correlation,
        'correlation_preservation_score': 1 - mean_abs_error,
        'original_correlation_matrix': corr_original,
        'oversampled_correlation_matrix': corr_oversampled
    }
```

### A.7.4 Comprehensive Quality Assessment

```python
def comprehensive_quality_assessment(X_original, X_oversampled,
                                     feature_names, strategy_name):
    """Run all quality metrics and compute composite score."""
    ks_results = calculate_ks_statistics(X_original, X_oversampled,
                                         feature_names)
    js_results = calculate_jensen_shannon_divergence(X_original, X_oversampled,
                                                     feature_names)
    corr_results = calculate_correlation_preservation(X_original, X_oversampled,
                                                      feature_names)

    ks_stats = [r['ks_statistic'] for r in ks_results.values()]
    similarity_scores = [r['similarity_score'] for r in js_results.values()]

    ks_quality = 1 - np.mean(ks_stats)
    js_quality = np.mean(similarity_scores)
    corr_quality = corr_results['correlation_preservation_score']
    overall_quality = (ks_quality + js_quality + corr_quality) / 3

    return {
        'strategy_name': strategy_name,
        'ks_results': ks_results,
        'js_results': js_results,
        'correlation_results': corr_results,
        'quality_scores': {
            'ks_quality': ks_quality,
            'js_quality': js_quality,
            'correlation_quality': corr_quality,
            'overall_quality': overall_quality
        }
    }
```

### A.7.5 Per-Class Quality Evaluation

```python
def sample_per_class(X, y, n, random_state=None):
    """Randomly sample n examples per class from (X, y)."""
    if random_state is not None:
        np.random.seed(random_state)
    unique_classes = [0, 1, 2, 4]
    X_list, y_list = [], []
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        chosen_idx = np.random.choice(idx, n, replace=False)
        X_list.append(X[chosen_idx])
        y_list.append(y[chosen_idx])
    return np.concatenate(X_list), np.concatenate(y_list)

# Evaluate quality per class for each strategy
feature_names = X.columns.tolist()
X_reference = X_train_scaled.copy()
quality_assessments = {}

for strategy_name, (X_oversampled, y_oversampled) in sampling_strategies.items():
    # Isolate synthetic samples
    orig_view = X_reference.view(
        [('', X_reference.dtype)] * X_reference.shape[1]
    )
    oversampled_view = X_oversampled.view(
        [('', X_oversampled.dtype)] * X_oversampled.shape[1]
    )
    synthetic_mask = ~np.isin(oversampled_view, orig_view)
    X_over_sample = X_oversampled[synthetic_mask.ravel()]
    y_over_sample = y_oversampled[synthetic_mask.ravel()]

    # Balanced sampling for fair comparison
    X_orig_sample, y_orig_sample = sample_per_class(
        X_reference, y_train, n=2526, random_state=42
    )
    X_over_sample, y_over_sample = sample_per_class(
        X_over_sample, y_over_sample, n=2526, random_state=42
    )

    class_results = {}
    for cls in [0, 1, 2, 4]:
        assessment = comprehensive_quality_assessment(
            X_orig_sample[y_orig_sample == cls],
            X_over_sample[y_over_sample == cls],
            feature_names, strategy_name
        )
        class_results[cls] = assessment
    quality_assessments[strategy_name] = class_results
```

## A.8 Model Configuration and Training

### A.8.1 Model Configurations

```python
models_config = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=100, max_depth=20,
            random_state=42, n_jobs=-1
        ),
        'class_weight': 'balanced'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            n_estimators=100, max_depth=10,
            random_state=42
        ),
        'class_weight': None
    },
    'HistGradient Boosting': {
        'model': HistGradientBoostingClassifier(
            max_iter=100, max_depth=15,
            random_state=42
        ),
        'class_weight': None
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(
            n_estimators=100, max_depth=20,
            random_state=42, n_jobs=-1
        ),
        'class_weight': 'balanced'
    },
}
```

### A.8.2 Training and Evaluation Function

```python
def train_evaluate_model(model, X_train, X_test, y_train, y_test,
                         model_name, class_weight=None):
    """Train model and compute classification metrics."""
    if class_weight and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = (model.predict_proba(X_test)
                    if hasattr(model, 'predict_proba') else None)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    class_report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_,
        output_dict=True
    )

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'classification_report': class_report
    }
```

### A.8.3 Training Across All Oversampling Strategies

```python
sampling_strategy_results = {}

for strategy_name, (X_resampled, y_resampled) in sampling_strategies.items():
    strategy_results = {}
    for model_name, config in models_config.items():
        result = train_evaluate_model(
            config['model'], X_resampled, X_test_scaled,
            y_resampled, y_test, model_name, config['class_weight']
        )
        strategy_results[model_name] = result
    sampling_strategy_results[strategy_name] = strategy_results
```

## A.9 Performance Evaluation

### A.9.1 Results Summary

```python
results_summary = []
for strategy, models in sampling_strategy_results.items():
    for model_name, result in models.items():
        results_summary.append({
            'Strategy': strategy,
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score']
        })

results_df = pd.DataFrame(results_summary)

# Top models by F1-Score
top_models = results_df.nlargest(10, 'F1_Score')

# Best model per strategy
best_by_strategy = results_df.loc[
    results_df.groupby('Strategy')['F1_Score'].idxmax()
]
```

### A.9.2 AUC-ROC Calculation

```python
for strategy in sampling_strategy_results:
    for model_name, res in sampling_strategy_results[strategy].items():
        auc = roc_auc_score(
            y_test, res['probabilities'],
            average='macro', multi_class='ovr'
        )
        print(f"{strategy} - {model_name}: AUC={auc:.4f}")
```

### A.9.3 Per-Class Accuracy

```python
def class_wise_accuracy(y_true, y_pred):
    """Compute accuracy for each class individually."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    class_acc = {}
    for cls in classes:
        cls_mask = (y_true == cls)
        correct = np.sum(y_pred[cls_mask] == cls)
        total = np.sum(cls_mask)
        class_acc[cls] = correct / total if total > 0 else np.nan
    return class_acc
```

### A.9.4 Confusion Matrix

```python
def get_confusion_matrix_table(y_true, y_pred, labels=None):
    """Generate confusion matrix as a labeled DataFrame."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {l}" for l in labels],
        columns=[f"Predicted: {l}" for l in labels]
    )
    return cm_df
```

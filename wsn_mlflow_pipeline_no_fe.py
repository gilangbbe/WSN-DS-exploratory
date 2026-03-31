"""
WSN Intrusion Detection System - MLflow Training Pipeline (No Feature Engineering)
==================================================================================
Stratified K-Fold Cross Validation pipeline for evaluating all combinations
of oversampling strategies and classifiers.

Oversampling Methods: SMOTE-ENN, Borderline-SMOTE, ADASYN, Conservative-SMOTE
Classifiers: Random Forest, Gradient Boosting, HistGradient Boosting, Extra Trees
Evaluation: Stratified 3-Fold Cross Validation

Oversampling is applied ONLY to the training fold of each CV iteration
to prevent data leakage.

Author: Machine Learning Engineering Team
Date: 2026-03-10
"""

import os
import warnings
from datetime import datetime
from collections import Counter
from typing import Dict, Any, Tuple, List, Optional

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, log_loss,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    average_precision_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import joblib
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the Stratified K-Fold CV training pipeline."""

    DATA_PATH = "data/WSN-DS.csv"
    MLFLOW_TRACKING_URI = "mlruns"
    EXPERIMENT_NAME = "WSN_IDS_No_FE_KFold_CV"
    K_FOLDS = 3
    RANDOM_STATE = 42
    MODEL_VERSION = "2.0.0"
    REDUNDANT_FEATURES = ['id', 'who CH']


# ============================================================================
# DATA PREPROCESSING (WITHOUT FEATURE ENGINEERING)
# ============================================================================

class DataPreprocessorNoFE:
    """
    Handles data loading, cleaning, and preprocessing WITHOUT feature engineering.
    Returns X, y arrays; splitting is handled by K-Fold CV.
    """

    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def load_data(self) -> pd.DataFrame:
        """Load the WSN-DS dataset."""
        print("Loading WSN-DS dataset...")
        df = pd.read_csv(self.config.DATA_PATH)
        df.columns = df.columns.str.strip()
        print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        print("Cleaning data...")
        original_size = len(df)
        df_clean = df.drop_duplicates()
        removed = original_size - len(df_clean)
        print(f"Removed {removed:,} duplicate records")
        print(f"Clean dataset: {len(df_clean):,} samples")
        return df_clean

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y arrays WITHOUT feature engineering.
        K-Fold CV handles train/val splitting, so no split is done here.
        """
        print("Preparing data WITHOUT feature engineering...")

        df_features = df.copy()
        df_features = df_features.drop(columns=self.config.REDUNDANT_FEATURES, errors='ignore')

        X = df_features.drop('Attack type', axis=1)
        y = df_features['Attack type']

        y_encoded = self.label_encoder.fit_transform(y)

        self.feature_names = list(X.columns)
        print(f"Using {len(self.feature_names)} original features (no engineering)")

        # Ensure all columns are numeric
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce')

        X_values = X.values.astype(np.float64)

        # Remove rows with inf/nan
        valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
        X_values = X_values[valid_mask]
        y_encoded = y_encoded[valid_mask]

        print(f"Final dataset: {X_values.shape[0]:,} samples, {X_values.shape[1]} features")
        return X_values, y_encoded

    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution with class names."""
        counter = Counter(y)
        return {self.label_encoder.classes_[k]: v for k, v in sorted(counter.items())}


# ============================================================================
# OVERSAMPLING STRATEGIES
# ============================================================================

class OversamplingStrategies:
    """Implements the four oversampling strategies for handling class imbalance."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_strategies(self) -> Dict[str, Any]:
        """Return a dict of strategy_name -> callable(X, y) -> (X_res, y_res)."""
        return {
            'SMOTE_ENN': self._apply_smote_enn,
            'BorderlineSMOTE': self._apply_borderline_smote,
            'ADASYN': self._apply_adasyn,
            'Conservative_SMOTE': self._apply_conservative_smote,
        }

    def _apply_smote_enn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sampler = SMOTEENN(random_state=self.random_state, sampling_strategy='auto')
        return sampler.fit_resample(X, y)

    def _apply_borderline_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sampler = BorderlineSMOTE(random_state=self.random_state, sampling_strategy='auto')
        return sampler.fit_resample(X, y)

    def _apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sampler = ADASYN(random_state=self.random_state, sampling_strategy='auto')
        return sampler.fit_resample(X, y)

    def _apply_conservative_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative SMOTE: limits minority augmentation to 20% of majority class count."""
        counts = Counter(y)
        majority_count = max(counts.values())
        target_count = int(majority_count * 0.2)

        conservative_strategy = {
            cl: target_count for cl, c in counts.items() if c < target_count
        }

        if conservative_strategy:
            sampler = SMOTE(random_state=self.random_state, sampling_strategy=conservative_strategy)
            return sampler.fit_resample(X, y)
        return X.copy(), y.copy()


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Return the four classifiers to evaluate."""
    return {
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20,
            random_state=random_state, n_jobs=-1, class_weight='balanced'
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        'HistGradient_Boosting': HistGradientBoostingClassifier(
            max_iter=100, max_depth=15, random_state=random_state
        ),
        'Extra_Trees': ExtraTreesClassifier(
            n_estimators=100, max_depth=20,
            random_state=random_state, n_jobs=-1, class_weight='balanced'
        ),
    }


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    class_names: List[str]
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for a single CV fold."""
    metrics = {}
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # ---- Primary aggregate metrics ----
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = p_w
    metrics['recall_weighted'] = r_w
    metrics['f1_weighted'] = f1_w

    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = p_m
    metrics['recall_macro'] = r_m
    metrics['f1_macro'] = f1_m

    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # ---- Per-class metrics ----
    p_pc, r_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    for i, cn in enumerate(class_names):
        safe = cn.replace(' ', '_')
        metrics[f'precision_{safe}'] = float(p_pc[i])
        metrics[f'recall_{safe}'] = float(r_pc[i])
        metrics[f'f1_{safe}'] = float(f1_pc[i])
        metrics[f'support_{safe}'] = float(sup_pc[i])

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        metrics[f'specificity_{safe}'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics[f'false_positive_rate_{safe}'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics[f'false_negative_rate_{safe}'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ---- Probability-based metrics ----
    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba, labels=labels)
        except Exception:
            pass
        try:
            metrics['roc_auc_ovr_weighted'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='weighted', labels=labels
            )
            metrics['roc_auc_ovr_macro'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='macro', labels=labels
            )
        except Exception:
            pass
        try:
            for i, cn in enumerate(class_names):
                safe = cn.replace(' ', '_')
                y_bin = (y_true == i).astype(int)
                metrics[f'roc_auc_{safe}'] = roc_auc_score(y_bin, y_pred_proba[:, i])
                metrics[f'avg_precision_{safe}'] = average_precision_score(y_bin, y_pred_proba[:, i])
        except Exception:
            pass

    # ---- Geometric mean of per-class recalls ----
    recalls = r_pc[r_pc > 0]
    if len(recalls) > 0:
        metrics['geometric_mean_recall'] = float(np.exp(np.mean(np.log(recalls))))

    return metrics


# ============================================================================
# K-FOLD CROSS VALIDATION PIPELINE
# ============================================================================

class KFoldCVPipeline:
    """
    Stratified K-Fold Cross Validation pipeline that evaluates every
    oversampling method x classifier combination.

    Oversampling is applied ONLY to the training fold to prevent data leakage.
    """

    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessorNoFE(config)
        self.oversampling = OversamplingStrategies(config.RANDOM_STATE)

    def setup_mlflow(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    def create_or_get_experiment(self, name: str) -> str:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
            print(f"Created experiment: {name} (ID: {exp_id})")
        else:
            exp_id = exp.experiment_id
            print(f"Using experiment: {name} (ID: {exp_id})")
        return exp_id

    # ------------------------------------------------------------------ #
    # Core: evaluate one (oversampling x model) combination via K-Fold
    # ------------------------------------------------------------------ #

    def evaluate_combination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        model_template,
        oversampling_name: str,
        oversampling_func,
        skf: StratifiedKFold,
        class_names: List[str],
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Evaluate one oversampling x classifier combination with K-Fold CV."""

        print(f"\n  [{oversampling_name}] x [{model_name}]")
        fold_metrics_list = []

        run_name = f"{oversampling_name}__{model_name}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("oversampling_strategy", oversampling_name)
            mlflow.set_tag("evaluation_method", f"Stratified_{self.config.K_FOLDS}-Fold_CV")
            mlflow.set_tag("feature_engineering", "None")
            mlflow.set_tag("version", self.config.MODEL_VERSION)

            # Log model hyper-parameters
            params = model_template.get_params()
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            mlflow.log_param("k_folds", self.config.K_FOLDS)
            mlflow.log_param("oversampling_method", oversampling_name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_classes", len(class_names))
            mlflow.log_param("total_samples", X.shape[0])

            # ---- K-Fold loop ----
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Scale: fit on training fold only (prevents data leakage)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)

                # Apply oversampling to training fold only
                try:
                    X_train_resampled, y_train_resampled = oversampling_func(
                        X_train_scaled, y_train_fold
                    )
                except Exception as e:
                    print(f"    Fold {fold_idx + 1}: Oversampling failed ({e}), using original training data")
                    X_train_resampled, y_train_resampled = X_train_scaled, y_train_fold

                # Fresh model instance per fold
                model = type(model_template)(**model_template.get_params())

                # Train
                start = datetime.now()
                model.fit(X_train_resampled, y_train_resampled)
                train_time = (datetime.now() - start).total_seconds()

                # Predict
                y_pred = model.predict(X_val_scaled)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_val_scaled)
                    except Exception:
                        pass

                # Compute metrics for this fold
                fold_metrics = compute_fold_metrics(y_val_fold, y_pred, y_pred_proba, class_names)
                fold_metrics['training_time_seconds'] = train_time
                fold_metrics_list.append(fold_metrics)

                print(
                    f"    Fold {fold_idx + 1}/{self.config.K_FOLDS}: "
                    f"Acc={fold_metrics['accuracy']:.4f}  "
                    f"F1w={fold_metrics['f1_weighted']:.4f}  "
                    f"F1m={fold_metrics['f1_macro']:.4f}  "
                    f"MCC={fold_metrics['matthews_corrcoef']:.4f}"
                )

            # ---- Aggregate across folds (mean ± std) ----
            all_keys = sorted(set().union(*(fm.keys() for fm in fold_metrics_list)))
            aggregated = {}
            for key in all_keys:
                vals = [
                    fm[key] for fm in fold_metrics_list
                    if key in fm and isinstance(fm[key], (int, float))
                    and np.isfinite(fm[key])
                ]
                if vals:
                    aggregated[f'mean_{key}'] = float(np.mean(vals))
                    aggregated[f'std_{key}'] = float(np.std(vals))

            # Log aggregated metrics to MLflow
            for mname, mval in aggregated.items():
                if np.isfinite(mval):
                    mlflow.log_metric(mname, mval)

            # Save fold details as artifact
            detail = {
                'oversampling': oversampling_name,
                'model': model_name,
                'k_folds': self.config.K_FOLDS,
                'fold_metrics': fold_metrics_list,
                'aggregated': aggregated,
            }
            detail_path = f"/tmp/{oversampling_name}_{model_name}_fold_details.json"
            with open(detail_path, 'w') as f:
                json.dump(detail, f, indent=2, default=str)
            mlflow.log_artifact(detail_path)

            run_id = mlflow.active_run().info.run_id

        # Print summary for this combination
        g = aggregated.get
        print(
            f"    --- Mean +/- Std ---\n"
            f"    Accuracy:      {g('mean_accuracy', 0):.4f} +/- {g('std_accuracy', 0):.4f}\n"
            f"    F1 (weighted): {g('mean_f1_weighted', 0):.4f} +/- {g('std_f1_weighted', 0):.4f}\n"
            f"    F1 (macro):    {g('mean_f1_macro', 0):.4f} +/- {g('std_f1_macro', 0):.4f}\n"
            f"    MCC:           {g('mean_matthews_corrcoef', 0):.4f} +/- {g('std_matthews_corrcoef', 0):.4f}\n"
            f"    Run ID: {run_id}"
        )

        return {
            'oversampling': oversampling_name,
            'model': model_name,
            'aggregated': aggregated,
            'fold_metrics': fold_metrics_list,
            'run_id': run_id,
        }

    # ------------------------------------------------------------------ #
    # Summary DataFrame
    # ------------------------------------------------------------------ #

    def build_summary_dataframe(self, all_results: List[Dict]) -> pd.DataFrame:
        """Build a comparison DataFrame from all combination results."""
        rows = []
        for res in all_results:
            agg = res['aggregated']
            rows.append({
                'Oversampling': res['oversampling'],
                'Model': res['model'],
                'Mean_Accuracy': agg.get('mean_accuracy', 0),
                'Std_Accuracy': agg.get('std_accuracy', 0),
                'Mean_Balanced_Accuracy': agg.get('mean_balanced_accuracy', 0),
                'Std_Balanced_Accuracy': agg.get('std_balanced_accuracy', 0),
                'Mean_F1_Weighted': agg.get('mean_f1_weighted', 0),
                'Std_F1_Weighted': agg.get('std_f1_weighted', 0),
                'Mean_F1_Macro': agg.get('mean_f1_macro', 0),
                'Std_F1_Macro': agg.get('std_f1_macro', 0),
                'Mean_Precision_Weighted': agg.get('mean_precision_weighted', 0),
                'Std_Precision_Weighted': agg.get('std_precision_weighted', 0),
                'Mean_Recall_Weighted': agg.get('mean_recall_weighted', 0),
                'Std_Recall_Weighted': agg.get('std_recall_weighted', 0),
                'Mean_Matthews_Corrcoef': agg.get('mean_matthews_corrcoef', 0),
                'Std_Matthews_Corrcoef': agg.get('std_matthews_corrcoef', 0),
                'Mean_Cohen_Kappa': agg.get('mean_cohen_kappa', 0),
                'Std_Cohen_Kappa': agg.get('std_cohen_kappa', 0),
                'Mean_Training_Time': agg.get('mean_training_time_seconds', 0),
                'Run_ID': res['run_id'],
            })
        df = pd.DataFrame(rows).sort_values('Mean_F1_Weighted', ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # LaTeX table generation
    # ------------------------------------------------------------------ #

    def generate_latex_table(self, df: pd.DataFrame) -> str:
        """
        Generate a LaTeX table comparing all oversampling x classifier
        combinations. Best value per metric column is bolded.
        """
        metrics_config = [
            ('Mean_Accuracy',          'Std_Accuracy',          'Accuracy'),
            ('Mean_Balanced_Accuracy', 'Std_Balanced_Accuracy', 'Balanced Acc.'),
            ('Mean_F1_Weighted',       'Std_F1_Weighted',       'F1 (Weighted)'),
            ('Mean_F1_Macro',          'Std_F1_Macro',          'F1 (Macro)'),
            ('Mean_Matthews_Corrcoef', 'Std_Matthews_Corrcoef', 'MCC'),
            ('Mean_Cohen_Kappa',       'Std_Cohen_Kappa',       "Cohen's $\\kappa$"),
        ]

        # Identify best (highest) index per metric
        best_per_metric = {m[0]: df[m[0]].idxmax() for m in metrics_config}

        col_spec = 'll' + 'c' * len(metrics_config)

        lines = [
            '% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}',
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Performance of oversampling--classifier combinations '
            r'evaluated via Stratified 3-Fold Cross-Validation (mean $\pm$ std). '
            r'Best values per metric are \textbf{bolded}.}',
            r'\label{tab:kfold_cv_results}',
            r'\resizebox{\textwidth}{!}{%',
            '\\begin{tabular}{' + col_spec + '}',
            r'\toprule',
        ]

        # Header row
        header = 'Oversampling & Classifier'
        for _, _, label in metrics_config:
            header += f' & {label}'
        header += r' \\'
        lines.append(header)
        lines.append(r'\midrule')

        # Data rows
        for idx, row in df.iterrows():
            os_display = row['Oversampling'].replace('_', '-')
            model_display = row['Model'].replace('_', ' ')
            cells = [os_display, model_display]

            for mean_col, std_col, _ in metrics_config:
                mean_val = row[mean_col]
                std_val = row[std_col]
                if idx == best_per_metric[mean_col]:
                    cell = f'$\\mathbf{{{mean_val:.4f} \\pm {std_val:.4f}}}$'
                else:
                    cell = f'${mean_val:.4f} \\pm {std_val:.4f}$'
                cells.append(cell)

            lines.append(' & '.join(cells) + r' \\')

        lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}',
        ])

        return '\n'.join(lines)

    def generate_per_class_latex(self, best_result: Dict, class_names: List[str]) -> str:
        """Generate per-class metrics LaTeX table for the best combination."""
        agg = best_result['aggregated']

        lines = [
            '% Requires: \\usepackage{booktabs}',
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{Per-class performance of the best combination '
            f'({best_result["oversampling"].replace("_", "-")} + '
            f'{best_result["model"].replace("_", " ")}) '
            f'via Stratified 3-Fold CV (mean $\\pm$ std).}}',
            r'\label{tab:kfold_per_class}',
            r'\begin{tabular}{lccc}',
            r'\toprule',
            r'Class & Precision & Recall & F1-Score \\',
            r'\midrule',
        ]

        for cn in class_names:
            safe = cn.replace(' ', '_')
            p_m = agg.get(f'mean_precision_{safe}', 0)
            p_s = agg.get(f'std_precision_{safe}', 0)
            r_m = agg.get(f'mean_recall_{safe}', 0)
            r_s = agg.get(f'std_recall_{safe}', 0)
            f_m = agg.get(f'mean_f1_{safe}', 0)
            f_s = agg.get(f'std_f1_{safe}', 0)

            lines.append(
                f'{cn} & ${p_m:.4f} \\pm {p_s:.4f}$ '
                f'& ${r_m:.4f} \\pm {r_s:.4f}$ '
                f'& ${f_m:.4f} \\pm {f_s:.4f}$ \\\\'
            )

        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ])

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Save artifacts
    # ------------------------------------------------------------------ #

    def save_preprocessor_info(self, output_dir: str = "mlflow_artifacts_no_fe"):
        """Save label encoder and feature name metadata."""
        os.makedirs(output_dir, exist_ok=True)

        encoder_path = os.path.join(output_dir, "label_encoder_no_fe.pkl")
        joblib.dump(self.preprocessor.label_encoder, encoder_path)

        features_path = os.path.join(output_dir, "feature_names_no_fe.json")
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.preprocessor.feature_names,
                'n_features': len(self.preprocessor.feature_names),
                'target_classes': list(self.preprocessor.label_encoder.classes_),
                'feature_engineering': False,
                'evaluation_method': f'Stratified_{self.config.K_FOLDS}-Fold_CV',
            }, f, indent=2)

        print(f"\nPreprocessor info saved to: {output_dir}")

    # ------------------------------------------------------------------ #
    # Main execution
    # ------------------------------------------------------------------ #

    def run(self):
        """Main pipeline execution."""
        print("\n" + "=" * 70)
        print("WSN IDS - STRATIFIED K-FOLD CV (NO FEATURE ENGINEERING)")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"K-Folds: {self.config.K_FOLDS}")

        # MLflow setup
        self.setup_mlflow()
        experiment_id = self.create_or_get_experiment(self.config.EXPERIMENT_NAME)

        # Load & prepare data (no train/test split — K-Fold handles it)
        df = self.preprocessor.load_data()
        df = self.preprocessor.clean_data(df)
        X, y = self.preprocessor.prepare_data(df)

        class_names = list(self.preprocessor.label_encoder.classes_)

        print(f"\nFeatures ({len(self.preprocessor.feature_names)}): {self.preprocessor.feature_names}")
        print(f"Classes ({len(class_names)}): {class_names}")
        print("\nClass distribution:")
        for cn, cnt in self.preprocessor.get_class_distribution(y).items():
            print(f"  {cn}: {cnt:,}")

        # Stratified K-Fold splitter (created once, reused for all combinations)
        skf = StratifiedKFold(
            n_splits=self.config.K_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE,
        )

        # Models & oversampling strategies
        models = get_models(self.config.RANDOM_STATE)
        strategies = self.oversampling.get_strategies()

        # ---- Evaluate ALL oversampling x classifier combinations ----
        all_results = []
        total = len(strategies) * len(models)
        combo_idx = 0

        for os_name, os_func in strategies.items():
            for model_name, model_template in models.items():
                combo_idx += 1
                print(f"\n{'─' * 60}")
                print(f"Combination {combo_idx}/{total}")

                result = self.evaluate_combination(
                    X=X, y=y,
                    model_name=model_name,
                    model_template=model_template,
                    oversampling_name=os_name,
                    oversampling_func=os_func,
                    skf=skf,
                    class_names=class_names,
                    experiment_id=experiment_id,
                )
                all_results.append(result)

        # ---- Build summary ----
        summary_df = self.build_summary_dataframe(all_results)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY (sorted by Mean F1 Weighted)")
        print("=" * 70)
        display_cols = [
            'Oversampling', 'Model',
            'Mean_Accuracy', 'Std_Accuracy',
            'Mean_F1_Weighted', 'Std_F1_Weighted',
            'Mean_F1_Macro', 'Std_F1_Macro',
            'Mean_Matthews_Corrcoef', 'Std_Matthews_Corrcoef',
        ]
        print(summary_df[display_cols].to_string(index=False))

        # Best combination
        best = summary_df.iloc[0]
        print(f"\n{'=' * 70}")
        print("BEST COMBINATION")
        print(f"{'=' * 70}")
        print(f"  Oversampling : {best['Oversampling']}")
        print(f"  Classifier   : {best['Model']}")
        print(f"  Accuracy     : {best['Mean_Accuracy']:.4f} +/- {best['Std_Accuracy']:.4f}")
        print(f"  F1 (Weighted): {best['Mean_F1_Weighted']:.4f} +/- {best['Std_F1_Weighted']:.4f}")
        print(f"  F1 (Macro)   : {best['Mean_F1_Macro']:.4f} +/- {best['Std_F1_Macro']:.4f}")
        print(f"  MCC          : {best['Mean_Matthews_Corrcoef']:.4f} +/- {best['Std_Matthews_Corrcoef']:.4f}")

        # ---- Save CSV ----
        csv_path = "experiment_summary_no_fe.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to: {csv_path}")

        # ---- Generate & save LaTeX tables ----
        latex_table = self.generate_latex_table(summary_df)
        latex_path = "kfold_cv_results_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")
        print("\n--- LaTeX Table (All Combinations) ---")
        print(latex_table)

        # Per-class table for the best combination
        best_result = next(
            r for r in all_results
            if r['oversampling'] == best['Oversampling'] and r['model'] == best['Model']
        )
        per_class_latex = self.generate_per_class_latex(best_result, class_names)
        per_class_path = "kfold_cv_per_class_table.tex"
        with open(per_class_path, 'w') as f:
            f.write(per_class_latex)
        print(f"\nPer-class LaTeX table saved to: {per_class_path}")
        print("\n--- Per-Class LaTeX Table (Best Combination) ---")
        print(per_class_latex)

        # Save preprocessor info
        self.save_preprocessor_info()

        print(f"\n{'=' * 70}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 70}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTo view results in MLflow UI, run:")
        print(f"  mlflow ui --backend-store-uri {self.config.MLFLOW_TRACKING_URI}")
        print(f"\nThen open http://localhost:5000 in your browser")

        return {
            'all_results': all_results,
            'summary': summary_df,
            'latex_table': latex_table,
            'per_class_latex': per_class_latex,
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    config = Config()
    pipeline = KFoldCVPipeline(config)
    results = pipeline.run()

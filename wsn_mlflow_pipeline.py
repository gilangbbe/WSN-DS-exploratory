"""
WSN Intrusion Detection System - MLflow Training Pipeline
=========================================================
This module implements a comprehensive ML training pipeline for WSN intrusion detection
with MLflow integration for experiment tracking, model versioning, and registry.

Two experiments are conducted:
1. Without Oversampling - Using class-weighted algorithms and specialized ensemble methods
2. With Oversampling - Using SMOTE variants (SMOTE-ENN, BorderlineSMOTE, ADASYN, Conservative SMOTE)

Author: Machine Learning Engineering Team
Date: 2026-01-29
"""

import os
import sys
import warnings
from datetime import datetime
from collections import Counter
from typing import Dict, Any, Tuple, List, Optional

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Core Data Science Libraries
import pandas as pd
import numpy as np
from scipy import stats

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Learning Libraries
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier

# MLflow for experiment tracking and model registry
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Serialization
import joblib
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the training pipeline"""
    
    # Paths
    DATA_PATH = "data/WSN-DS.csv"
    MLFLOW_TRACKING_URI = "mlruns"  # Local MLflow tracking
    EXPERIMENT_NAME_NO_OVERSAMPLING = "WSN_Intrusion_Detection_No_Oversampling"
    EXPERIMENT_NAME_WITH_OVERSAMPLING = "WSN_Intrusion_Detection_With_Oversampling"
    
    # Data split parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Cross-validation
    CV_FOLDS = 5
    
    # Model versioning
    MODEL_VERSION = "1.0.0"
    
    # Redundant features to remove
    REDUNDANT_FEATURES = ['id', 'who CH']


# ============================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for WSN intrusion detection.
    Follows the same preprocessing steps as the original notebook.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the WSN-DS dataset"""
        print("Loading WSN-DS dataset...")
        df = pd.read_csv(self.config.DATA_PATH)
        df.columns = df.columns.str.strip()
        print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        print("Cleaning data...")
        original_size = len(df)
        df_clean = df.drop_duplicates()
        removed = original_size - len(df_clean)
        print(f"Removed {removed:,} duplicate records")
        print(f"Clean dataset: {len(df_clean):,} samples")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        Creates WSN-specific features based on domain knowledge.
        """
        print("Applying feature engineering...")
        df_features = df.copy()
        
        # Remove redundant features
        df_features = df_features.drop(columns=self.config.REDUNDANT_FEATURES)
        
        # Distance-based features
        df_features['Distance_Efficiency'] = df_features['Dist_To_CH'] / (df_features['dist_CH_To_BS'] + 1e-8)
        df_features['Distance_Ratio'] = df_features['Dist_To_CH'] / (df_features['Dist_To_CH'].max() + 1e-8)
        
        # Energy consumption features
        df_features['Energy_Per_Data'] = df_features['Expaned Energy'] / (df_features['DATA_S'] + 1e-8)
        df_features['Energy_Efficiency'] = df_features['DATA_S'] / (df_features['Expaned Energy'] + 1e-8)
        df_features['Energy_Rank_Ratio'] = df_features['Expaned Energy'] / (df_features['Rank'] + 1e-8)
        
        # Communication pattern features
        df_features['Total_Messages_Sent'] = (
            df_features['ADV_S'] + df_features['JOIN_S'] + 
            df_features['SCH_S'] + df_features['DATA_S']
        )
        df_features['Total_Messages_Received'] = (
            df_features['ADV_R'] + df_features['JOIN_R'] + 
            df_features['SCH_R'] + df_features['DATA_R']
        )
        df_features['Message_Balance'] = df_features['Total_Messages_Sent'] - df_features['Total_Messages_Received']
        df_features['Communication_Ratio'] = df_features['Total_Messages_Sent'] / (df_features['Total_Messages_Received'] + 1e-8)
        
        # Temporal features
        df_features['Time_Normalized'] = df_features['Time'] / df_features['Time'].max()
        df_features['Time_Category'] = pd.cut(
            df_features['Time'], bins=5, 
            labels=['Early', 'Early_Mid', 'Mid', 'Late_Mid', 'Late']
        )
        time_mapping = {'Early': 0, 'Early_Mid': 1, 'Mid': 2, 'Late_Mid': 3, 'Late': 4}
        df_features['Time_Category_Num'] = df_features['Time_Category'].map(time_mapping)
        
        # Network role interaction features
        df_features['CH_Distance_Product'] = df_features['Is_CH'] * df_features['Dist_To_CH']
        df_features['CH_Energy_Product'] = df_features['Is_CH'] * df_features['Expaned Energy']
        df_features['NonCH_Rank_Product'] = (1 - df_features['Is_CH']) * df_features['Rank']
        
        # Drop categorical Time_Category
        df_features = df_features.drop(columns=['Time_Category'])
        
        print(f"Feature engineering complete: {df_features.shape[1] - 1} features")
        return df_features
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning:
        - Separate features and target
        - Encode target variable
        - Handle invalid values
        - Split into train/test
        - Scale features
        """
        print("Preparing data for machine learning...")
        
        # Separate features and target
        X = df.drop('Attack type', axis=1)
        y = df['Attack type']
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle numeric conversion and invalid values
        X_numeric = X.select_dtypes(include=[np.number])
        for col in X.columns:
            if col not in X_numeric.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove rows with inf/nan
        X_values = X.values
        valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution with class names"""
        counter = Counter(y)
        return {self.label_encoder.classes_[k]: v for k, v in counter.items()}


# ============================================================================
# OVERSAMPLING STRATEGIES
# ============================================================================

class OversamplingStrategies:
    """
    Implements various oversampling strategies for handling class imbalance.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def apply_smote_enn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE-ENN: Combines SMOTE with Edited Nearest Neighbors"""
        print("Applying SMOTE-ENN...")
        sampler = SMOTEENN(random_state=self.random_state, sampling_strategy='auto')
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"SMOTE-ENN: {len(y)} -> {len(y_resampled)} samples")
        return X_resampled, y_resampled
    
    def apply_borderline_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BorderlineSMOTE: Focuses on samples near decision boundary"""
        print("Applying BorderlineSMOTE...")
        sampler = BorderlineSMOTE(random_state=self.random_state, sampling_strategy='auto')
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"BorderlineSMOTE: {len(y)} -> {len(y_resampled)} samples")
        return X_resampled, y_resampled
    
    def apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ADASYN: Adaptive Synthetic Sampling"""
        print("Applying ADASYN...")
        sampler = ADASYN(random_state=self.random_state, sampling_strategy='auto')
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"ADASYN: {len(y)} -> {len(y_resampled)} samples")
        return X_resampled, y_resampled
    
    def apply_conservative_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conservative SMOTE: Novel approach limiting minority augmentation to 20% of majority.
        Prevents synthetic data dominance while addressing imbalance.
        """
        print("Applying Conservative SMOTE...")
        counts = Counter(y)
        majority_count = max(counts.values())
        target_count = int(majority_count * 0.2)
        
        conservative_strategy = {}
        for class_label, count in counts.items():
            if count < target_count:
                conservative_strategy[class_label] = target_count
        
        if conservative_strategy:
            sampler = SMOTE(random_state=self.random_state, sampling_strategy=conservative_strategy)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Conservative SMOTE: {len(y)} -> {len(y_resampled)} samples")
        return X_resampled, y_resampled


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_standard_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns dictionary of standard ML models with configurations.
    These are the same models used in the original notebook.
    """
    return {
        'Random_Forest': {
            'model': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=random_state, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'description': 'Random Forest with balanced class weights'
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=random_state
            ),
            'description': 'Gradient Boosting Classifier'
        },
        'HistGradient_Boosting': {
            'model': HistGradientBoostingClassifier(
                max_iter=100, 
                max_depth=15, 
                random_state=random_state
            ),
            'description': 'Histogram-based Gradient Boosting'
        },
        'Extra_Trees': {
            'model': ExtraTreesClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=random_state, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'description': 'Extra Trees with balanced class weights'
        },
        'Neural_Network': {
            'model': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=random_state, 
                early_stopping=True
            ),
            'description': 'Multi-layer Perceptron Neural Network'
        },
        'Logistic_Regression': {
            'model': LogisticRegression(
                max_iter=1000, 
                random_state=random_state, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'description': 'Logistic Regression with balanced class weights'
        }
    }


def get_ensemble_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns dictionary of specialized ensemble models for imbalanced data.
    These models are designed to work with imbalanced datasets without oversampling.
    """
    return {
        'Balanced_Random_Forest': {
            'model': BalancedRandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=random_state, 
                n_jobs=-1
            ),
            'description': 'Balanced Random Forest for imbalanced data'
        },
        'Balanced_Bagging': {
            'model': BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=15),
                n_estimators=50, 
                random_state=random_state, 
                n_jobs=-1
            ),
            'description': 'Balanced Bagging Classifier'
        }
    }


# ============================================================================
# MLFLOW TRAINING PIPELINE
# ============================================================================

class MLflowTrainingPipeline:
    """
    Main training pipeline with MLflow integration for experiment tracking
    and model registry.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.oversampling = OversamplingStrategies(config.RANDOM_STATE)
        self.client = None
        
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        print("\n" + "=" * 60)
        print("Setting up MLflow...")
        print("=" * 60)
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    def create_or_get_experiment(self, experiment_name: str) -> str:
        """Create or get existing experiment"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Add log loss if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                pass
        
        # Add per-class metrics
        class_names = self.preprocessor.label_encoder.classes_
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(class_names):
            safe_name = class_name.replace(' ', '_')
            metrics[f'precision_{safe_name}'] = precision_per_class[i]
            metrics[f'recall_{safe_name}'] = recall_per_class[i]
            metrics[f'f1_{safe_name}'] = f1_per_class[i]
        
        return metrics
    
    def train_single_model(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_id: str,
        sampling_strategy: str = "None",
        description: str = "",
        tags: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Train a single model and log everything to MLflow.
        """
        print(f"\n  Training {model_name}...")
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_{sampling_strategy}"):
            # Log tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("sampling_strategy", sampling_strategy)
            mlflow.set_tag("description", description)
            mlflow.set_tag("version", self.config.MODEL_VERSION)
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Log parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Log only serializable parameters
                for key, value in params.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        mlflow.log_param(key, value)
                    else:
                        mlflow.log_param(key, str(value))
            
            mlflow.log_param("training_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(self.preprocessor.label_encoder.classes_))
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            
            # Compute metrics
            metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
            metrics['training_time_seconds'] = training_time
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log classification report as artifact
            class_report = classification_report(
                y_test, y_pred, 
                target_names=self.preprocessor.label_encoder.classes_,
                output_dict=True
            )
            report_path = f"/tmp/{model_name}_{sampling_strategy}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            mlflow.log_artifact(report_path)
            
            # Log confusion matrix as artifact
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=self.preprocessor.label_encoder.classes_,
                columns=self.preprocessor.label_encoder.classes_
            )
            cm_path = f"/tmp/{model_name}_{sampling_strategy}_confusion_matrix.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(cm_path)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.preprocessor.feature_names,
                    model.feature_importances_
                ))
                fi_path = f"/tmp/{model_name}_{sampling_strategy}_feature_importance.json"
                with open(fi_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(fi_path)
            
            # Create model signature
            signature = infer_signature(X_train[:5], y_pred[:5])
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"WSN_IDS_{model_name}_{sampling_strategy}",
                input_example=X_train[:1]
            )
            
            # Get run ID for reference
            run_id = mlflow.active_run().info.run_id
            
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
            print(f"    Training time: {training_time:.2f}s")
            print(f"    Run ID: {run_id}")
            
            return {
                'model': model,
                'metrics': metrics,
                'run_id': run_id,
                'classification_report': class_report
            }
    
    def run_experiment_no_oversampling(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run experiment WITHOUT oversampling strategies.
        Uses class-weighted algorithms and specialized ensemble methods.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: WITHOUT OVERSAMPLING")
        print("=" * 60)
        print("Using class-weighted models and specialized ensemble methods")
        
        experiment_id = self.create_or_get_experiment(
            self.config.EXPERIMENT_NAME_NO_OVERSAMPLING
        )
        
        results = {}
        
        # Train standard models with class weights
        print("\n--- Standard Models (Class-Weighted) ---")
        standard_models = get_standard_models(self.config.RANDOM_STATE)
        
        for model_name, model_config in standard_models.items():
            result = self.train_single_model(
                model=model_config['model'],
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experiment_id=experiment_id,
                sampling_strategy="No_Oversampling",
                description=model_config['description'],
                tags={"experiment_type": "no_oversampling", "model_category": "standard"}
            )
            results[f"{model_name}_No_Oversampling"] = result
        
        # Train specialized ensemble models
        print("\n--- Specialized Ensemble Models ---")
        ensemble_models = get_ensemble_models(self.config.RANDOM_STATE)
        
        for model_name, model_config in ensemble_models.items():
            result = self.train_single_model(
                model=model_config['model'],
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experiment_id=experiment_id,
                sampling_strategy="Ensemble_Imbalanced",
                description=model_config['description'],
                tags={"experiment_type": "no_oversampling", "model_category": "ensemble"}
            )
            results[f"{model_name}_Ensemble"] = result
        
        return results
    
    def run_experiment_with_oversampling(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run experiment WITH various oversampling strategies.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: WITH OVERSAMPLING STRATEGIES")
        print("=" * 60)
        
        experiment_id = self.create_or_get_experiment(
            self.config.EXPERIMENT_NAME_WITH_OVERSAMPLING
        )
        
        # Define oversampling strategies
        oversampling_strategies = {
            'SMOTE_ENN': self.oversampling.apply_smote_enn,
            'BorderlineSMOTE': self.oversampling.apply_borderline_smote,
            'ADASYN': self.oversampling.apply_adasyn,
            'Conservative_SMOTE': self.oversampling.apply_conservative_smote
        }
        
        results = {}
        standard_models = get_standard_models(self.config.RANDOM_STATE)
        
        for strategy_name, strategy_func in oversampling_strategies.items():
            print(f"\n--- {strategy_name} ---")
            
            # Apply oversampling
            try:
                X_resampled, y_resampled = strategy_func(X_train, y_train)
            except Exception as e:
                print(f"  Warning: {strategy_name} failed: {e}")
                continue
            
            # Train all models with this strategy
            for model_name, model_config in standard_models.items():
                # Create a fresh model instance
                model_class = type(model_config['model'])
                model_params = model_config['model'].get_params()
                model = model_class(**model_params)
                
                result = self.train_single_model(
                    model=model,
                    model_name=model_name,
                    X_train=X_resampled,
                    X_test=X_test,
                    y_train=y_resampled,
                    y_test=y_test,
                    experiment_id=experiment_id,
                    sampling_strategy=strategy_name,
                    description=f"{model_config['description']} with {strategy_name}",
                    tags={
                        "experiment_type": "with_oversampling",
                        "oversampling_strategy": strategy_name,
                        "resampled_size": str(len(y_resampled))
                    }
                )
                results[f"{model_name}_{strategy_name}"] = result
        
        return results
    
    def save_preprocessors(self, output_dir: str = "mlflow_artifacts"):
        """Save preprocessing components for deployment"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, "standard_scaler.pkl")
        joblib.dump(self.preprocessor.scaler, scaler_path)
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "label_encoder.pkl")
        joblib.dump(self.preprocessor.label_encoder, encoder_path)
        
        # Save feature names
        features_path = os.path.join(output_dir, "feature_names.json")
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.preprocessor.feature_names,
                'n_features': len(self.preprocessor.feature_names),
                'target_classes': list(self.preprocessor.label_encoder.classes_)
            }, f, indent=2)
        
        print(f"\nPreprocessors saved to: {output_dir}")
    
    def generate_summary_report(
        self,
        results_no_oversampling: Dict[str, Any],
        results_with_oversampling: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate summary report of all experiments"""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY REPORT")
        print("=" * 60)
        
        all_results = []
        
        # Process no-oversampling results
        for model_key, result in results_no_oversampling.items():
            all_results.append({
                'Model': model_key,
                'Experiment': 'No Oversampling',
                'Accuracy': result['metrics']['accuracy'],
                'F1_Weighted': result['metrics']['f1_weighted'],
                'Precision_Weighted': result['metrics']['precision_weighted'],
                'Recall_Weighted': result['metrics']['recall_weighted'],
                'Training_Time': result['metrics']['training_time_seconds'],
                'Run_ID': result['run_id']
            })
        
        # Process with-oversampling results
        for model_key, result in results_with_oversampling.items():
            parts = model_key.rsplit('_', 1)
            strategy = parts[-1] if len(parts) > 1 else 'Unknown'
            all_results.append({
                'Model': model_key,
                'Experiment': 'With Oversampling',
                'Accuracy': result['metrics']['accuracy'],
                'F1_Weighted': result['metrics']['f1_weighted'],
                'Precision_Weighted': result['metrics']['precision_weighted'],
                'Recall_Weighted': result['metrics']['recall_weighted'],
                'Training_Time': result['metrics']['training_time_seconds'],
                'Run_ID': result['run_id']
            })
        
        # Create DataFrame and sort by F1 score
        df = pd.DataFrame(all_results)
        df = df.sort_values('F1_Weighted', ascending=False)
        
        # Display top models
        print("\nTop 10 Models by F1 Score:")
        print(df.head(10)[['Model', 'Experiment', 'Accuracy', 'F1_Weighted']].to_string(index=False))
        
        # Best model per experiment
        print("\nBest Model per Experiment:")
        for exp in df['Experiment'].unique():
            exp_best = df[df['Experiment'] == exp].iloc[0]
            print(f"  {exp}:")
            print(f"    Model: {exp_best['Model']}")
            print(f"    F1 Score: {exp_best['F1_Weighted']:.4f}")
            print(f"    Accuracy: {exp_best['Accuracy']:.4f}")
        
        # Save summary
        summary_path = "experiment_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nFull summary saved to: {summary_path}")
        
        return df
    
    def run(self):
        """Main execution method"""
        print("\n" + "=" * 60)
        print("WSN INTRUSION DETECTION - MLFLOW TRAINING PIPELINE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Load and preprocess data
        df = self.preprocessor.load_data()
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.engineer_features(df)
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data(df)
        
        # Log class distribution
        print("\nClass Distribution:")
        for class_name, count in self.preprocessor.get_class_distribution(y_train).items():
            print(f"  {class_name}: {count:,}")
        
        # Run experiments
        results_no_oversampling = self.run_experiment_no_oversampling(
            X_train, X_test, y_train, y_test
        )
        
        results_with_oversampling = self.run_experiment_with_oversampling(
            X_train, X_test, y_train, y_test
        )
        
        # Save preprocessors
        self.save_preprocessors()
        
        # Generate summary
        summary_df = self.generate_summary_report(
            results_no_oversampling,
            results_with_oversampling
        )
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTo view results in MLflow UI, run:")
        print(f"  mlflow ui --backend-store-uri {self.config.MLFLOW_TRACKING_URI}")
        print(f"\nThen open http://localhost:5000 in your browser")
        
        return {
            'no_oversampling': results_no_oversampling,
            'with_oversampling': results_with_oversampling,
            'summary': summary_df
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Create and run pipeline
    pipeline = MLflowTrainingPipeline(config)
    results = pipeline.run()

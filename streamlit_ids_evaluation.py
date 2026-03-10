"""
WSN Intrusion Detection System - Comparative Evaluation Dashboard
================================================================
A Streamlit application for comparing IDS models trained with and without
oversampling techniques, featuring comprehensive performance metrics and
distribution preservation analysis.

This application supports academic research by providing:
1. IDS-specific evaluation metrics (beyond accuracy)
2. Imbalance and distribution preservation analysis
3. Publication-ready visualizations

Author: ML Research Team
Date: 2026-01-30
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score
)
import mlflow
from mlflow.tracking import MlflowClient
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="WSN IDS Evaluation Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E40AF;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .interpretation-box {
        background-color: #FEF3C7;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING AND MLFLOW INTEGRATION
# ============================================================================

@st.cache_data
def load_mlflow_experiments():
    """Load experiment data from MLflow."""
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()
    
    experiments = {}
    
    # Get all experiments
    all_experiments = client.search_experiments()
    
    for exp in all_experiments:
        if "No_Feature_Engineering" in exp.name:
            experiments[exp.name] = {
                'id': exp.experiment_id,
                'name': exp.name
            }
    
    return experiments, client


@st.cache_data
def get_experiment_runs(_client, experiment_id):
    """Get all runs from an experiment."""
    runs = _client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_weighted DESC"]
    )
    return runs


@st.cache_data
def load_best_models_data():
    """
    Load data for the two best models:
    - Model A: Best model without oversampling
    - Model B: Best model with oversampling
    """
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()
    
    # Find experiments
    exp_no_os = client.get_experiment_by_name("WSN_IDS_No_Feature_Engineering_No_Oversampling")
    exp_with_os = client.get_experiment_by_name("WSN_IDS_No_Feature_Engineering_With_Oversampling")
    
    if not exp_no_os or not exp_with_os:
        return None, None
    
    # Get best run from each experiment
    runs_no_os = client.search_runs(
        experiment_ids=[exp_no_os.experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=1
    )
    
    runs_with_os = client.search_runs(
        experiment_ids=[exp_with_os.experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=1
    )
    
    model_a = runs_no_os[0] if runs_no_os else None
    model_b = runs_with_os[0] if runs_with_os else None
    
    return model_a, model_b


def get_run_artifacts(client, run_id):
    """Load artifacts from a run."""
    artifacts = {}
    try:
        artifact_path = client.download_artifacts(run_id, "")
        
        # Load classification report
        report_files = [f for f in os.listdir(artifact_path) if 'classification_report' in f]
        if report_files:
            with open(os.path.join(artifact_path, report_files[0]), 'r') as f:
                artifacts['classification_report'] = json.load(f)
        
        # Load confusion matrix
        cm_files = [f for f in os.listdir(artifact_path) if 'confusion_matrix' in f]
        if cm_files:
            artifacts['confusion_matrix'] = pd.read_csv(
                os.path.join(artifact_path, cm_files[0]), index_col=0
            )
        
        # Load feature importance
        fi_files = [f for f in os.listdir(artifact_path) if 'feature_importance' in f]
        if fi_files:
            with open(os.path.join(artifact_path, fi_files[0]), 'r') as f:
                artifacts['feature_importance'] = json.load(f)
                
    except Exception as e:
        st.warning(f"Could not load some artifacts: {e}")
    
    return artifacts


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_gmean(y_true, y_pred):
    """Compute Geometric Mean of class-wise recalls."""
    cm = confusion_matrix(y_true, y_pred)
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    recalls = recalls[recalls > 0]  # Avoid log(0)
    return np.exp(np.mean(np.log(recalls))) if len(recalls) > 0 else 0


def compute_ks_test(original_data, oversampled_data, feature_names):
    """
    Perform Kolmogorov-Smirnov test between original and oversampled distributions.
    """
    ks_results = {}
    
    for i, feature in enumerate(feature_names):
        if i < original_data.shape[1] and i < oversampled_data.shape[1]:
            stat, pvalue = stats.ks_2samp(original_data[:, i], oversampled_data[:, i])
            ks_results[feature] = {
                'statistic': stat,
                'pvalue': pvalue,
                'significant': pvalue < 0.05
            }
    
    return ks_results


def compute_jensen_shannon_divergence(original_data, oversampled_data, n_bins=50):
    """
    Compute Jensen-Shannon Divergence between distributions.
    """
    jsd_results = {}
    
    for i in range(min(original_data.shape[1], oversampled_data.shape[1])):
        # Create histograms with same bins
        combined = np.concatenate([original_data[:, i], oversampled_data[:, i]])
        bins = np.linspace(combined.min(), combined.max(), n_bins)
        
        hist_orig, _ = np.histogram(original_data[:, i], bins=bins, density=True)
        hist_over, _ = np.histogram(oversampled_data[:, i], bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        hist_orig = hist_orig + 1e-10
        hist_over = hist_over + 1e-10
        
        # Normalize
        hist_orig = hist_orig / hist_orig.sum()
        hist_over = hist_over / hist_over.sum()
        
        # Compute JSD
        jsd = jensenshannon(hist_orig, hist_over)
        jsd_results[i] = jsd if not np.isnan(jsd) else 0
    
    return jsd_results


def compute_correlation_preservation(original_data, oversampled_data):
    """
    Analyze correlation preservation between original and oversampled data.
    """
    # Compute correlation matrices
    corr_orig = np.corrcoef(original_data.T)
    corr_over = np.corrcoef(oversampled_data.T)
    
    # Handle NaN values
    corr_orig = np.nan_to_num(corr_orig)
    corr_over = np.nan_to_num(corr_over)
    
    # Mean Absolute Correlation Difference
    mae_corr = np.mean(np.abs(corr_orig - corr_over))
    
    # Frobenius Norm
    frobenius = np.linalg.norm(corr_orig - corr_over, 'fro')
    
    # Correlation of correlations
    upper_tri_idx = np.triu_indices(corr_orig.shape[0], k=1)
    corr_correlation = np.corrcoef(
        corr_orig[upper_tri_idx].flatten(),
        corr_over[upper_tri_idx].flatten()
    )[0, 1]
    
    return {
        'original_matrix': corr_orig,
        'oversampled_matrix': corr_over,
        'mae': mae_corr,
        'frobenius': frobenius,
        'correlation': corr_correlation if not np.isnan(corr_correlation) else 0
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_performance_comparison(metrics_a, metrics_b, model_a_name, model_b_name):
    """
    Create grouped bar chart comparing Model A vs Model B performance metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']
    labels = ['Macro Precision', 'Macro Recall', 'Macro F1', 'MCC']
    
    values_a = [metrics_a.get(m, 0) for m in metrics]
    values_b = [metrics_b.get(m, 0) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars_a = ax.bar(x - width/2, values_a, width, label=model_a_name, 
                    color='#3B82F6', alpha=0.8, edgecolor='black')
    bars_b = ax.bar(x + width/2, values_b, width, label=model_b_name,
                    color='#10B981', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metric Comparison: Model A vs Model B', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars_a:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    for bar in bars_b:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrices(cm_a, cm_b, class_names, model_a_name, model_b_name):
    """
    Plot side-by-side confusion matrices with highlighting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Normalize confusion matrices
    cm_a_norm = cm_a.astype('float') / cm_a.sum(axis=1)[:, np.newaxis]
    cm_b_norm = cm_b.astype('float') / cm_b.sum(axis=1)[:, np.newaxis]
    
    # Model A
    sns.heatmap(cm_a_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                cbar_kws={'label': 'Proportion'})
    axes[0].set_title(f'Confusion Matrix - {model_a_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    # Model B
    sns.heatmap(cm_b_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'Confusion Matrix - {model_b_name}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_per_class_recall(report_a, report_b, class_names, model_a_name, model_b_name):
    """
    Create per-class recall comparison chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    recalls_a = [report_a.get(cls, {}).get('recall', 0) for cls in class_names]
    recalls_b = [report_b.get(cls, {}).get('recall', 0) for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    bars_a = ax.bar(x - width/2, recalls_a, width, label=model_a_name,
                    color='#3B82F6', alpha=0.8, edgecolor='black')
    bars_b = ax.bar(x + width/2, recalls_b, width, label=model_b_name,
                    color='#10B981', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Recall (Detection Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Recall Comparison\n(Emphasis on Minority Attack Classes)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars_a:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    for bar in bars_b:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # Highlight minority classes
    minority_classes = ['Flooding', 'TDMA', 'Blackhole', 'Grayhole']
    for i, cls in enumerate(class_names):
        if cls in minority_classes:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red')
    
    plt.tight_layout()
    return fig


def plot_ks_test_results(ks_results, top_n=10):
    """
    Create bar chart of KS statistics per feature.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by KS statistic
    sorted_features = sorted(ks_results.items(), key=lambda x: x[1]['statistic'], reverse=True)[:top_n]
    
    features = [f[0] for f in sorted_features]
    statistics = [f[1]['statistic'] for f in sorted_features]
    pvalues = [f[1]['pvalue'] for f in sorted_features]
    significant = [f[1]['significant'] for f in sorted_features]
    
    colors = ['#EF4444' if sig else '#3B82F6' for sig in significant]
    
    bars = ax.barh(range(len(features)), statistics, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('KS Statistic', fontsize=12, fontweight='bold')
    ax.set_title('Kolmogorov-Smirnov Test: Distribution Shift Analysis\n(Red = Significant Difference, p < 0.05)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value annotations
    for i, (bar, pval) in enumerate(zip(bars, pvalues)):
        width = bar.get_width()
        ax.annotate(f'p={pval:.2e}', xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points", 
                   ha='left', va='center', fontsize=8)
    
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_jsd_results(jsd_results, feature_names):
    """
    Create bar chart of Jensen-Shannon Divergence per feature.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get top features by JSD
    jsd_with_names = [(feature_names[i] if i < len(feature_names) else f'Feature_{i}', v) 
                      for i, v in jsd_results.items()]
    jsd_sorted = sorted(jsd_with_names, key=lambda x: x[1], reverse=True)[:10]
    
    features = [f[0] for f in jsd_sorted]
    values = [f[1] for f in jsd_sorted]
    
    # Color gradient based on divergence magnitude
    colors = plt.cm.RdYlGn_r(np.array(values) / max(values) if max(values) > 0 else np.zeros_like(values))
    
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Jensen-Shannon Divergence', fontsize=12, fontweight='bold')
    ax.set_title('Jensen-Shannon Divergence: Pre vs Post-Oversampling\n(Higher = Greater Distribution Shift)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_correlation_heatmaps(corr_orig, corr_over, feature_names):
    """
    Plot correlation matrix heatmaps for original and oversampled data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Select subset of features for visibility
    n_features = min(10, corr_orig.shape[0])
    corr_orig_sub = corr_orig[:n_features, :n_features]
    corr_over_sub = corr_over[:n_features, :n_features]
    feature_names_sub = feature_names[:n_features] if len(feature_names) >= n_features else feature_names
    
    # Original correlation matrix
    mask = np.triu(np.ones_like(corr_orig_sub, dtype=bool), k=1)
    sns.heatmap(corr_orig_sub, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=axes[0], vmin=-1, vmax=1,
                xticklabels=feature_names_sub, yticklabels=feature_names_sub,
                annot_kws={'size': 8})
    axes[0].set_title('Original Dataset\nCorrelation Matrix', fontsize=12, fontweight='bold')
    
    # Oversampled correlation matrix
    sns.heatmap(corr_over_sub, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=axes[1], vmin=-1, vmax=1,
                xticklabels=feature_names_sub, yticklabels=feature_names_sub,
                annot_kws={'size': 8})
    axes[1].set_title('Oversampled Dataset\nCorrelation Matrix', fontsize=12, fontweight='bold')
    
    # Difference matrix
    diff_matrix = np.abs(corr_orig_sub - corr_over_sub)
    sns.heatmap(diff_matrix, mask=mask, annot=True, fmt='.2f', cmap='Reds',
                square=True, ax=axes[2], vmin=0, vmax=0.5,
                xticklabels=feature_names_sub, yticklabels=feature_names_sub,
                annot_kws={'size': 8})
    axes[2].set_title('Absolute Correlation\nDifference', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_correlation_drift_summary(corr_results):
    """
    Create summary bar chart of correlation drift metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ['Mean Abs. Diff.', 'Frobenius Norm', 'Corr. Preservation']
    values = [corr_results['mae'], corr_results['frobenius'] / 10, corr_results['correlation']]
    
    colors = ['#EF4444', '#F59E0B', '#10B981']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Preservation Summary\n(Lower Drift = Better Preservation)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# ACADEMIC INTERPRETATION TEXT
# ============================================================================

def get_accuracy_insufficiency_text():
    return """
    ### Why Accuracy is Insufficient for IDS Evaluation
    
    In intrusion detection systems operating on imbalanced datasets like WSN-DS, 
    accuracy is a misleading metric for several critical reasons:
    
    **1. Class Imbalance Masking Effect**
    The WSN-DS dataset exhibits severe class imbalance with a ratio of approximately 
    102:1 between the majority class (Normal) and minority classes (attacks). A naive 
    classifier predicting only "Normal" would achieve ~90% accuracy while completely 
    failing to detect any attacks—the primary objective of an IDS.
    
    **2. Cost Asymmetry in Security Applications**
    In IDS contexts, the cost of a false negative (missed attack) far exceeds that of 
    a false positive (false alarm). Accuracy treats all errors equally, failing to 
    capture this critical asymmetry (He & Garcia, 2009; Chawla et al., 2002).
    
    **3. Academic Consensus**
    The imbalanced learning literature strongly advocates for alternative metrics:
    - **F1-score**: Harmonic mean of precision and recall
    - **Matthews Correlation Coefficient (MCC)**: Balanced measure considering all 
      confusion matrix quadrants
    - **G-Mean**: Geometric mean of class-wise recalls
    - **ROC-AUC and PR-AUC**: Threshold-independent performance measures
    
    *References:*
    - He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
    - Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique.
    """


def get_oversampling_impact_text():
    return """
    ### Impact of Oversampling on Data Distribution Integrity
    
    While oversampling techniques like SMOTE effectively address class imbalance, 
    they introduce potential risks to data distribution integrity:
    
    **1. Synthetic Sample Bias**
    Oversampling generates synthetic samples by interpolating between existing 
    minority class instances. This process may:
    - Create samples in sparse regions of the feature space
    - Introduce artificial patterns not present in real attack data
    - Potentially overlap with majority class boundaries
    
    **2. Distribution Shift Concerns**
    The Kolmogorov-Smirnov test results indicate whether oversampling significantly 
    alters the underlying feature distributions. Significant shifts (p < 0.05) 
    suggest that the synthetic data may not accurately represent real-world attack 
    characteristics.
    
    **3. Correlation Structure Preservation**
    The correlation preservation analysis examines whether inter-feature relationships 
    are maintained after oversampling. High correlation drift may indicate:
    - Introduction of spurious feature dependencies
    - Loss of genuine attack signatures
    - Potential overfitting to synthetic patterns
    
    **4. Jensen-Shannon Divergence Interpretation**
    JSD provides a symmetric, bounded (0 to 1) measure of distribution divergence:
    - JSD ≈ 0: Distributions are nearly identical
    - JSD > 0.1: Moderate divergence requiring attention
    - JSD > 0.3: Substantial distribution shift
    
    **Implications for WSN IDS Deployment:**
    Models trained on oversampled data should be validated on purely real-world 
    data to ensure that learned patterns generalize beyond synthetic samples.
    """


def get_methodology_text():
    return """
    ## Visualization and Statistical Methodology
    
    ### Performance Evaluation Framework
    
    This analysis employs a comprehensive IDS evaluation framework incorporating 
    multiple complementary metrics:
    
    **Classification Metrics:**
    - **Precision (Macro)**: Average precision across all classes, treating each 
      class equally regardless of support
    - **Recall (Macro)**: Average detection rate across all attack types
    - **F1-Score (Macro)**: Harmonic mean providing balanced precision-recall trade-off
    - **Matthews Correlation Coefficient**: Correlation between predicted and actual 
      classifications, ranging from -1 to +1
    
    **Threshold-Independent Metrics:**
    - **ROC Curve**: Trade-off between true positive rate and false positive rate
    - **Precision-Recall Curve**: Particularly informative for imbalanced datasets
    
    ### Distribution Preservation Analysis
    
    **Kolmogorov-Smirnov (KS) Test:**
    A non-parametric test comparing empirical cumulative distribution functions 
    of original and oversampled features. The KS statistic measures the maximum 
    distance between distributions.
    
    **Jensen-Shannon Divergence (JSD):**
    Preferred over Kullback-Leibler divergence due to:
    - Symmetry: JSD(P||Q) = JSD(Q||P)
    - Boundedness: 0 ≤ JSD ≤ 1 (using log base 2)
    - Defined for distributions with non-overlapping support
    
    **Correlation Preservation:**
    Quantified using:
    - Mean Absolute Error between correlation matrices
    - Frobenius norm of the difference matrix
    - Pearson correlation between upper-triangular elements
    """


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🔒 WSN Intrusion Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #6B7280;">Comparative Evaluation Dashboard</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Load MLflow data
    with st.spinner("Loading MLflow experiments..."):
        try:
            model_a, model_b = load_best_models_data()
            
            if model_a is None or model_b is None:
                st.error("Could not load model data from MLflow. Please ensure experiments have been run.")
                st.info("Run `python wsn_mlflow_pipeline_no_fe.py` to generate the required experiments.")
                return
            
            mlflow.set_tracking_uri("mlruns")
            client = MlflowClient()
            
        except Exception as e:
            st.error(f"Error loading MLflow data: {e}")
            st.info("Make sure the MLflow experiments have been run successfully.")
            return
    
    # Extract model information
    model_a_name = f"Model A: {model_a.data.tags.get('model_name', 'Unknown')} (No Oversampling)"
    model_b_name = f"Model B: {model_b.data.tags.get('model_name', 'Unknown')} ({model_b.data.tags.get('sampling_strategy', 'Oversampling')})"
    
    st.sidebar.success("✅ Models loaded successfully")
    st.sidebar.markdown(f"**Model A:** No Oversampling")
    st.sidebar.markdown(f"**Model B:** With Oversampling")
    
    # Get metrics
    metrics_a = model_a.data.metrics
    metrics_b = model_b.data.metrics
    
    # Load artifacts
    artifacts_a = get_run_artifacts(client, model_a.info.run_id)
    artifacts_b = get_run_artifacts(client, model_b.info.run_id)
    
    # ========================================================================
    # Section 1: Why Accuracy is Insufficient
    # ========================================================================
    st.markdown('<h2 class="section-header">📊 1. IDS Evaluation Framework</h2>', 
                unsafe_allow_html=True)
    
    with st.expander("📖 Why Accuracy is Insufficient for IDS Under Class Imbalance", expanded=True):
        st.markdown(get_accuracy_insufficiency_text())
    
    # ========================================================================
    # Section 2: Performance Metric Comparison
    # ========================================================================
    st.markdown('<h2 class="section-header">📈 2. Performance Metric Comparison</h2>', 
                unsafe_allow_html=True)
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model A - F1 (Macro)",
            f"{metrics_a.get('f1_macro', 0):.4f}",
            delta=None
        )
    with col2:
        st.metric(
            "Model B - F1 (Macro)",
            f"{metrics_b.get('f1_macro', 0):.4f}",
            delta=f"{(metrics_b.get('f1_macro', 0) - metrics_a.get('f1_macro', 0)):.4f}"
        )
    with col3:
        st.metric(
            "Model A - MCC",
            f"{metrics_a.get('matthews_corrcoef', 0):.4f}"
        )
    with col4:
        st.metric(
            "Model B - MCC",
            f"{metrics_b.get('matthews_corrcoef', 0):.4f}",
            delta=f"{(metrics_b.get('matthews_corrcoef', 0) - metrics_a.get('matthews_corrcoef', 0)):.4f}"
        )
    
    # Performance comparison chart
    st.subheader("A. Performance Metric Comparison Chart")
    fig_perf = plot_performance_comparison(metrics_a, metrics_b, model_a_name, model_b_name)
    st.pyplot(fig_perf)
    st.caption("""
    **Figure 1:** Grouped bar chart comparing macro-averaged precision, recall, F1-score, 
    and Matthews Correlation Coefficient between Model A (no oversampling) and Model B 
    (with oversampling). These metrics provide balanced evaluation across all attack classes.
    """)
    
    # ========================================================================
    # Section 3: Confusion Matrix Analysis
    # ========================================================================
    st.markdown('<h2 class="section-header">🎯 3. Confusion Matrix Analysis</h2>', 
                unsafe_allow_html=True)
    
    st.subheader("C. Confusion Matrix Heatmaps")
    
    class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
    
    if 'confusion_matrix' in artifacts_a and 'confusion_matrix' in artifacts_b:
        cm_a = artifacts_a['confusion_matrix'].values
        cm_b = artifacts_b['confusion_matrix'].values
        
        fig_cm = plot_confusion_matrices(cm_a, cm_b, class_names, model_a_name, model_b_name)
        st.pyplot(fig_cm)
        st.caption("""
        **Figure 2:** Normalized confusion matrices showing prediction distributions. 
        The diagonal elements represent correct classifications (detection rate per class). 
        Off-diagonal elements highlight misclassifications—particularly important are 
        false negatives (attacks classified as Normal) which represent missed detections.
        """)
        
        # Interpretation
        st.markdown("""
        <div class="interpretation-box">
        <strong>🔍 Interpretation:</strong> Compare the diagonal values (correct classifications) 
        between models. Pay special attention to the attack classes (non-Normal rows) - higher 
        diagonal values indicate better attack detection. False negatives (attacks predicted as 
        Normal) are critical security failures.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Confusion matrix artifacts not available.")
    
    # ========================================================================
    # Section 4: Per-Class Recall Analysis
    # ========================================================================
    st.markdown('<h2 class="section-header">🎯 4. Per-Class Recall (Detection Rate)</h2>', 
                unsafe_allow_html=True)
    
    st.subheader("D. Per-Class Recall Chart")
    
    if 'classification_report' in artifacts_a and 'classification_report' in artifacts_b:
        report_a = artifacts_a['classification_report']
        report_b = artifacts_b['classification_report']
        
        fig_recall = plot_per_class_recall(report_a, report_b, class_names, model_a_name, model_b_name)
        st.pyplot(fig_recall)
        st.caption("""
        **Figure 3:** Per-class recall (detection rate) comparison. Minority attack classes 
        (highlighted) are particularly important for IDS evaluation. High recall on these 
        classes indicates effective detection of rare but critical attack types. The shaded 
        regions emphasize minority classes requiring special attention.
        """)
        
        # Create detailed per-class table
        st.subheader("Detailed Per-Class Metrics")
        
        per_class_data = []
        for cls in class_names:
            per_class_data.append({
                'Class': cls,
                'Model A Precision': report_a.get(cls, {}).get('precision', 0),
                'Model A Recall': report_a.get(cls, {}).get('recall', 0),
                'Model A F1': report_a.get(cls, {}).get('f1-score', 0),
                'Model B Precision': report_b.get(cls, {}).get('precision', 0),
                'Model B Recall': report_b.get(cls, {}).get('recall', 0),
                'Model B F1': report_b.get(cls, {}).get('f1-score', 0),
            })
        
        df_per_class = pd.DataFrame(per_class_data)
        st.dataframe(df_per_class.style.format({
            'Model A Precision': '{:.4f}',
            'Model A Recall': '{:.4f}',
            'Model A F1': '{:.4f}',
            'Model B Precision': '{:.4f}',
            'Model B Recall': '{:.4f}',
            'Model B F1': '{:.4f}'
        }), use_container_width=True)
    
    # ========================================================================
    # Section 5: Imbalance and Distribution Analysis
    # ========================================================================
    st.markdown('<h2 class="section-header">📊 5. Imbalance & Distribution Preservation Analysis</h2>', 
                unsafe_allow_html=True)
    
    with st.expander("📖 Impact of Oversampling on Data Distribution Integrity", expanded=True):
        st.markdown(get_oversampling_impact_text())
    
    # Generate synthetic data for demonstration (in production, load from MLflow)
    st.info("""
    **Note:** The distribution analysis below uses simulated data to demonstrate the 
    visualization framework. In production, these metrics should be computed from 
    actual training data stored as MLflow artifacts.
    """)
    
    # Simulated data for visualization demonstration
    np.random.seed(42)
    n_samples_orig = 1000
    n_samples_over = 3000
    n_features = 16
    
    feature_names = ['Time', 'Is_CH', 'Dist_To_CH', 'ADV_S', 'ADV_R', 'JOIN_S', 
                     'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 'DATA_S', 'DATA_R', 
                     'Data_Sent_To_BS', 'dist_CH_To_BS', 'send_code', 'Expaned_Energy']
    
    # Simulate original and oversampled data
    original_data = np.random.randn(n_samples_orig, n_features)
    oversampled_data = np.random.randn(n_samples_over, n_features) * 1.1 + 0.05  # Slight shift
    
    # ========================================================================
    # Section 5A: KS Test Visualization
    # ========================================================================
    st.subheader("E. Kolmogorov-Smirnov Test Analysis")
    
    ks_results = compute_ks_test(original_data, oversampled_data, feature_names)
    
    fig_ks = plot_ks_test_results(ks_results)
    st.pyplot(fig_ks)
    st.caption("""
    **Figure 4:** Kolmogorov-Smirnov test statistics per feature comparing original 
    and oversampled training data distributions. Red bars indicate statistically 
    significant differences (p < 0.05), suggesting the oversampling process has 
    altered the feature distribution. Higher KS statistics indicate greater 
    distributional divergence.
    """)
    
    # KS Summary Table
    ks_df = pd.DataFrame([
        {'Feature': k, 'KS Statistic': v['statistic'], 'P-Value': v['pvalue'], 
         'Significant': '✓' if v['significant'] else ''}
        for k, v in ks_results.items()
    ]).sort_values('KS Statistic', ascending=False).head(10)
    
    st.dataframe(ks_df.style.format({
        'KS Statistic': '{:.4f}',
        'P-Value': '{:.2e}'
    }), use_container_width=True)
    
    # ========================================================================
    # Section 5B: Jensen-Shannon Divergence
    # ========================================================================
    st.subheader("F. Jensen-Shannon Divergence Analysis")
    
    jsd_results = compute_jensen_shannon_divergence(original_data, oversampled_data)
    
    fig_jsd = plot_jsd_results(jsd_results, feature_names)
    st.pyplot(fig_jsd)
    st.caption("""
    **Figure 5:** Jensen-Shannon Divergence per feature between pre-oversampling 
    and post-oversampling distributions. JSD is preferred over KL-divergence due 
    to its symmetry and boundedness (0 to 1). Values approaching 0 indicate similar 
    distributions; higher values suggest substantial distribution shift requiring 
    careful interpretation.
    """)
    
    # JSD Summary
    avg_jsd = np.mean(list(jsd_results.values()))
    max_jsd = max(jsd_results.values())
    st.markdown(f"""
    **Summary Statistics:**
    - Average JSD across features: **{avg_jsd:.4f}**
    - Maximum JSD: **{max_jsd:.4f}**
    - Features with JSD > 0.1: **{sum(1 for v in jsd_results.values() if v > 0.1)}**
    """)
    
    # ========================================================================
    # Section 5C: Correlation Preservation Analysis
    # ========================================================================
    st.subheader("G. Correlation Preservation Analysis")
    
    corr_results = compute_correlation_preservation(original_data, oversampled_data)
    
    fig_corr = plot_correlation_heatmaps(
        corr_results['original_matrix'], 
        corr_results['oversampled_matrix'],
        feature_names
    )
    st.pyplot(fig_corr)
    st.caption("""
    **Figure 6:** Correlation matrix comparison between original (left) and oversampled 
    (center) datasets. The rightmost heatmap shows absolute differences, highlighting 
    features with altered correlation structures. Substantial differences may indicate 
    introduction of spurious dependencies or loss of genuine feature relationships.
    """)
    
    # Correlation drift summary
    fig_drift = plot_correlation_drift_summary(corr_results)
    st.pyplot(fig_drift)
    st.caption("""
    **Figure 7:** Summary of correlation preservation metrics. Lower MAE and Frobenius 
    norm values indicate better preservation of original correlation structure. 
    Correlation preservation score near 1.0 suggests minimal structural distortion.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Abs. Correlation Diff.", f"{corr_results['mae']:.4f}")
    with col2:
        st.metric("Frobenius Norm", f"{corr_results['frobenius']:.4f}")
    with col3:
        st.metric("Correlation Preservation", f"{corr_results['correlation']:.4f}")
    
    # ========================================================================
    # Section 6: Academic Methodology
    # ========================================================================
    st.markdown('<h2 class="section-header">📚 6. Methodology Documentation</h2>', 
                unsafe_allow_html=True)
    
    with st.expander("📖 Visualization and Statistical Methodology", expanded=False):
        st.markdown(get_methodology_text())
    
    # ========================================================================
    # Section 7: Summary and Conclusions
    # ========================================================================
    st.markdown('<h2 class="section-header">📋 7. Summary and Recommendations</h2>', 
                unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Comparative Analysis Summary
    
    | Metric | Model A (No Oversampling) | Model B (With Oversampling) | Difference |
    |--------|---------------------------|-----------------------------| -----------|
    | F1-Score (Macro) | {metrics_a.get('f1_macro', 0):.4f} | {metrics_b.get('f1_macro', 0):.4f} | {metrics_b.get('f1_macro', 0) - metrics_a.get('f1_macro', 0):+.4f} |
    | Precision (Macro) | {metrics_a.get('precision_macro', 0):.4f} | {metrics_b.get('precision_macro', 0):.4f} | {metrics_b.get('precision_macro', 0) - metrics_a.get('precision_macro', 0):+.4f} |
    | Recall (Macro) | {metrics_a.get('recall_macro', 0):.4f} | {metrics_b.get('recall_macro', 0):.4f} | {metrics_b.get('recall_macro', 0) - metrics_a.get('recall_macro', 0):+.4f} |
    | MCC | {metrics_a.get('matthews_corrcoef', 0):.4f} | {metrics_b.get('matthews_corrcoef', 0):.4f} | {metrics_b.get('matthews_corrcoef', 0) - metrics_a.get('matthews_corrcoef', 0):+.4f} |
    | Balanced Accuracy | {metrics_a.get('balanced_accuracy', 0):.4f} | {metrics_b.get('balanced_accuracy', 0):.4f} | {metrics_b.get('balanced_accuracy', 0) - metrics_a.get('balanced_accuracy', 0):+.4f} |
    
    ### Key Findings
    
    1. **Performance Comparison:** {"Model B (with oversampling) shows improved" if metrics_b.get('f1_macro', 0) > metrics_a.get('f1_macro', 0) else "Model A (without oversampling) shows comparable or better"} macro F1-score performance.
    
    2. **Detection Rate:** Oversampling {"enhances" if metrics_b.get('recall_macro', 0) > metrics_a.get('recall_macro', 0) else "does not significantly improve"} detection rates for minority attack classes.
    
    3. **Distribution Integrity:** The distribution analysis reveals the trade-offs between improved class balance and potential synthetic sample bias.
    
    ### Recommendations for WSN IDS Deployment
    
    - Validate models on purely real-world test data to ensure generalization
    - Monitor false negative rates for critical attack types
    - Consider ensemble approaches combining both models
    - Implement continuous model retraining with new attack patterns
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    WSN Intrusion Detection System - Comparative Evaluation Dashboard<br>
    Generated for Academic Research | MLflow Integration | Streamlit Visualization
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

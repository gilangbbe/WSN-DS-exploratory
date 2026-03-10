"""
WSN IDS Model Comparison Dashboard
==================================
Interactive Streamlit app for comparing ML models across different 
oversampling strategies with comprehensive metrics visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import jensenshannon
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
    page_title="WSN IDS Model Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 WSN IDS Model Comparison Dashboard")
st.markdown("Compare ML models across oversampling strategies with interactive charts")

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_all_runs():
    """Load all runs from MLflow experiments."""
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()
    
    all_runs = []
    experiments = client.search_experiments()
    
    for exp in experiments:
        if "WSN" in exp.name or "IDS" in exp.name:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                run_data = {
                    'run_id': run.info.run_id,
                    'experiment': exp.name,
                    'model_name': run.data.tags.get('model_name', 'Unknown'),
                    'sampling_strategy': run.data.tags.get('sampling_strategy', 'None'),
                    'feature_engineering': 'With_FE' if 'Feature_Engineering' not in exp.name else 'No_FE',
                }
                # Add all metrics
                for key, value in run.data.metrics.items():
                    run_data[key] = value
                all_runs.append(run_data)
    
    return pd.DataFrame(all_runs)


@st.cache_data
def load_dataset():
    """Load the WSN-DS dataset for distribution analysis."""
    df = pd.read_csv("/Users/biru/Documents/TugasAkhir/data/WSN-DS.csv")
    return df


@st.cache_data
def get_run_artifacts(run_id):
    """Load artifacts from a specific run."""
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()
    artifacts = {}
    
    try:
        artifact_path = client.download_artifacts(run_id, "")
        
        for f in os.listdir(artifact_path):
            if 'classification_report' in f and f.endswith('.json'):
                with open(os.path.join(artifact_path, f), 'r') as file:
                    artifacts['classification_report'] = json.load(file)
            elif 'confusion_matrix' in f and f.endswith('.csv'):
                artifacts['confusion_matrix'] = pd.read_csv(
                    os.path.join(artifact_path, f), index_col=0
                )
            elif 'feature_importance' in f and f.endswith('.json'):
                with open(os.path.join(artifact_path, f), 'r') as file:
                    artifacts['feature_importance'] = json.load(file)
    except Exception as e:
        pass
    
    return artifacts


# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner("Loading MLflow data..."):
    df_runs = load_all_runs()

if df_runs.empty:
    st.error("No runs found in MLflow. Please run the training pipeline first.")
    st.stop()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

st.sidebar.header("🔧 Filters")

# Filter by Feature Engineering
fe_options = df_runs['feature_engineering'].unique().tolist()
selected_fe = st.sidebar.selectbox("Feature Engineering", fe_options, index=0)

df_filtered = df_runs[df_runs['feature_engineering'] == selected_fe]

# Filter by Model Type
model_options = sorted(df_filtered['model_name'].unique().tolist())
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    model_options,
    default=model_options[:3] if len(model_options) >= 3 else model_options
)

# Filter by Sampling Strategy
sampling_options = sorted(df_filtered['sampling_strategy'].unique().tolist())
selected_sampling = st.sidebar.multiselect(
    "Select Sampling Strategies",
    sampling_options,
    default=sampling_options
)

# Apply filters
df_display = df_filtered[
    (df_filtered['model_name'].isin(selected_models)) &
    (df_filtered['sampling_strategy'].isin(selected_sampling))
]

if df_display.empty:
    st.warning("No runs match the selected filters. Please adjust your selections.")
    st.stop()

# Create combined label for display
df_display = df_display.copy()
df_display['label'] = df_display['model_name'] + ' (' + df_display['sampling_strategy'] + ')'

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing {len(df_display)} model runs**")

# ============================================================================
# TAB LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance Metrics", 
    "🎯 Per-Class Metrics",
    "📊 Confusion Matrices",
    "📉 Distribution Analysis",
    "🔍 Feature Importance"
])

# ============================================================================
# TAB 1: PERFORMANCE METRICS COMPARISON
# ============================================================================

with tab1:
    st.header("Performance Metrics Comparison")
    
    # Metric selection
    metric_options = ['f1_weighted', 'f1_macro', 'precision_macro', 'recall_macro', 
                      'matthews_corrcoef', 'balanced_accuracy', 'roc_auc_weighted_ovr',
                      'g_mean', 'accuracy']
    
    available_metrics = [m for m in metric_options if m in df_display.columns]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_metric = st.selectbox("Primary Metric", available_metrics, index=0)
        sort_order = st.radio("Sort Order", ["Descending", "Ascending"])
        ascending = sort_order == "Ascending"
    
    with col2:
        # Bar chart comparison
        df_sorted = df_display.sort_values(selected_metric, ascending=ascending)
        
        fig = px.bar(
            df_sorted,
            x='label',
            y=selected_metric,
            color='sampling_strategy',
            title=f'{selected_metric} Comparison',
            labels={'label': 'Model', selected_metric: selected_metric.replace('_', ' ').title()},
            text=df_sorted[selected_metric].round(4)
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Multi-metric radar chart
    st.subheader("Multi-Metric Radar Comparison")
    
    radar_metrics = st.multiselect(
        "Select metrics for radar chart",
        available_metrics,
        default=available_metrics[:5] if len(available_metrics) >= 5 else available_metrics
    )
    
    if radar_metrics:
        fig_radar = go.Figure()
        
        for idx, row in df_display.iterrows():
            values = [row[m] for m in radar_metrics]
            values.append(values[0])  # Close the radar
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=row['label'],
                opacity=0.6
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600,
            title="Multi-Metric Radar Comparison"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    # Grouped bar chart for multiple metrics
    st.subheader("Side-by-Side Metric Comparison")
    
    compare_metrics = st.multiselect(
        "Select metrics to compare side by side",
        available_metrics,
        default=['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef'][:min(4, len(available_metrics))]
    )
    
    if compare_metrics:
        df_melted = df_display.melt(
            id_vars=['label', 'model_name', 'sampling_strategy'],
            value_vars=compare_metrics,
            var_name='Metric',
            value_name='Score'
        )
        
        fig_grouped = px.bar(
            df_melted,
            x='label',
            y='Score',
            color='Metric',
            barmode='group',
            title='Multi-Metric Comparison',
            labels={'label': 'Model'}
        )
        fig_grouped.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_grouped, use_container_width=True)

# ============================================================================
# TAB 2: PER-CLASS METRICS
# ============================================================================

with tab2:
    st.header("Per-Class Metrics Comparison")
    
    class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
    class_metric = st.selectbox("Select Metric", ['recall', 'precision', 'f1-score'])
    
    # Build per-class data
    per_class_data = []
    
    for idx, row in df_display.iterrows():
        artifacts = get_run_artifacts(row['run_id'])
        if 'classification_report' in artifacts:
            report = artifacts['classification_report']
            for cls in class_names:
                if cls in report:
                    per_class_data.append({
                        'Model': row['label'],
                        'Sampling': row['sampling_strategy'],
                        'Class': cls,
                        'Value': report[cls].get(class_metric, 0)
                    })
    
    if per_class_data:
        df_class = pd.DataFrame(per_class_data)
        
        # Grouped bar chart
        fig_class = px.bar(
            df_class,
            x='Class',
            y='Value',
            color='Model',
            barmode='group',
            title=f'Per-Class {class_metric.title()} Comparison',
            labels={'Value': class_metric.title()}
        )
        fig_class.update_layout(height=500)
        st.plotly_chart(fig_class, use_container_width=True)
        
        # Heatmap view
        st.subheader("Heatmap View")
        
        df_pivot = df_class.pivot(index='Model', columns='Class', values='Value')
        
        fig_heat = px.imshow(
            df_pivot,
            labels=dict(x="Class", y="Model", color=class_metric.title()),
            aspect="auto",
            color_continuous_scale="Blues",
            title=f'{class_metric.title()} Heatmap'
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No per-class data available for selected models.")

# ============================================================================
# TAB 3: CONFUSION MATRICES
# ============================================================================

with tab3:
    st.header("Confusion Matrix Comparison")
    
    # Select models to compare
    col1, col2 = st.columns(2)
    
    labels_list = df_display['label'].tolist()
    
    with col1:
        model_a_label = st.selectbox("Model A", labels_list, index=0, key='cm_a')
    with col2:
        model_b_label = st.selectbox("Model B", labels_list, 
                                     index=min(1, len(labels_list)-1), key='cm_b')
    
    # Get confusion matrices
    run_a = df_display[df_display['label'] == model_a_label].iloc[0]
    run_b = df_display[df_display['label'] == model_b_label].iloc[0]
    
    artifacts_a = get_run_artifacts(run_a['run_id'])
    artifacts_b = get_run_artifacts(run_b['run_id'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'confusion_matrix' in artifacts_a:
            cm_a = artifacts_a['confusion_matrix']
            # Normalize
            cm_a_norm = cm_a.div(cm_a.sum(axis=1), axis=0)
            
            fig_cm_a = px.imshow(
                cm_a_norm,
                labels=dict(x="Predicted", y="Actual", color="Rate"),
                x=cm_a.columns,
                y=cm_a.index,
                color_continuous_scale="Blues",
                title=f'Model A: {model_a_label}',
                text_auto='.2%'
            )
            fig_cm_a.update_layout(height=450)
            st.plotly_chart(fig_cm_a, use_container_width=True)
        else:
            st.warning(f"No confusion matrix for {model_a_label}")
    
    with col2:
        if 'confusion_matrix' in artifacts_b:
            cm_b = artifacts_b['confusion_matrix']
            cm_b_norm = cm_b.div(cm_b.sum(axis=1), axis=0)
            
            fig_cm_b = px.imshow(
                cm_b_norm,
                labels=dict(x="Predicted", y="Actual", color="Rate"),
                x=cm_b.columns,
                y=cm_b.index,
                color_continuous_scale="Greens",
                title=f'Model B: {model_b_label}',
                text_auto='.2%'
            )
            fig_cm_b.update_layout(height=450)
            st.plotly_chart(fig_cm_b, use_container_width=True)
        else:
            st.warning(f"No confusion matrix for {model_b_label}")
    
    # Difference heatmap
    if 'confusion_matrix' in artifacts_a and 'confusion_matrix' in artifacts_b:
        st.subheader("Confusion Matrix Difference (Model B - Model A)")
        
        cm_diff = cm_b_norm - cm_a_norm
        
        fig_diff = px.imshow(
            cm_diff,
            labels=dict(x="Predicted", y="Actual", color="Difference"),
            x=cm_a.columns,
            y=cm_a.index,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title='Difference (Green = B better, Red = A better)',
            text_auto='.2%'
        )
        fig_diff.update_layout(height=450)
        st.plotly_chart(fig_diff, use_container_width=True)

# ============================================================================
# TAB 4: DISTRIBUTION ANALYSIS
# ============================================================================

with tab4:
    st.header("Distribution Analysis")
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        df_data = load_dataset()
    
    feature_cols = ['Time', 'Is CH', 'who CH', 'Dist To CH', 'ADV_S', 'ADV_R', 
                    'JOIN_S', 'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 'DATA_S', 
                    'DATA_R', 'Data_Sent_To_BS', 'dist_CH_To_BS', 'send_code', 
                    'Expaned Energy']
    
    available_features = [f for f in feature_cols if f in df_data.columns]
    
    # Attack type filter
    attack_types = df_data['Attack type'].unique().tolist()
    
    st.subheader("Distribution Comparison by Attack Type")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_feature = st.selectbox("Select Feature", available_features)
        
        compare_classes = st.multiselect(
            "Compare Classes",
            attack_types,
            default=['Normal', 'Blackhole'] if 'Normal' in attack_types and 'Blackhole' in attack_types else attack_types[:2]
        )
    
    with col2:
        if selected_feature and compare_classes:
            # Histogram comparison
            df_subset = df_data[df_data['Attack type'].isin(compare_classes)]
            
            fig_hist = px.histogram(
                df_subset,
                x=selected_feature,
                color='Attack type',
                barmode='overlay',
                title=f'{selected_feature} Distribution by Attack Type',
                opacity=0.6,
                nbins=50
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # ECDF Comparison (for KS test visualization)
    st.subheader("Empirical CDF Comparison (KS Test Visualization)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ecdf_feature = st.selectbox("Feature for ECDF", available_features, key='ecdf_feat')
        class_a = st.selectbox("Class A", attack_types, index=0, key='ecdf_a')
        class_b = st.selectbox("Class B", attack_types, 
                               index=min(1, len(attack_types)-1), key='ecdf_b')
    
    with col2:
        if ecdf_feature and class_a and class_b:
            data_a = df_data[df_data['Attack type'] == class_a][ecdf_feature].dropna()
            data_b = df_data[df_data['Attack type'] == class_b][ecdf_feature].dropna()
            
            # Compute ECDF
            def compute_ecdf(data):
                x = np.sort(data)
                y = np.arange(1, len(data) + 1) / len(data)
                return x, y
            
            x_a, y_a = compute_ecdf(data_a)
            x_b, y_b = compute_ecdf(data_b)
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(data_a, data_b)
            
            fig_ecdf = go.Figure()
            
            fig_ecdf.add_trace(go.Scatter(
                x=x_a, y=y_a, mode='lines', name=class_a,
                line=dict(color='blue', width=2)
            ))
            fig_ecdf.add_trace(go.Scatter(
                x=x_b, y=y_b, mode='lines', name=class_b,
                line=dict(color='red', width=2)
            ))
            
            fig_ecdf.update_layout(
                title=f'ECDF: {ecdf_feature}<br>KS Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.2e}',
                xaxis_title=ecdf_feature,
                yaxis_title='Cumulative Probability',
                height=450
            )
            st.plotly_chart(fig_ecdf, use_container_width=True)
            
            # KS Test interpretation
            if ks_pvalue < 0.05:
                st.error(f"⚠️ Significant difference (p < 0.05): The distributions are statistically different.")
            else:
                st.success(f"✅ No significant difference (p ≥ 0.05): Distributions are similar.")
    
    st.markdown("---")
    
    # KS Test Summary Table
    st.subheader("KS Test Summary: All Features")
    
    ks_class_a = st.selectbox("Reference Class", attack_types, index=0, key='ks_ref')
    ks_class_b = st.selectbox("Comparison Class", attack_types, 
                              index=min(1, len(attack_types)-1), key='ks_comp')
    
    if st.button("Compute KS Tests"):
        ks_results = []
        
        data_ref = df_data[df_data['Attack type'] == ks_class_a]
        data_comp = df_data[df_data['Attack type'] == ks_class_b]
        
        for feat in available_features:
            try:
                stat, pval = stats.ks_2samp(
                    data_ref[feat].dropna(),
                    data_comp[feat].dropna()
                )
                ks_results.append({
                    'Feature': feat,
                    'KS Statistic': stat,
                    'P-Value': pval,
                    'Significant': '✓' if pval < 0.05 else ''
                })
            except:
                pass
        
        df_ks = pd.DataFrame(ks_results).sort_values('KS Statistic', ascending=False)
        
        # Bar chart
        fig_ks = px.bar(
            df_ks,
            x='KS Statistic',
            y='Feature',
            orientation='h',
            color='KS Statistic',
            color_continuous_scale='Reds',
            title=f'KS Statistics: {ks_class_a} vs {ks_class_b}'
        )
        fig_ks.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ks, use_container_width=True)
        
        st.dataframe(df_ks, use_container_width=True)
    
    st.markdown("---")
    
    # Jensen-Shannon Divergence
    st.subheader("Jensen-Shannon Divergence")
    
    jsd_class_a = st.selectbox("Class A for JSD", attack_types, index=0, key='jsd_a')
    jsd_class_b = st.selectbox("Class B for JSD", attack_types, 
                               index=min(1, len(attack_types)-1), key='jsd_b')
    
    if st.button("Compute JSD"):
        jsd_results = []
        
        data_a = df_data[df_data['Attack type'] == jsd_class_a]
        data_b = df_data[df_data['Attack type'] == jsd_class_b]
        
        for feat in available_features:
            try:
                vals_a = data_a[feat].dropna().values
                vals_b = data_b[feat].dropna().values
                
                # Create histograms
                combined = np.concatenate([vals_a, vals_b])
                bins = np.linspace(combined.min(), combined.max(), 50)
                
                hist_a, _ = np.histogram(vals_a, bins=bins, density=True)
                hist_b, _ = np.histogram(vals_b, bins=bins, density=True)
                
                hist_a = hist_a + 1e-10
                hist_b = hist_b + 1e-10
                hist_a = hist_a / hist_a.sum()
                hist_b = hist_b / hist_b.sum()
                
                jsd = jensenshannon(hist_a, hist_b)
                
                jsd_results.append({
                    'Feature': feat,
                    'JSD': jsd if not np.isnan(jsd) else 0
                })
            except:
                pass
        
        df_jsd = pd.DataFrame(jsd_results).sort_values('JSD', ascending=False)
        
        fig_jsd = px.bar(
            df_jsd,
            x='JSD',
            y='Feature',
            orientation='h',
            color='JSD',
            color_continuous_scale='Viridis',
            title=f'Jensen-Shannon Divergence: {jsd_class_a} vs {jsd_class_b}'
        )
        fig_jsd.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_jsd, use_container_width=True)
        
        st.dataframe(df_jsd, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Matrix Comparison
    st.subheader("Correlation Matrix Comparison")
    
    corr_class_a = st.selectbox("Class A for Correlation", attack_types, index=0, key='corr_a')
    corr_class_b = st.selectbox("Class B for Correlation", attack_types, 
                                index=min(1, len(attack_types)-1), key='corr_b')
    
    if st.button("Compare Correlations"):
        data_a = df_data[df_data['Attack type'] == corr_class_a][available_features]
        data_b = df_data[df_data['Attack type'] == corr_class_b][available_features]
        
        corr_a = data_a.corr()
        corr_b = data_b.corr()
        corr_diff = (corr_b - corr_a).abs()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_corr_a = px.imshow(
                corr_a,
                title=f'{corr_class_a}',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                aspect='equal'
            )
            fig_corr_a.update_layout(height=400)
            st.plotly_chart(fig_corr_a, use_container_width=True)
        
        with col2:
            fig_corr_b = px.imshow(
                corr_b,
                title=f'{corr_class_b}',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                aspect='equal'
            )
            fig_corr_b.update_layout(height=400)
            st.plotly_chart(fig_corr_b, use_container_width=True)
        
        with col3:
            fig_corr_diff = px.imshow(
                corr_diff,
                title='Absolute Difference',
                color_continuous_scale='Reds',
                aspect='equal'
            )
            fig_corr_diff.update_layout(height=400)
            st.plotly_chart(fig_corr_diff, use_container_width=True)
        
        # Summary metrics
        mae = corr_diff.values[np.triu_indices(len(available_features), k=1)].mean()
        st.metric("Mean Absolute Correlation Difference", f"{mae:.4f}")

# ============================================================================
# TAB 5: FEATURE IMPORTANCE
# ============================================================================

with tab5:
    st.header("Feature Importance Comparison")
    
    # Select models
    fi_models = st.multiselect(
        "Select models to compare feature importance",
        df_display['label'].tolist(),
        default=df_display['label'].tolist()[:2] if len(df_display) >= 2 else df_display['label'].tolist()
    )
    
    if fi_models:
        fi_data = []
        
        for label in fi_models:
            run = df_display[df_display['label'] == label].iloc[0]
            artifacts = get_run_artifacts(run['run_id'])
            
            if 'feature_importance' in artifacts:
                fi = artifacts['feature_importance']
                for feat, imp in fi.items():
                    fi_data.append({
                        'Model': label,
                        'Feature': feat,
                        'Importance': imp
                    })
        
        if fi_data:
            df_fi = pd.DataFrame(fi_data)
            
            # Grouped bar chart
            fig_fi = px.bar(
                df_fi,
                x='Feature',
                y='Importance',
                color='Model',
                barmode='group',
                title='Feature Importance Comparison'
            )
            fig_fi.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            # Heatmap
            st.subheader("Feature Importance Heatmap")
            
            df_fi_pivot = df_fi.pivot(index='Model', columns='Feature', values='Importance')
            
            fig_fi_heat = px.imshow(
                df_fi_pivot,
                labels=dict(x="Feature", y="Model", color="Importance"),
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig_fi_heat.update_layout(height=400)
            st.plotly_chart(fig_fi_heat, use_container_width=True)
        else:
            st.warning("No feature importance data available for selected models.")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

st.markdown("---")
st.header("📋 Summary Table")

# Select columns to display
display_cols = ['label', 'model_name', 'sampling_strategy'] + [
    c for c in df_display.columns if c in available_metrics
]

st.dataframe(
    df_display[display_cols].sort_values('f1_weighted' if 'f1_weighted' in df_display.columns else display_cols[-1], ascending=False),
    use_container_width=True
)

# Download button
csv = df_display[display_cols].to_csv(index=False)
st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="model_comparison_results.csv",
    mime="text/csv"
)

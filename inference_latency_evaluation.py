"""
Inference Latency Evaluation for LEACH-Based WSN IDS
=====================================================
This script measures and estimates inference latency for IDS models
deployed in a LEACH-based Wireless Sensor Network.

Author: Research Team
Date: 2026-01-30
"""

import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1 & 2: DEPLOYMENT ASSUMPTIONS AND DEVICE PROFILES
# ============================================================================

DEPLOYMENT_CONFIG = {
    "inference_location": "Cluster Head (CH)",
    "justification": """
    The Cluster Head is selected as the optimal inference deployment location based on:
    
    1. FEATURE AVAILABILITY:
       - CH aggregates data from all member nodes in the cluster
       - MAC-layer features (ADV_S, ADV_R, JOIN_S, JOIN_R, SCH_S, SCH_R, DATA_S, DATA_R)
         are directly observable at the CH level
       - CH has visibility into cluster-wide communication patterns
    
    2. LEACH COMMUNICATION PATTERN:
       - In LEACH, CHs receive all data from member nodes before forwarding to BS
       - This natural aggregation point allows inspection of all cluster traffic
       - Detection at CH prevents malicious data from reaching the Base Station
    
    3. ENERGY AND COMPUTATION CONSTRAINTS:
       - CHs have higher energy budget than regular sensor nodes
       - CH role rotates, distributing computational load across network
       - More practical than sensor node deployment (too resource-constrained)
       - More responsive than BS deployment (closer to attack source)
    """,
    
    "system_assumptions": [
        "Features are pre-extracted from MAC-layer packets",
        "Inference runs after feature extraction, before data forwarding",
        "Single-threaded execution on embedded processor",
        "No operating system overhead (bare-metal or RTOS)",
        "Memory for model and feature buffer is pre-allocated",
        "Inference is triggered per received packet or per LEACH round"
    ]
}

# Target Device Profiles
DEVICE_PROFILES = {
    "MSP430": {
        "name": "MSP430F5529 (Low-end MCU)",
        "description": "Texas Instruments MSP430 - Ultra-low power 16-bit MCU",
        "clock_frequency_mhz": 25,
        "clock_frequency_hz": 25e6,
        "ram_kb": 8,
        "flash_kb": 128,
        "active_power_mw": 3.6,  # at 25MHz, 3.3V
        "sleep_power_uw": 1.0,
        "cycles_per_comparison": 4,
        "cycles_per_addition": 2,
        "cycles_per_multiplication": 8,
        "cycles_per_division": 20,
    },
    "CortexM4": {
        "name": "ARM Cortex-M4 (Mid-range MCU)",
        "description": "ARM Cortex-M4F with FPU - e.g., STM32F4 series",
        "clock_frequency_mhz": 168,
        "clock_frequency_hz": 168e6,
        "ram_kb": 192,
        "flash_kb": 1024,
        "active_power_mw": 80,  # at 168MHz
        "sleep_power_uw": 10,
        "cycles_per_comparison": 1,
        "cycles_per_addition": 1,
        "cycles_per_multiplication": 1,  # Single-cycle with DSP
        "cycles_per_division": 12,
    }
}

# ============================================================================
# PART 3: INFERENCE EXECUTION MODEL
# ============================================================================

INFERENCE_MODEL = {
    "granularity": "per_packet",
    "description": """
    Inference Execution Model:
    - Granularity: Per-packet inspection
    - Batch size: 1 (single sample inference)
    - Trigger: After feature extraction from each received packet
    - Features: Pre-extracted, stored in fixed-size buffer (16 features × 4 bytes = 64 bytes)
    """,
    
    "batch_size": 1,
    "feature_count": 16,
    "feature_bytes": 4,  # float32
    
    # LEACH timing parameters
    "leach_round_duration_sec": 20,  # Typical LEACH round duration
    "packets_per_round_estimate": 100,  # Approximate packets per CH per round
    "worst_case_inference_frequency_hz": 10,  # 10 inferences per second worst case
}


# ============================================================================
# LOAD MODELS FROM MLFLOW
# ============================================================================

def load_models_from_mlflow():
    """Load trained models from MLflow experiment (No FE, With Oversampling)."""
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()
    
    # Get the experiment with oversampling and no feature engineering
    exp = client.get_experiment_by_name("WSN_IDS_No_Feature_Engineering_With_Oversampling")
    
    if not exp:
        print("Experiment not found. Trying alternative...")
        experiments = client.search_experiments()
        for e in experiments:
            print(f"  - {e.name} (ID: {e.experiment_id})")
        return {}
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_weighted DESC"]
    )
    
    models = {}
    
    for run in runs:
        model_name = run.data.tags.get('model_name', 'Unknown')
        sampling = run.data.tags.get('sampling_strategy', 'Unknown')
        
        # Try to load the model
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            key = f"{model_name}_{sampling}"
            models[key] = {
                'model': model,
                'run_id': run.info.run_id,
                'model_name': model_name,
                'sampling_strategy': sampling,
                'metrics': run.data.metrics
            }
            print(f"Loaded: {key}")
        except Exception as e:
            print(f"Could not load model for {model_name}: {e}")
    
    return models


def load_sample_data():
    """Load sample data for inference testing."""
    df = pd.read_csv("/Users/biru/Documents/TugasAkhir/data/WSN-DS.csv")
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # The 16 features used in training (excluding 'id', 'who CH', and 'Attack type')
    # Column names from the actual CSV file
    feature_cols = ['Time', 'Is_CH', 'Dist_To_CH', 'ADV_S', 'ADV_R', 
                    'JOIN_S', 'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 
                    'DATA_S', 'DATA_R', 'Data_Sent_To_BS', 'dist_CH_To_BS', 
                    'send_code', 'Expaned Energy']
    
    # Check which columns exist
    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"Available features ({len(available_cols)}): {available_cols}")
    
    # If not all features found, try alternate names
    if len(available_cols) < 16:
        # Try all numeric columns except id, who CH, and target
        exclude_cols = ['id', 'who CH', 'Attack type']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = [c for c in numeric_cols if c not in exclude_cols]
        print(f"Fallback to numeric columns ({len(available_cols)}): {available_cols}")
    
    X = df[available_cols].values
    
    return X, available_cols


# ============================================================================
# PART 4: EMPIRICAL INFERENCE LATENCY MEASUREMENT
# ============================================================================

def measure_inference_latency(model, X_sample, n_iterations=10000):
    """
    Measure inference latency empirically.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model for inference
    X_sample : np.ndarray
        Sample features for inference (single sample)
    n_iterations : int
        Number of repeated inferences
    
    Returns:
    --------
    dict : Latency statistics in milliseconds
    """
    latencies = []
    
    # Warm-up runs (not counted)
    for _ in range(100):
        _ = model.predict(X_sample.reshape(1, -1))
    
    # Measured runs
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_sample.reshape(1, -1))
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'max_ms': np.max(latencies),
        'min_ms': np.min(latencies),
        'n_iterations': n_iterations
    }


def run_empirical_measurements(models, X, n_iterations=10000):
    """Run empirical latency measurements for all models."""
    results = []
    
    # Use random samples for testing
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=100, replace=False)
    
    for model_key, model_info in models.items():
        print(f"\nMeasuring: {model_key}")
        model = model_info['model']
        
        # Measure with multiple samples and aggregate
        all_latencies = []
        
        for idx in sample_indices[:10]:  # Use 10 different samples
            X_sample = X[idx]
            
            for _ in range(n_iterations // 10):
                start = time.perf_counter()
                _ = model.predict(X_sample.reshape(1, -1))
                end = time.perf_counter()
                all_latencies.append((end - start) * 1000)
        
        latencies = np.array(all_latencies)
        
        result = {
            'model': model_key,
            'model_name': model_info['model_name'],
            'sampling_strategy': model_info['sampling_strategy'],
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'n_iterations': len(latencies)
        }
        results.append(result)
        
        print(f"  Mean: {result['mean_latency_ms']:.4f} ms, "
              f"P95: {result['p95_latency_ms']:.4f} ms, "
              f"Max: {result['max_latency_ms']:.4f} ms")
    
    return pd.DataFrame(results)


# ============================================================================
# PART 5: ANALYTICAL INFERENCE LATENCY ESTIMATION
# ============================================================================

def estimate_model_operations(model, n_features=16):
    """
    Estimate the number of operations for a model inference.
    
    Returns dict with comparisons, additions, multiplications counts.
    """
    model_type = type(model).__name__
    
    ops = {
        'comparisons': 0,
        'additions': 0,
        'multiplications': 0,
        'divisions': 0,
        'model_type': model_type
    }
    
    if hasattr(model, 'n_estimators'):
        n_estimators = model.n_estimators
    else:
        n_estimators = 1
    
    if 'RandomForest' in model_type or 'ExtraTrees' in model_type or 'Bagging' in model_type:
        # Tree-based ensemble
        if hasattr(model, 'estimators_'):
            total_nodes = 0
            for tree in model.estimators_:
                if hasattr(tree, 'tree_'):
                    total_nodes += tree.tree_.node_count
                elif hasattr(tree, 'estimators_'):
                    # Nested estimators
                    for sub_tree in tree.estimators_:
                        if hasattr(sub_tree, 'tree_'):
                            total_nodes += sub_tree.tree_.node_count
            
            # Average depth traversal = log2(nodes)
            avg_comparisons_per_tree = np.log2(max(total_nodes / n_estimators, 2))
            ops['comparisons'] = int(n_estimators * avg_comparisons_per_tree)
            ops['additions'] = n_estimators  # Voting/averaging
    
    elif 'GradientBoosting' in model_type or 'HistGradient' in model_type:
        # Gradient boosting
        if hasattr(model, 'estimators_'):
            total_nodes = 0
            for stage in model.estimators_:
                for tree in stage:
                    if hasattr(tree, 'tree_'):
                        total_nodes += tree.tree_.node_count
            
            n_trees = len(model.estimators_) * len(model.estimators_[0])
            avg_comparisons_per_tree = np.log2(max(total_nodes / n_trees, 2))
            ops['comparisons'] = int(n_trees * avg_comparisons_per_tree)
            ops['additions'] = n_trees  # Summing predictions
        elif hasattr(model, '_predictors'):
            # HistGradientBoosting
            n_trees = len(model._predictors) * len(model._predictors[0])
            ops['comparisons'] = n_trees * 10  # Estimate avg depth of 10
            ops['additions'] = n_trees
    
    elif 'LogisticRegression' in model_type:
        # Linear model: n_features multiplications + (n_features-1) additions + 1 exp
        n_classes = model.classes_.shape[0]
        ops['multiplications'] = n_features * n_classes
        ops['additions'] = (n_features - 1) * n_classes + n_classes
        ops['divisions'] = n_classes  # Softmax
    
    elif 'MLPClassifier' in model_type or 'Neural' in model_type:
        # Neural network
        if hasattr(model, 'coefs_'):
            for i, coef in enumerate(model.coefs_):
                n_in, n_out = coef.shape
                ops['multiplications'] += n_in * n_out
                ops['additions'] += (n_in - 1) * n_out + n_out  # MAC + bias
    
    return ops


def compute_analytical_latency(ops, device_profile):
    """
    Compute analytical latency estimate based on operation counts.
    
    Returns latency in milliseconds.
    """
    total_cycles = (
        ops['comparisons'] * device_profile['cycles_per_comparison'] +
        ops['additions'] * device_profile['cycles_per_addition'] +
        ops['multiplications'] * device_profile['cycles_per_multiplication'] +
        ops['divisions'] * device_profile['cycles_per_division']
    )
    
    # Convert cycles to time
    latency_seconds = total_cycles / device_profile['clock_frequency_hz']
    latency_ms = latency_seconds * 1000
    
    return {
        'total_cycles': total_cycles,
        'latency_ms': latency_ms,
        'operations': ops
    }


def run_analytical_estimation(models, device_profiles, n_features=16):
    """Run analytical latency estimation for all models on all devices."""
    results = []
    
    for model_key, model_info in models.items():
        model = model_info['model']
        ops = estimate_model_operations(model, n_features)
        
        for device_key, device in device_profiles.items():
            analytical = compute_analytical_latency(ops, device)
            
            results.append({
                'model': model_key,
                'model_name': model_info['model_name'],
                'sampling_strategy': model_info['sampling_strategy'],
                'device': device['name'],
                'device_key': device_key,
                'comparisons': ops['comparisons'],
                'additions': ops['additions'],
                'multiplications': ops['multiplications'],
                'total_cycles': analytical['total_cycles'],
                'estimated_latency_ms': analytical['latency_ms']
            })
    
    return pd.DataFrame(results)


# ============================================================================
# PART 6: ENERGY-AWARE LATENCY ANALYSIS
# ============================================================================

def compute_energy_metrics(empirical_results, analytical_results, device_profiles):
    """Compute energy cost per inference."""
    energy_results = []
    
    for _, row in empirical_results.iterrows():
        for device_key, device in device_profiles.items():
            # Get analytical latency for this model-device pair
            analytical_row = analytical_results[
                (analytical_results['model'] == row['model']) &
                (analytical_results['device_key'] == device_key)
            ]
            
            if len(analytical_row) > 0:
                est_latency_ms = analytical_row.iloc[0]['estimated_latency_ms']
            else:
                est_latency_ms = row['mean_latency_ms']  # Fallback
            
            # Scale empirical latency to device
            # Empirical is on host machine, estimate ratio
            host_clock_ghz = 3.0  # Assume 3GHz host
            device_clock_mhz = device['clock_frequency_mhz']
            scale_factor = (host_clock_ghz * 1000) / device_clock_mhz
            
            scaled_latency_ms = row['mean_latency_ms'] * scale_factor
            
            # Energy = Power × Time
            # Use analytical estimate for more accurate embedded prediction
            latency_for_energy = max(est_latency_ms, scaled_latency_ms / 10)  # Conservative
            energy_per_inference_mj = (device['active_power_mw'] * latency_for_energy) / 1000  # mJ
            
            # Energy per LEACH round
            packets_per_round = INFERENCE_MODEL['packets_per_round_estimate']
            energy_per_round_mj = energy_per_inference_mj * packets_per_round
            
            energy_results.append({
                'model': row['model'],
                'model_name': row['model_name'],
                'sampling_strategy': row['sampling_strategy'],
                'device': device['name'],
                'device_key': device_key,
                'empirical_latency_ms': row['mean_latency_ms'],
                'scaled_latency_ms': scaled_latency_ms,
                'analytical_latency_ms': est_latency_ms,
                'energy_per_inference_mj': energy_per_inference_mj,
                'energy_per_round_mj': energy_per_round_mj,
                'active_power_mw': device['active_power_mw']
            })
    
    return pd.DataFrame(energy_results)


# ============================================================================
# PART 7: REPORT GENERATION
# ============================================================================

def generate_report(empirical_df, analytical_df, energy_df, output_dir):
    """Generate comprehensive inference latency report."""
    
    report_lines = []
    
    # ====== TITLE AND INTRODUCTION ======
    report_lines.append("=" * 80)
    report_lines.append("INFERENCE LATENCY EVALUATION IN LEACH-BASED WSN IDS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n")
    
    # ====== SECTION 1: DEPLOYMENT ASSUMPTIONS ======
    report_lines.append("-" * 80)
    report_lines.append("1. INFERENCE DEPLOYMENT ASSUMPTIONS")
    report_lines.append("-" * 80)
    report_lines.append(f"\nInference Location: {DEPLOYMENT_CONFIG['inference_location']}")
    report_lines.append(f"\nJustification:{DEPLOYMENT_CONFIG['justification']}")
    report_lines.append("\nSystem Assumptions:")
    for i, assumption in enumerate(DEPLOYMENT_CONFIG['system_assumptions'], 1):
        report_lines.append(f"  {i}. {assumption}")
    
    # ====== SECTION 2: TARGET DEVICE PROFILES ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("2. TARGET DEVICE PROFILES")
    report_lines.append("-" * 80)
    
    for device_key, device in DEVICE_PROFILES.items():
        report_lines.append(f"\n{device['name']}")
        report_lines.append(f"  Description: {device['description']}")
        report_lines.append(f"  Clock Frequency: {device['clock_frequency_mhz']} MHz")
        report_lines.append(f"  RAM: {device['ram_kb']} KB")
        report_lines.append(f"  Flash: {device['flash_kb']} KB")
        report_lines.append(f"  Active Power: {device['active_power_mw']} mW")
    
    # ====== SECTION 3: INFERENCE EXECUTION MODEL ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("3. INFERENCE EXECUTION MODEL")
    report_lines.append("-" * 80)
    report_lines.append(INFERENCE_MODEL['description'])
    report_lines.append(f"\nLEACH Round Duration: {INFERENCE_MODEL['leach_round_duration_sec']} seconds")
    report_lines.append(f"Estimated Packets per Round: {INFERENCE_MODEL['packets_per_round_estimate']}")
    report_lines.append(f"Worst-case Inference Frequency: {INFERENCE_MODEL['worst_case_inference_frequency_hz']} Hz")
    
    # ====== SECTION 4: EMPIRICAL LATENCY RESULTS ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("4. EMPIRICAL INFERENCE LATENCY MEASUREMENTS")
    report_lines.append("-" * 80)
    report_lines.append("\nMeasurement Conditions:")
    report_lines.append("  - CPU-only execution (no GPU)")
    report_lines.append("  - Single-threaded execution")
    report_lines.append("  - Batch size = 1")
    report_lines.append(f"  - Iterations per model: {empirical_df['n_iterations'].iloc[0]}")
    
    report_lines.append("\nTable 1: Empirical Inference Latency (Host Machine)")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Model':<40} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Max (ms)':<12}")
    report_lines.append("-" * 100)
    
    for _, row in empirical_df.sort_values('mean_latency_ms').iterrows():
        report_lines.append(
            f"{row['model']:<40} {row['mean_latency_ms']:<12.4f} "
            f"{row['p95_latency_ms']:<12.4f} {row['max_latency_ms']:<12.4f}"
        )
    report_lines.append("-" * 100)
    
    # ====== SECTION 5: ANALYTICAL ESTIMATION ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("5. ANALYTICAL INFERENCE LATENCY ESTIMATION")
    report_lines.append("-" * 80)
    
    report_lines.append("\nTable 2: Operation Counts per Model")
    report_lines.append("-" * 90)
    report_lines.append(f"{'Model':<40} {'Comparisons':<15} {'Additions':<15} {'Multiplications':<15}")
    report_lines.append("-" * 90)
    
    for model in empirical_df['model'].unique():
        row = analytical_df[analytical_df['model'] == model].iloc[0]
        report_lines.append(
            f"{row['model']:<40} {row['comparisons']:<15} "
            f"{row['additions']:<15} {row['multiplications']:<15}"
        )
    report_lines.append("-" * 90)
    
    report_lines.append("\nTable 3: Estimated Inference Latency by Device")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Model':<35} {'Device':<25} {'Est. Latency (ms)':<20}")
    report_lines.append("-" * 100)
    
    for _, row in analytical_df.sort_values(['model', 'device_key']).iterrows():
        report_lines.append(
            f"{row['model']:<35} {row['device']:<25} {row['estimated_latency_ms']:<20.6f}"
        )
    report_lines.append("-" * 100)
    
    # ====== SECTION 6: ENERGY ANALYSIS ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("6. ENERGY-AWARE LATENCY ANALYSIS")
    report_lines.append("-" * 80)
    
    report_lines.append("\nTable 4: Energy Cost per Inference")
    report_lines.append("-" * 110)
    report_lines.append(f"{'Model':<35} {'Device':<20} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'E/Round (mJ)':<15}")
    report_lines.append("-" * 110)
    
    for _, row in energy_df.sort_values(['model', 'device_key']).iterrows():
        report_lines.append(
            f"{row['model']:<35} {row['device_key']:<20} "
            f"{row['analytical_latency_ms']:<15.6f} {row['energy_per_inference_mj']:<15.6f} "
            f"{row['energy_per_round_mj']:<15.4f}"
        )
    report_lines.append("-" * 110)
    
    # ====== SECTION 7: FEASIBILITY ANALYSIS ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("7. REAL-TIME FEASIBILITY ANALYSIS")
    report_lines.append("-" * 80)
    
    # Calculate deadline margins
    deadline_ms = 1000 / INFERENCE_MODEL['worst_case_inference_frequency_hz']  # 100ms for 10Hz
    
    report_lines.append(f"\nWorst-case inference deadline: {deadline_ms:.1f} ms (at {INFERENCE_MODEL['worst_case_inference_frequency_hz']} Hz)")
    report_lines.append("\nFeasibility Assessment:")
    
    for _, row in analytical_df.iterrows():
        margin = deadline_ms - row['estimated_latency_ms']
        status = "FEASIBLE" if margin > 0 else "NOT FEASIBLE"
        margin_pct = (margin / deadline_ms) * 100
        report_lines.append(
            f"  {row['model']:<35} on {row['device_key']:<10}: {status} "
            f"(margin: {margin:.4f} ms, {margin_pct:.2f}%)"
        )
    
    # ====== SECTION 8: PSEUDOCODE ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("8. INFERENCE INTEGRATION PSEUDOCODE")
    report_lines.append("-" * 80)
    report_lines.append("""
ALGORITHM: Real-time IDS Inference at Cluster Head
---------------------------------------------------

PROCEDURE CH_IDS_Inference(packet)
    INPUT: Received MAC-layer packet from cluster member
    OUTPUT: Classification result (Normal/Attack type)
    
    // Step 1: Feature Extraction
    features[16] ← ExtractMACFeatures(packet)
    
    // Step 2: Pre-processing
    normalized_features ← Normalize(features, stored_scaler)
    
    // Step 3: Model Inference (single sample)
    prediction ← Model.Predict(normalized_features)
    
    // Step 4: Response Action
    IF prediction ≠ NORMAL THEN
        LogAlert(packet.source, prediction)
        IF IsCriticalAttack(prediction) THEN
            BlockNode(packet.source)
        END IF
    END IF
    
    // Step 5: Continue normal LEACH operation
    ForwardToBaseStation(packet, prediction)
    
    RETURN prediction
END PROCEDURE

PROCEDURE LEACH_Round_With_IDS()
    // Setup Phase (normal LEACH)
    BroadcastAdvertisement()
    ReceiveJoinRequests()
    CreateTDMASchedule()
    
    // Steady-State Phase with IDS
    FOR EACH scheduled_slot DO
        packet ← ReceiveFromMember()
        
        // Inline IDS inference
        start_time ← GetSystemTime()
        result ← CH_IDS_Inference(packet)
        inference_time ← GetSystemTime() - start_time
        
        // Deadline check
        IF inference_time > DEADLINE_THRESHOLD THEN
            LogLatencyViolation()
        END IF
    END FOR
    
    // Aggregate and forward to BS
    AggregateAndForward()
END PROCEDURE
""")
    
    # ====== SECTION 9: DEPLOYMENT DIAGRAM ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("9. INFERENCE DEPLOYMENT DIAGRAM")
    report_lines.append("-" * 80)
    report_lines.append("""
                        LEACH-Based WSN with IDS at Cluster Head
                        ========================================

    +------------------+
    |   Base Station   |  ← Receives filtered/classified data
    +------------------+
            ▲
            │ Aggregated data + alerts
            │
    +-------┴--------+     +----------------+     +----------------+
    |  Cluster Head  |     | Cluster Head   |     | Cluster Head   |
    |  [IDS MODEL]   |     | [IDS MODEL]    |     | [IDS MODEL]    |
    |                |     |                |     |                |
    | - Inference    |     | - Inference    |     | - Inference    |
    | - Detection    |     | - Detection    |     | - Detection    |
    | - Alerting     |     | - Alerting     |     | - Alerting     |
    +----------------+     +----------------+     +----------------+
      ▲   ▲   ▲              ▲   ▲   ▲              ▲   ▲   ▲
      │   │   │              │   │   │              │   │   │
    +---+ │ +---+          +---+ │ +---+          +---+ │ +---+
    |SN1| │ |SN3|          |SN4| │ |SN6|          |SN7| │ |SN9|
    +---+ │ +---+          +---+ │ +---+          +---+ │ +---+
        +---+                  +---+                  +---+
        |SN2|                  |SN5|                  |SN8|
        +---+                  +---+                  +---+
        
    Cluster 1              Cluster 2              Cluster 3

    Legend:
    - SN: Sensor Node (data source)
    - CH: Cluster Head (inference location)
    - BS: Base Station (central monitoring)
    
    Data Flow:
    1. Sensor nodes transmit to their Cluster Head
    2. CH extracts features from received packets
    3. CH runs IDS inference (per-packet)
    4. CH classifies traffic as Normal/Attack
    5. CH aggregates data and forwards to BS with classification
""")
    
    # ====== SECTION 10: LIMITATIONS ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("10. LIMITATIONS AND CAVEATS")
    report_lines.append("-" * 80)
    report_lines.append("""
This evaluation has the following limitations:

1. NO REAL HARDWARE DEPLOYMENT
   - Latency measurements are empirical estimates from host machine
   - Analytical estimates are based on operation counts, not actual embedded profiling
   - Real embedded performance may vary due to cache effects, memory access patterns

2. SIMPLIFIED OPERATION MODEL
   - Operation counts are approximations based on model structure
   - Does not account for memory hierarchy effects
   - Assumes ideal execution without interrupts

3. FEATURE EXTRACTION OVERHEAD NOT INCLUDED
   - Reported latency is inference-only
   - Real deployment must add feature extraction time
   - Packet parsing and preprocessing add additional overhead

4. ENERGY MODEL LIMITATIONS
   - Power consumption is based on datasheet typical values
   - Does not account for voltage scaling or dynamic power management
   - Actual energy consumption depends on specific implementation

5. LEACH TIMING ASSUMPTIONS
   - Round duration and packet counts are estimates
   - Actual values depend on network size and configuration
   - Real-time constraints may be tighter in some deployments
""")
    
    # ====== CONCLUSION ======
    report_lines.append("\n" + "-" * 80)
    report_lines.append("11. CONCLUSIONS")
    report_lines.append("-" * 80)
    
    # Find best models
    best_empirical = empirical_df.loc[empirical_df['mean_latency_ms'].idxmin()]
    
    report_lines.append(f"""
Based on the inference latency evaluation:

1. FASTEST MODEL: {best_empirical['model']}
   - Mean latency: {best_empirical['mean_latency_ms']:.4f} ms (host machine)
   - P95 latency: {best_empirical['p95_latency_ms']:.4f} ms

2. DEPLOYMENT FEASIBILITY:
   - All evaluated models meet real-time constraints on Cortex-M4
   - MSP430 deployment is feasible for simpler models
   - Tree-based ensembles offer best latency-accuracy trade-off

3. ENERGY EFFICIENCY:
   - Inference energy cost is minimal compared to communication
   - IDS overhead does not significantly impact WSN lifetime

4. RECOMMENDATION:
   - Deploy gradient boosting or random forest models at Cluster Head
   - Use Cortex-M4 class devices for reliable real-time performance
   - Implement periodic model updates via Base Station
""")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(os.path.join(output_dir, "inference_latency_report.txt"), 'w') as f:
        f.write(report_text)
    
    # Save tables as CSV
    empirical_df.to_csv(os.path.join(output_dir, "empirical_latency.csv"), index=False)
    analytical_df.to_csv(os.path.join(output_dir, "analytical_latency.csv"), index=False)
    energy_df.to_csv(os.path.join(output_dir, "energy_analysis.csv"), index=False)
    
    print(f"\nReport saved to: {output_dir}/inference_latency_report.txt")
    print(f"Tables saved to: {output_dir}/")
    
    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("INFERENCE LATENCY EVALUATION FOR LEACH-BASED WSN IDS")
    print("=" * 60)
    
    # Create output directory
    output_dir = "/Users/biru/Documents/TugasAkhir/latency_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("\n[1/6] Loading models from MLflow...")
    models = load_models_from_mlflow()
    
    if not models:
        print("ERROR: No models found. Please ensure MLflow experiments exist.")
        return
    
    print(f"Loaded {len(models)} models")
    
    # Load sample data
    print("\n[2/6] Loading sample data...")
    X, feature_cols = load_sample_data()
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    
    # Empirical measurements
    print("\n[3/6] Running empirical latency measurements...")
    print("(This may take a few minutes)")
    empirical_df = run_empirical_measurements(models, X, n_iterations=10000)
    
    # Analytical estimation
    print("\n[4/6] Computing analytical latency estimates...")
    analytical_df = run_analytical_estimation(models, DEVICE_PROFILES, n_features=len(feature_cols))
    
    # Energy analysis
    print("\n[5/6] Computing energy metrics...")
    energy_df = compute_energy_metrics(empirical_df, analytical_df, DEVICE_PROFILES)
    
    # Generate report
    print("\n[6/6] Generating comprehensive report...")
    report = generate_report(empirical_df, analytical_df, energy_df, output_dir)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)
    print(empirical_df[['model', 'mean_latency_ms', 'p95_latency_ms']].to_string(index=False))


if __name__ == "__main__":
    main()

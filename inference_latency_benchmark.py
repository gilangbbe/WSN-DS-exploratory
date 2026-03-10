#!/usr/bin/env python3
"""
Inference Latency Benchmark — Host Machine Evaluation
Measures actual inference time of Conservative SMOTE models (no feature engineering)
on the host machine using scikit-learn.

Models loaded from MLflow pipeline artifacts.
"""

import os
import sys
import time
import json
import platform
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ─────────────────────── Configuration ───────────────────────

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "WSN-DS.csv"
OUTPUT_DIR = BASE_DIR / "latency_evaluation"

# No-FE preprocessors
SCALER_PATH = BASE_DIR / "mlflow_artifacts_no_fe" / "standard_scaler_no_fe.pkl"
ENCODER_PATH = BASE_DIR / "mlflow_artifacts_no_fe" / "label_encoder_no_fe.pkl"

# MLflow tracking
MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")

FEATURE_NAMES = [
    "Time", "Is_CH", "Dist_To_CH", "ADV_S", "ADV_R", "JOIN_S", "JOIN_R",
    "SCH_S", "SCH_R", "Rank", "DATA_S", "DATA_R", "Data_Sent_To_BS",
    "dist_CH_To_BS", "send_code", "Expaned Energy"
]

TARGET_COL = "Attack type"
REDUNDANT_FEATURES = ["id", "who CH"]

# Models to evaluate — MLflow registered model names (Conservative SMOTE, no FE)
MLFLOW_MODELS = {
    "Extra Trees": "WSN_IDS_NoFE_Extra_Trees_Conservative_SMOTE",
    "Random Forest": "WSN_IDS_NoFE_Random_Forest_Conservative_SMOTE",
    "Gradient Boosting": "WSN_IDS_NoFE_Gradient_Boosting_Conservative_SMOTE",
    "HistGradient Boosting": "WSN_IDS_NoFE_HistGradient_Boosting_Conservative_SMOTE",
    "Logistic Regression": "WSN_IDS_NoFE_Logistic_Regression_Conservative_SMOTE",
    "Neural Network": "WSN_IDS_NoFE_Neural_Network_Conservative_SMOTE",
}

# Benchmark parameters
N_WARMUP = 100          # Warmup iterations (not timed)
N_REPEAT = 1000         # Timed iterations per sample
N_SAMPLES_BATCH = [1, 10, 100, 1000]  # Batch sizes to test
N_SINGLE_REPEAT = 5000  # Repeats for single-sample latency


# ─────────────────────── System Info ───────────────────────

def get_system_info() -> Dict:
    """Collect host system information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # macOS specific
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True
        ).strip()
        info["cpu"] = cpu
    except Exception:
        info["cpu"] = platform.processor()

    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
        info["ram_gb"] = round(mem_bytes / (1024**3), 1)
    except Exception:
        info["ram_gb"] = "unknown"

    try:
        cores_perf = int(subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True
        ).strip())
        cores_eff = int(subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel1.logicalcpu"], text=True
        ).strip())
        info["cores"] = f"{cores_perf}P + {cores_eff}E"
    except Exception:
        try:
            cores = int(subprocess.check_output(
                ["sysctl", "-n", "hw.logicalcpu"], text=True
            ).strip())
            info["cores"] = str(cores)
        except Exception:
            info["cores"] = "unknown"

    # Package versions
    import sklearn
    info["sklearn_version"] = sklearn.__version__
    info["numpy_version"] = np.__version__
    info["joblib_version"] = joblib.__version__

    return info


# ─────────────────────── Data Loading ───────────────────────

def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Load WSN-DS dataset and apply preprocessing (matching MLflow pipeline)."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Drop redundant
    df = df.drop(columns=[c for c in REDUNDANT_FEATURES if c in df.columns], errors="ignore")

    # Clean
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate features and target
    X = df[FEATURE_NAMES].values
    y = df[TARGET_COL].values

    print(f"  Dataset shape: {X.shape}")
    print(f"  Classes: {np.unique(y)}")

    # Load no-FE preprocessors
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print(f"  Scaler expects {scaler.n_features_in_} features")

    y_encoded = encoder.transform(y)

    # Stratified split (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


# ─────────────────────── Benchmarking ───────────────────────

def benchmark_single_sample(model, X_single: np.ndarray, n_warmup: int, n_repeat: int) -> Dict:
    """Benchmark single-sample inference latency."""
    # Warmup
    for _ in range(n_warmup):
        model.predict(X_single)

    # Timed runs
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter_ns()
        model.predict(X_single)
        end = time.perf_counter_ns()
        times.append(end - start)

    times_us = np.array(times) / 1000.0  # ns -> µs

    return {
        "n_repeat": n_repeat,
        "mean_us": float(np.mean(times_us)),
        "median_us": float(np.median(times_us)),
        "std_us": float(np.std(times_us)),
        "min_us": float(np.min(times_us)),
        "max_us": float(np.max(times_us)),
        "p5_us": float(np.percentile(times_us, 5)),
        "p25_us": float(np.percentile(times_us, 25)),
        "p75_us": float(np.percentile(times_us, 75)),
        "p95_us": float(np.percentile(times_us, 95)),
        "p99_us": float(np.percentile(times_us, 99)),
    }


def benchmark_batch(model, X_batch: np.ndarray, n_warmup: int, n_repeat: int) -> Dict:
    """Benchmark batch inference latency."""
    batch_size = X_batch.shape[0]

    # Warmup
    for _ in range(min(n_warmup, 50)):
        model.predict(X_batch)

    # Timed runs
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter_ns()
        model.predict(X_batch)
        end = time.perf_counter_ns()
        times.append(end - start)

    times_us = np.array(times) / 1000.0
    per_sample_us = times_us / batch_size

    return {
        "batch_size": batch_size,
        "n_repeat": n_repeat,
        "total_mean_us": float(np.mean(times_us)),
        "total_median_us": float(np.median(times_us)),
        "per_sample_mean_us": float(np.mean(per_sample_us)),
        "per_sample_median_us": float(np.median(per_sample_us)),
        "throughput_per_sec": float(1_000_000 / np.mean(per_sample_us)) if np.mean(per_sample_us) > 0 else 0,
    }


def evaluate_model(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray, encoder) -> Dict:
    """Full evaluation: accuracy + latency."""
    print(f"\n  Evaluating: {model_name}")

    # ── Accuracy ──
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    class_names = encoder.classes_
    per_class = {str(cls): float(f1) for cls, f1 in zip(class_names, f1_per_class)}

    print(f"    Accuracy: {accuracy:.4f}, F1_macro: {f1_macro:.4f}")

    # ── Single-sample latency ──
    # Use first test sample
    X_single = X_test[0:1]
    print(f"    Benchmarking single-sample ({N_SINGLE_REPEAT} iterations)...")
    single_result = benchmark_single_sample(model, X_single, N_WARMUP, N_SINGLE_REPEAT)
    print(f"    -> Median: {single_result['median_us']:.1f} µs, "
          f"Mean: {single_result['mean_us']:.1f} µs, "
          f"P95: {single_result['p95_us']:.1f} µs")

    # ── Batch latency ──
    batch_results = {}
    for batch_size in N_SAMPLES_BATCH:
        if batch_size > X_test.shape[0]:
            continue
        X_batch = X_test[:batch_size]
        n_rep = max(100, N_REPEAT // batch_size)
        print(f"    Benchmarking batch={batch_size} ({n_rep} iterations)...")
        batch_results[batch_size] = benchmark_batch(model, X_batch, N_WARMUP, n_rep)
        tp = batch_results[batch_size]["throughput_per_sec"]
        print(f"    -> {batch_results[batch_size]['per_sample_mean_us']:.2f} µs/sample, "
              f"{tp:,.0f} inf/s")

    # ── Model metadata ──
    model_info = {}
    if hasattr(model, "n_estimators"):
        model_info["n_estimators"] = model.n_estimators
    if hasattr(model, "max_depth"):
        model_info["max_depth"] = model.max_depth
    if hasattr(model, "estimators_"):
        model_info["n_estimators_actual"] = len(model.estimators_)
    # HistGB stores estimators differently
    if hasattr(model, "_predictors"):
        n_iters = len(model._predictors)
        model_info["n_iterations"] = n_iters
        if n_iters > 0:
            model_info["n_trees_per_iteration"] = len(model._predictors[0])
            model_info["n_trees_total"] = n_iters * len(model._predictors[0])

    return {
        "model_name": model_name,
        "model_info": model_info,
        "quality": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_per_class": per_class,
        },
        "single_sample": single_result,
        "batch_results": {str(k): v for k, v in batch_results.items()},
    }


# ─────────────────────── Main ───────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("INFERENCE LATENCY BENCHMARK — HOST MACHINE")
    print("Conservative SMOTE Models (No Feature Engineering)")
    print("=" * 80)

    # System info
    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['cpu']}")
    print(f"RAM: {sys_info['ram_gb']} GB")
    print(f"Cores: {sys_info['cores']}")
    print(f"Python: {sys_info['python_version']}")
    print(f"scikit-learn: {sys_info['sklearn_version']}")

    # Load data
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()

    # Load models from MLflow registry and evaluate
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    all_results = []

    for model_name, registry_name in MLFLOW_MODELS.items():
        print(f"\n{'─' * 60}")
        print(f"  Loading from MLflow: {registry_name}")
        try:
            model = mlflow.sklearn.load_model(f"models:/{registry_name}/1")
            print(f"  -> {type(model).__name__}, n_features_in_={getattr(model, 'n_features_in_', '?')}")
        except Exception as e:
            print(f"  SKIP: {model_name} — {e}")
            continue

        result = evaluate_model(model_name, model, X_test, y_test, encoder)
        result["mlflow_registry"] = registry_name
        all_results.append(result)

    # ── Save JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"host_inference_benchmark_{timestamp}.json"
    output_data = {
        "benchmark_info": {
            "description": "Inference latency benchmark on host machine",
            "models": "Conservative SMOTE (no feature engineering)",
            "dataset": "WSN-DS (test set, 20%)",
            "timestamp": datetime.now().isoformat(),
            "n_warmup": N_WARMUP,
            "n_single_repeat": N_SINGLE_REPEAT,
            "n_batch_repeat": N_REPEAT,
            "batch_sizes": N_SAMPLES_BATCH,
        },
        "system_info": sys_info,
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # ── Generate Markdown Report ──
    md_path = OUTPUT_DIR / "host_inference_benchmark.md"
    generate_markdown_report(output_data, md_path)
    print(f"Markdown report saved: {md_path}")

    # ── Summary Table ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1_macro':>10} {'Median(µs)':>12} {'P95(µs)':>10} {'Throughput':>14}")
    print("─" * 85)
    for r in all_results:
        q = r["quality"]
        s = r["single_sample"]
        tp = 1_000_000 / s["median_us"] if s["median_us"] > 0 else 0
        print(f"{r['model_name']:<25} {q['accuracy']:>10.4f} {q['f1_macro']:>10.4f} "
              f"{s['median_us']:>12.1f} {s['p95_us']:>10.1f} {tp:>12,.0f} /s")


def generate_markdown_report(data: Dict, output_path: Path):
    """Generate comprehensive markdown report."""
    sys_info = data["system_info"]
    bench_info = data["benchmark_info"]
    results = data["results"]

    lines = []
    lines.append("# Inference Latency Benchmark — Host Machine Evaluation")
    lines.append("")
    lines.append("Empirical measurement of inference latency for Conservative SMOTE tree ensemble models")
    lines.append("(no feature engineering) on the host development machine.")
    lines.append("")

    # System info
    lines.append("## System Specifications")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| **CPU** | {sys_info['cpu']} |")
    lines.append(f"| **Architecture** | {sys_info['machine']} |")
    lines.append(f"| **RAM** | {sys_info['ram_gb']} GB |")
    lines.append(f"| **Cores** | {sys_info['cores']} |")
    lines.append(f"| **OS** | {sys_info['platform']} |")
    lines.append(f"| **Python** | {sys_info['python_version']} |")
    lines.append(f"| **scikit-learn** | {sys_info['sklearn_version']} |")
    lines.append(f"| **NumPy** | {sys_info['numpy_version']} |")
    lines.append("")

    # Benchmark config
    lines.append("## Benchmark Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Dataset** | WSN-DS (test set, 20% stratified split) |")
    lines.append(f"| **Features** | 16 (no feature engineering) |")
    lines.append(f"| **Oversampling** | Conservative SMOTE |")
    lines.append(f"| **Warmup iterations** | {bench_info['n_warmup']} |")
    lines.append(f"| **Single-sample repeats** | {bench_info['n_single_repeat']} |")
    lines.append(f"| **Batch sizes tested** | {bench_info['batch_sizes']} |")
    lines.append(f"| **Timestamp** | {bench_info['timestamp']} |")
    lines.append("")

    # Model quality summary
    lines.append("## Model Quality (Test Set)")
    lines.append("")
    lines.append("| Model | Accuracy | F1 Macro | F1 Weighted |")
    lines.append("|---|---|---|---|")
    for r in results:
        q = r["quality"]
        lines.append(f"| {r['model_name']} | {q['accuracy']:.4f} | {q['f1_macro']:.4f} | {q['f1_weighted']:.4f} |")
    lines.append("")

    # Per-class F1
    lines.append("### Per-Class F1 Scores")
    lines.append("")
    classes = list(results[0]["quality"]["f1_per_class"].keys()) if results else []
    header = "| Model | " + " | ".join(classes) + " |"
    sep = "|---|" + "|".join(["---"] * len(classes)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        vals = [f"{r['quality']['f1_per_class'][c]:.4f}" for c in classes]
        lines.append(f"| {r['model_name']} | " + " | ".join(vals) + " |")
    lines.append("")

    # Single-sample latency
    lines.append("## Single-Sample Inference Latency")
    lines.append("")
    lines.append("Latency for predicting a single sample (1 row, 16 features).")
    lines.append("")
    lines.append("| Model | Mean (µs) | Median (µs) | Std (µs) | Min (µs) | P5 (µs) | P95 (µs) | P99 (µs) | Max (µs) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r["single_sample"]
        lines.append(
            f"| {r['model_name']} "
            f"| {s['mean_us']:.1f} "
            f"| {s['median_us']:.1f} "
            f"| {s['std_us']:.1f} "
            f"| {s['min_us']:.1f} "
            f"| {s['p5_us']:.1f} "
            f"| {s['p95_us']:.1f} "
            f"| {s['p99_us']:.1f} "
            f"| {s['max_us']:.1f} |"
        )
    lines.append("")

    # Throughput summary
    lines.append("### Single-Sample Throughput")
    lines.append("")
    lines.append("| Model | Median Latency (µs) | Throughput (inferences/sec) |")
    lines.append("|---|---|---|")
    for r in results:
        med = r["single_sample"]["median_us"]
        tp = 1_000_000 / med if med > 0 else 0
        lines.append(f"| {r['model_name']} | {med:.1f} | {tp:,.0f} |")
    lines.append("")

    # Batch latency
    lines.append("## Batch Inference Latency")
    lines.append("")
    lines.append("Amortized per-sample latency when processing batches of samples.")
    lines.append("")

    for r in results:
        lines.append(f"### {r['model_name']}")
        lines.append("")
        lines.append("| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |")
        lines.append("|---|---|---|---|")

        for bs_str, b in sorted(r["batch_results"].items(), key=lambda x: int(x[0])):
            lines.append(
                f"| {b['batch_size']} "
                f"| {b['total_mean_us']:.1f} "
                f"| {b['per_sample_mean_us']:.2f} "
                f"| {b['throughput_per_sec']:,.0f} |"
            )
        lines.append("")

    # Model info
    lines.append("## Model Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r['model_name']}")
        lines.append("")
        if r["model_info"]:
            lines.append("| Parameter | Value |")
            lines.append("|---|---|")
            for k, v in r["model_info"].items():
                k_nice = k.replace("_", " ").title()
                lines.append(f"| {k_nice} | {v} |")
            lines.append("")
        lines.append(f"MLflow Registry: `{r.get('mlflow_registry', 'N/A')}`")
        lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Timing method**: `time.perf_counter_ns()` (nanosecond precision)")
    lines.append("- **Warmup**: Each model is warmed up before timing to ensure JIT and cache effects are minimized")
    lines.append("- **Single-sample**: Measures the overhead of calling `model.predict()` with a single row — "
                 "this includes Python function call overhead, NumPy array processing, and the actual tree traversal")
    lines.append("- **Batch inference**: Amortizes per-sample cost by processing multiple samples at once, "
                 "which is more representative of real-world throughput")
    lines.append("- **Host vs embedded**: These measurements reflect scikit-learn's Python implementation on a "
                 "modern desktop CPU and are NOT directly comparable to embedded C implementations on Cortex-M4 or MSP430. "
                 "The Python overhead is significant for single-sample inference.")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

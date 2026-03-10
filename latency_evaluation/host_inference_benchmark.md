# Inference Latency Benchmark — Host Machine Evaluation

Empirical measurement of inference latency for Conservative SMOTE tree ensemble models
(no feature engineering) on the host development machine.

## System Specifications

| Parameter | Value |
|---|---|
| **CPU** | Apple M4 Pro |
| **Architecture** | arm64 |
| **RAM** | 24.0 GB |
| **Cores** | 10P + 4E |
| **OS** | macOS-26.3-arm64-arm-64bit |
| **Python** | 3.9.6 |
| **scikit-learn** | 1.6.1 |
| **NumPy** | 2.0.2 |

## Benchmark Configuration

| Parameter | Value |
|---|---|
| **Dataset** | WSN-DS (test set, 20% stratified split) |
| **Features** | 16 (no feature engineering) |
| **Oversampling** | Conservative SMOTE |
| **Warmup iterations** | 100 |
| **Single-sample repeats** | 5000 |
| **Batch sizes tested** | [1, 10, 100, 1000] |
| **Timestamp** | 2026-02-18T11:20:34.963422 |

## Model Quality (Test Set)

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| Extra Trees | 0.9920 | 0.9512 | 0.9921 |
| Random Forest | 0.9977 | 0.9847 | 0.9977 |
| Gradient Boosting | 0.9991 | 0.9943 | 0.9991 |
| HistGradient Boosting | 0.9955 | 0.9720 | 0.9956 |
| Logistic Regression | 0.9643 | 0.8392 | 0.9676 |
| Neural Network | 0.9955 | 0.9662 | 0.9955 |

### Per-Class F1 Scores

| Model | Blackhole | Flooding | Grayhole | Normal | TDMA |
|---|---|---|---|---|---|
| Extra Trees | 0.9380 | 0.9432 | 0.9251 | 0.9968 | 0.9530 |
| Random Forest | 0.9943 | 0.9670 | 0.9887 | 0.9988 | 0.9746 |
| Gradient Boosting | 0.9973 | 0.9906 | 0.9961 | 0.9996 | 0.9880 |
| HistGradient Boosting | 0.9868 | 0.9626 | 0.9827 | 0.9978 | 0.9303 |
| Logistic Regression | 0.7589 | 0.9333 | 0.5983 | 0.9863 | 0.9193 |
| Neural Network | 0.9572 | 0.9474 | 0.9706 | 0.9983 | 0.9575 |

## Single-Sample Inference Latency

Latency for predicting a single sample (1 row, 16 features).

| Model | Mean (µs) | Median (µs) | Std (µs) | Min (µs) | P5 (µs) | P95 (µs) | P99 (µs) | Max (µs) |
|---|---|---|---|---|---|---|---|---|
| Extra Trees | 13942.0 | 14126.9 | 1114.6 | 11455.0 | 12398.6 | 14304.4 | 14432.6 | 35879.2 |
| Random Forest | 13964.8 | 14124.7 | 1102.1 | 11452.3 | 12479.6 | 14298.5 | 14440.2 | 36278.4 |
| Gradient Boosting | 308.4 | 308.1 | 15.8 | 287.5 | 291.5 | 326.4 | 350.7 | 744.1 |
| HistGradient Boosting | 8495.1 | 8303.4 | 1854.4 | 6498.9 | 7468.3 | 9422.8 | 15266.4 | 92696.3 |
| Logistic Regression | 35.5 | 35.2 | 3.0 | 32.9 | 34.7 | 36.1 | 40.2 | 118.2 |
| Neural Network | 55.2 | 54.3 | 4.7 | 53.0 | 53.6 | 60.1 | 64.8 | 261.4 |

### Single-Sample Throughput

| Model | Median Latency (µs) | Throughput (inferences/sec) |
|---|---|---|
| Extra Trees | 14126.9 | 71 |
| Random Forest | 14124.7 | 71 |
| Gradient Boosting | 308.1 | 3,245 |
| HistGradient Boosting | 8303.4 | 120 |
| Logistic Regression | 35.2 | 28,369 |
| Neural Network | 54.3 | 18,405 |

## Batch Inference Latency

Amortized per-sample latency when processing batches of samples.

### Extra Trees

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 13893.9 | 13893.89 | 72 |
| 10 | 14137.0 | 1413.70 | 707 |
| 100 | 13529.9 | 135.30 | 7,391 |
| 1000 | 13967.6 | 13.97 | 71,594 |

### Random Forest

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 14117.7 | 14117.74 | 71 |
| 10 | 14091.1 | 1409.11 | 710 |
| 100 | 13321.3 | 133.21 | 7,507 |
| 1000 | 14094.3 | 14.09 | 70,951 |

### Gradient Boosting

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 306.9 | 306.87 | 3,259 |
| 10 | 394.1 | 39.41 | 25,377 |
| 100 | 1901.3 | 19.01 | 52,595 |
| 1000 | 12476.2 | 12.48 | 80,152 |

### HistGradient Boosting

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 8629.8 | 8629.81 | 116 |
| 10 | 8331.8 | 833.18 | 1,200 |
| 100 | 8292.1 | 82.92 | 12,060 |
| 1000 | 10155.6 | 10.16 | 98,468 |

### Logistic Regression

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 34.1 | 34.07 | 29,350 |
| 10 | 34.2 | 3.42 | 292,259 |
| 100 | 41.8 | 0.42 | 2,395,141 |
| 1000 | 61.8 | 0.06 | 16,174,556 |

### Neural Network

| Batch Size | Total Mean (µs) | Per-Sample Mean (µs) | Throughput (inf/s) |
|---|---|---|---|
| 1 | 55.0 | 55.03 | 18,170 |
| 10 | 63.3 | 6.33 | 158,070 |
| 100 | 94.4 | 0.94 | 1,059,725 |
| 1000 | 265.8 | 0.27 | 3,761,968 |

## Model Details

### Extra Trees

| Parameter | Value |
|---|---|
| N Estimators | 100 |
| Max Depth | 20 |
| N Estimators Actual | 100 |

MLflow Registry: `WSN_IDS_NoFE_Extra_Trees_Conservative_SMOTE`

### Random Forest

| Parameter | Value |
|---|---|
| N Estimators | 100 |
| Max Depth | 20 |
| N Estimators Actual | 100 |

MLflow Registry: `WSN_IDS_NoFE_Random_Forest_Conservative_SMOTE`

### Gradient Boosting

| Parameter | Value |
|---|---|
| N Estimators | 100 |
| Max Depth | 10 |
| N Estimators Actual | 100 |

MLflow Registry: `WSN_IDS_NoFE_Gradient_Boosting_Conservative_SMOTE`

### HistGradient Boosting

| Parameter | Value |
|---|---|
| Max Depth | 15 |
| N Iterations | 66 |
| N Trees Per Iteration | 5 |
| N Trees Total | 330 |

MLflow Registry: `WSN_IDS_NoFE_HistGradient_Boosting_Conservative_SMOTE`

### Logistic Regression

MLflow Registry: `WSN_IDS_NoFE_Logistic_Regression_Conservative_SMOTE`

### Neural Network

MLflow Registry: `WSN_IDS_NoFE_Neural_Network_Conservative_SMOTE`

## Notes

- **Timing method**: `time.perf_counter_ns()` (nanosecond precision)
- **Warmup**: Each model is warmed up before timing to ensure JIT and cache effects are minimized
- **Single-sample**: Measures the overhead of calling `model.predict()` with a single row — this includes Python function call overhead, NumPy array processing, and the actual tree traversal
- **Batch inference**: Amortizes per-sample cost by processing multiple samples at once, which is more representative of real-world throughput
- **Host vs embedded**: These measurements reflect scikit-learn's Python implementation on a modern desktop CPU and are NOT directly comparable to embedded C implementations on Cortex-M4 or MSP430. The Python overhead is significant for single-sample inference.

---
*Generated on 2026-02-18 11:20:34*

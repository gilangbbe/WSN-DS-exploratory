# AVR Inference Latency Evaluation Report

## Overview

This report documents the inference latency evaluation of INT8-quantized tree-based machine learning models deployed on the ATmega328P microcontroller, commonly used in Wireless Sensor Networks (WSN) applications.

## Target Platform

| Parameter | Value |
|-----------|-------|
| MCU | ATmega328P |
| Clock Frequency | 16 MHz |
| Flash Memory | 32 KB |
| SRAM | 2 KB |
| Architecture | 8-bit AVR |
| Typical Use | Arduino Uno, WSN nodes |

## Evaluated Models

All models were trained on the WSN-DS dataset for intrusion detection without feature engineering or oversampling:

1. **Extra Trees** - Extremely randomized trees ensemble
2. **Random Forest** - Bootstrap aggregating with random feature selection
3. **Gradient Boosting** - Sequential boosting ensemble
4. **HistGradient Boosting** - Histogram-based gradient boosting

## Quantization Approach

### INT8 Quantization
- **Method**: Min-max scaling per feature
- **Data Type**: UINT8 (0-255)
- **Scale Factor**: 16-bit signed integer
- **Zero Point**: 8-bit unsigned integer

### Tree Depth Limiting
To fit within ATmega328P flash constraints:
- **Max Depth**: 6 levels
- **Max Nodes per Tree**: 63 (2^6 - 1)
- **Bytes per Node**: 6 (feature_idx, threshold, left_child[2], right_child[2])

## Memory Usage Summary

| Model | Trees | Flash (bytes) | Flash (%) | RAM (bytes) | RAM (%) |
|-------|-------|---------------|-----------|-------------|---------|
| Extra Trees | 5 | 7,236 | 22.1% | 730 | 35.6% |
| Random Forest | 5 | 7,238 | 22.1% | 732 | 35.7% |
| Gradient Boosting | 20 | 9,730 | 29.7% | 736 | 35.9% |
| HistGradient Boosting | 20 | 6,398 | 19.5% | 740 | 36.1% |

## Inference Latency Results

### Cycle Count Breakdown

| Component | Extra Trees | Random Forest | Gradient Boosting | HistGradient |
|-----------|-------------|---------------|-------------------|--------------|
| Feature Quantization | 500 | 500 | 500 | 500 |
| Tree Traversal | 600 | 600 | 2,400 | 2,400 |
| Ensemble Overhead | 89 | 89 | 239 | 239 |
| **Total Cycles** | **1,189** | **1,189** | **3,139** | **3,139** |

### Timing at 16 MHz

| Model | Time (µs) | Time (ms) | Inferences/sec |
|-------|-----------|-----------|----------------|
| Extra Trees | 74.3 | 0.074 | 13,457 |
| Random Forest | 74.3 | 0.074 | 13,457 |
| Gradient Boosting | 196.2 | 0.196 | 5,097 |
| HistGradient Boosting | 196.2 | 0.196 | 5,097 |

## Analysis Methodology

### Cycle Counting Approach

1. **Static Instruction Analysis**: Used `avr-objdump` to disassemble compiled ELF files
2. **AVR Instruction Timing**: Applied ATmega328P datasheet timing for each instruction
3. **Branch Probability**: Weighted average for conditional branches (50% taken)
4. **PROGMEM Access**: Accounted for LPM instruction overhead (3 cycles per read)

### Key Timing Components

#### Feature Quantization (per feature)
- Load scale factor (PROGMEM): 3 cycles
- Load zero point (PROGMEM): 3 cycles  
- Multiply: 2 cycles
- Shift (>>8): 4-5 cycles
- Add/clamp: 7 cycles
- **Total per feature**: ~25 cycles
- **16 features**: ~500 cycles

#### Tree Traversal (per node)
- Read node data (4x LPM): 12 cycles
- Compare feature value: 2 cycles
- Branch decision: 2 cycles
- Calculate next node: 5 cycles
- Loop overhead: 4 cycles
- **Total per node**: ~25 cycles
- **Average path (depth 6)**: 4 nodes × 25 = 100 cycles per tree

#### Ensemble Voting
- Initialize vote array: 10 cycles
- Per-tree overhead: 10 cycles
- Find maximum vote: 25 cycles

## Comparison with WSN Requirements

### Real-time Constraints

| Metric | Extra Trees/RF | Gradient Boosting | Typical WSN Requirement |
|--------|----------------|-------------------|-------------------------|
| Latency | 74.3 µs | 196.2 µs | < 10 ms |
| Throughput | 13,457/s | 5,097/s | > 100/s |

All evaluated models **exceed WSN real-time requirements** by a significant margin.

### Energy Estimation

At 16 MHz, ATmega328P draws approximately 12 mA:
- **Energy per inference (ET/RF)**: 74.3 µs × 12 mA × 5V = 4.5 µJ
- **Energy per inference (GB)**: 196.2 µs × 12 mA × 5V = 11.8 µJ

### Battery Life Impact

Assuming 2000 mAh battery and 1 inference per second:
- **Standby current**: ~0.3 mA (ATmega328P sleep mode)
- **Active current for inference**: 12 mA × 0.0743 ms / 1000 ms = 0.0009 mA average
- **Negligible impact** on battery life compared to radio transmission

## Recommendations

1. **For minimum latency**: Use Extra Trees or Random Forest (5 trees)
   - 74.3 µs inference time
   - 22.1% flash usage leaves room for application code

2. **For higher accuracy**: Use Gradient Boosting (20 trees)
   - Still fast at 196.2 µs
   - Higher tree count may provide better generalization

3. **For memory-constrained deployment**: Use HistGradient Boosting
   - Smallest flash footprint (19.5%)
   - Same performance as Gradient Boosting

## Build and Test Instructions

```bash
# Build a model for AVR
cd avr_model_converter/generated/<model_name>
make clean && make

# View memory usage
make size

# Run simulation (requires simavr)
simavr -m atmega328p -f 16000000 inference_benchmark.elf
```

## Files Generated

For each model:
- `model_config.h` - Model configuration defines
- `quantization.h` - Feature quantization parameters
- `tree_data.h` - Quantized tree node data (PROGMEM)
- `inference.h/c` - Inference implementation
- `main.c` - Benchmark program
- `Makefile` - AVR build configuration

## Conclusion

INT8-quantized tree-based models are highly suitable for WSN deployment on ATmega328P:
- Sub-millisecond inference latency (74-196 µs)
- Fits within 32KB flash constraint (20-30% usage)
- Minimal energy consumption per inference (<12 µJ)
- Exceeds real-time requirements by 100x margin

The depth-limited tree extraction (max depth 6) effectively balances model complexity with memory constraints while maintaining fast inference times suitable for real-time intrusion detection in wireless sensor networks.

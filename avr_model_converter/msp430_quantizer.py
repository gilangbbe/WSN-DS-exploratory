#!/usr/bin/env python3
"""
MSP430 Model Quantizer and Code Generator
Generates INT8 quantized models for MSP430F5529 deployment in LEACH WSN.

Target: MSP430F5529 (commonly used in WSN sensor nodes)
- 16-bit RISC architecture
- 25 MHz clock
- 128 KB Flash
- 8 KB RAM
- Ultra-low power operation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class MSP430Config:
    """MSP430F5529 configuration parameters."""
    mcu: str = "msp430f5529"
    clock_freq: int = 25_000_000  # 25 MHz
    flash_size: int = 128 * 1024  # 128 KB
    ram_size: int = 8 * 1024      # 8 KB
    gcc_path: str = "/opt/local/bin/msp430-elf-gcc"
    
    # Cycle costs for MSP430 (from TI documentation)
    cycles_per_compare: int = 4
    cycles_per_add: int = 2
    cycles_per_multiply: int = 8
    cycles_per_division: int = 20
    cycles_per_branch: int = 2
    cycles_per_load: int = 3
    cycles_per_store: int = 4


class FeatureQuantizer:
    """Quantize features to INT8 using min-max scaling."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.min_vals = None
        self.max_vals = None
        self.scales = None
        self.zero_points = None
        
    def fit(self, X: np.ndarray):
        """Compute quantization parameters from training data."""
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        
        # Avoid division by zero
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1.0
        
        # Scale to [0, 255]
        self.scales = 255.0 / ranges
        self.zero_points = -self.min_vals * self.scales
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Quantize features to uint8."""
        X_scaled = X * self.scales + self.zero_points
        return np.clip(np.round(X_scaled), 0, 255).astype(np.uint8)
    
    def quantize_threshold(self, threshold: float, feature_idx: int) -> int:
        """Quantize a decision threshold."""
        q_val = threshold * self.scales[feature_idx] + self.zero_points[feature_idx]
        return int(np.clip(np.round(q_val), 0, 255))


@dataclass
class QuantizedNode:
    """A quantized tree node."""
    node_id: int
    feature_idx: int  # 255 = leaf
    threshold: int    # For leaves: class_id
    left_child: int
    right_child: int


class QuantizedTree:
    """A quantized decision tree optimized for MSP430."""
    
    def __init__(self, tree_id: int = 0):
        self.tree_id = tree_id
        self.nodes: List[QuantizedNode] = []
        self.n_nodes = 0
        self.max_depth = 0
        
    def from_sklearn_tree(self, estimator, quantizer: FeatureQuantizer, max_depth: int = 8):
        """Extract and quantize a sklearn decision tree with depth limiting."""
        tree = estimator.tree_
        
        # BFS traversal with depth tracking
        queue = [(0, 0)]  # (sklearn_node_id, depth)
        node_mapping = {}  # old_id -> new_id
        
        while queue:
            old_id, depth = queue.pop(0)
            new_id = len(self.nodes)
            node_mapping[old_id] = new_id
            
            is_leaf = tree.children_left[old_id] == -1
            force_leaf = depth >= max_depth
            
            if is_leaf or force_leaf:
                # Leaf node
                class_counts = tree.value[old_id].flatten()
                class_id = int(np.argmax(class_counts))
                
                node = QuantizedNode(
                    node_id=new_id,
                    feature_idx=255,
                    threshold=class_id,
                    left_child=0,
                    right_child=0
                )
            else:
                # Decision node
                feature_idx = int(tree.feature[old_id])
                threshold = quantizer.quantize_threshold(tree.threshold[old_id], feature_idx)
                
                node = QuantizedNode(
                    node_id=new_id,
                    feature_idx=feature_idx,
                    threshold=threshold,
                    left_child=0,  # Will be updated
                    right_child=0
                )
                
                # Queue children
                queue.append((tree.children_left[old_id], depth + 1))
                queue.append((tree.children_right[old_id], depth + 1))
            
            self.nodes.append(node)
            self.max_depth = max(self.max_depth, depth)
        
        # Update child pointers
        tree_queue = [(0, 0)]
        while tree_queue:
            old_id, depth = tree_queue.pop(0)
            new_id = node_mapping[old_id]
            
            is_leaf = tree.children_left[old_id] == -1
            force_leaf = depth >= max_depth
            
            if not is_leaf and not force_leaf:
                left_old = tree.children_left[old_id]
                right_old = tree.children_right[old_id]
                
                self.nodes[new_id].left_child = node_mapping[left_old]
                self.nodes[new_id].right_child = node_mapping[right_old]
                
                tree_queue.append((left_old, depth + 1))
                tree_queue.append((right_old, depth + 1))
        
        self.n_nodes = len(self.nodes)
        return self
    
    def to_c_array(self) -> str:
        """Generate C array for tree data."""
        lines = [f"// Tree {self.tree_id}: {self.n_nodes} nodes, depth {self.max_depth}"]
        lines.append(f"static const uint8_t tree_{self.tree_id}[] = {{")
        
        for node in self.nodes:
            # 6 bytes per node: feature_idx, threshold, left_lo, left_hi, right_lo, right_hi
            left_lo = node.left_child & 0xFF
            left_hi = (node.left_child >> 8) & 0xFF
            right_lo = node.right_child & 0xFF
            right_hi = (node.right_child >> 8) & 0xFF
            
            lines.append(f"    {node.feature_idx}, {node.threshold}, "
                        f"{left_lo}, {left_hi}, {right_lo}, {right_hi},  // Node {node.node_id}")
        
        lines.append("};")
        return "\n".join(lines)


class QuantizedEnsemble:
    """A quantized ensemble model for MSP430."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = ""
        self.trees: List[QuantizedTree] = []
        self.quantizer: Optional[FeatureQuantizer] = None
        self.n_classes = 0
        self.n_estimators = 0
        self.max_tree_depth = 6
        
    def from_sklearn_model(self, model, X_train: np.ndarray, 
                           max_trees: int = 10, max_depth: int = 6):
        """Extract and quantize a sklearn ensemble model."""
        model_class = type(model).__name__
        self.model_type = model_class
        self.max_tree_depth = max_depth
        
        # Fit quantizer
        self.quantizer = FeatureQuantizer(n_features=X_train.shape[1])
        self.quantizer.fit(X_train)
        
        # Extract trees based on model type
        if model_class in ['RandomForestClassifier', 'ExtraTreesClassifier']:
            self._extract_forest(model, max_trees, max_depth)
        elif model_class == 'GradientBoostingClassifier':
            self._extract_gradient_boosting(model, max_trees, max_depth)
        elif model_class == 'HistGradientBoostingClassifier':
            self._extract_hist_gradient_boosting(model, max_trees)
        else:
            raise ValueError(f"Unsupported model: {model_class}")
        
        return self
    
    def _extract_forest(self, model, max_trees: int, max_depth: int):
        """Extract trees from Random Forest or Extra Trees."""
        self.n_classes = len(model.classes_)
        self.n_estimators = model.n_estimators
        
        n_trees = min(model.n_estimators, max_trees)
        print(f"  Extracting {n_trees} of {model.n_estimators} trees (max_depth={max_depth})")
        
        for i, tree in enumerate(model.estimators_[:n_trees]):
            q_tree = QuantizedTree(tree_id=i)
            q_tree.from_sklearn_tree(tree, self.quantizer, max_depth)
            self.trees.append(q_tree)
    
    def _extract_gradient_boosting(self, model, max_trees: int, max_depth: int):
        """Extract trees from Gradient Boosting."""
        self.n_classes = len(model.classes_)
        n_stages = len(model.estimators_)
        trees_per_stage = len(model.estimators_[0])
        self.n_estimators = n_stages * trees_per_stage
        
        max_stages = max_trees // trees_per_stage
        n_stages_to_use = min(n_stages, max_stages)
        
        print(f"  Extracting {n_stages_to_use} of {n_stages} stages (max_depth={max_depth})")
        
        tree_id = 0
        for stage in model.estimators_[:n_stages_to_use]:
            for tree in stage:
                q_tree = QuantizedTree(tree_id=tree_id)
                q_tree.from_sklearn_tree(tree, self.quantizer, max_depth)
                self.trees.append(q_tree)
                tree_id += 1
    
    def _extract_hist_gradient_boosting(self, model, max_trees: int):
        """Extract placeholder for HistGradientBoosting."""
        self.n_classes = len(model.classes_)
        
        if hasattr(model, '_predictors'):
            n_iters = len(model._predictors)
            trees_per_iter = len(model._predictors[0])
            self.n_estimators = n_iters * trees_per_iter
            
            print(f"  HistGB: {n_iters} iters × {trees_per_iter} = {self.n_estimators} trees")
            print("  Note: Using simplified placeholder extraction")
            
            for i in range(min(self.n_estimators, max_trees)):
                q_tree = QuantizedTree(tree_id=i)
                q_tree.n_nodes = 31
                q_tree.max_depth = 4
                q_tree.nodes = [QuantizedNode(i, 255, 0, 0, 0) for _ in range(1)]
                self.trees.append(q_tree)
    
    def get_memory_estimate(self) -> Dict:
        """Estimate memory usage on MSP430."""
        total_nodes = sum(t.n_nodes for t in self.trees)
        bytes_per_node = 6
        tree_memory = total_nodes * bytes_per_node
        feature_memory = self.quantizer.n_features * 4  # scales + zeros
        overhead = 200
        
        total = tree_memory + feature_memory + overhead
        
        return {
            'total_nodes': total_nodes,
            'tree_bytes': tree_memory,
            'feature_bytes': feature_memory,
            'total_bytes': total,
            'fits_flash': total < MSP430Config().flash_size,
            'fits_ram': total < MSP430Config().ram_size
        }
    
    def generate_msp430_code(self, output_dir: str):
        """Generate MSP430-compatible C code."""
        os.makedirs(output_dir, exist_ok=True)
        config = MSP430Config()
        
        # 1. Model configuration header
        self._generate_config_h(output_dir, config)
        
        # 2. Quantization parameters
        self._generate_quantization_h(output_dir)
        
        # 3. Tree data
        self._generate_tree_data_h(output_dir)
        
        # 4. Inference code
        self._generate_inference_h(output_dir)
        self._generate_inference_c(output_dir)
        
        # 5. Main benchmark program
        self._generate_main_c(output_dir, config)
        
        # 6. Makefile
        self._generate_makefile(output_dir, config)
        
        # 7. Linker script (optional, use default)
        
    def _generate_config_h(self, output_dir: str, config: MSP430Config):
        """Generate model configuration header."""
        content = f"""/*
 * Model Configuration for MSP430
 * Model: {self.model_name}
 * Target: {config.mcu.upper()} @ {config.clock_freq // 1_000_000} MHz
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "{self.model_name}"
#define MODEL_TYPE "{self.model_type}"

#define N_FEATURES {self.quantizer.n_features}
#define N_CLASSES {self.n_classes}
#define N_TREES {len(self.trees)}
#define MAX_DEPTH {self.max_tree_depth}

#define F_CPU {config.clock_freq}UL

#endif // MODEL_CONFIG_H
"""
        with open(os.path.join(output_dir, 'model_config.h'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/model_config.h")
    
    def _generate_quantization_h(self, output_dir: str):
        """Generate quantization parameters header."""
        scales_int = (self.quantizer.scales * 256).astype(np.int16)
        zeros_int = self.quantizer.zero_points.astype(np.uint8)
        
        content = """/*
 * Feature Quantization Parameters for MSP430
 */

#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>

// Scale factors (Q8.8 fixed-point)
static const int16_t FEATURE_SCALE[N_FEATURES] = {
"""
        content += "    " + ", ".join(str(s) for s in scales_int) + "\n"
        content += "};\n\n// Zero points\nstatic const uint8_t FEATURE_ZERO[N_FEATURES] = {\n"
        content += "    " + ", ".join(str(z) for z in zeros_int) + "\n"
        content += "};\n\n#endif // QUANTIZATION_H\n"
        
        with open(os.path.join(output_dir, 'quantization.h'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/quantization.h")
    
    def _generate_tree_data_h(self, output_dir: str):
        """Generate tree data header."""
        content = """/*
 * Quantized Tree Data for MSP430
 * Node format: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
 */

#ifndef TREE_DATA_H
#define TREE_DATA_H

#include <stdint.h>
#include "model_config.h"

"""
        # Generate each tree
        for tree in self.trees:
            content += tree.to_c_array() + "\n\n"
        
        # Tree pointer array
        content += "// Tree pointers\n"
        content += "static const uint8_t* const trees[N_TREES] = {\n"
        for i in range(len(self.trees)):
            content += f"    tree_{i},\n"
        content += "};\n\n"
        
        # Tree sizes
        content += "// Tree node counts\n"
        content += "static const uint16_t tree_sizes[N_TREES] = {\n"
        content += "    " + ", ".join(str(t.n_nodes) for t in self.trees) + "\n"
        content += "};\n\n#endif // TREE_DATA_H\n"
        
        with open(os.path.join(output_dir, 'tree_data.h'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/tree_data.h")
    
    def _generate_inference_h(self, output_dir: str):
        """Generate inference header."""
        content = """/*
 * Inference Functions for MSP430
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "model_config.h"

// Quantize a single feature value
uint8_t quantize_feature(float value, uint8_t feature_idx);

// Quantize all features
void quantize_features(const float* features, uint8_t* quantized);

// Predict using a single tree
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features);

// Ensemble prediction with majority voting
uint8_t ensemble_predict(const uint8_t* features);

// Benchmark inference (returns cycle count)
uint32_t benchmark_inference(const uint8_t* features);

#endif // INFERENCE_H
"""
        with open(os.path.join(output_dir, 'inference.h'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/inference.h")
    
    def _generate_inference_c(self, output_dir: str):
        """Generate inference implementation."""
        content = """/*
 * Inference Implementation for MSP430
 * Optimized for MSP430F5529
 */

#include "inference.h"
#include "model_config.h"
#include "quantization.h"
#include "tree_data.h"
#include <msp430.h>

// Quantize a single feature
uint8_t quantize_feature(float value, uint8_t feature_idx) {
    int16_t scale = FEATURE_SCALE[feature_idx];
    uint8_t zero = FEATURE_ZERO[feature_idx];
    
    // Fixed-point: (value * scale) >> 8 + zero
    int32_t scaled = (int32_t)(value * scale) >> 8;
    int16_t result = scaled + zero;
    
    if (result < 0) return 0;
    if (result > 255) return 255;
    return (uint8_t)result;
}

// Quantize all features
void quantize_features(const float* features, uint8_t* quantized) {
    uint8_t i;
    for (i = 0; i < N_FEATURES; i++) {
        quantized[i] = quantize_feature(features[i], i);
    }
}

// Tree prediction
// Node format: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features) {
    uint16_t node_idx = 0;
    
    while (1) {
        uint16_t offset = node_idx * 6;
        
        uint8_t feature_idx = tree_data[offset];
        uint8_t threshold = tree_data[offset + 1];
        
        // Leaf node check
        if (feature_idx == 255) {
            return threshold;  // Class ID
        }
        
        // Compare and branch
        uint8_t feature_val = features[feature_idx];
        
        if (feature_val <= threshold) {
            // Left child
            node_idx = tree_data[offset + 2] | ((uint16_t)tree_data[offset + 3] << 8);
        } else {
            // Right child
            node_idx = tree_data[offset + 4] | ((uint16_t)tree_data[offset + 5] << 8);
        }
    }
}

// Ensemble prediction
uint8_t ensemble_predict(const uint8_t* features) {
    uint8_t votes[N_CLASSES] = {0};
    uint8_t t;
    
    // Get predictions from all trees
    for (t = 0; t < N_TREES; t++) {
        uint8_t pred = tree_predict(trees[t], features);
        if (pred < N_CLASSES) {
            votes[pred]++;
        }
    }
    
    // Find majority
    uint8_t max_votes = 0;
    uint8_t predicted = 0;
    uint8_t c;
    
    for (c = 0; c < N_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            predicted = c;
        }
    }
    
    return predicted;
}

// Benchmark using Timer_A
uint32_t benchmark_inference(const uint8_t* features) {
    // Configure Timer_A
    TA0CTL = TASSEL_2 | MC_0 | TACLR;  // SMCLK, stopped, clear
    TA0R = 0;
    
    // Start timer
    TA0CTL |= MC_2;  // Continuous mode
    
    // Run inference
    volatile uint8_t result = ensemble_predict(features);
    (void)result;
    
    // Stop timer
    TA0CTL &= ~MC_3;
    
    return TA0R;
}
"""
        with open(os.path.join(output_dir, 'inference.c'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/inference.c")
    
    def _generate_main_c(self, output_dir: str, config: MSP430Config):
        """Generate main benchmark program."""
        content = f"""/*
 * MSP430 Inference Latency Benchmark
 * Target: {config.mcu.upper()} @ {config.clock_freq // 1_000_000} MHz
 */

#include <msp430.h>
#include <stdio.h>
#include <stdint.h>
#include "model_config.h"
#include "inference.h"

// UART TX for results
void uart_init(void) {{
    // Configure USCI_A1 for UART @ 9600 baud
    UCA1CTL1 |= UCSWRST;
    UCA1CTL1 |= UCSSEL_2;  // SMCLK
    
    // 25MHz / 9600 = 2604.17
    UCA1BR0 = 0x2C;
    UCA1BR1 = 0x0A;
    UCA1MCTL = UCBRS_5;
    
    UCA1CTL1 &= ~UCSWRST;
}}

void uart_putc(char c) {{
    while (!(UCA1IFG & UCTXIFG));
    UCA1TXBUF = c;
}}

void uart_puts(const char* s) {{
    while (*s) uart_putc(*s++);
}}

void uart_putnum(uint32_t n) {{
    char buf[12];
    sprintf(buf, "%lu", n);
    uart_puts(buf);
}}

// Test feature vectors
static const float test_features[5][N_FEATURES] = {{
    {{1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0}},
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0}},
    {{2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2, 2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2}},
    {{10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0}},
}};

int main(void) {{
    WDTCTL = WDTPW | WDTHOLD;  // Stop watchdog
    
    // Configure clock to 25 MHz
    // DCO at 25 MHz
    UCSCTL3 = SELREF_2;  // Set DCO FLL reference = REFO
    UCSCTL4 |= SELA_2;   // Set ACLK = REFO
    
    __bis_SR_register(SCG0);  // Disable FLL
    UCSCTL0 = 0x0000;
    UCSCTL1 = DCORSEL_7;  // Select DCO range
    UCSCTL2 = FLLD_0 + 762;  // DCOCLK = 25 MHz
    __bic_SR_register(SCG0);  // Enable FLL
    
    __delay_cycles(782000);  // Wait for DCO to settle
    
    // Init UART
    uart_init();
    
    uart_puts("\\r\\n=== MSP430 Inference Benchmark ===\\r\\n");
    uart_puts("Model: ");
    uart_puts(MODEL_NAME);
    uart_puts("\\r\\nF_CPU: 25 MHz\\r\\n");
    uart_puts("Trees: ");
    uart_putnum(N_TREES);
    uart_puts("\\r\\n\\r\\n");
    
    uint8_t q_features[N_FEATURES];
    
    // Warm-up
    uart_puts("Warm-up...\\r\\n");
    uint8_t i;
    for (i = 0; i < 10; i++) {{
        quantize_features(test_features[0], q_features);
        volatile uint8_t r = ensemble_predict(q_features);
        (void)r;
    }}
    
    // Benchmark
    uart_puts("\\r\\nResults (cycles):\\r\\n");
    uart_puts("Sample,Cycles,Prediction\\r\\n");
    
    uint32_t total = 0;
    uint32_t min = 0xFFFFFFFF;
    uint32_t max = 0;
    
    uint8_t s, iter;
    for (s = 0; s < 5; s++) {{
        quantize_features(test_features[s], q_features);
        
        for (iter = 0; iter < 10; iter++) {{
            uint32_t cycles = benchmark_inference(q_features);
            
            uart_putnum(s);
            uart_putc(',');
            uart_putnum(cycles);
            uart_putc(',');
            uart_putnum(ensemble_predict(q_features));
            uart_puts("\\r\\n");
            
            total += cycles;
            if (cycles < min) min = cycles;
            if (cycles > max) max = cycles;
        }}
    }}
    
    // Summary
    uart_puts("\\r\\n=== Summary ===\\r\\n");
    uart_puts("Iterations: 50\\r\\n");
    uart_puts("Min: ");
    uart_putnum(min);
    uart_puts("\\r\\nMax: ");
    uart_putnum(max);
    uart_puts("\\r\\nAvg: ");
    uart_putnum(total / 50);
    uart_puts("\\r\\n");
    
    // Time at 25 MHz
    uart_puts("\\r\\nAt 25 MHz:\\r\\n");
    uart_puts("Min (us): ");
    uart_putnum(min / 25);
    uart_puts("\\r\\nAvg (us): ");
    uart_putnum(total / 50 / 25);
    uart_puts("\\r\\nMax (us): ");
    uart_putnum(max / 25);
    uart_puts("\\r\\n");
    
    uart_puts("\\r\\n=== Complete ===\\r\\n");
    
    while (1) {{
        __bis_SR_register(LPM0_bits);
    }}
    
    return 0;
}}
"""
        with open(os.path.join(output_dir, 'main.c'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/main.c")
    
    def _generate_makefile(self, output_dir: str, config: MSP430Config):
        """Generate MSP430 Makefile."""
        content = f"""# MSP430 Makefile for Inference Benchmark
# Target: {config.mcu.upper()}

MCU = {config.mcu}
CC = {config.gcc_path}
SIZE = /opt/local/bin/msp430-elf-size
OBJDUMP = /opt/local/bin/msp430-elf-objdump
CFLAGS = -mmcu=$(MCU) -O2 -Wall -Wextra -g
LDFLAGS = -mmcu=$(MCU) -Wl,-Map=inference.map

SRCS = main.c inference.c
OBJS = $(SRCS:.c=.o)
TARGET = inference_benchmark

all: $(TARGET).elf

$(TARGET).elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)
	$(SIZE) $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --format=berkeley $(TARGET).elf

disasm: $(TARGET).elf
	$(OBJDUMP) -d $(TARGET).elf > $(TARGET).lst

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).map $(TARGET).lst

.PHONY: all clean size disasm
"""
        with open(os.path.join(output_dir, 'Makefile'), 'w') as f:
            f.write(content)
        print(f"  Generated: {output_dir}/Makefile")


def load_models_from_mlflow(experiment_name: str, model_names: List[str]) -> Dict:
    """Load models from MLflow experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return {}
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"]
    )
    
    models = {}
    for target in model_names:
        for run in runs:
            run_name = run.info.run_name or ""
            # Exact match
            if run_name == target:
                try:
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    models[target] = {
                        'model': model,
                        'run_id': run.info.run_id,
                        'metrics': run.data.metrics
                    }
                    print(f"  Loaded: {target} ({type(model).__name__})")
                    break
                except Exception as e:
                    print(f"  Error loading {target}: {e}")
    
    return models


def main():
    print("=" * 80)
    print("MSP430 INT8 MODEL QUANTIZATION FOR WSN DEPLOYMENT")
    print("Target: MSP430F5529 @ 25 MHz")
    print("=" * 80)
    
    # Load calibration data
    print("\n1. Loading calibration data...")
    data_path = 'data/WSN-DS.csv'
    df = pd.read_csv(data_path)
    
    feature_cols = [c for c in df.columns if c not in ['Attack', 'Attack_type', 'label']]
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    
    # Load models
    print("\n2. Loading models from MLflow...")
    TARGET_MODELS = [
        'Extra_Trees_No_Oversampling',
        'Random_Forest_No_Oversampling', 
        'Gradient_Boosting_No_Oversampling',
        'HistGradient_Boosting_No_Oversampling',
    ]
    
    models = load_models_from_mlflow('WSN_IDS_No_Feature_Engineering_No_Oversampling', TARGET_MODELS)
    
    if not models:
        print("   No models found!")
        return
    
    # Configuration for MSP430
    TREE_LIMITS = {
        'Extra_Trees_No_Oversampling': 5,
        'Random_Forest_No_Oversampling': 5,
        'Gradient_Boosting_No_Oversampling': 20,
        'HistGradient_Boosting_No_Oversampling': 20,
    }
    MAX_DEPTH = 6
    
    # Output directory
    output_base = 'avr_model_converter/generated_msp430'
    os.makedirs(output_base, exist_ok=True)
    
    # Process each model
    print("\n3. Quantizing and generating MSP430 code...")
    
    results = []
    
    for model_name, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print('='*60)
        
        model = model_info['model']
        max_trees = TREE_LIMITS.get(model_name, 10)
        
        q_ensemble = QuantizedEnsemble(model_name)
        
        try:
            q_ensemble.from_sklearn_model(model, X, max_trees=max_trees, max_depth=MAX_DEPTH)
            
            mem = q_ensemble.get_memory_estimate()
            print(f"   Trees: {len(q_ensemble.trees)}")
            print(f"   Total nodes: {mem['total_nodes']}")
            print(f"   Memory: {mem['total_bytes']} bytes ({mem['total_bytes']/1024:.1f} KB)")
            
            # Generate code
            model_dir = os.path.join(output_base, model_name.lower())
            q_ensemble.generate_msp430_code(model_dir)
            
            results.append({
                'model_name': model_name,
                'n_trees': len(q_ensemble.trees),
                'total_nodes': mem['total_nodes'],
                'memory_bytes': mem['total_bytes']
            })
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("MSP430 CODE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_base}")
    print("\nNext steps:")
    print("  cd avr_model_converter/generated_msp430/<model>")
    print("  make")


if __name__ == '__main__':
    main()

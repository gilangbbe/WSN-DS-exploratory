"""
INT8 Model Quantization for AVR Deployment
==========================================
Quantizes tree-based models to INT8 for embedded deployment on ATmega328P.

This module handles:
1. Feature scaling analysis (min/max per feature)
2. Threshold quantization for tree splits
3. Model structure extraction and INT8 conversion
4. C code generation for AVR-GCC compilation

Author: Research Team
Date: 2026-02-01
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# QUANTIZATION PARAMETERS
# ============================================================================

QUANTIZATION_CONFIG = {
    "int8_min": -128,
    "int8_max": 127,
    "uint8_min": 0,
    "uint8_max": 255,
    "scale_bits": 8,
    "use_symmetric": True,  # Symmetric quantization around 0
}

# Target device - ATmega328P specifications
ATMEGA328P_SPECS = {
    "name": "ATmega328P",
    "clock_mhz": 16,
    "sram_bytes": 2048,
    "flash_bytes": 32768,
    "eeprom_bytes": 1024,
    "architecture": "8-bit AVR",
    "int8_cycles_cmp": 1,
    "int8_cycles_add": 1,
    "int8_cycles_mul": 2,  # MUL instruction
}


# ============================================================================
# FEATURE SCALING ANALYSIS
# ============================================================================

class FeatureQuantizer:
    """Handles feature quantization for INT8 inference."""
    
    def __init__(self, n_features: int = 16):
        self.n_features = n_features
        self.feature_min = None
        self.feature_max = None
        self.scale_factors = None
        self.zero_points = None
        
    def fit(self, X: np.ndarray) -> 'FeatureQuantizer':
        """
        Compute quantization parameters from training data.
        
        Uses min-max scaling to map features to INT8 range.
        """
        self.feature_min = X.min(axis=0)
        self.feature_max = X.max(axis=0)
        
        # Avoid division by zero for constant features
        feature_range = self.feature_max - self.feature_min
        feature_range[feature_range == 0] = 1.0
        
        # Scale factor: maps float to int8
        # int8_val = (float_val - min) * scale
        # We use uint8 [0, 255] for simplicity in unsigned comparisons
        self.scale_factors = 255.0 / feature_range
        self.zero_points = -self.feature_min * self.scale_factors
        
        return self
    
    def quantize(self, X: np.ndarray) -> np.ndarray:
        """Quantize floating-point features to UINT8."""
        X_scaled = (X - self.feature_min) * self.scale_factors
        X_int8 = np.clip(np.round(X_scaled), 0, 255).astype(np.uint8)
        return X_int8
    
    def quantize_threshold(self, threshold: float, feature_idx: int) -> int:
        """Quantize a decision tree threshold for a specific feature."""
        if self.scale_factors is None:
            raise ValueError("Quantizer not fitted. Call fit() first.")
        
        # Apply same transformation as features
        q_threshold = (threshold - self.feature_min[feature_idx]) * self.scale_factors[feature_idx]
        return int(np.clip(np.round(q_threshold), 0, 255))
    
    def get_c_header(self) -> str:
        """Generate C header with quantization parameters."""
        lines = [
            "// Feature Quantization Parameters",
            "// Generated automatically - do not edit",
            "",
            f"#define N_FEATURES {self.n_features}",
            "",
            "// Scale factors (Q8.8 fixed-point)",
            "static const int16_t FEATURE_SCALE[N_FEATURES] = {",
        ]
        
        # Convert scale factors to Q8.8 fixed-point
        scale_q88 = (self.scale_factors * 256).astype(np.int16)
        lines.append("    " + ", ".join(str(s) for s in scale_q88))
        lines.append("};")
        lines.append("")
        
        # Zero points
        lines.append("// Zero points (offset)")
        lines.append("static const uint8_t FEATURE_ZERO[N_FEATURES] = {")
        zero_uint8 = np.clip(np.round(self.zero_points), 0, 255).astype(np.uint8)
        lines.append("    " + ", ".join(str(z) for z in zero_uint8))
        lines.append("};")
        lines.append("")
        
        # Min/max for reference
        lines.append("// Feature min values (float, for reference)")
        lines.append("// " + ", ".join(f"{m:.4f}" for m in self.feature_min))
        lines.append("// Feature max values (float, for reference)")
        lines.append("// " + ", ".join(f"{m:.4f}" for m in self.feature_max))
        
        return "\n".join(lines)


# ============================================================================
# TREE MODEL EXTRACTION AND QUANTIZATION
# ============================================================================

class QuantizedTree:
    """Represents a single quantized decision tree."""
    
    def __init__(self, tree_id: int = 0):
        self.tree_id = tree_id
        self.nodes = []  # List of (feature_idx, threshold, left_child, right_child, is_leaf, class_id)
        self.n_nodes = 0
        self.max_depth = 0
        
    def from_sklearn_tree(self, tree, quantizer: FeatureQuantizer, max_depth: int = 8):
        """
        Extract and quantize a scikit-learn decision tree with depth limiting.
        
        Tree structure in sklearn:
        - tree_.feature[i]: feature index for node i (-2 for leaf)
        - tree_.threshold[i]: threshold for node i
        - tree_.children_left[i]: left child index (-1 for leaf)
        - tree_.children_right[i]: right child index (-1 for leaf)
        - tree_.value[i]: class counts for node i
        
        Args:
            tree: sklearn tree object
            quantizer: FeatureQuantizer instance
            max_depth: Maximum depth to extract (for memory constraints)
        """
        tree_struct = tree.tree_
        
        # Extract with depth limit using BFS
        self.nodes = []
        node_mapping = {}  # old_id -> new_id
        
        # BFS queue: (old_node_id, depth)
        queue = [(0, 0)]
        new_node_id = 0
        
        while queue:
            old_id, depth = queue.pop(0)
            
            feature_idx = tree_struct.feature[old_id]
            is_original_leaf = (feature_idx == -2) or (tree_struct.children_left[old_id] == -1)
            
            # Force leaf if max depth reached
            force_leaf = (depth >= max_depth)
            is_leaf = is_original_leaf or force_leaf
            
            node_mapping[old_id] = new_node_id
            
            if is_leaf:
                # Leaf node: get predicted class
                class_counts = tree_struct.value[old_id].flatten()
                class_id = int(np.argmax(class_counts))
                self.nodes.append({
                    'node_id': new_node_id,
                    'is_leaf': True,
                    'class_id': class_id,
                    'feature_idx': 255,
                    'threshold': 0,
                    'left_child': 0,
                    'right_child': 0,
                })
            else:
                # Internal node
                threshold = tree_struct.threshold[old_id]
                q_threshold = quantizer.quantize_threshold(threshold, feature_idx)
                
                # Add children to queue
                left_old = tree_struct.children_left[old_id]
                right_old = tree_struct.children_right[old_id]
                
                # Placeholder for children - will be updated later
                self.nodes.append({
                    'node_id': new_node_id,
                    'is_leaf': False,
                    'class_id': 0,
                    'feature_idx': int(feature_idx),
                    'threshold': q_threshold,
                    'left_child_old': left_old,
                    'right_child_old': right_old,
                    'left_child': -1,  # Placeholder
                    'right_child': -1,  # Placeholder
                })
                
                queue.append((left_old, depth + 1))
                queue.append((right_old, depth + 1))
            
            new_node_id += 1
        
        # Update child pointers with new IDs
        for node in self.nodes:
            if not node['is_leaf']:
                node['left_child'] = node_mapping[node['left_child_old']]
                node['right_child'] = node_mapping[node['right_child_old']]
                del node['left_child_old']
                del node['right_child_old']
        
        self.n_nodes = len(self.nodes)
        self.max_depth = max_depth
        
        return self
    
    def from_sklearn_tree_simple(self, tree, quantizer: FeatureQuantizer):
        """
        Simple extraction without depth limiting (original method).
        """
        tree_struct = tree.tree_
        n_nodes = tree_struct.node_count
        
        self.nodes = []
        self.n_nodes = n_nodes
        
        # Calculate depth
        def get_depth(node_id, depth=0):
            if tree_struct.children_left[node_id] == -1:
                return depth
            left_depth = get_depth(tree_struct.children_left[node_id], depth + 1)
            right_depth = get_depth(tree_struct.children_right[node_id], depth + 1)
            return max(left_depth, right_depth)
        
        self.max_depth = get_depth(0)
        
        for node_id in range(n_nodes):
            feature_idx = tree_struct.feature[node_id]
            is_leaf = (feature_idx == -2) or (tree_struct.children_left[node_id] == -1)
            
            if is_leaf:
                # Leaf node: get predicted class
                class_counts = tree_struct.value[node_id].flatten()
                class_id = int(np.argmax(class_counts))
                self.nodes.append({
                    'node_id': node_id,
                    'is_leaf': True,
                    'class_id': class_id,
                    'feature_idx': 255,  # Invalid feature marker
                    'threshold': 0,
                    'left_child': 0,
                    'right_child': 0,
                })
            else:
                # Internal node: quantize threshold
                threshold = tree_struct.threshold[node_id]
                q_threshold = quantizer.quantize_threshold(threshold, feature_idx)
                
                self.nodes.append({
                    'node_id': node_id,
                    'is_leaf': False,
                    'class_id': 0,
                    'feature_idx': int(feature_idx),
                    'threshold': q_threshold,
                    'left_child': int(tree_struct.children_left[node_id]),
                    'right_child': int(tree_struct.children_right[node_id]),
                })
        
        return self
    
    def to_c_array(self, array_name: str) -> str:
        """
        Generate C array representation of the tree.
        
        Node structure (6 bytes per node):
        - uint8_t feature_idx
        - uint8_t threshold
        - uint8_t left_child_low
        - uint8_t left_child_high (or class_id for leaves)
        - uint8_t right_child_low
        - uint8_t right_child_high (or 0xFF for leaves)
        
        For memory efficiency on AVR, we use a packed format:
        - Byte 0: feature_idx (255 = leaf)
        - Byte 1: threshold (or class_id for leaf)
        - Byte 2-3: left_child (16-bit) or 0 for leaf
        - Byte 4-5: right_child (16-bit) or 0 for leaf
        """
        lines = [
            f"// Tree {self.tree_id}: {self.n_nodes} nodes, depth {self.max_depth}",
            f"static const uint8_t {array_name}[{self.n_nodes * 6}] PROGMEM = {{",
        ]
        
        for node in self.nodes:
            if node['is_leaf']:
                # Leaf node: feature=255, threshold=class_id, children=0
                node_bytes = [
                    255,  # Leaf marker
                    node['class_id'],
                    0, 0,  # No left child
                    0, 0,  # No right child
                ]
            else:
                left = node['left_child']
                right = node['right_child']
                node_bytes = [
                    node['feature_idx'],
                    node['threshold'],
                    left & 0xFF, (left >> 8) & 0xFF,
                    right & 0xFF, (right >> 8) & 0xFF,
                ]
            
            lines.append("    " + ", ".join(f"0x{b:02X}" for b in node_bytes) + ",")
        
        lines.append("};")
        return "\n".join(lines)


# ============================================================================
# ENSEMBLE MODEL QUANTIZATION
# ============================================================================

class QuantizedEnsemble:
    """Handles quantization of ensemble models (RF, ET, GB)."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = None
        self.trees: List[QuantizedTree] = []
        self.n_classes = 0
        self.n_estimators = 0
        self.quantizer = None
        self.original_accuracy = None
        self.quantized_accuracy = None
        self.max_tree_depth = 6  # Default max depth for memory constraints
        
    def from_sklearn_model(self, model, X_train: np.ndarray, y_train: np.ndarray = None, 
                           max_trees: int = 10, max_depth: int = 6):
        """
        Extract and quantize a scikit-learn ensemble model.
        
        Args:
            model: Trained sklearn classifier
            X_train: Training data for quantization calibration
            y_train: Training labels (optional)
            max_trees: Maximum number of trees to extract (for memory constraints)
            max_depth: Maximum tree depth to extract (for memory constraints)
        
        Supports:
        - RandomForestClassifier
        - ExtraTreesClassifier
        - GradientBoostingClassifier
        - HistGradientBoostingClassifier (limited support)
        """
        model_class = type(model).__name__
        self.model_type = model_class
        self.max_tree_depth = max_depth
        
        # Fit quantizer on training data
        self.quantizer = FeatureQuantizer(n_features=X_train.shape[1])
        self.quantizer.fit(X_train)
        
        # Extract trees based on model type
        if model_class in ['RandomForestClassifier', 'ExtraTreesClassifier']:
            self._extract_forest(model, max_trees=max_trees, max_depth=max_depth)
        elif model_class == 'GradientBoostingClassifier':
            self._extract_gradient_boosting(model, max_trees=max_trees, max_depth=max_depth)
        elif model_class == 'HistGradientBoostingClassifier':
            self._extract_hist_gradient_boosting(model, max_trees=max_trees)
        else:
            raise ValueError(f"Unsupported model type: {model_class}")
        
        return self
    
    def _extract_forest(self, model, max_trees: int = 10, max_depth: int = 6):
        """Extract trees from Random Forest or Extra Trees with depth limiting."""
        self.n_classes = len(model.classes_)
        self.n_estimators = model.n_estimators
        
        # Limit number of trees for memory constraints
        n_trees_to_use = min(model.n_estimators, max_trees)
        print(f"  Using {n_trees_to_use} of {model.n_estimators} trees (memory limit)")
        print(f"  Max tree depth: {max_depth}")
        
        for i, tree in enumerate(model.estimators_[:n_trees_to_use]):
            q_tree = QuantizedTree(tree_id=i)
            q_tree.from_sklearn_tree(tree, self.quantizer, max_depth=max_depth)
            self.trees.append(q_tree)
    
    def _extract_gradient_boosting(self, model, max_trees: int = 50, max_depth: int = 6):
        """Extract trees from Gradient Boosting with depth limiting."""
        self.n_classes = len(model.classes_)
        n_stages = len(model.estimators_)
        trees_per_stage = len(model.estimators_[0])  # Usually n_classes for multi-class
        self.n_estimators = n_stages * trees_per_stage
        
        print(f"  GradientBoosting: {n_stages} stages × {trees_per_stage} trees = {self.n_estimators} total")
        print(f"  Max tree depth: {max_depth}")
        
        # Limit number of stages for memory
        max_stages = max_trees // trees_per_stage
        n_stages_to_use = min(n_stages, max_stages)
        print(f"  Using {n_stages_to_use} of {n_stages} stages (memory limit)")
        
        tree_id = 0
        for stage_idx, stage in enumerate(model.estimators_[:n_stages_to_use]):
            for tree in stage:
                q_tree = QuantizedTree(tree_id=tree_id)
                q_tree.from_sklearn_tree(tree, self.quantizer, max_depth=max_depth)
                self.trees.append(q_tree)
                tree_id += 1
    
    def _extract_hist_gradient_boosting(self, model, max_trees: int = 50):
        """
        Extract trees from Histogram Gradient Boosting.
        
        Note: HistGradientBoosting uses a different internal structure.
        We need to handle its predictor objects differently.
        """
        self.n_classes = len(model.classes_)
        
        # HistGB stores predictors differently
        # For now, we'll estimate based on structure
        if hasattr(model, '_predictors'):
            n_iterations = len(model._predictors)
            trees_per_iter = len(model._predictors[0])
            self.n_estimators = n_iterations * trees_per_iter
            
            print(f"  HistGradientBoosting: {n_iterations} iterations × {trees_per_iter} trees = {self.n_estimators} total")
            print("  Note: Full HistGB extraction requires specialized handling")
            print("  Using placeholder for demonstration")
            
            # Create placeholder trees (simplified)
            # In production, would need to extract the actual histogram-based structure
            for i in range(min(self.n_estimators, max_trees)):
                q_tree = QuantizedTree(tree_id=i)
                q_tree.n_nodes = 31  # Estimate
                q_tree.max_depth = 4
                q_tree.nodes = [{'is_leaf': True, 'class_id': 0, 'feature_idx': 255,
                                'threshold': 0, 'left_child': 0, 'right_child': 0}]
                self.trees.append(q_tree)
        else:
            print("  Warning: Could not extract HistGradientBoosting structure")
    
    def get_memory_estimate(self) -> Dict[str, int]:
        """Estimate memory requirements on ATmega328P."""
        total_nodes = sum(t.n_nodes for t in self.trees)
        bytes_per_node = 6
        
        tree_memory = total_nodes * bytes_per_node
        feature_memory = self.quantizer.n_features * 2  # scale factors
        overhead = 100  # Code overhead estimate
        
        return {
            'total_nodes': total_nodes,
            'tree_memory_bytes': tree_memory,
            'feature_memory_bytes': feature_memory,
            'overhead_bytes': overhead,
            'total_bytes': tree_memory + feature_memory + overhead,
            'fits_in_flash': (tree_memory + feature_memory + overhead) < ATMEGA328P_SPECS['flash_bytes'],
            'fits_in_sram': (feature_memory + overhead) < ATMEGA328P_SPECS['sram_bytes'],
        }
    
    def generate_c_code(self, output_dir: str) -> Dict[str, str]:
        """Generate complete C code for AVR deployment."""
        files = {}
        
        # 1. Generate model header
        files['model_config.h'] = self._generate_config_header()
        
        # 2. Generate quantization header
        files['quantization.h'] = self.quantizer.get_c_header()
        
        # 3. Generate tree data
        files['tree_data.h'] = self._generate_tree_data()
        
        # 4. Generate inference code
        files['inference.h'] = self._generate_inference_header()
        files['inference.c'] = self._generate_inference_code()
        
        # 5. Generate main benchmark code
        files['main.c'] = self._generate_main_code()
        
        # 6. Generate Makefile
        files['Makefile'] = self._generate_makefile()
        
        # Write files
        os.makedirs(output_dir, exist_ok=True)
        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Generated: {filepath}")
        
        return files
    
    def _generate_config_header(self) -> str:
        """Generate model configuration header."""
        mem = self.get_memory_estimate()
        
        return f"""/*
 * Model Configuration Header
 * Model: {self.model_name}
 * Type: {self.model_type}
 * Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>
#include <avr/pgmspace.h>

// Model parameters
#define MODEL_NAME "{self.model_name}"
#define MODEL_TYPE "{self.model_type}"
#define N_FEATURES 16
#define N_CLASSES {self.n_classes}
#define N_TREES {len(self.trees)}
#define N_ESTIMATORS {self.n_estimators}

// Memory estimates
#define TOTAL_NODES {mem['total_nodes']}
#define TREE_MEMORY_BYTES {mem['tree_memory_bytes']}
#define TOTAL_MEMORY_BYTES {mem['total_bytes']}

// Class labels
#define CLASS_NORMAL 0
#define CLASS_BLACKHOLE 1
#define CLASS_GRAYHOLE 2
#define CLASS_FLOODING 3
#define CLASS_TDMA 4

// Inference timing
#define TIMER_PRESCALER 1
#define F_CPU 16000000UL

#endif // MODEL_CONFIG_H
"""

    def _generate_tree_data(self) -> str:
        """Generate tree data arrays."""
        lines = [
            "/*",
            " * Tree Data Arrays (stored in PROGMEM)",
            f" * Model: {self.model_name}",
            f" * Trees: {len(self.trees)}",
            " */",
            "",
            "#ifndef TREE_DATA_H",
            "#define TREE_DATA_H",
            "",
            "#include <avr/pgmspace.h>",
            "#include <stdint.h>",
            "",
        ]
        
        # Limit trees for ATmega328P memory
        max_trees = min(len(self.trees), 10)  # Start with 10 trees for testing
        
        if len(self.trees) > max_trees:
            lines.append(f"// Note: Using {max_trees} of {len(self.trees)} trees for memory constraints")
            lines.append(f"#define ACTIVE_TREES {max_trees}")
            lines.append("")
        
        # Generate tree arrays
        for i, tree in enumerate(self.trees[:max_trees]):
            lines.append(tree.to_c_array(f"tree_{i}"))
            lines.append("")
        
        # Generate tree pointer array
        lines.append("// Tree pointer array")
        lines.append(f"static const uint8_t* const trees[{max_trees}] PROGMEM = {{")
        for i in range(max_trees):
            lines.append(f"    tree_{i},")
        lines.append("};")
        lines.append("")
        
        # Tree sizes
        lines.append("// Tree node counts")
        lines.append(f"static const uint16_t tree_sizes[{max_trees}] PROGMEM = {{")
        lines.append("    " + ", ".join(str(t.n_nodes) for t in self.trees[:max_trees]))
        lines.append("};")
        lines.append("")
        
        lines.append("#endif // TREE_DATA_H")
        
        return "\n".join(lines)
    
    def _generate_inference_header(self) -> str:
        """Generate inference function header."""
        return """/*
 * Inference Functions Header
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

// Quantize a single feature value
uint8_t quantize_feature(float value, uint8_t feature_idx);

// Quantize all features
void quantize_features(const float* features, uint8_t* quantized);

// Traverse a single tree
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features);

// Ensemble prediction (majority voting)
uint8_t ensemble_predict(const uint8_t* features);

// Get timing for single inference
uint32_t benchmark_inference(const uint8_t* features);

#endif // INFERENCE_H
"""

    def _generate_inference_code(self) -> str:
        """Generate inference implementation code."""
        max_trees = min(len(self.trees), 10)
        
        return f"""/*
 * Inference Implementation for AVR
 * Optimized for ATmega328P
 */

#include "inference.h"
#include "model_config.h"
#include "quantization.h"
#include "tree_data.h"
#include <avr/pgmspace.h>

// Quantize a single feature
uint8_t quantize_feature(float value, uint8_t feature_idx) {{
    int16_t scale = pgm_read_word(&FEATURE_SCALE[feature_idx]);
    uint8_t zero = pgm_read_byte(&FEATURE_ZERO[feature_idx]);
    
    // Fixed-point multiplication: (value * scale) >> 8 + zero
    int32_t scaled = (int32_t)(value * scale) >> 8;
    int16_t result = scaled + zero;
    
    // Clamp to uint8
    if (result < 0) return 0;
    if (result > 255) return 255;
    return (uint8_t)result;
}}

// Quantize all features
void quantize_features(const float* features, uint8_t* quantized) {{
    for (uint8_t i = 0; i < N_FEATURES; i++) {{
        quantized[i] = quantize_feature(features[i], i);
    }}
}}

// Traverse a single tree and return predicted class
// Tree node format: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features) {{
    uint16_t node_idx = 0;
    
    while (1) {{
        uint16_t offset = node_idx * 6;
        
        // Read node from PROGMEM
        uint8_t feature_idx = pgm_read_byte(&tree_data[offset]);
        uint8_t threshold = pgm_read_byte(&tree_data[offset + 1]);
        
        // Check if leaf node
        if (feature_idx == 255) {{
            // threshold field contains class_id for leaves
            return threshold;
        }}
        
        // Get feature value and compare
        uint8_t feature_val = features[feature_idx];
        
        if (feature_val <= threshold) {{
            // Go left
            uint8_t left_lo = pgm_read_byte(&tree_data[offset + 2]);
            uint8_t left_hi = pgm_read_byte(&tree_data[offset + 3]);
            node_idx = left_lo | (left_hi << 8);
        }} else {{
            // Go right
            uint8_t right_lo = pgm_read_byte(&tree_data[offset + 4]);
            uint8_t right_hi = pgm_read_byte(&tree_data[offset + 5]);
            node_idx = right_lo | (right_hi << 8);
        }}
    }}
}}

// Ensemble prediction using majority voting
uint8_t ensemble_predict(const uint8_t* features) {{
    uint8_t votes[N_CLASSES] = {{0}};
    
    // Get predictions from all trees
    for (uint8_t t = 0; t < {max_trees}; t++) {{
        const uint8_t* tree_ptr = (const uint8_t*)pgm_read_ptr(&trees[t]);
        uint8_t pred = tree_predict(tree_ptr, features);
        if (pred < N_CLASSES) {{
            votes[pred]++;
        }}
    }}
    
    // Find majority vote
    uint8_t max_votes = 0;
    uint8_t predicted_class = 0;
    
    for (uint8_t c = 0; c < N_CLASSES; c++) {{
        if (votes[c] > max_votes) {{
            max_votes = votes[c];
            predicted_class = c;
        }}
    }}
    
    return predicted_class;
}}

// Benchmark inference timing using Timer1
uint32_t benchmark_inference(const uint8_t* features) {{
    // Reset Timer1
    TCNT1 = 0;
    
    // Start timing
    TCCR1B = (1 << CS10);  // No prescaler, start timer
    
    // Run inference
    volatile uint8_t result = ensemble_predict(features);
    (void)result;  // Prevent optimization
    
    // Stop timer
    TCCR1B = 0;
    
    return TCNT1;
}}
"""

    def _generate_main_code(self) -> str:
        """Generate main benchmark code."""
        return """/*
 * Main Benchmark Program for AVR
 * Measures inference latency using Timer1
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdio.h>
#include "model_config.h"
#include "inference.h"

// UART configuration for output
#define BAUD 9600
#define UBRR_VALUE ((F_CPU / 16 / BAUD) - 1)

void uart_init(void) {
    UBRR0H = (uint8_t)(UBRR_VALUE >> 8);
    UBRR0L = (uint8_t)UBRR_VALUE;
    UCSR0B = (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void uart_putchar(char c) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = c;
}

void uart_puts(const char* s) {
    while (*s) {
        uart_putchar(*s++);
    }
}

void uart_putnum(uint32_t n) {
    char buf[12];
    sprintf(buf, "%lu", n);
    uart_puts(buf);
}

// Timer1 initialization for cycle counting
void timer_init(void) {
    // Timer1: Normal mode, no prescaler
    TCCR1A = 0;
    TCCR1B = 0;  // Stopped initially
}

// Test feature vectors (example values)
// These should be replaced with actual test data
static const float test_features[5][16] = {
    {1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
    {2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2, 2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2},
    {10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0},
};

int main(void) {
    // Initialize
    uart_init();
    timer_init();
    
    // Print header
    uart_puts("\\r\\n=== AVR Inference Latency Benchmark ===\\r\\n");
    uart_puts("Model: ");
    uart_puts(MODEL_NAME);
    uart_puts("\\r\\n");
    uart_puts("F_CPU: 16 MHz\\r\\n");
    uart_puts("Trees: ");
    uart_putnum(N_TREES);
    uart_puts("\\r\\n\\r\\n");
    
    // Quantized feature buffer
    uint8_t q_features[N_FEATURES];
    
    // Warm-up run
    uart_puts("Warm-up...\\r\\n");
    for (uint8_t i = 0; i < 10; i++) {
        quantize_features(test_features[0], q_features);
        volatile uint8_t r = ensemble_predict(q_features);
        (void)r;
    }
    
    // Benchmark runs
    uart_puts("\\r\\nBenchmark Results (cycles):\\r\\n");
    uart_puts("Sample,Cycles,Prediction\\r\\n");
    
    uint32_t total_cycles = 0;
    uint32_t min_cycles = 0xFFFFFFFF;
    uint32_t max_cycles = 0;
    
    for (uint8_t s = 0; s < 5; s++) {
        // Quantize features
        quantize_features(test_features[s], q_features);
        
        // Run multiple iterations per sample
        for (uint8_t iter = 0; iter < 10; iter++) {
            uint32_t cycles = benchmark_inference(q_features);
            
            uart_putnum(s);
            uart_putchar(',');
            uart_putnum(cycles);
            uart_putchar(',');
            
            uint8_t pred = ensemble_predict(q_features);
            uart_putnum(pred);
            uart_puts("\\r\\n");
            
            total_cycles += cycles;
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
        }
    }
    
    // Print summary
    uart_puts("\\r\\n=== Summary ===\\r\\n");
    uart_puts("Total iterations: 50\\r\\n");
    uart_puts("Min cycles: ");
    uart_putnum(min_cycles);
    uart_puts("\\r\\n");
    uart_puts("Max cycles: ");
    uart_putnum(max_cycles);
    uart_puts("\\r\\n");
    uart_puts("Avg cycles: ");
    uart_putnum(total_cycles / 50);
    uart_puts("\\r\\n");
    
    // Convert to time
    uart_puts("\\r\\nAt 16 MHz:\\r\\n");
    uart_puts("Min time (us): ");
    uart_putnum(min_cycles / 16);
    uart_puts("\\r\\n");
    uart_puts("Avg time (us): ");
    uart_putnum(total_cycles / 50 / 16);
    uart_puts("\\r\\n");
    uart_puts("Max time (us): ");
    uart_putnum(max_cycles / 16);
    uart_puts("\\r\\n");
    
    uart_puts("\\r\\n=== Benchmark Complete ===\\r\\n");
    
    // Infinite loop
    while (1) {
        _delay_ms(1000);
    }
    
    return 0;
}
"""

    def _generate_makefile(self) -> str:
        """Generate Makefile for AVR compilation and simulation."""
        return f"""# Makefile for AVR Inference Benchmark
# Target: ATmega328P

MCU = atmega328p
F_CPU = 16000000UL
BAUD = 9600

CC = avr-gcc
OBJCOPY = avr-objcopy
SIZE = avr-size
SIMAVR = simavr

CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -Wall -Wextra
CFLAGS += -ffunction-sections -fdata-sections
LDFLAGS = -Wl,--gc-sections

TARGET = inference_benchmark
SRCS = main.c inference.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean size sim

all: $(TARGET).hex size

$(TARGET).elf: $(OBJS)
\t$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET).hex: $(TARGET).elf
\t$(OBJCOPY) -O ihex -R .eeprom $< $@

%.o: %.c
\t$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
\t$(SIZE) --mcu=$(MCU) -C $(TARGET).elf

# Run simulation with simavr
sim: $(TARGET).elf
\t$(SIMAVR) -m $(MCU) -f $(F_CPU) $(TARGET).elf

# Clean up
clean:
\trm -f $(OBJS) $(TARGET).elf $(TARGET).hex

# Show memory usage
memory: $(TARGET).elf
\t@echo "Memory Usage:"
\t@$(SIZE) -A $(TARGET).elf
"""


# ============================================================================
# MAIN QUANTIZATION PIPELINE
# ============================================================================

def load_models_from_mlflow(experiment_name: str, target_models: List[str]) -> Dict[str, Any]:
    """Load models from MLflow experiment."""
    mlflow.set_tracking_uri('mlruns')
    client = MlflowClient()
    
    # Find experiment
    experiments = client.search_experiments()
    exp_id = None
    for exp in experiments:
        if experiment_name in exp.name:
            exp_id = exp.experiment_id
            break
    
    if exp_id is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[exp_id])
    
    models = {}
    for model_name in target_models:
        # Use exact match to avoid loading wrong model
        mask = runs['tags.mlflow.runName'] == model_name
        matching = runs[mask]
        
        if len(matching) > 0:
            run = matching.iloc[0]
            run_id = run['run_id']
            
            # Load model
            model_uri = f"runs:/{run_id}/model"
            try:
                model = mlflow.sklearn.load_model(model_uri)
                models[model_name] = {
                    'model': model,
                    'run_id': run_id,
                    'run_name': run['tags.mlflow.runName'],
                }
                print(f"  Loaded: {run['tags.mlflow.runName']}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
        else:
            print(f"  Not found: {model_name}")
    
    return models


def main():
    """Main quantization and conversion pipeline."""
    print("=" * 80)
    print("INT8 MODEL QUANTIZATION FOR AVR DEPLOYMENT")
    print("=" * 80)
    
    # Target models
    TARGET_MODELS = [
        'Extra_Trees_No_Oversampling',
        'Random_Forest_No_Oversampling', 
        'Gradient_Boosting_No_Oversampling',
        'HistGradient_Boosting_No_Oversampling'
    ]
    
    # Load dataset for quantization calibration
    print("\n1. Loading calibration data...")
    df = pd.read_csv('data/WSN-DS.csv')
    df.columns = df.columns.str.strip()
    
    # Prepare features - exclude non-numeric columns
    exclude_cols = ['id', 'who CH', 'Attack', 'Attack Type', 'Attacked']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Keep only numeric columns
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    feature_cols = list(numeric_df.columns)
    X = numeric_df.values.astype(np.float32)
    
    print(f"   Feature columns: {feature_cols}")
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    
    # Load models from MLflow
    print("\n2. Loading models from MLflow...")
    models = load_models_from_mlflow('No_Feature_Engineering_No_Oversampling', TARGET_MODELS)
    
    if len(models) == 0:
        print("   No models found. Check experiment name.")
        return
    
    # Create output directory
    output_base = 'avr_model_converter/generated'
    os.makedirs(output_base, exist_ok=True)
    
    # Tree limits based on model type for ATmega328P memory
    # Flash = 32KB, need to fit code + data
    # Rough estimate: ~6 bytes per node, target ~20KB for trees
    # Max depth: 6 gives max 63 nodes per tree (2^6 - 1 = 63)
    # 5 trees × 63 nodes × 6 bytes = 1890 bytes for trees
    TREE_LIMITS = {
        'Extra_Trees_No_Oversampling': 5,          # Large trees, depth-limited
        'Random_Forest_No_Oversampling': 5,        # Large trees, depth-limited
        'Gradient_Boosting_No_Oversampling': 20,   # Smaller trees
        'HistGradient_Boosting_No_Oversampling': 20,  # Smaller trees
    }
    
    # Max tree depth for each model type
    # Depth 6 = max 63 nodes, Depth 7 = max 127 nodes, Depth 8 = max 255 nodes
    MAX_DEPTHS = {
        'Extra_Trees_No_Oversampling': 6,          # Limit deep trees
        'Random_Forest_No_Oversampling': 6,        # Limit deep trees
        'Gradient_Boosting_No_Oversampling': 6,    # Usually shallow anyway
        'HistGradient_Boosting_No_Oversampling': 6,  # Usually shallow
    }
    
    # Process each model
    print("\n3. Quantizing and converting models...")
    
    for model_name, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print('='*60)
        
        model = model_info['model']
        max_trees = TREE_LIMITS.get(model_name, 10)
        max_depth = MAX_DEPTHS.get(model_name, 6)
        
        # Create quantized ensemble
        q_ensemble = QuantizedEnsemble(model_name)
        
        try:
            q_ensemble.from_sklearn_model(model, X, max_trees=max_trees, max_depth=max_depth)
            
            # Memory estimate
            mem = q_ensemble.get_memory_estimate()
            print(f"   Trees extracted: {len(q_ensemble.trees)}")
            print(f"   Total nodes: {mem['total_nodes']}")
            print(f"   Memory: {mem['total_bytes']} bytes ({mem['total_bytes']/1024:.1f} KB)")
            print(f"   Fits in Flash (32KB): {mem['fits_in_flash']}")
            print(f"   Fits in SRAM (2KB): {mem['fits_in_sram']}")
            
            # Generate C code
            model_dir = os.path.join(output_base, model_name.lower())
            q_ensemble.generate_c_code(model_dir)
            
            print(f"   Output: {model_dir}/")
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE")
    print("=" * 80)
    print("""
Next steps:
1. Install avr-gcc and simavr:
   brew install avr-gcc simavr   # macOS
   
2. Build and simulate:
   cd avr_model_converter/generated/<model_name>
   make
   make sim

3. Collect timing results from UART output
""")


if __name__ == '__main__':
    main()

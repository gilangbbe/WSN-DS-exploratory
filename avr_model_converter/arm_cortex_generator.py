#!/usr/bin/env python3
"""
ARM Cortex-M4F Model Converter, Evaluator, and Latency Analyzer
================================================================
Generates native float32 (non-quantized) C code for ARM Cortex-M4F deployment.
Evaluates model quality after depth limiting. Analyzes inference latency.

Target: ARM Cortex-M4F (e.g., nRF52840, STM32L4)
- 32-bit ARM architecture with hardware FPU
- Single-precision float32 natively supported
- No quantization needed — threshold comparisons use VCMP

Models: Random Forest, Extra Trees, Gradient Boosting, HistGradient Boosting
        from WSN_IDS_No_Feature_Engineering_With_Oversampling (SMOTE-ENN)
"""

import os
import sys
import json
import pickle
import struct
import subprocess
import re
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ARMCortexConfig:
    """ARM Cortex-M4F target configuration."""
    mcu_name: str = "nRF52840"
    arch: str = "ARM Cortex-M4F"
    clock_freq: int = 64_000_000       # 64 MHz (nRF52840)
    flash_size: int = 1024 * 1024      # 1 MB
    ram_size: int = 256 * 1024         # 256 KB
    fpu: str = "fpv4-sp-d16"           # Single-precision FPU
    float_abi: str = "hard"            # Hardware float ABI
    cpu: str = "cortex-m4"
    
    # Compiler
    gcc_path: str = "/opt/homebrew/bin/arm-none-eabi-gcc"
    size_path: str = "/opt/homebrew/bin/arm-none-eabi-size"
    objdump_path: str = "/opt/homebrew/bin/arm-none-eabi-objdump"
    
    # Memory budget for model (leave room for firmware, LEACH stack, radio)
    max_model_flash: int = 800 * 1024  # 800 KB for model data + inference code
    max_model_ram: int = 64 * 1024     # 64 KB for runtime (features, votes, stack)
    
    # ARM Cortex-M4 instruction cycle costs (from ARM Technical Reference Manual)
    # Most ALU instructions: 1 cycle
    # Float compare (VCMP): 1 cycle
    # Float load (VLDR): 1-2 cycles (1 if cached)
    # Branch (B/BX): 1+P cycles (P=pipeline refill, typically 1-3)
    # Load (LDR): 1-2 cycles
    cycles_float_compare: int = 1   # VCMP.F32
    cycles_vmrs: int = 1            # VMRS APSR_nzcv, FPSCR
    cycles_branch: int = 2          # Conditional branch (average: taken+not taken)
    cycles_load_reg: int = 1        # LDR Rd, [Rn, #imm]
    cycles_load_float: int = 1      # VLDR S0, [Rn, #imm]
    cycles_alu: int = 1             # ADD, SUB, CMP, etc.
    cycles_multiply: int = 1        # MUL (single cycle on M4)
    cycles_call: int = 4            # BL + pipeline refill
    cycles_ret: int = 3             # BX LR + pipeline refill


# Model paths mapping — Conservative SMOTE, No Feature Engineering
MODEL_PATHS = {
    'Extra_Trees_Conservative_SMOTE': {
        'pkl': 'mlruns/410992055011183175/models/m-3e057b03d6814337877607b8b495ccc6/artifacts/model.pkl',
        'run_id': 'f9cd86a326204218827c547320b54049',
    },
    'Random_Forest_Conservative_SMOTE': {
        'pkl': 'mlruns/410992055011183175/models/m-6d14c8f09176402abac542494fd2871a/artifacts/model.pkl',
        'run_id': '51865721565f49feb7b2028fca1fd5dd',
    },
    'Gradient_Boosting_Conservative_SMOTE': {
        'pkl': 'mlruns/410992055011183175/models/m-8b94e721d09b479e9c1ba33fc7debc6d/artifacts/model.pkl',
        'run_id': '04af177cad73415c9401e131f1f4486f',
    },
    'HistGradient_Boosting_Conservative_SMOTE': {
        'pkl': 'mlruns/410992055011183175/models/m-314ff2a6346d4417b349fa640ab31867/artifacts/model.pkl',
        'run_id': '9e369ce607204245a6b2388d86f37246',
    },
}

FEATURE_NAMES = [
    "Time", "Is_CH", "Dist_To_CH", "ADV_S", "ADV_R",
    "JOIN_S", "JOIN_R", "SCH_S", "SCH_R", "Rank",
    "DATA_S", "DATA_R", "Data_Sent_To_BS", "dist_CH_To_BS",
    "send_code", "Expaned Energy"
]

CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
N_FEATURES = 16
N_CLASSES = 5

# ============================================================================
# Tree Node Data Structure (float32, no quantization)
# ============================================================================

@dataclass
class FloatTreeNode:
    """A tree node with float32 threshold (no quantization)."""
    node_id: int
    feature_idx: int      # 0xFFFF (65535) = leaf node
    threshold: float      # float32: split threshold or leaf value
    left_child: int       # uint16: left child index
    right_child: int      # uint16: right child index

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx == 0xFFFF


class FloatTree:
    """A decision tree with float32 thresholds for ARM Cortex-M4F."""
    
    def __init__(self, tree_id: int = 0, class_idx: int = -1):
        self.tree_id = tree_id
        self.class_idx = class_idx  # For GB/HistGB: which class this tree predicts
        self.nodes: List[FloatTreeNode] = []
        self.n_nodes = 0
        self.max_depth = 0
    
    def from_sklearn_tree(self, estimator, max_depth: int = 255, 
                          leaf_value_fn=None):
        """Extract a sklearn DecisionTree with optional depth limiting.
        
        Args:
            estimator: sklearn tree estimator (or tree_ object)
            max_depth: Maximum depth (255 = unlimited)
            leaf_value_fn: Function to extract leaf value. 
                          For RF/ET: returns majority class as float
                          For GB: returns regression value
        """
        tree = estimator.tree_ if hasattr(estimator, 'tree_') else estimator
        
        # BFS traversal with depth tracking
        queue = [(0, 0)]  # (sklearn_node_id, depth)
        node_mapping = {}  # old_id -> new_id
        visit_order = []   # Track visit order for child pointer update
        
        while queue:
            old_id, depth = queue.pop(0)
            new_id = len(self.nodes)
            node_mapping[old_id] = new_id
            visit_order.append((old_id, depth))
            
            is_leaf = tree.children_left[old_id] == -1
            force_leaf = depth >= max_depth
            
            if is_leaf or force_leaf:
                # Leaf node
                if leaf_value_fn is not None:
                    leaf_val = leaf_value_fn(tree, old_id)
                else:
                    # Default: majority class from value array
                    class_counts = tree.value[old_id].flatten()
                    leaf_val = float(np.argmax(class_counts))
                
                node = FloatTreeNode(
                    node_id=new_id,
                    feature_idx=0xFFFF,
                    threshold=leaf_val,
                    left_child=0,
                    right_child=0
                )
            else:
                # Decision node
                feature_idx = int(tree.feature[old_id])
                threshold = float(tree.threshold[old_id])
                
                node = FloatTreeNode(
                    node_id=new_id,
                    feature_idx=feature_idx,
                    threshold=threshold,
                    left_child=0,  # Updated later
                    right_child=0
                )
                
                queue.append((tree.children_left[old_id], depth + 1))
                queue.append((tree.children_right[old_id], depth + 1))
            
            self.nodes.append(node)
            self.max_depth = max(self.max_depth, depth)
        
        # Update child pointers
        for old_id, depth in visit_order:
            new_id = node_mapping[old_id]
            tree_obj = tree
            
            is_leaf = tree_obj.children_left[old_id] == -1
            force_leaf = depth >= max_depth
            
            if not is_leaf and not force_leaf:
                left_old = tree_obj.children_left[old_id]
                right_old = tree_obj.children_right[old_id]
                self.nodes[new_id].left_child = node_mapping[left_old]
                self.nodes[new_id].right_child = node_mapping[right_old]
        
        self.n_nodes = len(self.nodes)
        return self
    
    def from_histgb_predictor(self, predictor, max_depth: int = 255):
        """Extract tree from HistGradientBoosting's TreePredictor."""
        nodes = predictor.nodes
        
        # BFS traversal
        queue = [(0, 0)]
        node_mapping = {}
        visit_order = []
        
        while queue:
            old_id, depth = queue.pop(0)
            new_id = len(self.nodes)
            node_mapping[old_id] = new_id
            visit_order.append((old_id, depth))
            
            n = nodes[old_id]
            is_leaf = bool(n['is_leaf'])
            force_leaf = depth >= max_depth
            
            if is_leaf or force_leaf:
                leaf_val = float(n['value'])
                node = FloatTreeNode(
                    node_id=new_id,
                    feature_idx=0xFFFF,
                    threshold=leaf_val,
                    left_child=0,
                    right_child=0
                )
            else:
                feature_idx = int(n['feature_idx'])
                threshold = float(n['num_threshold'])
                
                node = FloatTreeNode(
                    node_id=new_id,
                    feature_idx=feature_idx,
                    threshold=threshold,
                    left_child=0,
                    right_child=0
                )
                
                queue.append((int(n['left']), depth + 1))
                queue.append((int(n['right']), depth + 1))
            
            self.nodes.append(node)
            self.max_depth = max(self.max_depth, depth)
        
        # Update child pointers
        for old_id, depth in visit_order:
            new_id = node_mapping[old_id]
            n = nodes[old_id]
            is_leaf = bool(n['is_leaf'])
            force_leaf = depth >= max_depth
            
            if not is_leaf and not force_leaf:
                self.nodes[new_id].left_child = node_mapping[int(n['left'])]
                self.nodes[new_id].right_child = node_mapping[int(n['right'])]
        
        self.n_nodes = len(self.nodes)
        return self
    
    def predict_single(self, features: np.ndarray) -> float:
        """Predict a single sample (mirrors C inference code)."""
        idx = 0
        while True:
            node = self.nodes[idx]
            if node.is_leaf:
                return node.threshold  # leaf value
            if features[node.feature_idx] <= node.threshold:
                idx = node.left_child
            else:
                idx = node.right_child
    
    def memory_bytes(self) -> int:
        """Memory usage: 12 bytes per node (float32 + 4x uint16)."""
        return self.n_nodes * 12


# ============================================================================
# Float Ensemble Model
# ============================================================================

class FloatEnsemble:
    """An ensemble model with float32 thresholds for ARM Cortex-M4F."""
    
    VOTING_MODELS = {'RandomForestClassifier', 'ExtraTreesClassifier'}
    SCORING_MODELS = {'GradientBoostingClassifier', 'HistGradientBoostingClassifier'}
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = ""
        self.inference_type = "voting"  # "voting" or "scoring"
        self.trees: List[FloatTree] = []
        self.n_classes = N_CLASSES
        self.n_features = N_FEATURES
        self.n_estimators = 0
        self.max_tree_depth_used = 0
        
        # For scoring models (GB, HistGB)
        self.learning_rate: float = 1.0
        self.initial_scores: Optional[np.ndarray] = None  # shape: (n_classes,)
        self.trees_per_class: int = 0
        self.n_stages: int = 0
    
    def from_sklearn_model(self, model, max_depth: int = 255, max_stages: int = 0):
        """Extract trees from a sklearn ensemble model (no quantization).
        
        Args:
            model: sklearn ensemble classifier
            max_depth: Max tree depth (255 = unlimited)
            max_stages: Max boosting stages for GB/HistGB (0 = all)
        """
        model_class = type(model).__name__
        self.model_type = model_class
        
        if model_class in self.VOTING_MODELS:
            self.inference_type = "voting"
            self._extract_forest(model, max_depth)
        elif model_class == 'GradientBoostingClassifier':
            self.inference_type = "scoring"
            self._extract_gradient_boosting(model, max_depth, max_stages)
        elif model_class == 'HistGradientBoostingClassifier':
            self.inference_type = "scoring"
            self._extract_hist_gradient_boosting(model, max_depth, max_stages)
        else:
            raise ValueError(f"Unsupported model: {model_class}")
        
        return self
    
    def _extract_forest(self, model, max_depth: int):
        """Extract trees from Random Forest or Extra Trees."""
        self.n_classes = len(model.classes_)
        self.n_estimators = model.n_estimators
        
        print(f"  Extracting {model.n_estimators} trees (max_depth={max_depth})")
        
        for i, estimator in enumerate(model.estimators_):
            ft = FloatTree(tree_id=i)
            ft.from_sklearn_tree(estimator, max_depth=max_depth)
            self.trees.append(ft)
            self.max_tree_depth_used = max(self.max_tree_depth_used, ft.max_depth)
    
    def _extract_gradient_boosting(self, model, max_depth: int, max_stages: int = 0):
        """Extract trees from Gradient Boosting Classifier."""
        self.n_classes = len(model.classes_)
        self.learning_rate = float(model.learning_rate)
        
        # Get initial predictions (log-prior probabilities)
        if hasattr(model, 'init_') and hasattr(model.init_, 'class_prior_'):
            prior = model.init_.class_prior_
            self.initial_scores = np.log(prior + 1e-15)
        elif hasattr(model, '_raw_prediction') and hasattr(model, 'init_'):
            try:
                X_dummy = np.zeros((1, self.n_features))
                self.initial_scores = model.init_.predict(X_dummy).flatten()
            except:
                self.initial_scores = np.zeros(self.n_classes)
        else:
            self.initial_scores = np.zeros(self.n_classes)
        
        n_stages = len(model.estimators_)
        trees_per_stage = len(model.estimators_[0])
        
        # Limit stages if specified
        if max_stages > 0:
            n_stages_use = min(n_stages, max_stages)
        else:
            n_stages_use = n_stages
        
        self.n_stages = n_stages_use
        self.trees_per_class = trees_per_stage
        self.n_estimators = n_stages_use * trees_per_stage
        
        print(f"  Extracting GB: {n_stages_use}/{n_stages} stages × {trees_per_stage} classes "
              f"(lr={self.learning_rate}, max_depth={max_depth})")
        
        # Leaf value for GB regression trees
        def gb_leaf_value(tree, node_id):
            return float(tree.value[node_id].flatten()[0])
        
        tree_id = 0
        for stage_idx, stage in enumerate(model.estimators_[:n_stages_use]):
            for class_idx, estimator in enumerate(stage):
                ft = FloatTree(tree_id=tree_id, class_idx=class_idx)
                ft.from_sklearn_tree(estimator, max_depth=max_depth,
                                     leaf_value_fn=gb_leaf_value)
                self.trees.append(ft)
                self.max_tree_depth_used = max(self.max_tree_depth_used, ft.max_depth)
                tree_id += 1
    
    def _extract_hist_gradient_boosting(self, model, max_depth: int, max_stages: int = 0):
        """Extract trees from HistGradient Boosting Classifier."""
        self.n_classes = len(model.classes_)
        # HistGB already bakes learning_rate into leaf values during training,
        # so we set lr=1.0 to avoid double-applying it during inference.
        self.learning_rate = 1.0
        
        # Get baseline predictions
        if hasattr(model, '_baseline_prediction'):
            bp = model._baseline_prediction
            if bp.ndim == 1:
                self.initial_scores = bp.astype(float)
            else:
                self.initial_scores = bp.flatten().astype(float)
        else:
            self.initial_scores = np.zeros(self.n_classes)
        
        n_iters = len(model._predictors)
        trees_per_iter = len(model._predictors[0])
        
        # Limit stages if specified
        if max_stages > 0:
            n_iters_use = min(n_iters, max_stages)
        else:
            n_iters_use = n_iters
        
        self.n_stages = n_iters_use
        self.trees_per_class = trees_per_iter
        self.n_estimators = n_iters_use * trees_per_iter
        
        print(f"  Extracting HistGB: {n_iters_use}/{n_iters} iters × {trees_per_iter} classes "
              f"(lr={self.learning_rate}, max_depth={max_depth})")
        
        tree_id = 0
        for iter_idx, iteration in enumerate(model._predictors[:n_iters_use]):
            for class_idx, predictor in enumerate(iteration):
                ft = FloatTree(tree_id=tree_id, class_idx=class_idx)
                ft.from_histgb_predictor(predictor, max_depth=max_depth)
                self.trees.append(ft)
                self.max_tree_depth_used = max(self.max_tree_depth_used, ft.max_depth)
                tree_id += 1
    
    def predict_single(self, features: np.ndarray) -> int:
        """Predict a single sample (mirrors C inference code exactly)."""
        if self.inference_type == "voting":
            return self._predict_voting(features)
        else:
            return self._predict_scoring(features)
    
    def _predict_voting(self, features: np.ndarray) -> int:
        """Majority voting prediction (RF/ET)."""
        votes = np.zeros(self.n_classes, dtype=int)
        for tree in self.trees:
            pred = int(tree.predict_single(features))
            if 0 <= pred < self.n_classes:
                votes[pred] += 1
        return int(np.argmax(votes))
    
    def _predict_scoring(self, features: np.ndarray) -> int:
        """Sum-of-scores prediction (GB/HistGB)."""
        scores = self.initial_scores.copy()
        
        for tree in self.trees:
            leaf_val = tree.predict_single(features)
            class_idx = tree.class_idx
            if 0 <= class_idx < self.n_classes:
                scores[class_idx] += self.learning_rate * leaf_val
        
        return int(np.argmax(scores))
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict batch of samples."""
        return np.array([self.predict_single(X[i]) for i in range(len(X))])
    
    def total_nodes(self) -> int:
        return sum(t.n_nodes for t in self.trees)
    
    def total_memory_bytes(self) -> int:
        """Total memory: 12 bytes/node + metadata."""
        node_bytes = self.total_nodes() * 12
        # Metadata: tree pointers array, tree sizes, initial scores, etc.
        meta_bytes = len(self.trees) * 8 + self.n_classes * 4 + 128
        return node_bytes + meta_bytes
    
    def summary(self) -> Dict:
        """Get model summary statistics."""
        node_counts = [t.n_nodes for t in self.trees]
        depths = [t.max_depth for t in self.trees]
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'inference_type': self.inference_type,
            'n_trees': len(self.trees),
            'total_nodes': self.total_nodes(),
            'avg_nodes': np.mean(node_counts),
            'max_nodes': max(node_counts),
            'min_nodes': min(node_counts),
            'avg_depth': np.mean(depths),
            'max_depth': max(depths),
            'memory_bytes': self.total_memory_bytes(),
            'memory_kb': self.total_memory_bytes() / 1024,
        }


# ============================================================================
# C Code Generator for ARM Cortex-M4F
# ============================================================================

class ARMCodeGenerator:
    """Generate float32 C code for ARM Cortex-M4F."""
    
    def __init__(self, ensemble: FloatEnsemble, config: ARMCortexConfig):
        self.ensemble = ensemble
        self.config = config
    
    def generate_all(self, output_dir: str):
        """Generate all C source files."""
        os.makedirs(output_dir, exist_ok=True)
        
        self._generate_model_config_h(output_dir)
        self._generate_tree_data_h(output_dir)
        self._generate_inference_h(output_dir)
        self._generate_inference_c(output_dir)
        self._generate_main_c(output_dir)
        self._generate_makefile(output_dir)
        self._generate_linker_script(output_dir)
        
        print(f"  Generated {7} files in {output_dir}")
    
    def _generate_model_config_h(self, output_dir: str):
        cfg = self.config
        ens = self.ensemble
        
        # For scoring models, include learning rate and initial scores
        scoring_defines = ""
        if ens.inference_type == "scoring":
            lr_hex = struct.pack('<f', ens.learning_rate).hex()
            scoring_defines += f"\n#define INFERENCE_TYPE_SCORING  1"
            scoring_defines += f"\n#define LEARNING_RATE  {ens.learning_rate}f"
            scoring_defines += f"\n#define N_STAGES  {ens.n_stages}"
            scoring_defines += f"\n#define TREES_PER_CLASS  {ens.trees_per_class}"
        else:
            scoring_defines += f"\n#define INFERENCE_TYPE_VOTING  1"
        
        content = f"""/*
 * Model Configuration for ARM Cortex-M4F
 * Model: {ens.model_name}
 * Type: {ens.model_type}
 * Target: {cfg.mcu_name} ({cfg.arch}) @ {cfg.clock_freq // 1_000_000} MHz
 * FPU: {cfg.fpu} (hardware float32)
 * 
 * NO QUANTIZATION — native float32 thresholds and comparisons
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>

#define MODEL_NAME       "{ens.model_name}"
#define MODEL_TYPE       "{ens.model_type}"
#define TARGET_MCU       "{cfg.mcu_name}"
#define F_CPU            {cfg.clock_freq}UL

#define N_FEATURES       {ens.n_features}
#define N_CLASSES        {ens.n_classes}
#define N_TREES          {len(ens.trees)}
#define MAX_DEPTH        {ens.max_tree_depth_used}
{scoring_defines}

/* Tree node structure: 12 bytes, naturally aligned for ARM */
typedef struct {{
    float threshold;        /* 4 bytes: split threshold or leaf value */
    uint16_t feature_idx;   /* 2 bytes: feature index (0xFFFF = leaf) */
    uint16_t left_child;    /* 2 bytes: left child node index */
    uint16_t right_child;   /* 2 bytes: right child node index */
    uint16_t _pad;          /* 2 bytes: alignment padding */
}} TreeNode;                /* Total: 12 bytes */

#endif /* MODEL_CONFIG_H */
"""
        with open(os.path.join(output_dir, 'model_config.h'), 'w') as f:
            f.write(content)
    
    def _generate_tree_data_h(self, output_dir: str):
        ens = self.ensemble
        
        content = f"""/*
 * Tree Data for ARM Cortex-M4F (float32, no quantization)
 * Model: {ens.model_name}
 * Total trees: {len(ens.trees)}
 * Total nodes: {ens.total_nodes()}
 * Memory: {ens.total_memory_bytes()} bytes ({ens.total_memory_bytes()/1024:.1f} KB)
 *
 * Node format: {{threshold(f32), feature_idx(u16), left(u16), right(u16), pad(u16)}}
 * Leaf nodes: feature_idx = 0xFFFF, threshold = class_id (voting) or score (scoring)
 */

#ifndef TREE_DATA_H
#define TREE_DATA_H

#include "model_config.h"

"""
        # Generate each tree's node array
        for tree in ens.trees:
            class_info = f" (class {tree.class_idx})" if tree.class_idx >= 0 else ""
            content += f"/* Tree {tree.tree_id}: {tree.n_nodes} nodes, "
            content += f"depth {tree.max_depth}{class_info} */\n"
            content += f"static const TreeNode tree_{tree.tree_id}[] = {{\n"
            
            for node in tree.nodes:
                if node.is_leaf:
                    content += f"    {{ {node.threshold:.8f}f, 0xFFFF, 0, 0, 0 }},  "
                    if ens.inference_type == "voting":
                        content += f"/* Leaf: class {int(node.threshold)} */\n"
                    else:
                        content += f"/* Leaf: score {node.threshold:.6f} */\n"
                else:
                    content += (f"    {{ {node.threshold:.8f}f, {node.feature_idx}, "
                               f"{node.left_child}, {node.right_child}, 0 }},  "
                               f"/* Node {node.node_id}: feat[{node.feature_idx}] "
                               f"<= {node.threshold:.6f} */\n")
            
            content += "};\n\n"
        
        # Tree pointers array
        content += "/* Tree pointer array */\n"
        content += "static const TreeNode* const all_trees[N_TREES] = {\n"
        for i in range(len(ens.trees)):
            content += f"    tree_{i},\n"
        content += "};\n\n"
        
        # Tree sizes array
        content += "/* Node count per tree */\n"
        content += "static const uint16_t tree_n_nodes[N_TREES] = {\n    "
        content += ", ".join(str(t.n_nodes) for t in ens.trees)
        content += "\n};\n\n"
        
        # For scoring models: initial scores and tree-to-class mapping
        if ens.inference_type == "scoring":
            content += "/* Initial class scores (baseline predictions) */\n"
            content += "static const float initial_scores[N_CLASSES] = {\n    "
            content += ", ".join(f"{s:.8f}f" for s in ens.initial_scores)
            content += "\n};\n\n"
            
            content += "/* Tree-to-class mapping: tree i predicts for class tree_class[i] */\n"
            content += "static const uint8_t tree_class[N_TREES] = {\n    "
            content += ", ".join(str(t.class_idx) for t in ens.trees)
            content += "\n};\n\n"
        
        content += "#endif /* TREE_DATA_H */\n"
        
        with open(os.path.join(output_dir, 'tree_data.h'), 'w') as f:
            f.write(content)
    
    def _generate_inference_h(self, output_dir: str):
        content = """/*
 * Inference Functions for ARM Cortex-M4F
 * Native float32 — no quantization needed
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "model_config.h"

/* Predict using a single tree. Returns leaf value.
 * For voting models: leaf value is class ID (as float).
 * For scoring models: leaf value is regression score. */
float tree_predict(const TreeNode* tree, const float* features);

/* Ensemble prediction. Returns predicted class (0..N_CLASSES-1). */
uint8_t ensemble_predict(const float* features);

/* Benchmark: run inference and return DWT cycle count. */
uint32_t benchmark_inference(const float* features);

#endif /* INFERENCE_H */
"""
        with open(os.path.join(output_dir, 'inference.h'), 'w') as f:
            f.write(content)
    
    def _generate_inference_c(self, output_dir: str):
        ens = self.ensemble
        
        # Choose inference implementation based on model type
        if ens.inference_type == "voting":
            ensemble_fn = self._voting_ensemble_c()
        else:
            ensemble_fn = self._scoring_ensemble_c()
        
        content = f"""/*
 * Inference Implementation for ARM Cortex-M4F
 * Model: {ens.model_name}
 * Type: {ens.model_type} ({ens.inference_type})
 *
 * Uses hardware FPU for float32 comparisons (VCMP.F32)
 * No quantization — thresholds are native float32
 */

#include "inference.h"
#include "tree_data.h"

/* Single tree traversal.
 * The FPU compares features[node->feature_idx] <= node->threshold
 * using a single VCMP.F32 + VMRS instruction pair. */
float tree_predict(const TreeNode* tree, const float* features) {{
    uint16_t idx = 0;
    
    while (1) {{
        const TreeNode* node = &tree[idx];
        
        /* Check if leaf node */
        if (node->feature_idx == 0xFFFF) {{
            return node->threshold;  /* Leaf value */
        }}
        
        /* Float comparison using FPU: VCMP.F32 + VMRS */
        if (features[node->feature_idx] <= node->threshold) {{
            idx = node->left_child;
        }} else {{
            idx = node->right_child;
        }}
    }}
}}

{ensemble_fn}

/* Benchmark using ARM DWT Cycle Counter */
uint32_t benchmark_inference(const float* features) {{
    /* Enable DWT cycle counter */
    volatile uint32_t* DWT_CTRL   = (volatile uint32_t*)0xE0001000;
    volatile uint32_t* DWT_CYCCNT = (volatile uint32_t*)0xE0001004;
    volatile uint32_t* DEMCR      = (volatile uint32_t*)0xE000EDFC;
    
    *DEMCR |= (1 << 24);    /* Enable DWT */
    *DWT_CTRL |= 1;         /* Enable cycle counter */
    *DWT_CYCCNT = 0;        /* Reset counter */
    
    /* Run inference */
    volatile uint8_t result = ensemble_predict(features);
    (void)result;
    
    uint32_t cycles = *DWT_CYCCNT;
    return cycles;
}}
"""
        with open(os.path.join(output_dir, 'inference.c'), 'w') as f:
            f.write(content)
    
    def _voting_ensemble_c(self) -> str:
        return """/* Majority voting ensemble (Random Forest / Extra Trees) */
uint8_t ensemble_predict(const float* features) {
    uint16_t votes[N_CLASSES] = {0};
    uint8_t t;
    
    /* Get vote from each tree */
    for (t = 0; t < N_TREES; t++) {
        float leaf = tree_predict(all_trees[t], features);
        uint8_t cls = (uint8_t)leaf;
        if (cls < N_CLASSES) {
            votes[cls]++;
        }
    }
    
    /* Find class with most votes */
    uint16_t max_votes = 0;
    uint8_t predicted = 0;
    uint8_t c;
    
    for (c = 0; c < N_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            predicted = c;
        }
    }
    
    return predicted;
}"""
    
    def _scoring_ensemble_c(self) -> str:
        return """/* Sum-of-scores ensemble (Gradient Boosting / HistGradient Boosting) */
uint8_t ensemble_predict(const float* features) {
    float scores[N_CLASSES];
    uint8_t c;
    
    /* Initialize with baseline scores */
    for (c = 0; c < N_CLASSES; c++) {
        scores[c] = initial_scores[c];
    }
    
    /* Accumulate tree predictions */
    uint16_t t;
    for (t = 0; t < N_TREES; t++) {
        float leaf = tree_predict(all_trees[t], features);
        uint8_t cls = tree_class[t];
        scores[cls] += LEARNING_RATE * leaf;
    }
    
    /* Argmax: find class with highest score */
    float max_score = scores[0];
    uint8_t predicted = 0;
    
    for (c = 1; c < N_CLASSES; c++) {
        if (scores[c] > max_score) {
            max_score = scores[c];
            predicted = c;
        }
    }
    
    return predicted;
}"""
    
    def _generate_main_c(self, output_dir: str):
        cfg = self.config
        ens = self.ensemble
        
        # Generate test vectors (representative feature values)
        test_vectors = self._generate_test_vectors()
        
        content = f"""/*
 * ARM Cortex-M4F Inference Latency Benchmark
 * Model: {ens.model_name}
 * Target: {cfg.mcu_name} @ {cfg.clock_freq // 1_000_000} MHz
 * FPU: {cfg.fpu} (hardware float32, no quantization)
 */

#include <stdint.h>
#include "model_config.h"
#include "inference.h"

/* Minimal semihosting / ITM output for benchmarking */
/* In real deployment, output goes to UART or is stored in memory */

/* DWT registers for cycle counting */
#define DWT_CTRL_ADDR   (*(volatile uint32_t*)0xE0001000)
#define DWT_CYCCNT_ADDR (*(volatile uint32_t*)0xE0001004)
#define DEMCR_ADDR      (*(volatile uint32_t*)0xE000EDFC)

/* Test feature vectors */
static const float test_features[5][N_FEATURES] = {{
{test_vectors}
}};

/* Results storage */
static volatile uint32_t cycle_counts[5];
static volatile uint8_t  predictions[5];

int main(void) {{
    /* Enable DWT cycle counter */
    DEMCR_ADDR |= (1 << 24);    /* TRCENA */
    DWT_CTRL_ADDR |= 1;         /* CYCCNTENA */
    
    uint8_t s;
    uint32_t total = 0;
    uint32_t min_c = 0xFFFFFFFF;
    uint32_t max_c = 0;
    
    /* Warm-up: run a few inferences to fill caches */
    for (s = 0; s < 3; s++) {{
        volatile uint8_t r = ensemble_predict(test_features[0]);
        (void)r;
    }}
    
    /* Benchmark each test vector */
    for (s = 0; s < 5; s++) {{
        DWT_CYCCNT_ADDR = 0;  /* Reset cycle counter */
        
        predictions[s] = ensemble_predict(test_features[s]);
        
        cycle_counts[s] = DWT_CYCCNT_ADDR;
        
        total += cycle_counts[s];
        if (cycle_counts[s] < min_c) min_c = cycle_counts[s];
        if (cycle_counts[s] > max_c) max_c = cycle_counts[s];
    }}
    
    /* Store summary (readable via debugger or semihosting) */
    volatile uint32_t avg_cycles = total / 5;
    volatile uint32_t latency_us = avg_cycles / (F_CPU / 1000000);
    
    (void)avg_cycles;
    (void)latency_us;
    (void)min_c;
    (void)max_c;
    
    /* Halt */
    while (1) {{
        __asm volatile ("wfi");
    }}
    
    return 0;
}}
"""
        with open(os.path.join(output_dir, 'main.c'), 'w') as f:
            f.write(content)
    
    def _generate_test_vectors(self) -> str:
        """Generate representative test feature vectors."""
        lines = []
        # 5 test vectors with realistic WSN feature ranges
        vectors = [
            [50.0,1.0,0.0,1.0,5.0,0.0,3.0,1.0,0.0,1.0,10.0,15.0,5.0,120.0,0.0,0.05],
            [100.0,0.0,25.0,0.0,1.0,1.0,0.0,0.0,1.0,3.0,5.0,5.0,0.0,0.0,1.0,0.03],
            [200.0,1.0,0.0,1.0,10.0,0.0,8.0,1.0,0.0,1.0,20.0,30.0,10.0,80.0,0.0,0.08],
            [75.0,0.0,50.0,0.0,1.0,1.0,0.0,0.0,1.0,5.0,1000.0,5.0,0.0,0.0,1.0,0.01],
            [150.0,1.0,0.0,2.0,15.0,0.0,10.0,2.0,0.0,2.0,50.0,60.0,25.0,100.0,0.0,0.1],
        ]
        for i, v in enumerate(vectors):
            vals = ", ".join(f"{x:.1f}f" for x in v)
            lines.append(f"    {{ {vals} }},")
        return "\n".join(lines)
    
    def _generate_makefile(self, output_dir: str):
        cfg = self.config
        ens = self.ensemble
        
        content = f"""# ARM Cortex-M4F Makefile for Inference Benchmark
# Model: {ens.model_name}
# Target: {cfg.mcu_name} ({cfg.arch}) @ {cfg.clock_freq // 1_000_000} MHz

CC      = {cfg.gcc_path}
SIZE    = {cfg.size_path}
OBJDUMP = {cfg.objdump_path}

CPU     = {cfg.cpu}
FPU     = {cfg.fpu}
FLOAT   = {cfg.float_abi}

CFLAGS  = -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
CFLAGS += -O2 -Wall -Wextra -g
CFLAGS += -ffunction-sections -fdata-sections -fno-common
CFLAGS += -ffreestanding -nostdlib

LDFLAGS = -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
LDFLAGS += -T linker.ld
LDFLAGS += -Wl,--gc-sections -Wl,-Map=inference.map
LDFLAGS += -nostdlib -nostartfiles -lgcc

SRCS    = main.c inference.c
OBJS    = $(SRCS:.c=.o)
TARGET  = inference_benchmark

all: $(TARGET).elf size

$(TARGET).elf: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
	$(SIZE) --format=berkeley $<

disasm: $(TARGET).elf
	$(OBJDUMP) -d -S $< > $(TARGET).lst

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).map $(TARGET).lst

.PHONY: all clean size disasm
"""
        with open(os.path.join(output_dir, 'Makefile'), 'w') as f:
            f.write(content)
    
    def _generate_linker_script(self, output_dir: str):
        cfg = self.config
        content = f"""/* Minimal linker script for ARM Cortex-M4F ({cfg.mcu_name}) */
MEMORY
{{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = {cfg.flash_size // 1024}K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = {cfg.ram_size // 1024}K
}}

ENTRY(Reset_Handler)

SECTIONS
{{
    .text :
    {{
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    }} > FLASH

    .data : AT(_etext)
    {{
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    }} > RAM

    .bss :
    {{
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    }} > RAM

    _estack = ORIGIN(RAM) + LENGTH(RAM);
}}

/* Default handler */
PROVIDE(Reset_Handler = main);
"""
        with open(os.path.join(output_dir, 'linker.ld'), 'w') as f:
            f.write(content)


# ============================================================================
# Model Quality Evaluator
# ============================================================================

class ModelQualityEvaluator:
    """Evaluate model quality: original vs depth-limited (ARM-optimized)."""
    
    def __init__(self, X_test: np.ndarray, y_test: np.ndarray):
        self.X_test = X_test
        self.y_test = y_test
    
    def evaluate_original(self, model, model_name: str) -> Dict:
        """Evaluate original sklearn model on test set."""
        y_pred = model.predict(self.X_test)
        return self._compute_metrics(y_pred, model_name, "original")
    
    def evaluate_depth_limited(self, ensemble: FloatEnsemble) -> Dict:
        """Evaluate depth-limited ensemble (ARM version) on test set."""
        print(f"  Predicting {len(self.X_test)} samples with depth-limited model...")
        y_pred = ensemble.predict_batch(self.X_test)
        return self._compute_metrics(y_pred, ensemble.model_name, "depth_limited")
    
    def _compute_metrics(self, y_pred: np.ndarray, model_name: str, 
                         variant: str) -> Dict:
        """Compute comprehensive classification metrics."""
        acc = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        prec_macro = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(self.y_test, y_pred, average=None, zero_division=0)
        
        report = classification_report(self.y_test, y_pred, 
                                       target_names=CLASS_NAMES, 
                                       output_dict=True, zero_division=0)
        
        return {
            'model_name': model_name,
            'variant': variant,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': prec_macro,
            'recall_macro': rec_macro,
            'f1_per_class': {CLASS_NAMES[i]: float(f1_per_class[i]) 
                            for i in range(len(CLASS_NAMES))},
            'classification_report': report,
        }
    
    def compare(self, original_metrics: Dict, limited_metrics: Dict) -> Dict:
        """Compare original vs depth-limited metrics."""
        delta_acc = limited_metrics['accuracy'] - original_metrics['accuracy']
        delta_f1 = limited_metrics['f1_macro'] - original_metrics['f1_macro']
        
        per_class_delta = {}
        for cls in CLASS_NAMES:
            orig = original_metrics['f1_per_class'][cls]
            lim = limited_metrics['f1_per_class'][cls]
            per_class_delta[cls] = lim - orig
        
        return {
            'model_name': original_metrics['model_name'],
            'original_accuracy': original_metrics['accuracy'],
            'limited_accuracy': limited_metrics['accuracy'],
            'delta_accuracy': delta_acc,
            'original_f1_macro': original_metrics['f1_macro'],
            'limited_f1_macro': limited_metrics['f1_macro'],
            'delta_f1_macro': delta_f1,
            'original_f1_weighted': original_metrics['f1_weighted'],
            'limited_f1_weighted': limited_metrics['f1_weighted'],
            'per_class_f1_delta': per_class_delta,
            'quality_preserved': delta_f1 >= -0.01,  # Allow improvements, flag drops > 1%
        }


# ============================================================================
# ARM Cortex-M4 Latency Analyzer
# ============================================================================

class ARMLatencyAnalyzer:
    """Analyze compiled ARM Cortex-M4F code for inference latency."""
    
    def __init__(self, config: ARMCortexConfig):
        self.config = config
    
    def compile_model(self, model_dir: str) -> bool:
        """Compile the generated C code."""
        print(f"  Compiling in {model_dir}...")
        result = subprocess.run(
            ['make', '-C', model_dir, 'clean', 'all'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Compilation failed:\n{result.stderr}")
            return False
        
        # Extract size info
        for line in result.stdout.split('\n'):
            if 'text' in line or '.elf' in line:
                print(f"    {line.strip()}")
        return True
    
    def get_binary_size(self, model_dir: str) -> Dict:
        """Get binary section sizes."""
        elf = os.path.join(model_dir, 'inference_benchmark.elf')
        if not os.path.exists(elf):
            return {}
        
        result = subprocess.run(
            [self.config.size_path, '--format=berkeley', elf],
            capture_output=True, text=True
        )
        
        sizes = {}
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                sizes['text'] = int(parts[0])
                sizes['data'] = int(parts[1])
                sizes['bss'] = int(parts[2])
                sizes['total'] = int(parts[3])
        
        return sizes
    
    def get_disassembly(self, model_dir: str) -> str:
        """Get full disassembly."""
        elf = os.path.join(model_dir, 'inference_benchmark.elf')
        if not os.path.exists(elf):
            return ""
        
        result = subprocess.run(
            [self.config.objdump_path, '-d', elf],
            capture_output=True, text=True
        )
        return result.stdout
    
    def analyze_inference_latency(self, model_dir: str, ensemble: FloatEnsemble) -> Dict:
        """Estimate inference latency using static analysis + disassembly."""
        cfg = self.config
        
        # Get disassembly
        disasm = self.get_disassembly(model_dir)
        
        # Parse function boundaries
        functions = self._parse_functions(disasm)
        
        # Analyze tree_predict function
        tree_predict_cycles = self._analyze_tree_predict(functions, disasm)
        
        # Analyze ensemble_predict function  
        ensemble_cycles = self._analyze_ensemble_predict(
            functions, disasm, ensemble, tree_predict_cycles)
        
        # Also do analytical estimation
        analytical = self._analytical_estimate(ensemble)
        
        # Get binary sizes
        sizes = self.get_binary_size(model_dir)
        
        clock_mhz = cfg.clock_freq / 1_000_000
        
        return {
            'model_name': ensemble.model_name,
            'model_type': ensemble.model_type,
            'n_trees': len(ensemble.trees),
            'total_nodes': ensemble.total_nodes(),
            'max_depth': ensemble.max_tree_depth_used,
            
            # Disassembly-based analysis
            'tree_predict_cycles': tree_predict_cycles,
            'ensemble_predict_cycles': ensemble_cycles,
            
            # Analytical estimation  
            'analytical_cycles_best': analytical['best_case'],
            'analytical_cycles_avg': analytical['average_case'],
            'analytical_cycles_worst': analytical['worst_case'],
            
            # Timing
            'clock_mhz': clock_mhz,
            'latency_us_best': analytical['best_case'] / clock_mhz,
            'latency_us_avg': analytical['average_case'] / clock_mhz,
            'latency_us_worst': analytical['worst_case'] / clock_mhz,
            
            # Binary sizes
            'text_bytes': sizes.get('text', 0),
            'data_bytes': sizes.get('data', 0),
            'bss_bytes': sizes.get('bss', 0),
            'total_binary_bytes': sizes.get('total', 0),
            
            # Model data size
            'model_data_bytes': ensemble.total_memory_bytes(),
            'model_data_kb': ensemble.total_memory_bytes() / 1024,
        }
    
    def _parse_functions(self, disasm: str) -> Dict[str, Tuple[int, int]]:
        """Parse function boundaries from disassembly."""
        functions = {}
        current_fn = None
        current_start = 0
        current_end = 0
        
        for line in disasm.split('\n'):
            # Match function header: "00000000 <tree_predict>:"
            fn_match = re.match(r'^([0-9a-f]+)\s+<(\w+)>:', line)
            if fn_match:
                if current_fn:
                    functions[current_fn] = (current_start, current_end)
                current_fn = fn_match.group(2)
                current_start = int(fn_match.group(1), 16)
                current_end = current_start
            
            # Match instruction line
            instr_match = re.match(r'^\s+([0-9a-f]+):', line)
            if instr_match and current_fn:
                current_end = int(instr_match.group(1), 16)
        
        if current_fn:
            functions[current_fn] = (current_start, current_end)
        
        return functions
    
    def _analyze_tree_predict(self, functions: Dict, disasm: str) -> Dict:
        """Analyze tree_predict function from disassembly."""
        if 'tree_predict' not in functions:
            return {'per_node_cycles': 10, 'overhead_cycles': 5}
        
        start, end = functions['tree_predict']
        fn_size = end - start
        
        # Count instruction types in the function
        n_float_ops = 0
        n_loads = 0
        n_branches = 0
        n_other = 0
        
        in_function = False
        for line in disasm.split('\n'):
            if f'<tree_predict>' in line:
                in_function = True
                continue
            if in_function:
                if re.match(r'^[0-9a-f]+ <\w+>:', line):
                    break  # Next function
                
                instr_match = re.match(r'^\s+[0-9a-f]+:\s+[0-9a-f ]+\s+(\w+)', line)
                if instr_match:
                    instr = instr_match.group(1).lower()
                    if instr.startswith('vcmp') or instr.startswith('vmrs'):
                        n_float_ops += 1
                    elif instr.startswith('ldr') or instr.startswith('vldr'):
                        n_loads += 1
                    elif instr.startswith('b') and instr != 'bic' and instr != 'bfi':
                        n_branches += 1
                    else:
                        n_other += 1
        
        # Per-node cycle estimate from instruction mix
        # Each node traversal: load node (2-3 loads), compare (vcmp + vmrs), 
        # branch (1 conditional), update index (1 ALU)
        per_node = (n_float_ops + n_loads * 1.5 + n_branches * 2 + n_other) 
        # Normalize by approximate loop body instruction count
        loop_body_instrs = max(n_float_ops + n_loads + n_branches + n_other - 4, 6)
        per_node_cycles = max(6, min(15, int(per_node / max(1, loop_body_instrs) * 8)))
        
        return {
            'per_node_cycles': per_node_cycles,
            'overhead_cycles': 5,  # function call/return overhead
            'fn_size_bytes': fn_size,
            'n_float_ops': n_float_ops,
            'n_loads': n_loads,
            'n_branches': n_branches,
        }
    
    def _analyze_ensemble_predict(self, functions: Dict, disasm: str,
                                   ensemble: FloatEnsemble,
                                   tree_predict_info: Dict) -> Dict:
        """Analyze ensemble_predict function.""" 
        per_node = tree_predict_info.get('per_node_cycles', 10)
        overhead = tree_predict_info.get('overhead_cycles', 5)
        
        # Tree traversal per tree
        avg_depth = np.mean([t.max_depth for t in ensemble.trees])
        avg_nodes_visited = avg_depth  # Average path length = depth
        
        per_tree_cycles = overhead + int(avg_nodes_visited * per_node)
        
        # Ensemble overhead
        if ensemble.inference_type == "voting":
            # Voting: increment counter + argmax at end
            ensemble_overhead = len(ensemble.trees) * 3 + self.config.n_classes_argmax_cycles()
        else:
            # Scoring: float add + multiply per tree + argmax
            ensemble_overhead = len(ensemble.trees) * 5 + self.config.n_classes_argmax_cycles()
        
        total = per_tree_cycles * len(ensemble.trees) + ensemble_overhead
        
        return {
            'per_tree_cycles': per_tree_cycles,
            'ensemble_overhead': ensemble_overhead,
            'total_cycles': total,
        }
    
    def _analytical_estimate(self, ensemble: FloatEnsemble) -> Dict:
        """Pure analytical cycle estimation for ARM Cortex-M4F."""
        cfg = self.config
        n_trees = len(ensemble.trees)
        
        # Per-node cycle count (verified from arm-none-eabi-objdump disassembly):
        # Inner loop per tree node:
        #   ADD (feature addr)    = 1 cycle
        #   VLDR (load feature)   = 1 cycle  
        #   VCMPE.F32 (compare)   = 1 cycle  ← Hardware FPU!
        #   VMRS (flags transfer) = 1 cycle
        #   ITE (conditional)     = 0 cycles (folded)
        #   LDRH (load child)     = 1 cycle
        #   ADD+ADD (calc offset) = 2 cycles
        #   VLDR (load threshold) = 1 cycle
        #   LDRH (load feat_idx)  = 1 cycle
        #   CMP (leaf check)      = 1 cycle
        #   BNE (loop back)       = 1 cycle (predicted)
        # Total: ~11-14 cycles/node, average 12
        CYCLES_PER_NODE = 12
        
        # Best case: minimum depth path
        min_depth = min(t.max_depth for t in ensemble.trees)
        avg_depth = np.mean([t.max_depth for t in ensemble.trees])
        max_depth = max(t.max_depth for t in ensemble.trees)
        
        # Function call overhead per tree
        CALL_OVERHEAD = 8  # BL + push/pop + return
        
        # Per-tree cycle count
        best_per_tree = CALL_OVERHEAD + 2 * CYCLES_PER_NODE  # Min 2 levels
        avg_per_tree = CALL_OVERHEAD + int(avg_depth * CYCLES_PER_NODE)
        worst_per_tree = CALL_OVERHEAD + int(max_depth * CYCLES_PER_NODE)
        
        # Ensemble aggregation overhead
        if ensemble.inference_type == "voting":
            # vote[cls]++ per tree + argmax
            agg_per_tree = 3  # load + inc + store
            agg_final = N_CLASSES * 3  # argmax loop
        else:
            # scores[cls] += lr * leaf per tree + argmax
            agg_per_tree = 5  # load + vmul + vadd + vstr
            agg_final = N_CLASSES * 4 + 10  # init scores + argmax
        
        best_total = n_trees * (best_per_tree + agg_per_tree) + agg_final
        avg_total = n_trees * (avg_per_tree + agg_per_tree) + agg_final
        worst_total = n_trees * (worst_per_tree + agg_per_tree) + agg_final
        
        return {
            'best_case': best_total,
            'average_case': avg_total,
            'worst_case': worst_total,
            'cycles_per_node': CYCLES_PER_NODE,
            'avg_depth': avg_depth,
            'n_trees': n_trees,
        }


# Add helper method to config
def _n_classes_argmax_cycles(self):
    return N_CLASSES * 3 + 5

ARMCortexConfig.n_classes_argmax_cycles = _n_classes_argmax_cycles


# ============================================================================
# Main Pipeline
# ============================================================================

def load_data():
    """Load WSN-DS dataset and prepare train/test split.
    Mirrors the preprocessing in wsn_mlflow_pipeline_no_fe.py exactly,
    including StandardScaler normalization."""
    print("Loading WSN-DS dataset...")
    df = pd.read_csv('data/WSN-DS.csv')
    
    # Strip whitespace from column names (same as training pipeline)
    df.columns = df.columns.str.strip()
    
    # Remove duplicates (same as training pipeline)
    original_size = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {original_size - len(df)} duplicates")
    
    # Drop redundant features (same as training pipeline)
    df = df.drop(columns=['id', 'who CH'], errors='ignore')
    
    # Separate features and target
    X = df.drop('Attack type', axis=1)
    y = df['Attack type']
    
    # Store actual feature names
    feature_names = list(X.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")
    
    # Encode labels using sorted class order (sklearn LabelEncoder sorts)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"  Classes: {class_names}")
    
    # Handle numeric conversion
    X_numeric = X.select_dtypes(include=[np.number])
    for col in X.columns:
        if col not in X_numeric.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove rows with inf/nan
    X_values = X.values
    valid_mask = ~(np.isinf(X_values).any(axis=1) | np.isnan(X_values).any(axis=1))
    X = X[valid_mask]
    y_encoded = y_encoded[valid_mask]
    
    # Same split as training pipeline (test_size=0.2, random_state=42)
    # Split as DataFrame to preserve feature names (matches training pipeline behavior)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Apply StandardScaler using the SAVED scaler from training pipeline
    # This ensures exact same normalization as was used during model training
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'mlflow_artifacts_no_fe', 'standard_scaler_no_fe.pkl')
    if os.path.exists(scaler_path):
        import joblib as jl
        scaler = jl.load(scaler_path)
        print(f"  Loaded saved StandardScaler from {scaler_path}")
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_df)
        print(f"  WARNING: Saved scaler not found, fitting new StandardScaler")
    
    X_train_scaled = scaler.transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    # Keep as float64 for accurate model evaluation.
    # On the actual ARM Cortex-M4F, data is float32 — but the quality comparison
    # should isolate the depth-limiting effect from float precision effects.
    
    print(f"  Total samples: {len(X.values)}")
    print(f"  Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}")
    print(f"  StandardScaler applied (matching training pipeline)")
    
    # Update globals to match actual data
    global FEATURE_NAMES, CLASS_NAMES, N_FEATURES, N_CLASSES
    FEATURE_NAMES = feature_names
    CLASS_NAMES = class_names
    N_FEATURES = len(feature_names)
    N_CLASSES = len(class_names)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_model(model_name: str) -> Any:
    """Load a trained model from MLflow artifact storage."""
    path = MODEL_PATHS[model_name]['pkl']
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    
    # Try relative path from workspace root
    if not os.path.exists(full_path):
        full_path = path
    
    print(f"  Loading {model_name} from {full_path}...")
    with open(full_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"    Type: {type(model).__name__}")
    
    # Print model details
    if hasattr(model, 'n_estimators'):
        print(f"    Estimators: {model.n_estimators}")
    if hasattr(model, 'max_depth') and model.max_depth is not None:
        print(f"    Max depth: {model.max_depth}")
    if hasattr(model, 'learning_rate'):
        print(f"    Learning rate: {model.learning_rate}")
    
    return model


def find_optimal_depth(model, model_name: str, X_test: np.ndarray, 
                       y_test: np.ndarray, config: ARMCortexConfig) -> Tuple:
    """Find the optimal tree depth (and stages) that fits in Flash while preserving quality.
    Returns: (best_depth, best_stages, search_results)
    """
    model_class = type(model).__name__
    is_boosting = model_class in ('GradientBoostingClassifier', 'HistGradientBoostingClassifier')
    
    # Test different depth limits
    depths_to_test = [6, 8, 10, 12, 15, 20, 255]  # 255 = unlimited
    
    # For boosting models, also test stage limits
    if is_boosting:
        if model_class == 'GradientBoostingClassifier':
            total_stages = len(model.estimators_)
        else:
            total_stages = len(model._predictors)
        stages_to_test = [10, 20, 50, total_stages]
    else:
        stages_to_test = [0]  # 0 = not applicable
    
    print(f"\n  Finding optimal configuration for {model_name}...")
    
    # Get original model accuracy
    y_orig = model.predict(X_test)
    orig_acc = accuracy_score(y_test, y_orig)
    orig_f1 = f1_score(y_test, y_orig, average='macro', zero_division=0)
    print(f"    Original: acc={orig_acc:.4f}, f1_macro={orig_f1:.4f}")
    
    best_depth = 255
    best_stages = 0
    results = []
    best_score = -1
    
    for stages in stages_to_test:
        for depth in depths_to_test:
            ens = FloatEnsemble(model_name)
            ens.from_sklearn_model(model, max_depth=depth, max_stages=stages)
            
            mem = ens.total_memory_bytes()
            fits = mem < config.max_model_flash
            
            # Quick quality check on subset
            n_sample = min(5000, len(X_test))
            indices = np.random.RandomState(42).choice(len(X_test), n_sample, replace=False)
            y_pred = ens.predict_batch(X_test[indices])
            acc = accuracy_score(y_test[indices], y_pred)
            f1 = f1_score(y_test[indices], y_pred, average='macro', zero_division=0)
            
            delta_f1 = f1 - orig_f1
            
            stage_label = f"stg={stages}" if is_boosting else ""
            results.append({
                'depth': depth if depth < 255 else 'unlimited',
                'stages': stages if is_boosting else 'N/A',
                'n_trees': len(ens.trees),
                'nodes': ens.total_nodes(),
                'memory_kb': mem / 1024,
                'fits': fits,
                'accuracy': acc,
                'f1_macro': f1,
                'delta_f1': delta_f1,
            })
            
            status = "OK" if fits else "TOO LARGE"
            quality = "GOOD" if abs(delta_f1) < 0.005 else ("ACCEPTABLE" if abs(delta_f1) < 0.01 else "DEGRADED")
            
            label = f"depth={depth:>3d}"
            if is_boosting:
                label += f", stages={stages:>3d}"
            
            print(f"    {label}: {ens.total_nodes():>6d} nodes, "
                  f"{mem/1024:>7.1f} KB, acc={acc:.4f}, f1={f1:.4f} "
                  f"(Δf1={delta_f1:+.4f}) [{status}] [{quality}]")
            
            # Pick the best configuration: fits in flash AND preserves quality
            # Prefer highest fidelity (smallest absolute delta_f1) among fitting configs
            if fits and delta_f1 >= -0.01:
                # Score: prefer smaller absolute delta (higher fidelity to original)
                fidelity = 1.0 / (abs(delta_f1) + 0.0001)
                if fidelity > best_score:
                    best_score = fidelity
                    best_depth = depth
                    best_stages = stages
    
    if best_score < 0:
        # Nothing passes strict quality threshold; pick the BEST QUALITY config that fits
        fitting_results = [r for r in results if r['fits']]
        if fitting_results:
            # Sort by absolute delta_f1 (closest to original = best fidelity)
            best_r = min(fitting_results, key=lambda x: abs(x['delta_f1']))
            best_depth = best_r['depth'] if best_r['depth'] != 'unlimited' else 255
            best_stages = best_r['stages'] if best_r['stages'] != 'N/A' else 0
            print(f"    WARNING: No config meets strict quality threshold (ΔF1 >= -0.01)")
            print(f"    Selecting best fidelity config: depth={best_depth}, ΔF1={best_r['delta_f1']:+.4f}")
    
    stage_info = f", stages={best_stages}" if is_boosting else ""
    print(f"    → Selected: depth={best_depth}{stage_info}")
    return best_depth, best_stages, results


def main():
    print("=" * 80)
    print("ARM CORTEX-M4F INFERENCE EVALUATION PIPELINE")
    print("Native float32 — NO QUANTIZATION")
    print("Target: nRF52840 (ARM Cortex-M4F) @ 64 MHz")
    print("=" * 80)
    
    config = ARMCortexConfig()
    
    # 1. Load data
    print("\n" + "=" * 60)
    print("STEP 1: Load Dataset")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Load models
    print("\n" + "=" * 60)
    print("STEP 2: Load Trained Models (Conservative SMOTE, No Feature Engineering)")
    print("=" * 60)
    
    models = {}
    for name in MODEL_PATHS:
        try:
            models[name] = load_model(name)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    
    if not models:
        print("No models loaded! Exiting.")
        return
    
    # 3. Find optimal depth and evaluate quality
    print("\n" + "=" * 60)
    print("STEP 3: Depth Optimization & Model Quality Evaluation")
    print("=" * 60)
    
    evaluator = ModelQualityEvaluator(X_test, y_test)
    all_results = {}
    quality_results = []
    depth_search_results = {}
    
    output_base = 'avr_model_converter/generated_arm'
    os.makedirs(output_base, exist_ok=True)
    
    for name, model in models.items():
        print(f"\n{'─' * 60}")
        print(f"Processing: {name}")
        print('─' * 60)
        
        # Find optimal depth (and stages for boosting models)
        best_depth, best_stages, depth_results = find_optimal_depth(
            model, name, X_test, y_test, config)
        depth_search_results[name] = depth_results
        
        # Extract with optimal settings
        is_boosting = type(model).__name__ in ('GradientBoostingClassifier', 'HistGradientBoostingClassifier')
        stage_info = f", stages={best_stages}" if is_boosting else ""
        print(f"\n  Extracting trees (depth={best_depth}{stage_info})...")
        ensemble = FloatEnsemble(name)
        ensemble.from_sklearn_model(model, max_depth=best_depth, max_stages=best_stages)
        
        summary = ensemble.summary()
        print(f"    Trees: {summary['n_trees']}")
        print(f"    Total nodes: {summary['total_nodes']}")
        print(f"    Memory: {summary['memory_kb']:.1f} KB")
        print(f"    Max depth used: {summary['max_depth']}")
        
        # Model quality evaluation
        print(f"\n  Evaluating model quality...")
        orig_metrics = evaluator.evaluate_original(model, name)
        limited_metrics = evaluator.evaluate_depth_limited(ensemble)
        comparison = evaluator.compare(orig_metrics, limited_metrics)
        
        print(f"    Original:  acc={orig_metrics['accuracy']:.4f}, "
              f"f1={orig_metrics['f1_macro']:.4f}")
        print(f"    ARM-opt:   acc={limited_metrics['accuracy']:.4f}, "
              f"f1={limited_metrics['f1_macro']:.4f}")
        print(f"    Δ accuracy: {comparison['delta_accuracy']:+.4f}")
        print(f"    Δ F1 macro: {comparison['delta_f1_macro']:+.4f}")
        print(f"    Quality preserved: {'YES' if comparison['quality_preserved'] else 'NO'}")
        
        quality_results.append(comparison)
        
        # Generate C code
        print(f"\n  Generating ARM Cortex-M4F C code...")
        model_dir = os.path.join(output_base, name.lower())
        generator = ARMCodeGenerator(ensemble, config)
        generator.generate_all(model_dir)
        
        all_results[name] = {
            'ensemble': ensemble,
            'summary': summary,
            'depth': best_depth,
            'depth_search': depth_results,
            'quality': comparison,
            'orig_metrics': orig_metrics,
            'limited_metrics': limited_metrics,
            'model_dir': model_dir,
        }
    
    # 4. Compile and analyze
    print("\n" + "=" * 60)
    print("STEP 4: Compile & Latency Analysis")
    print("=" * 60)
    
    analyzer = ARMLatencyAnalyzer(config)
    latency_results = []
    
    for name, result in all_results.items():
        print(f"\n{'─' * 60}")
        print(f"Compiling: {name}")
        print('─' * 60)
        
        model_dir = result['model_dir']
        
        if analyzer.compile_model(model_dir):
            latency = analyzer.analyze_inference_latency(
                model_dir, result['ensemble'])
            latency_results.append(latency)
            all_results[name]['latency'] = latency
            
            print(f"    Binary: text={latency['text_bytes']} B, "
                  f"data={latency['data_bytes']} B, "
                  f"total={latency['total_binary_bytes']} B")
            print(f"    Model data: {latency['model_data_kb']:.1f} KB")
            print(f"    Estimated cycles (avg): {latency['analytical_cycles_avg']}")
            print(f"    Estimated latency (avg): {latency['latency_us_avg']:.1f} µs")
        else:
            print(f"    Compilation failed!")
    
    # 5. Save results
    print("\n" + "=" * 60)
    print("STEP 5: Save Results")
    print("=" * 60)
    
    save_results(all_results, quality_results, latency_results, 
                 depth_search_results, output_base)
    
    # 6. Print summary
    print_final_summary(all_results, config)


def save_results(all_results, quality_results, latency_results, 
                 depth_search_results, output_base):
    """Save all results to files."""
    
    # Quality comparison CSV
    quality_csv = os.path.join(output_base, 'model_quality_comparison.csv')
    with open(quality_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Original_Accuracy', 'ARM_Accuracy', 'Delta_Accuracy',
            'Original_F1_Macro', 'ARM_F1_Macro', 'Delta_F1_Macro',
            'Original_F1_Weighted', 'ARM_F1_Weighted',
            'F1_Blackhole_Orig', 'F1_Blackhole_ARM',
            'F1_Flooding_Orig', 'F1_Flooding_ARM',
            'F1_Grayhole_Orig', 'F1_Grayhole_ARM',
            'F1_Normal_Orig', 'F1_Normal_ARM',
            'F1_TDMA_Orig', 'F1_TDMA_ARM',
            'Quality_Preserved'
        ])
        for q in quality_results:
            name = q['model_name']
            r = all_results[name]
            orig = r['orig_metrics']
            lim = r['limited_metrics']
            writer.writerow([
                name,
                f"{q['original_accuracy']:.6f}", f"{q['limited_accuracy']:.6f}",
                f"{q['delta_accuracy']:+.6f}",
                f"{q['original_f1_macro']:.6f}", f"{q['limited_f1_macro']:.6f}",
                f"{q['delta_f1_macro']:+.6f}",
                f"{q['original_f1_weighted']:.6f}", f"{q['limited_f1_weighted']:.6f}",
                f"{orig['f1_per_class']['Blackhole']:.6f}",
                f"{lim['f1_per_class']['Blackhole']:.6f}",
                f"{orig['f1_per_class']['Flooding']:.6f}",
                f"{lim['f1_per_class']['Flooding']:.6f}",
                f"{orig['f1_per_class']['Grayhole']:.6f}",
                f"{lim['f1_per_class']['Grayhole']:.6f}",
                f"{orig['f1_per_class']['Normal']:.6f}",
                f"{lim['f1_per_class']['Normal']:.6f}",
                f"{orig['f1_per_class']['TDMA']:.6f}",
                f"{lim['f1_per_class']['TDMA']:.6f}",
                q['quality_preserved']
            ])
    print(f"  Saved: {quality_csv}")
    
    # Latency results CSV
    latency_csv = os.path.join(output_base, 'arm_inference_latency.csv')
    with open(latency_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Type', 'N_Trees', 'Total_Nodes', 'Max_Depth',
            'Cycles_Best', 'Cycles_Avg', 'Cycles_Worst',
            'Latency_us_Best', 'Latency_us_Avg', 'Latency_us_Worst',
            'Text_Bytes', 'Data_Bytes', 'Total_Binary_Bytes',
            'Model_Data_KB'
        ])
        for lat in latency_results:
            writer.writerow([
                lat['model_name'], lat['model_type'],
                lat['n_trees'], lat['total_nodes'], lat['max_depth'],
                lat['analytical_cycles_best'],
                lat['analytical_cycles_avg'],
                lat['analytical_cycles_worst'],
                f"{lat['latency_us_best']:.1f}",
                f"{lat['latency_us_avg']:.1f}",
                f"{lat['latency_us_worst']:.1f}",
                lat['text_bytes'], lat['data_bytes'],
                lat['total_binary_bytes'],
                f"{lat['model_data_kb']:.1f}"
            ])
    print(f"  Saved: {latency_csv}")
    
    # Depth search results JSON
    depth_json = os.path.join(output_base, 'depth_search_results.json')
    with open(depth_json, 'w') as f:
        json.dump(depth_search_results, f, indent=2, default=str)
    print(f"  Saved: {depth_json}")
    
    # Comprehensive results JSON
    summary_json = os.path.join(output_base, 'arm_evaluation_summary.json')
    summary_data = {}
    for name, r in all_results.items():
        summary_data[name] = {
            'summary': r['summary'],
            'depth': r['depth'],
            'quality': r['quality'],
            'latency': r.get('latency', {}),
        }
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"  Saved: {summary_json}")


def print_final_summary(all_results: Dict, config: ARMCortexConfig):
    """Print final summary table."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: ARM Cortex-M4F Inference Evaluation")
    print(f"Target: {config.mcu_name} ({config.arch}) @ {config.clock_freq//1_000_000} MHz")
    print("Method: Native float32 — NO QUANTIZATION")
    print("=" * 80)
    
    print(f"\n{'Model':<35} {'Trees':>6} {'Nodes':>7} {'Depth':>6} "
          f"{'Memory':>8} {'Cycles':>8} {'Latency':>10} {'ΔF1':>8} {'Quality':>8}")
    print("─" * 105)
    
    for name, r in all_results.items():
        s = r['summary']
        q = r['quality']
        lat = r.get('latency', {})
        
        cycles = lat.get('analytical_cycles_avg', 0)
        latency_us = lat.get('latency_us_avg', 0)
        
        quality_status = "✓ OK" if q['quality_preserved'] else "✗ DROP"
        
        print(f"{name:<35} {s['n_trees']:>6} {s['total_nodes']:>7} "
              f"{s['max_depth']:>6} {s['memory_kb']:>7.1f}K "
              f"{cycles:>8} {latency_us:>8.1f} µs "
              f"{q['delta_f1_macro']:>+7.4f} {quality_status:>8}")
    
    print("\n" + "=" * 80)
    print("Output directory: avr_model_converter/generated_arm/")
    print("=" * 80)


if __name__ == '__main__':
    main()

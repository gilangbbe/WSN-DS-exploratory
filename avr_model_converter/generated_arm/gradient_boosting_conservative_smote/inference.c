/*
 * Inference Implementation for ARM Cortex-M4F
 * Model: Gradient_Boosting_Conservative_SMOTE
 * Type: GradientBoostingClassifier (scoring)
 *
 * Uses hardware FPU for float32 comparisons (VCMP.F32)
 * No quantization — thresholds are native float32
 */

#include "inference.h"
#include "tree_data.h"

/* Single tree traversal.
 * The FPU compares features[node->feature_idx] <= node->threshold
 * using a single VCMP.F32 + VMRS instruction pair. */
float tree_predict(const TreeNode* tree, const float* features) {
    uint16_t idx = 0;
    
    while (1) {
        const TreeNode* node = &tree[idx];
        
        /* Check if leaf node */
        if (node->feature_idx == 0xFFFF) {
            return node->threshold;  /* Leaf value */
        }
        
        /* Float comparison using FPU: VCMP.F32 + VMRS */
        if (features[node->feature_idx] <= node->threshold) {
            idx = node->left_child;
        } else {
            idx = node->right_child;
        }
    }
}

/* Sum-of-scores ensemble (Gradient Boosting / HistGradient Boosting) */
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
}

/* Benchmark using ARM DWT Cycle Counter */
uint32_t benchmark_inference(const float* features) {
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
}

/*
 * Inference Implementation for ARM Cortex-M4F
 * Model: Random_Forest_Conservative_SMOTE
 * Type: RandomForestClassifier (voting)
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

/* Majority voting ensemble (Random Forest / Extra Trees) */
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

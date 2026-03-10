/*
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

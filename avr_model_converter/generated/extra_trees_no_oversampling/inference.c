/*
 * Inference Implementation for AVR
 * Optimized for ATmega328P
 */

#include "inference.h"
#include "model_config.h"
#include "quantization.h"
#include "tree_data.h"
#include <avr/pgmspace.h>

// Quantize a single feature
uint8_t quantize_feature(float value, uint8_t feature_idx) {
    int16_t scale = pgm_read_word(&FEATURE_SCALE[feature_idx]);
    uint8_t zero = pgm_read_byte(&FEATURE_ZERO[feature_idx]);
    
    // Fixed-point multiplication: (value * scale) >> 8 + zero
    int32_t scaled = (int32_t)(value * scale) >> 8;
    int16_t result = scaled + zero;
    
    // Clamp to uint8
    if (result < 0) return 0;
    if (result > 255) return 255;
    return (uint8_t)result;
}

// Quantize all features
void quantize_features(const float* features, uint8_t* quantized) {
    for (uint8_t i = 0; i < N_FEATURES; i++) {
        quantized[i] = quantize_feature(features[i], i);
    }
}

// Traverse a single tree and return predicted class
// Tree node format: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features) {
    uint16_t node_idx = 0;
    
    while (1) {
        uint16_t offset = node_idx * 6;
        
        // Read node from PROGMEM
        uint8_t feature_idx = pgm_read_byte(&tree_data[offset]);
        uint8_t threshold = pgm_read_byte(&tree_data[offset + 1]);
        
        // Check if leaf node
        if (feature_idx == 255) {
            // threshold field contains class_id for leaves
            return threshold;
        }
        
        // Get feature value and compare
        uint8_t feature_val = features[feature_idx];
        
        if (feature_val <= threshold) {
            // Go left
            uint8_t left_lo = pgm_read_byte(&tree_data[offset + 2]);
            uint8_t left_hi = pgm_read_byte(&tree_data[offset + 3]);
            node_idx = left_lo | (left_hi << 8);
        } else {
            // Go right
            uint8_t right_lo = pgm_read_byte(&tree_data[offset + 4]);
            uint8_t right_hi = pgm_read_byte(&tree_data[offset + 5]);
            node_idx = right_lo | (right_hi << 8);
        }
    }
}

// Ensemble prediction using majority voting
uint8_t ensemble_predict(const uint8_t* features) {
    uint8_t votes[N_CLASSES] = {0};
    
    // Get predictions from all trees
    for (uint8_t t = 0; t < 5; t++) {
        const uint8_t* tree_ptr = (const uint8_t*)pgm_read_ptr(&trees[t]);
        uint8_t pred = tree_predict(tree_ptr, features);
        if (pred < N_CLASSES) {
            votes[pred]++;
        }
    }
    
    // Find majority vote
    uint8_t max_votes = 0;
    uint8_t predicted_class = 0;
    
    for (uint8_t c = 0; c < N_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            predicted_class = c;
        }
    }
    
    return predicted_class;
}

// Benchmark inference timing using Timer1
uint32_t benchmark_inference(const uint8_t* features) {
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
}

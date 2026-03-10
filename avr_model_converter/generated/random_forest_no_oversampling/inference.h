/*
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

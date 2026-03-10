/*
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

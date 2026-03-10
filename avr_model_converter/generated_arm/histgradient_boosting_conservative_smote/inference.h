/*
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

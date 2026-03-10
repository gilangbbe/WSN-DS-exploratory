/*
 * Feature Quantization Parameters for MSP430
 */

#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>

// Scale factors (Q8.8 fixed-point)
static const int16_t FEATURE_SCALE[N_FEATURES] = {
    0, 18, -256, 0, 304, 672, 557, -256, 526, 659, -256, 659, 270, 43, 270, 323, 4352, 1447
};

// Zero points
static const uint8_t FEATURE_ZERO[N_FEATURES] = {
    249, 253, 0, 249, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

#endif // QUANTIZATION_H

/*
 * Model Configuration for ARM Cortex-M4F
 * Model: Extra_Trees_SMOTE_ENN
 * Type: ExtraTreesClassifier
 * Target: nRF52840 (ARM Cortex-M4F) @ 64 MHz
 * FPU: fpv4-sp-d16 (hardware float32)
 * 
 * NO QUANTIZATION — native float32 thresholds and comparisons
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>

#define MODEL_NAME       "Extra_Trees_SMOTE_ENN"
#define MODEL_TYPE       "ExtraTreesClassifier"
#define TARGET_MCU       "nRF52840"
#define F_CPU            64000000UL

#define N_FEATURES       16
#define N_CLASSES        5
#define N_TREES          100
#define MAX_DEPTH        6

#define INFERENCE_TYPE_VOTING  1

/* Tree node structure: 12 bytes, naturally aligned for ARM */
typedef struct {
    float threshold;        /* 4 bytes: split threshold or leaf value */
    uint16_t feature_idx;   /* 2 bytes: feature index (0xFFFF = leaf) */
    uint16_t left_child;    /* 2 bytes: left child node index */
    uint16_t right_child;   /* 2 bytes: right child node index */
    uint16_t _pad;          /* 2 bytes: alignment padding */
} TreeNode;                /* Total: 12 bytes */

#endif /* MODEL_CONFIG_H */

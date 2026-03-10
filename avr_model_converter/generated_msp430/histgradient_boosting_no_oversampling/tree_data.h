/*
 * Quantized Tree Data for MSP430
 * Node format: [feature_idx, threshold, left_lo, left_hi, right_lo, right_hi]
 */

#ifndef TREE_DATA_H
#define TREE_DATA_H

#include <stdint.h>
#include "model_config.h"

// Tree 0: 31 nodes, depth 4
static const uint8_t tree_0[] = {
    255, 0, 0, 0, 0, 0,  // Node 0
};

// Tree 1: 31 nodes, depth 4
static const uint8_t tree_1[] = {
    255, 0, 0, 0, 0, 0,  // Node 1
};

// Tree 2: 31 nodes, depth 4
static const uint8_t tree_2[] = {
    255, 0, 0, 0, 0, 0,  // Node 2
};

// Tree 3: 31 nodes, depth 4
static const uint8_t tree_3[] = {
    255, 0, 0, 0, 0, 0,  // Node 3
};

// Tree 4: 31 nodes, depth 4
static const uint8_t tree_4[] = {
    255, 0, 0, 0, 0, 0,  // Node 4
};

// Tree 5: 31 nodes, depth 4
static const uint8_t tree_5[] = {
    255, 0, 0, 0, 0, 0,  // Node 5
};

// Tree 6: 31 nodes, depth 4
static const uint8_t tree_6[] = {
    255, 0, 0, 0, 0, 0,  // Node 6
};

// Tree 7: 31 nodes, depth 4
static const uint8_t tree_7[] = {
    255, 0, 0, 0, 0, 0,  // Node 7
};

// Tree 8: 31 nodes, depth 4
static const uint8_t tree_8[] = {
    255, 0, 0, 0, 0, 0,  // Node 8
};

// Tree 9: 31 nodes, depth 4
static const uint8_t tree_9[] = {
    255, 0, 0, 0, 0, 0,  // Node 9
};

// Tree 10: 31 nodes, depth 4
static const uint8_t tree_10[] = {
    255, 0, 0, 0, 0, 0,  // Node 10
};

// Tree 11: 31 nodes, depth 4
static const uint8_t tree_11[] = {
    255, 0, 0, 0, 0, 0,  // Node 11
};

// Tree 12: 31 nodes, depth 4
static const uint8_t tree_12[] = {
    255, 0, 0, 0, 0, 0,  // Node 12
};

// Tree 13: 31 nodes, depth 4
static const uint8_t tree_13[] = {
    255, 0, 0, 0, 0, 0,  // Node 13
};

// Tree 14: 31 nodes, depth 4
static const uint8_t tree_14[] = {
    255, 0, 0, 0, 0, 0,  // Node 14
};

// Tree 15: 31 nodes, depth 4
static const uint8_t tree_15[] = {
    255, 0, 0, 0, 0, 0,  // Node 15
};

// Tree 16: 31 nodes, depth 4
static const uint8_t tree_16[] = {
    255, 0, 0, 0, 0, 0,  // Node 16
};

// Tree 17: 31 nodes, depth 4
static const uint8_t tree_17[] = {
    255, 0, 0, 0, 0, 0,  // Node 17
};

// Tree 18: 31 nodes, depth 4
static const uint8_t tree_18[] = {
    255, 0, 0, 0, 0, 0,  // Node 18
};

// Tree 19: 31 nodes, depth 4
static const uint8_t tree_19[] = {
    255, 0, 0, 0, 0, 0,  // Node 19
};

// Tree pointers
static const uint8_t* const trees[N_TREES] = {
    tree_0,
    tree_1,
    tree_2,
    tree_3,
    tree_4,
    tree_5,
    tree_6,
    tree_7,
    tree_8,
    tree_9,
    tree_10,
    tree_11,
    tree_12,
    tree_13,
    tree_14,
    tree_15,
    tree_16,
    tree_17,
    tree_18,
    tree_19,
};

// Tree node counts
static const uint16_t tree_sizes[N_TREES] = {
    31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31
};

#endif // TREE_DATA_H

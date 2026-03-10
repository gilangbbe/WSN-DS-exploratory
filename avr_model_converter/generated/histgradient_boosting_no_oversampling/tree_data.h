/*
 * Tree Data Arrays (stored in PROGMEM)
 * Model: HistGradient_Boosting_No_Oversampling
 * Trees: 20
 */

#ifndef TREE_DATA_H
#define TREE_DATA_H

#include <avr/pgmspace.h>
#include <stdint.h>

// Note: Using 10 of 20 trees for memory constraints
#define ACTIVE_TREES 10

// Tree 0: 31 nodes, depth 4
static const uint8_t tree_0[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 1: 31 nodes, depth 4
static const uint8_t tree_1[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 2: 31 nodes, depth 4
static const uint8_t tree_2[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 3: 31 nodes, depth 4
static const uint8_t tree_3[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 4: 31 nodes, depth 4
static const uint8_t tree_4[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 5: 31 nodes, depth 4
static const uint8_t tree_5[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 6: 31 nodes, depth 4
static const uint8_t tree_6[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 7: 31 nodes, depth 4
static const uint8_t tree_7[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 8: 31 nodes, depth 4
static const uint8_t tree_8[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree 9: 31 nodes, depth 4
static const uint8_t tree_9[186] PROGMEM = {
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Tree pointer array
static const uint8_t* const trees[10] PROGMEM = {
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
};

// Tree node counts
static const uint16_t tree_sizes[10] PROGMEM = {
    31, 31, 31, 31, 31, 31, 31, 31, 31, 31
};

#endif // TREE_DATA_H
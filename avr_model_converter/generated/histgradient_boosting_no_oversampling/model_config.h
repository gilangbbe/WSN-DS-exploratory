/*
 * Model Configuration Header
 * Model: HistGradient_Boosting_No_Oversampling
 * Type: HistGradientBoostingClassifier
 * Generated: 2026-02-01 16:40:33
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>
#include <avr/pgmspace.h>

// Model parameters
#define MODEL_NAME "HistGradient_Boosting_No_Oversampling"
#define MODEL_TYPE "HistGradientBoostingClassifier"
#define N_FEATURES 16
#define N_CLASSES 5
#define N_TREES 20
#define N_ESTIMATORS 205

// Memory estimates
#define TOTAL_NODES 620
#define TREE_MEMORY_BYTES 3720
#define TOTAL_MEMORY_BYTES 3852

// Class labels
#define CLASS_NORMAL 0
#define CLASS_BLACKHOLE 1
#define CLASS_GRAYHOLE 2
#define CLASS_FLOODING 3
#define CLASS_TDMA 4

// Inference timing
#define TIMER_PRESCALER 1
#define F_CPU 16000000UL

#endif // MODEL_CONFIG_H

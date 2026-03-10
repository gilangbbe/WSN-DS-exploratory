/*
 * Model Configuration for MSP430
 * Model: Extra_Trees_No_Oversampling
 * Target: MSP430F5529 @ 25 MHz
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "Extra_Trees_No_Oversampling"
#define MODEL_TYPE "ExtraTreesClassifier"

#define N_FEATURES 18
#define N_CLASSES 5
#define N_TREES 5
#define MAX_DEPTH 6

#define F_CPU 25000000UL

#endif // MODEL_CONFIG_H

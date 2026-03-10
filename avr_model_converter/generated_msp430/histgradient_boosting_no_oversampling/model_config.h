/*
 * Model Configuration for MSP430
 * Model: HistGradient_Boosting_No_Oversampling
 * Target: MSP430F5529 @ 25 MHz
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "HistGradient_Boosting_No_Oversampling"
#define MODEL_TYPE "HistGradientBoostingClassifier"

#define N_FEATURES 18
#define N_CLASSES 5
#define N_TREES 20
#define MAX_DEPTH 6

#define F_CPU 25000000UL

#endif // MODEL_CONFIG_H

/*
 * ARM Cortex-M4F Inference Latency Benchmark
 * Model: Gradient_Boosting_SMOTE_ENN
 * Target: nRF52840 @ 64 MHz
 * FPU: fpv4-sp-d16 (hardware float32, no quantization)
 */

#include <stdint.h>
#include "model_config.h"
#include "inference.h"

/* Minimal semihosting / ITM output for benchmarking */
/* In real deployment, output goes to UART or is stored in memory */

/* DWT registers for cycle counting */
#define DWT_CTRL_ADDR   (*(volatile uint32_t*)0xE0001000)
#define DWT_CYCCNT_ADDR (*(volatile uint32_t*)0xE0001004)
#define DEMCR_ADDR      (*(volatile uint32_t*)0xE000EDFC)

/* Test feature vectors */
static const float test_features[5][N_FEATURES] = {
    { 50.0f, 1.0f, 0.0f, 1.0f, 5.0f, 0.0f, 3.0f, 1.0f, 0.0f, 1.0f, 10.0f, 15.0f, 5.0f, 120.0f, 0.0f, 0.1f },
    { 100.0f, 0.0f, 25.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 3.0f, 5.0f, 5.0f, 0.0f, 0.0f, 1.0f, 0.0f },
    { 200.0f, 1.0f, 0.0f, 1.0f, 10.0f, 0.0f, 8.0f, 1.0f, 0.0f, 1.0f, 20.0f, 30.0f, 10.0f, 80.0f, 0.0f, 0.1f },
    { 75.0f, 0.0f, 50.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 1000.0f, 5.0f, 0.0f, 0.0f, 1.0f, 0.0f },
    { 150.0f, 1.0f, 0.0f, 2.0f, 15.0f, 0.0f, 10.0f, 2.0f, 0.0f, 2.0f, 50.0f, 60.0f, 25.0f, 100.0f, 0.0f, 0.1f },
};

/* Results storage */
static volatile uint32_t cycle_counts[5];
static volatile uint8_t  predictions[5];

int main(void) {
    /* Enable DWT cycle counter */
    DEMCR_ADDR |= (1 << 24);    /* TRCENA */
    DWT_CTRL_ADDR |= 1;         /* CYCCNTENA */
    
    uint8_t s;
    uint32_t total = 0;
    uint32_t min_c = 0xFFFFFFFF;
    uint32_t max_c = 0;
    
    /* Warm-up: run a few inferences to fill caches */
    for (s = 0; s < 3; s++) {
        volatile uint8_t r = ensemble_predict(test_features[0]);
        (void)r;
    }
    
    /* Benchmark each test vector */
    for (s = 0; s < 5; s++) {
        DWT_CYCCNT_ADDR = 0;  /* Reset cycle counter */
        
        predictions[s] = ensemble_predict(test_features[s]);
        
        cycle_counts[s] = DWT_CYCCNT_ADDR;
        
        total += cycle_counts[s];
        if (cycle_counts[s] < min_c) min_c = cycle_counts[s];
        if (cycle_counts[s] > max_c) max_c = cycle_counts[s];
    }
    
    /* Store summary (readable via debugger or semihosting) */
    volatile uint32_t avg_cycles = total / 5;
    volatile uint32_t latency_us = avg_cycles / (F_CPU / 1000000);
    
    (void)avg_cycles;
    (void)latency_us;
    (void)min_c;
    (void)max_c;
    
    /* Halt */
    while (1) {
        __asm volatile ("wfi");
    }
    
    return 0;
}

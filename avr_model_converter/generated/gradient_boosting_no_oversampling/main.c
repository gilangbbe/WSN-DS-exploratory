/*
 * Main Benchmark Program for AVR
 * Measures inference latency using Timer1
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdio.h>
#include "model_config.h"
#include "inference.h"

// UART configuration for output
#define BAUD 9600
#define UBRR_VALUE ((F_CPU / 16 / BAUD) - 1)

void uart_init(void) {
    UBRR0H = (uint8_t)(UBRR_VALUE >> 8);
    UBRR0L = (uint8_t)UBRR_VALUE;
    UCSR0B = (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void uart_putchar(char c) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = c;
}

void uart_puts(const char* s) {
    while (*s) {
        uart_putchar(*s++);
    }
}

void uart_putnum(uint32_t n) {
    char buf[12];
    sprintf(buf, "%lu", n);
    uart_puts(buf);
}

// Timer1 initialization for cycle counting
void timer_init(void) {
    // Timer1: Normal mode, no prescaler
    TCCR1A = 0;
    TCCR1B = 0;  // Stopped initially
}

// Test feature vectors (example values)
// These should be replaced with actual test data
static const float test_features[5][16] = {
    {1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
    {2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2, 2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2},
    {10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0},
};

int main(void) {
    // Initialize
    uart_init();
    timer_init();
    
    // Print header
    uart_puts("\r\n=== AVR Inference Latency Benchmark ===\r\n");
    uart_puts("Model: ");
    uart_puts(MODEL_NAME);
    uart_puts("\r\n");
    uart_puts("F_CPU: 16 MHz\r\n");
    uart_puts("Trees: ");
    uart_putnum(N_TREES);
    uart_puts("\r\n\r\n");
    
    // Quantized feature buffer
    uint8_t q_features[N_FEATURES];
    
    // Warm-up run
    uart_puts("Warm-up...\r\n");
    for (uint8_t i = 0; i < 10; i++) {
        quantize_features(test_features[0], q_features);
        volatile uint8_t r = ensemble_predict(q_features);
        (void)r;
    }
    
    // Benchmark runs
    uart_puts("\r\nBenchmark Results (cycles):\r\n");
    uart_puts("Sample,Cycles,Prediction\r\n");
    
    uint32_t total_cycles = 0;
    uint32_t min_cycles = 0xFFFFFFFF;
    uint32_t max_cycles = 0;
    
    for (uint8_t s = 0; s < 5; s++) {
        // Quantize features
        quantize_features(test_features[s], q_features);
        
        // Run multiple iterations per sample
        for (uint8_t iter = 0; iter < 10; iter++) {
            uint32_t cycles = benchmark_inference(q_features);
            
            uart_putnum(s);
            uart_putchar(',');
            uart_putnum(cycles);
            uart_putchar(',');
            
            uint8_t pred = ensemble_predict(q_features);
            uart_putnum(pred);
            uart_puts("\r\n");
            
            total_cycles += cycles;
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
        }
    }
    
    // Print summary
    uart_puts("\r\n=== Summary ===\r\n");
    uart_puts("Total iterations: 50\r\n");
    uart_puts("Min cycles: ");
    uart_putnum(min_cycles);
    uart_puts("\r\n");
    uart_puts("Max cycles: ");
    uart_putnum(max_cycles);
    uart_puts("\r\n");
    uart_puts("Avg cycles: ");
    uart_putnum(total_cycles / 50);
    uart_puts("\r\n");
    
    // Convert to time
    uart_puts("\r\nAt 16 MHz:\r\n");
    uart_puts("Min time (us): ");
    uart_putnum(min_cycles / 16);
    uart_puts("\r\n");
    uart_puts("Avg time (us): ");
    uart_putnum(total_cycles / 50 / 16);
    uart_puts("\r\n");
    uart_puts("Max time (us): ");
    uart_putnum(max_cycles / 16);
    uart_puts("\r\n");
    
    uart_puts("\r\n=== Benchmark Complete ===\r\n");
    
    // Infinite loop
    while (1) {
        _delay_ms(1000);
    }
    
    return 0;
}

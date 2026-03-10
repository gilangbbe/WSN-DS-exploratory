/*
 * MSP430 Inference Latency Benchmark
 * Target: MSP430F5529 @ 25 MHz
 */

#include <msp430.h>
#include <stdio.h>
#include <stdint.h>
#include "model_config.h"
#include "inference.h"

// UART TX for results
void uart_init(void) {
    // Configure USCI_A1 for UART @ 9600 baud
    UCA1CTL1 |= UCSWRST;
    UCA1CTL1 |= UCSSEL_2;  // SMCLK
    
    // 25MHz / 9600 = 2604.17
    UCA1BR0 = 0x2C;
    UCA1BR1 = 0x0A;
    UCA1MCTL = UCBRS_5;
    
    UCA1CTL1 &= ~UCSWRST;
}

void uart_putc(char c) {
    while (!(UCA1IFG & UCTXIFG));
    UCA1TXBUF = c;
}

void uart_puts(const char* s) {
    while (*s) uart_putc(*s++);
}

void uart_putnum(uint32_t n) {
    char buf[12];
    sprintf(buf, "%lu", n);
    uart_puts(buf);
}

// Test feature vectors
static const float test_features[5][N_FEATURES] = {
    {1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
    {2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2, 2.5, 1.0, 3.0, 0.5, 2.0, 1.5, 0.8, 1.2},
    {10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0},
};

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;  // Stop watchdog
    
    // Configure clock to 25 MHz
    // DCO at 25 MHz
    UCSCTL3 = SELREF_2;  // Set DCO FLL reference = REFO
    UCSCTL4 |= SELA_2;   // Set ACLK = REFO
    
    __bis_SR_register(SCG0);  // Disable FLL
    UCSCTL0 = 0x0000;
    UCSCTL1 = DCORSEL_7;  // Select DCO range
    UCSCTL2 = FLLD_0 + 762;  // DCOCLK = 25 MHz
    __bic_SR_register(SCG0);  // Enable FLL
    
    __delay_cycles(782000);  // Wait for DCO to settle
    
    // Init UART
    uart_init();
    
    uart_puts("\r\n=== MSP430 Inference Benchmark ===\r\n");
    uart_puts("Model: ");
    uart_puts(MODEL_NAME);
    uart_puts("\r\nF_CPU: 25 MHz\r\n");
    uart_puts("Trees: ");
    uart_putnum(N_TREES);
    uart_puts("\r\n\r\n");
    
    uint8_t q_features[N_FEATURES];
    
    // Warm-up
    uart_puts("Warm-up...\r\n");
    uint8_t i;
    for (i = 0; i < 10; i++) {
        quantize_features(test_features[0], q_features);
        volatile uint8_t r = ensemble_predict(q_features);
        (void)r;
    }
    
    // Benchmark
    uart_puts("\r\nResults (cycles):\r\n");
    uart_puts("Sample,Cycles,Prediction\r\n");
    
    uint32_t total = 0;
    uint32_t min = 0xFFFFFFFF;
    uint32_t max = 0;
    
    uint8_t s, iter;
    for (s = 0; s < 5; s++) {
        quantize_features(test_features[s], q_features);
        
        for (iter = 0; iter < 10; iter++) {
            uint32_t cycles = benchmark_inference(q_features);
            
            uart_putnum(s);
            uart_putc(',');
            uart_putnum(cycles);
            uart_putc(',');
            uart_putnum(ensemble_predict(q_features));
            uart_puts("\r\n");
            
            total += cycles;
            if (cycles < min) min = cycles;
            if (cycles > max) max = cycles;
        }
    }
    
    // Summary
    uart_puts("\r\n=== Summary ===\r\n");
    uart_puts("Iterations: 50\r\n");
    uart_puts("Min: ");
    uart_putnum(min);
    uart_puts("\r\nMax: ");
    uart_putnum(max);
    uart_puts("\r\nAvg: ");
    uart_putnum(total / 50);
    uart_puts("\r\n");
    
    // Time at 25 MHz
    uart_puts("\r\nAt 25 MHz:\r\n");
    uart_puts("Min (us): ");
    uart_putnum(min / 25);
    uart_puts("\r\nAvg (us): ");
    uart_putnum(total / 50 / 25);
    uart_puts("\r\nMax (us): ");
    uart_putnum(max / 25);
    uart_puts("\r\n");
    
    uart_puts("\r\n=== Complete ===\r\n");
    
    while (1) {
        __bis_SR_register(LPM0_bits);
    }
    
    return 0;
}

#!/bin/bash
# ============================================================================
# SimAVR Runner Script for Inference Latency Measurement
# ============================================================================
# This script builds and runs AVR simulations for quantized ML models
# using simavr to measure inference latency in CPU cycles
#
# Usage: ./run_simavr.sh [model_name]
# Example: ./run_simavr.sh random_forest_no_oversampling
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_DIR="$SCRIPT_DIR/generated"
RESULTS_DIR="$SCRIPT_DIR/simulation_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v avr-gcc &> /dev/null; then
        echo -e "${RED}Error: avr-gcc not found${NC}"
        echo "Install with: brew install avr-gcc"
        exit 1
    fi
    
    if ! command -v simavr &> /dev/null; then
        echo -e "${YELLOW}Warning: simavr not found${NC}"
        echo "Install with: brew install simavr"
        echo "Continuing with build only..."
        SIMAVR_AVAILABLE=false
    else
        SIMAVR_AVAILABLE=true
    fi
    
    echo -e "${GREEN}Dependencies OK${NC}"
}

# Build a single model
build_model() {
    local model_dir="$1"
    local model_name=$(basename "$model_dir")
    
    echo ""
    echo "=============================================="
    echo "Building: $model_name"
    echo "=============================================="
    
    cd "$model_dir"
    
    # Clean previous build
    make clean 2>/dev/null || true
    
    # Build
    if make; then
        echo -e "${GREEN}Build successful${NC}"
        make size
        return 0
    else
        echo -e "${RED}Build failed${NC}"
        return 1
    fi
}

# Run simulation for a single model
run_simulation() {
    local model_dir="$1"
    local model_name=$(basename "$model_dir")
    local output_file="$RESULTS_DIR/${model_name}_results.txt"
    
    echo ""
    echo "=============================================="
    echo "Simulating: $model_name"
    echo "=============================================="
    
    cd "$model_dir"
    
    if [ ! -f "inference_benchmark.elf" ]; then
        echo -e "${RED}ELF file not found. Build first.${NC}"
        return 1
    fi
    
    mkdir -p "$RESULTS_DIR"
    
    # Run simavr with timeout
    # Capture UART output (simavr outputs to stdout)
    echo "Running simulation (30 second timeout)..."
    
    timeout 30s simavr -m atmega328p -f 16000000 inference_benchmark.elf 2>&1 | tee "$output_file" || true
    
    if [ -f "$output_file" ]; then
        echo -e "${GREEN}Results saved to: $output_file${NC}"
        
        # Extract summary
        echo ""
        echo "--- Summary ---"
        grep -A 10 "=== Summary ===" "$output_file" || echo "Summary not found"
    fi
}

# Process all models
process_all() {
    echo "Processing all models in $GENERATED_DIR"
    
    for model_dir in "$GENERATED_DIR"/*/; do
        if [ -d "$model_dir" ]; then
            build_model "$model_dir"
            if [ "$SIMAVR_AVAILABLE" = true ]; then
                run_simulation "$model_dir"
            fi
        fi
    done
}

# Process single model
process_single() {
    local model_name="$1"
    local model_dir="$GENERATED_DIR/$model_name"
    
    if [ ! -d "$model_dir" ]; then
        echo -e "${RED}Model directory not found: $model_dir${NC}"
        echo "Available models:"
        ls -1 "$GENERATED_DIR"
        exit 1
    fi
    
    build_model "$model_dir"
    if [ "$SIMAVR_AVAILABLE" = true ]; then
        run_simulation "$model_dir"
    fi
}

# Generate summary report
generate_report() {
    echo ""
    echo "=============================================="
    echo "GENERATING SUMMARY REPORT"
    echo "=============================================="
    
    local report_file="$RESULTS_DIR/summary_report.txt"
    
    {
        echo "AVR Inference Latency Simulation Results"
        echo "========================================"
        echo "Date: $(date)"
        echo "Target: ATmega328P @ 16MHz"
        echo ""
        
        for result_file in "$RESULTS_DIR"/*_results.txt; do
            if [ -f "$result_file" ]; then
                model_name=$(basename "$result_file" _results.txt)
                echo ""
                echo "Model: $model_name"
                echo "----------------------------------------"
                
                # Extract key metrics
                avg_cycles=$(grep "Avg cycles:" "$result_file" | awk '{print $3}')
                min_cycles=$(grep "Min cycles:" "$result_file" | awk '{print $3}')
                max_cycles=$(grep "Max cycles:" "$result_file" | awk '{print $3}')
                
                if [ -n "$avg_cycles" ]; then
                    echo "  Avg cycles: $avg_cycles"
                    echo "  Min cycles: $min_cycles"
                    echo "  Max cycles: $max_cycles"
                    
                    # Calculate time at 16MHz
                    avg_us=$((avg_cycles / 16))
                    echo "  Avg time: ${avg_us} µs"
                else
                    echo "  No timing data found"
                fi
            fi
        done
        
    } | tee "$report_file"
    
    echo ""
    echo -e "${GREEN}Report saved to: $report_file${NC}"
}

# Main
main() {
    echo "=============================================="
    echo "AVR Inference Latency Simulator"
    echo "=============================================="
    
    check_dependencies
    
    if [ $# -eq 0 ]; then
        # Process all models
        process_all
    elif [ "$1" = "report" ]; then
        generate_report
    else
        # Process specific model
        process_single "$1"
    fi
    
    if [ "$SIMAVR_AVAILABLE" = true ]; then
        generate_report
    fi
    
    echo ""
    echo "=============================================="
    echo "Done!"
    echo "=============================================="
}

main "$@"

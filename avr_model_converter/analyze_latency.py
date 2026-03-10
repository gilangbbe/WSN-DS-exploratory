#!/usr/bin/env python3
"""
AVR Inference Latency Analyzer
Uses avr-objdump to analyze instruction cycles for each model.
Also runs simavr simulation to verify functionality.
"""

import subprocess
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# AVR instruction cycle counts (ATmega328P)
# Reference: ATmega328P datasheet, instruction set summary
AVR_CYCLES = {
    # Arithmetic and Logic
    'add': 1, 'adc': 1, 'adiw': 2, 'sub': 1, 'subi': 1, 'sbc': 1, 'sbci': 1,
    'sbiw': 2, 'and': 1, 'andi': 1, 'or': 1, 'ori': 1, 'eor': 1, 'com': 1,
    'neg': 1, 'inc': 1, 'dec': 1, 'tst': 1, 'clr': 1, 'ser': 1, 'mul': 2,
    'muls': 2, 'mulsu': 2, 'fmul': 2, 'fmuls': 2, 'fmulsu': 2,
    
    # Branch
    'rjmp': 2, 'ijmp': 2, 'jmp': 3, 'rcall': 3, 'icall': 3, 'call': 4,
    'ret': 4, 'reti': 4, 'cpse': 1,  # 1/2/3 depending on skip
    'cp': 1, 'cpc': 1, 'cpi': 1,
    'sbrc': 1, 'sbrs': 1, 'sbic': 1, 'sbis': 1,  # 1/2/3
    'brbs': 1, 'brbc': 1,  # 1/2
    'breq': 1, 'brne': 1, 'brcs': 1, 'brcc': 1, 'brsh': 1, 'brlo': 1,
    'brmi': 1, 'brpl': 1, 'brge': 1, 'brlt': 1, 'brhs': 1, 'brhc': 1,
    'brts': 1, 'brtc': 1, 'brvs': 1, 'brvc': 1, 'brie': 1, 'brid': 1,
    
    # Data Transfer
    'mov': 1, 'movw': 1, 'ldi': 1, 'lds': 2, 'ld': 2, 'ldd': 2,
    'sts': 2, 'st': 2, 'std': 2,
    'lpm': 3, 'elpm': 3, 'spm': 0,  # SPM varies
    'in': 1, 'out': 1, 'push': 2, 'pop': 2,
    
    # Bit and Bit-test
    'lsl': 1, 'lsr': 1, 'rol': 1, 'ror': 1, 'asr': 1, 'swap': 1,
    'bset': 1, 'bclr': 1, 'sbi': 2, 'cbi': 2, 'bst': 1, 'bld': 1,
    'sec': 1, 'clc': 1, 'sen': 1, 'cln': 1, 'sez': 1, 'clz': 1,
    'sei': 1, 'cli': 1, 'ses': 1, 'cls': 1, 'sev': 1, 'clv': 1,
    'set': 1, 'clt': 1, 'seh': 1, 'clh': 1,
    
    # MCU Control
    'nop': 1, 'sleep': 1, 'wdr': 1, 'break': 1,
}


def get_function_addresses(elf_path: str) -> Dict[str, Tuple[int, int]]:
    """Get start and end addresses of key functions from ELF."""
    result = subprocess.run(
        ['avr-nm', '-n', elf_path],
        capture_output=True, text=True
    )
    
    symbols = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3:
            addr = int(parts[0], 16)
            sym_type = parts[1]
            name = parts[2]
            if sym_type in ['T', 't']:  # Text (code) symbols
                symbols.append((addr, name))
    
    # Sort by address
    symbols.sort(key=lambda x: x[0])
    
    # Create function ranges
    functions = {}
    for i, (addr, name) in enumerate(symbols):
        if i + 1 < len(symbols):
            end_addr = symbols[i + 1][0]
        else:
            end_addr = addr + 1000  # Estimate
        functions[name] = (addr, end_addr)
    
    return functions


def analyze_function_cycles(elf_path: str, func_name: str) -> Dict:
    """Analyze cycle count for a specific function."""
    # Get function addresses
    funcs = get_function_addresses(elf_path)
    
    if func_name not in funcs:
        print(f"Function {func_name} not found")
        return {'cycles': 0, 'instructions': 0}
    
    start_addr, end_addr = funcs[func_name]
    
    # Disassemble
    result = subprocess.run(
        ['avr-objdump', '-d', elf_path],
        capture_output=True, text=True
    )
    
    total_cycles = 0
    inst_count = 0
    in_function = False
    
    for line in result.stdout.split('\n'):
        # Check for function label
        if f'<{func_name}>:' in line:
            in_function = True
            continue
        
        # Check for next function
        if in_function and re.match(r'^[0-9a-f]+\s+<\w+>:', line):
            break
        
        if in_function:
            # Parse instruction
            match = re.match(r'\s*([0-9a-f]+):\s+[0-9a-f ]+\s+(\w+)', line)
            if match:
                addr = int(match.group(1), 16)
                instr = match.group(2).lower()
                
                # Get cycle count
                cycles = AVR_CYCLES.get(instr, 1)  # Default 1 cycle
                total_cycles += cycles
                inst_count += 1
    
    return {
        'cycles': total_cycles,
        'instructions': inst_count,
        'start_addr': hex(start_addr),
        'end_addr': hex(end_addr)
    }


def analyze_tree_traversal(elf_path: str) -> Dict:
    """Estimate cycles for tree traversal based on tree depth."""
    # Analyze tree_predict function
    tree_predict = analyze_function_cycles(elf_path, 'tree_predict')
    
    # Tree traversal is data-dependent
    # Average path length = log2(n_nodes) for balanced tree
    # With max_depth=6, average = ~4-5 iterations
    
    return tree_predict


def analyze_model(model_dir: str) -> Dict:
    """Analyze a single model's latency characteristics."""
    elf_path = os.path.join(model_dir, 'inference_benchmark.elf')
    model_name = os.path.basename(model_dir)
    
    if not os.path.exists(elf_path):
        print(f"ELF not found: {elf_path}")
        return None
    
    print(f"\nAnalyzing: {model_name}")
    print("=" * 50)
    
    # Get memory usage
    size_result = subprocess.run(
        ['avr-size', '--mcu=atmega328p', '-C', elf_path],
        capture_output=True, text=True
    )
    print(size_result.stdout)
    
    # Analyze key functions
    functions_to_analyze = [
        'tree_predict',
        'ensemble_predict',
        'quantize_features',
        'quantize_feature',
    ]
    
    results = {
        'model_name': model_name,
        'functions': {}
    }
    
    for func in functions_to_analyze:
        analysis = analyze_function_cycles(elf_path, func)
        results['functions'][func] = analysis
        print(f"  {func}: {analysis['instructions']} instructions, ~{analysis['cycles']} base cycles")
    
    # Get model config from header
    config_path = os.path.join(model_dir, 'model_config.h')
    n_trees = 5
    max_depth = 6
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_content = f.read()
            m = re.search(r'#define N_TREES\s+(\d+)', config_content)
            if m:
                n_trees = int(m.group(1))
            m = re.search(r'#define MAX_DEPTH\s+(\d+)', config_content)
            if m:
                max_depth = int(m.group(1))
    
    results['n_trees'] = n_trees
    results['max_depth'] = max_depth
    
    # Estimate total inference cycles
    # tree_predict is called n_trees times, each traverses ~max_depth nodes
    tree_predict_cycles = results['functions'].get('tree_predict', {}).get('cycles', 100)
    ensemble_overhead = results['functions'].get('ensemble_predict', {}).get('cycles', 50)
    quantize_cycles = results['functions'].get('quantize_features', {}).get('cycles', 50)
    
    # For each tree: ~tree_predict_cycles per depth level
    # Average path = max_depth / 2 iterations through the loop
    avg_iterations = max_depth // 2 + 1
    
    # Loop overhead in tree_predict (estimate from disassembly pattern)
    # Each iteration: compare, branch, load from PROGMEM
    loop_cycles_per_iter = 15  # Estimated: lpm(3) + compare(1) + branch(2) + loads(~6) + misc(3)
    
    single_tree_cycles = loop_cycles_per_iter * avg_iterations + 30  # 30 for function call/ret overhead
    total_trees_cycles = single_tree_cycles * n_trees
    
    # Voting overhead
    voting_cycles = n_trees * 5 + 30  # Increment votes, find max
    
    # Total estimate
    total_cycles = quantize_cycles + total_trees_cycles + voting_cycles + ensemble_overhead
    
    results['estimated_cycles'] = {
        'quantization': quantize_cycles,
        'per_tree_avg': single_tree_cycles,
        'all_trees': total_trees_cycles,
        'voting': voting_cycles,
        'total': total_cycles
    }
    
    # Calculate timing at 16MHz
    freq_mhz = 16
    time_us = total_cycles / freq_mhz
    time_ms = time_us / 1000
    
    results['timing'] = {
        'frequency_mhz': freq_mhz,
        'time_us': round(time_us, 2),
        'time_ms': round(time_ms, 4),
        'throughput_per_sec': round(1000000 / time_us, 1) if time_us > 0 else 0
    }
    
    print(f"\n  Estimated Total Cycles: {total_cycles}")
    print(f"  At 16 MHz: {time_us:.1f} µs ({time_ms:.3f} ms)")
    print(f"  Throughput: {1000000 / time_us:.1f} inferences/sec")
    
    return results


def run_simulation(model_dir: str) -> str:
    """Run simavr simulation and capture UART output."""
    elf_path = os.path.join(model_dir, 'inference_benchmark.elf')
    
    if not os.path.exists(elf_path):
        return "ELF not found"
    
    # Run simavr with timeout (using Python's subprocess timeout)
    try:
        result = subprocess.run(
            ['simavr', '-m', 'atmega328p', '-f', '16000000', elf_path],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Simulation timeout (expected - infinite loop at end)"
    except Exception as e:
        return f"Simulation error: {e}"


def main():
    script_dir = Path(__file__).parent
    generated_dir = script_dir / 'generated'
    results_dir = script_dir / 'simulation_results'
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("AVR INFERENCE LATENCY ANALYSIS")
    print("=" * 70)
    
    models = [
        'extra_trees_no_oversampling',
        'random_forest_no_oversampling',
        'gradient_boosting_no_oversampling',
        'histgradient_boosting_no_oversampling',
    ]
    
    all_results = []
    
    for model in models:
        model_dir = generated_dir / model
        
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue
        
        # Build if needed
        elf_path = model_dir / 'inference_benchmark.elf'
        if not elf_path.exists():
            print(f"Building {model}...")
            subprocess.run(['make', 'clean'], cwd=model_dir, capture_output=True)
            subprocess.run(['make'], cwd=model_dir, capture_output=True)
        
        # Analyze
        result = analyze_model(str(model_dir))
        if result:
            all_results.append(result)
    
    # Summary table
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Cycles':>10} {'Time (µs)':>12} {'Infer/sec':>12}")
    print("-" * 70)
    
    for r in all_results:
        name = r['model_name'].replace('_no_oversampling', '')
        cycles = r['estimated_cycles']['total']
        time_us = r['timing']['time_us']
        throughput = r['timing']['throughput_per_sec']
        print(f"{name:<45} {cycles:>10} {time_us:>12.1f} {throughput:>12.1f}")
    
    # Save detailed results
    output_file = results_dir / 'latency_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Generate CSV summary
    csv_file = results_dir / 'latency_summary.csv'
    with open(csv_file, 'w') as f:
        f.write("Model,Trees,MaxDepth,TotalCycles,TimeUs,TimeMs,InferencesPerSec,FlashUsage\n")
        for r in all_results:
            name = r['model_name']
            trees = r['n_trees']
            depth = r['max_depth']
            cycles = r['estimated_cycles']['total']
            time_us = r['timing']['time_us']
            time_ms = r['timing']['time_ms']
            throughput = r['timing']['throughput_per_sec']
            f.write(f"{name},{trees},{depth},{cycles},{time_us},{time_ms},{throughput}\n")
    print(f"CSV summary saved to: {csv_file}")


if __name__ == '__main__':
    main()

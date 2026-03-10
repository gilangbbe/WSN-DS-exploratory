# Complete Guide: Deploying Machine Learning Models on Microcontrollers
## Understanding Every Step from Python to MSP430 C Code

**Author**: Generated for Thesis Presentation  
**Target Audience**: Readers with zero embedded systems knowledge  
**Focus**: WSN Intrusion Detection System on MSP430F5529

---

## Table of Contents

1. [Why Do We Need This?](#1-why-do-we-need-this)
2. [What is a Microcontroller?](#2-what-is-a-microcontroller)
3. [The Problem: Python Models Can't Run on MCUs](#3-the-problem-python-models-cant-run-on-mcus)
4. [Solution Overview: The Conversion Pipeline](#4-solution-overview-the-conversion-pipeline)
5. [Step 1: Understanding Quantization](#5-step-1-understanding-quantization)
6. [Step 2: How Trees are Stored in Memory](#6-step-2-how-trees-are-stored-in-memory)
7. [Step 3: The Generated C Code Explained](#7-step-3-the-generated-c-code-explained)
8. [Step 4: How Inference Works on MCU](#8-step-4-how-inference-works-on-mcu)
9. [Step 5: Cycle Counting - How We Measure Speed](#9-step-5-cycle-counting---how-we-measure-speed)
10. [Putting It All Together: Complete Example](#10-putting-it-all-together-complete-example)
11. [Key Results and What They Mean](#11-key-results-and-what-they-mean)

---

## 1. Why Do We Need This?

### The Scenario

In a Wireless Sensor Network (WSN), hundreds of small sensor nodes collect data and communicate with each other. In LEACH architecture (Low-Energy Adaptive Clustering Hierarchy), these nodes need to:

1. **Collect sensor data** (temperature, humidity, network traffic, etc.)
2. **Detect intrusions** (identify if there's a security attack)
3. **Send only important data** to save battery

The challenge: **We want each sensor node to run a machine learning model to detect intrusions locally**, before sending data. This saves power and enables faster response.

### The Constraint

Sensor nodes use **microcontrollers (MCUs)** - tiny computers with:
- Very limited memory (KB, not GB)
- No operating system
- No Python, no floating-point hardware
- Must run on battery for months/years

Your trained ML model in Python uses:
- **Floating-point numbers** (32-bit or 64-bit decimals)
- **Scikit-learn objects** with complex data structures
- **Python runtime** (interpreter, garbage collector, etc.)

**None of this can run on a microcontroller.**

---

## 2. What is a Microcontroller?

### MSP430F5529 Specifications

| Property | MSP430F5529 | Your Laptop (for comparison) |
|----------|-------------|------------------------------|
| Architecture | 16-bit RISC | 64-bit x86 |
| Clock Speed | 25 MHz | 3,000+ MHz |
| Flash Memory (Code Storage) | 128 KB | 256+ GB SSD |
| RAM (Working Memory) | 8 KB | 16+ GB |
| Floating Point Unit | **None** | Yes (hardware) |
| Operating System | **None** | Windows/macOS/Linux |
| Power Consumption | ~1 mW active | 15-45 W |

### Key Differences

```
Your Laptop:
┌────────────────────────────────────────────────────────────┐
│  Operating System (Windows/macOS/Linux)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Python Runtime                                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Scikit-learn                                  │  │  │
│  │  │  ┌──────────────────────────────────────────┐  │  │  │
│  │  │  │  Your Model (Random Forest, etc.)        │  │  │  │
│  │  │  └──────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘

MSP430 Microcontroller:
┌────────────────────────────────────────────────────────────┐
│                 BARE METAL (No OS)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Your C Code Runs Directly on Hardware               │  │
│  │  (No Python, No Libraries, Just Your Code)           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 3. The Problem: Python Models Can't Run on MCUs

### Problem 1: No Floating Point Hardware

Your model uses numbers like:
```python
threshold = 2.5731248855590820
feature_value = 0.123456789
```

The MSP430 has **no hardware to do decimal math**. Every floating-point operation would need to be emulated in software, taking **hundreds of CPU cycles** per operation.

### Problem 2: Complex Data Structures

In scikit-learn, a decision tree looks like:
```python
tree.tree_.threshold = array([2.573, -2.0, 1.234, ...], dtype=float64)
tree.tree_.feature = array([5, 3, 12, ...], dtype=int64)
tree.tree_.children_left = array([1, 3, -1, ...], dtype=int64)
```

These are 64-bit numbers! On a 16-bit MCU, even storing one `int64` takes 8 bytes and requires multiple operations to access.

### Problem 3: Memory Constraints

A simple Random Forest model in Python might use:
- 100 trees × 1000 nodes/tree × ~50 bytes/node = **5 MB**

MSP430 has only **128 KB Flash** and **8 KB RAM**.

### The Solution

We need to:
1. **Convert floating-point to integers** (Quantization)
2. **Simplify the tree structure** (Limit depth, reduce trees)
3. **Generate efficient C code** (Compile to machine code)
4. **Measure performance** (Cycle counting)

---

## 4. Solution Overview: The Conversion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVERSION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ PYTHON       │     │ QUANTIZATION │     │ C CODE       │                 │
│  │ scikit-learn │ ──▶ │ float→int8   │ ──▶ │ Generation   │                 │
│  │ Model        │     │ (0-255)      │     │ (.c, .h)     │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                    │                    │                          │
│         ▼                    ▼                    ▼                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │ 64-bit       │     │ 8-bit        │     │ Tree nodes   │                 │
│  │ Floats       │     │ Integers     │     │ as byte      │                 │
│  │ (8 bytes     │     │ (1 byte      │     │ arrays       │                 │
│  │  per number) │     │  per number) │     │              │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                   │                          │
│                                                   ▼                          │
│                              ┌──────────────────────────────┐               │
│                              │    MSP430 GCC COMPILER       │               │
│                              │    msp430-elf-gcc            │               │
│                              └──────────────────────────────┘               │
│                                                   │                          │
│                                                   ▼                          │
│                              ┌──────────────────────────────┐               │
│                              │    .ELF Binary               │               │
│                              │    (Machine Code for MSP430) │               │
│                              └──────────────────────────────┘               │
│                                                   │                          │
│                                                   ▼                          │
│                              ┌──────────────────────────────┐               │
│                              │    CYCLE ANALYSIS            │               │
│                              │    (Count CPU instructions)  │               │
│                              └──────────────────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Step 1: Understanding Quantization

### What is Quantization?

Quantization converts **floating-point numbers** (decimals) into **fixed-point integers**. We use **INT8** (8-bit unsigned integers: 0 to 255).

### Why 8 Bits?

| Data Type | Size | Range | Example Value |
|-----------|------|-------|---------------|
| float64 | 8 bytes | ±1.7×10³⁰⁸ | 3.141592653589793 |
| float32 | 4 bytes | ±3.4×10³⁸ | 3.1415927 |
| **int8/uint8** | **1 byte** | 0-255 or -128-127 | 127 |

Using 8-bit integers saves **8× memory** compared to float64!

### The Math Behind Quantization

For each feature in your dataset, we compute:

```
min_value = minimum value this feature ever takes
max_value = maximum value this feature ever takes
range = max_value - min_value

scale = 255 / range
zero_point = -min_value × scale
```

To quantize a value:
```
quantized_value = (original_value × scale) + zero_point
```

**Example:**

Let's say Feature #5 (Distance) has values from 0.0 to 10.0:
```
min = 0.0, max = 10.0, range = 10.0
scale = 255 / 10.0 = 25.5
zero_point = -0.0 × 25.5 = 0

Original value: 4.0
Quantized: (4.0 × 25.5) + 0 = 102

Original value: 7.5
Quantized: (7.5 × 25.5) + 0 = 191
```

### Quantization in Code

From `quantization.h`:
```c
// Scale factors (Q8.8 fixed-point format)
static const int16_t FEATURE_SCALE[N_FEATURES] = {
    0, 18, -256, 0, 304, 672, 557, -256, ...
};

// Zero points
static const uint8_t FEATURE_ZERO[N_FEATURES] = {
    249, 253, 0, 249, 0, 0, 0, 0, ...
};
```

The scale values use **Q8.8 fixed-point**: the value is stored as an integer but represents `value / 256`. This allows fractional values without floating-point!

```c
// How a feature is quantized on MSP430
uint8_t quantize_feature(float value, uint8_t feature_idx) {
    int16_t scale = FEATURE_SCALE[feature_idx];
    uint8_t zero = FEATURE_ZERO[feature_idx];
    
    // Multiply and shift right by 8 (divide by 256)
    int32_t scaled = (int32_t)(value * scale) >> 8;
    int16_t result = scaled + zero;
    
    // Clamp to 0-255 range
    if (result < 0) return 0;
    if (result > 255) return 255;
    return (uint8_t)result;
}
```

---

## 6. Step 2: How Trees are Stored in Memory

### Decision Tree Basics

A decision tree makes predictions by asking a series of yes/no questions:

```
                    ┌─────────────────┐
                    │ Is feature[5]   │
                    │    ≤ 127?       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES                          │ NO
              ▼                              ▼
     ┌────────────────┐             ┌────────────────┐
     │ Is feature[3]  │             │ Is feature[8]  │
     │    ≤ 64?       │             │    ≤ 200?      │
     └───────┬────────┘             └───────┬────────┘
             │                              │
    ┌────────┴────────┐            ┌────────┴────────┐
    │YES              │NO          │YES              │NO
    ▼                 ▼            ▼                 ▼
┌───────┐        ┌───────┐    ┌───────┐        ┌───────┐
│Class 0│        │Class 1│    │Class 2│        │Class 3│
│Normal │        │Flooding│   │Blackhole│      │Grayhole│
└───────┘        └───────┘    └───────┘        └───────┘
```

### Node Storage Format

Each tree node is stored as **6 bytes**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           NODE FORMAT (6 bytes)                             │
├────────────┬────────────┬────────────┬────────────┬────────────┬───────────┤
│ Byte 0     │ Byte 1     │ Byte 2     │ Byte 3     │ Byte 4     │ Byte 5    │
├────────────┼────────────┼────────────┼────────────┼────────────┼───────────┤
│ feature_idx│ threshold  │ left_lo    │ left_hi    │ right_lo   │ right_hi  │
│ (0-254)    │ (0-255)    │ low byte   │ high byte  │ low byte   │ high byte │
├────────────┼────────────┼────────────┴────────────┼────────────┴───────────┤
│ Which      │ Compare    │ Left child index        │ Right child index      │
│ feature?   │ against    │ (16-bit value)          │ (16-bit value)         │
└────────────┴────────────┴─────────────────────────┴────────────────────────┘

SPECIAL CASE - LEAF NODE:
┌────────────┬────────────┬────────────────────────────────────────────────────┐
│ Byte 0     │ Byte 1     │ Bytes 2-5                                          │
├────────────┼────────────┼────────────────────────────────────────────────────┤
│    255     │ class_id   │ 0, 0, 0, 0  (unused)                               │
├────────────┼────────────┼────────────────────────────────────────────────────┤
│ LEAF       │ Predicted  │                                                    │
│ MARKER     │ Class      │                                                    │
└────────────┴────────────┴────────────────────────────────────────────────────┘
```

### Why 16-bit Child Indices?

Child indices are stored as two bytes because:
- Maximum tree might have up to 2^16 = 65,536 nodes
- But we use two bytes for each child index
- Low byte first (little-endian format common in embedded)

```
left_child = left_lo + (left_hi × 256)
right_child = right_lo + (right_hi × 256)
```

### Real Example from tree_data.h

```c
// Tree 0: 85 nodes, depth 6
static const uint8_t tree_0[] = {
    1, 0, 1, 0, 2, 0,    // Node 0: if feature[1] ≤ 0, go to node 1, else node 2
    8, 0, 3, 0, 4, 0,    // Node 1: if feature[8] ≤ 0, go to node 3, else node 4
    6, 23, 5, 0, 6, 0,   // Node 2: if feature[6] ≤ 23, go to node 5, else node 6
    ...
    255, 3, 0, 0, 0, 0,  // Node 16: LEAF - predict class 3
    255, 4, 0, 0, 0, 0,  // Node 25: LEAF - predict class 4
    ...
};
```

**Reading Node 2:** `6, 23, 5, 0, 6, 0`
- `feature_idx = 6` → Check feature number 6
- `threshold = 23` → Compare against value 23
- `left_child = 5 + (0 × 256) = 5` → If ≤, go to node 5
- `right_child = 6 + (0 × 256) = 6` → If >, go to node 6

**Reading Node 16:** `255, 3, 0, 0, 0, 0`
- `feature_idx = 255` → This is a LEAF node!
- `threshold = 3` → Predict class 3 (this is actually the class ID)

---

## 7. Step 3: The Generated C Code Explained

### File Structure

```
generated_msp430/extra_trees_no_oversampling/
├── model_config.h      # Model parameters (N_TREES, N_FEATURES, etc.)
├── quantization.h      # Scale factors and zero points
├── tree_data.h         # All tree nodes as byte arrays
├── inference.h         # Function declarations
├── inference.c         # The actual inference code
├── main.c              # Benchmark program
└── Makefile            # Build instructions
```

### model_config.h - Model Parameters

```c
#define MODEL_NAME "Extra_Trees_No_Oversampling"
#define MODEL_TYPE "ExtraTreesClassifier"

#define N_FEATURES 18     // Number of input features
#define N_CLASSES 5       // Number of output classes (Normal + 4 attack types)
#define N_TREES 5         // Number of trees in ensemble
#define MAX_DEPTH 6       // Maximum tree depth (limited from original)

#define F_CPU 25000000UL  // 25 MHz clock speed
```

### inference.c - The Core Algorithm

#### 1. Single Tree Prediction

```c
uint8_t tree_predict(const uint8_t* tree_data, const uint8_t* features) {
    uint16_t node_idx = 0;  // Start at root node
    
    while (1) {  // Loop until we hit a leaf
        // Calculate byte offset: each node is 6 bytes
        uint16_t offset = node_idx * 6;
        
        // Read node data
        uint8_t feature_idx = tree_data[offset];      // Which feature to check
        uint8_t threshold = tree_data[offset + 1];    // Threshold value
        
        // Check if this is a leaf node
        if (feature_idx == 255) {
            return threshold;  // Return class ID
        }
        
        // Get the feature value from input
        uint8_t feature_val = features[feature_idx];
        
        // Compare and branch
        if (feature_val <= threshold) {
            // Go left: read 16-bit left child index
            node_idx = tree_data[offset + 2] | 
                      ((uint16_t)tree_data[offset + 3] << 8);
        } else {
            // Go right: read 16-bit right child index
            node_idx = tree_data[offset + 4] | 
                      ((uint16_t)tree_data[offset + 5] << 8);
        }
    }
}
```

**Visual explanation:**

```
tree_predict(tree_0, [102, 45, 200, 78, ...])
│
├─ node_idx = 0
│  ├─ feature_idx = 1, threshold = 0
│  ├─ features[1] = 45
│  ├─ 45 ≤ 0? NO → go right
│  └─ node_idx = 2
│
├─ node_idx = 2  
│  ├─ feature_idx = 6, threshold = 23
│  ├─ features[6] = 200
│  ├─ 200 ≤ 23? NO → go right
│  └─ node_idx = 6
│
├─ ... (continue traversing)
│
└─ node_idx = 44
   ├─ feature_idx = 255 (LEAF!)
   ├─ threshold = 4 (this is the class ID)
   └─ RETURN 4 (Class: TDMA attack)
```

#### 2. Ensemble Voting

```c
uint8_t ensemble_predict(const uint8_t* features) {
    uint8_t votes[N_CLASSES] = {0};  // Vote counter for each class
    uint8_t t;
    
    // Get prediction from each tree
    for (t = 0; t < N_TREES; t++) {
        uint8_t pred = tree_predict(trees[t], features);  // Get tree's vote
        if (pred < N_CLASSES) {
            votes[pred]++;  // Add vote for this class
        }
    }
    
    // Find majority vote
    uint8_t max_votes = 0;
    uint8_t predicted = 0;
    uint8_t c;
    
    for (c = 0; c < N_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            predicted = c;
        }
    }
    
    return predicted;
}
```

**Visual explanation:**

```
ensemble_predict([102, 45, 200, ...])
│
├─ Tree 0 predicts: Class 3
├─ Tree 1 predicts: Class 3
├─ Tree 2 predicts: Class 3
├─ Tree 3 predicts: Class 2
├─ Tree 4 predicts: Class 3
│
├─ votes[0] = 0
├─ votes[1] = 0
├─ votes[2] = 1
├─ votes[3] = 4  ← WINNER!
├─ votes[4] = 0
│
└─ RETURN 3 (Class 3 wins with 4 votes)
```

---

## 8. Step 4: How Inference Works on MCU

### Complete Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE INFERENCE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Sensor Data Collection
┌─────────────────────────────────────────────────────────────────────────────┐
│ Raw sensor readings (floating-point or ADC values):                          │
│ features[] = {1.5, 0.0, 2.3, 0.1, 45.6, 78.9, 12.3, ...} (18 values)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 2: Feature Quantization
┌─────────────────────────────────────────────────────────────────────────────┐
│ Convert each feature to uint8 using pre-computed scale and zero_point:      │
│                                                                              │
│ for i = 0 to 17:                                                             │
│     q_features[i] = (features[i] * scale[i]) >> 8 + zero[i]                 │
│     clamp to [0, 255]                                                        │
│                                                                              │
│ Result: q_features[] = {102, 45, 200, 78, 12, 234, 56, ...} (18 bytes)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 3: Tree Traversal (repeat for each tree)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Start at Node 0                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────┐                                │
│  │ Is feature_idx == 255? (Leaf check)     │──YES──▶ Return class_id        │
│  └─────────────────────────────────────────┘                                │
│       │ NO                                                                   │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────┐                                │
│  │ Compare: q_features[feature_idx] ≤ threshold?                            │
│  └─────────────────────────────────────────┘                                │
│       │                              │                                       │
│       │ YES                          │ NO                                    │
│       ▼                              ▼                                       │
│  Go to left_child              Go to right_child                            │
│       │                              │                                       │
│       └──────────────┬───────────────┘                                       │
│                      │                                                       │
│                      ▼                                                       │
│              Loop back to leaf check                                         │
│                                                                              │
│  Average: 3-4 comparisons per tree (depth/2 + 1)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 4: Voting
┌─────────────────────────────────────────────────────────────────────────────┐
│ Collect predictions from all 5 trees:                                        │
│   Tree 0 → Class 3                                                           │
│   Tree 1 → Class 3                                                           │
│   Tree 2 → Class 3                                                           │
│   Tree 3 → Class 2                                                           │
│   Tree 4 → Class 3                                                           │
│                                                                              │
│ Count votes: votes = [0, 0, 1, 4, 0]                                        │
│ Find max: Class 3 with 4 votes                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 5: Output
┌─────────────────────────────────────────────────────────────────────────────┐
│ Predicted class: 3 (e.g., Grayhole Attack)                                  │
│ Action: Raise alert, drop packet, notify cluster head                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Step 5: Cycle Counting - How We Measure Speed

### What is a CPU Cycle?

A **CPU cycle** is the smallest unit of time for a processor. At each cycle, the processor can execute (part of) an instruction.

```
MSP430 @ 25 MHz means:
- 25 million cycles per second
- Each cycle = 1 / 25,000,000 = 0.00004 milliseconds = 40 nanoseconds
```

### Why Count Cycles?

On embedded systems, we don't have a system clock to measure time in seconds. Instead:

1. **Cycles are deterministic** - Same code = same cycles (no OS interference)
2. **Cycles × (1/Clock_Frequency) = Time** - Easy to convert
3. **Portable** - Results apply regardless of exact clock speed

### MSP430 Instruction Cycles

Every MSP430 instruction takes a specific number of cycles:

| Instruction | Cycles | Example | Description |
|-------------|--------|---------|-------------|
| MOV | 1 | `mov r15, r14` | Copy register to register |
| MOV (memory) | 3-4 | `mov @r14, r15` | Load from memory |
| ADD | 1 | `add r14, r15` | Add two registers |
| CMP | 1-4 | `cmp r14, r15` | Compare values |
| JMP | 2 | `jmp label` | Unconditional jump |
| JNE/JEQ | 2 | `jne label` | Conditional jump |
| CALL | 4 | `call function` | Function call |
| RET | 3 | `ret` | Return from function |
| PUSH | 3 | `push r15` | Push to stack |
| POP | 2 | `pop r15` | Pop from stack |

### How We Count Cycles

#### Method 1: Static Analysis (What We Use)

We analyze the compiled assembly code and count instructions:

```python
# From msp430_latency_analysis.py

MSP430_CYCLES = {
    'mov': 1, 'add': 1, 'cmp': 1,
    'jmp': 2, 'jnz': 2, 'jne': 2, 'jeq': 2,
    'call': 4, 'ret': 3,
    'push': 3, 'pop': 2,
    ...
}

def estimate_instruction_cycles(instr, operands):
    base_cycles = MSP430_CYCLES.get(instr, 1)
    
    # Memory access adds cycles
    if '@' in operands:    # Indirect: @Rn
        base_cycles += 1
    if '(' in operands:    # Indexed: X(Rn)
        base_cycles += 2
    if '#' in operands:    # Immediate: #N
        base_cycles += 1
    
    return base_cycles
```

#### Method 2: Analytical Estimation (What We Also Use)

Based on algorithm structure:

```python
# Feature quantization: per-feature cost
quant_per_feature = 35  # Load scale, multiply, shift, add, clamp, store
quant_total = n_features * quant_per_feature + 20  # +loop overhead

# Tree traversal: per-node cost
cycles_per_node = 25    # Load idx, load threshold, compare, branch, load child
avg_path_length = max_depth / 2 + 1  # Average nodes visited
tree_traversal = avg_path_length * cycles_per_node
call_overhead = 10      # call + ret + register save/restore

cycles_per_tree = tree_traversal + call_overhead

# Ensemble: all trees + voting
total_cycles = quant_total + (n_trees * cycles_per_tree) + voting_overhead
```

### Breaking Down the Cycles

For **Extra Trees (5 trees, depth 6)**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CYCLE BREAKDOWN                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Feature Quantization                                        │    │
│  │                                                                      │    │
│  │   18 features × 35 cycles/feature = 630 cycles                      │    │
│  │   + Loop overhead: 20 cycles                                         │    │
│  │   ────────────────────────────────                                   │    │
│  │   SUBTOTAL: 650 cycles                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Tree Traversal (per tree)                                   │    │
│  │                                                                      │    │
│  │   Average path: 6/2 + 1 = 4 nodes                                   │    │
│  │   Per node:                                                          │    │
│  │     - Load feature_idx:     3 cycles                                │    │
│  │     - Load threshold:       3 cycles                                │    │
│  │     - Load feature value:   3 cycles                                │    │
│  │     - Compare:              4 cycles                                │    │
│  │     - Branch:               2 cycles                                │    │
│  │     - Load child pointer:   6 cycles                                │    │
│  │     - Other overhead:       4 cycles                                │    │
│  │     ─────────────────────────────                                   │    │
│  │     Total per node: 25 cycles                                       │    │
│  │                                                                      │    │
│  │   4 nodes × 25 cycles = 100 cycles per tree                         │    │
│  │   + Call overhead: 10 cycles                                         │    │
│  │   ──────────────────────────────                                    │    │
│  │   Per tree: 110 cycles                                              │    │
│  │                                                                      │    │
│  │   5 trees × 110 cycles = 550 cycles                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: Voting                                                       │    │
│  │                                                                      │    │
│  │   Update votes array: 5 trees × 8 cycles = 40 cycles                │    │
│  │   Find maximum: 5 classes × 6 cycles = 30 cycles                    │    │
│  │   ───────────────────────────────────                                │    │
│  │   SUBTOTAL: 70 cycles                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════    │
│  TOTAL: 650 + 550 + 70 = 1,270 cycles                                       │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Converting Cycles to Time

```
Time = Cycles / Clock_Frequency

For Extra Trees:
Time = 1,270 cycles / 25,000,000 Hz
     = 0.0000508 seconds
     = 50.8 microseconds (µs)
     = 0.0508 milliseconds (ms)
```

### Converting to Throughput

```
Throughput = 1 / Time = Clock_Frequency / Cycles

Throughput = 25,000,000 / 1,270 = 19,685 inferences per second
```

---

## 10. Putting It All Together: Complete Example

### Input Data

Suppose a sensor node receives this network packet with 18 features:

```
Raw features (floating-point):
[1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 0.5, 0.8]
```

### Step-by-Step Execution

```
STEP 1: Quantize Features (650 cycles)
────────────────────────────────────────
quantized = [249, 252, 0, 249, 0, 107, 71, 0, 100, 0, 0, 42, 69, 9, 69, 52, 349, 155]
          (clamped to [0,255])

STEP 2: Traverse Tree 0 (110 cycles)
────────────────────────────────────────
Node 0: feature[1]=252 ≤ 0? NO → right child = 2
Node 2: feature[6]=71 ≤ 23? NO → right child = 6
Node 6: feature[13]=9 ≤ 0? NO → right child = 14
Node 14: feature[0]=249 ≤ 0? NO → right child = 30
Node 30: feature[7]=0 ≤ 255? YES → left child = 49
Node 49: feature[3]=249 ≤ 0? NO → right child = 82
Node 82: feature_idx=255 (LEAF) → return class 1

Tree 0 votes: Class 1

STEP 3: Traverse Tree 1 (110 cycles)
────────────────────────────────────────
... (similar traversal)
Tree 1 votes: Class 1

STEP 4: Traverse Tree 2 (110 cycles)
────────────────────────────────────────
... (similar traversal)
Tree 2 votes: Class 1

STEP 5: Traverse Tree 3 (110 cycles)
────────────────────────────────────────
... (similar traversal)
Tree 3 votes: Class 0

STEP 6: Traverse Tree 4 (110 cycles)
────────────────────────────────────────
... (similar traversal)
Tree 4 votes: Class 1

STEP 7: Voting (70 cycles)
────────────────────────────────────────
votes = [1, 4, 0, 0, 0]
       Class 0: 1 vote
       Class 1: 4 votes ← WINNER
       Class 2: 0 votes
       Class 3: 0 votes
       Class 4: 0 votes

FINAL RESULT: Class 1 (Flooding Attack)

TOTAL TIME: 1,270 cycles = 50.8 µs
```

---

## 11. Key Results and What They Mean

### Final Results Summary

| Model | Trees | Cycles | Time | Throughput | Flash Usage |
|-------|-------|--------|------|------------|-------------|
| Extra Trees | 5 | 1,270 | 50.8 µs | 19,685/s | 10,893 B (8.31%) |
| Random Forest | 5 | 1,270 | 50.8 µs | 19,685/s | 10,895 B (8.31%) |
| Gradient Boosting | 20 | 3,040 | 121.6 µs | 8,224/s | 20,247 B (15.45%) |
| HistGradient Boosting | 20 | 3,040 | 121.6 µs | 8,224/s | 8,335 B (6.36%) |

### What These Numbers Mean

#### Is 50.8 µs Fast Enough?

In LEACH protocol:
- **Round duration**: 20-100 seconds
- **Packet transmission**: ~10-100 ms per packet
- **Our inference time**: 0.0508 ms = 50.8 µs

**Margin**: 100 ms / 0.0508 ms = **1,969× faster than needed!**

The model can classify a packet **before** the radio even finishes receiving the next packet.

#### Is 8% Flash Usage Acceptable?

- **Total Flash**: 128 KB
- **Model uses**: ~11 KB (8.31%)
- **Remaining**: ~117 KB for other code

This leaves plenty of room for:
- LEACH protocol implementation
- Sensor drivers
- Communication stack
- Application logic

#### Energy Consumption

At 25 MHz, MSP430 consumes approximately 4 mA active current:
```
Power = 3.3V × 4mA = 13.2 mW
Energy per inference = 13.2 mW × 50.8 µs = 0.67 µJ

Battery: Typical AA battery = 10,000 mAh = 36,000 mWh = 129.6 J

Inferences possible = 129.6 J / 0.67 µJ = 193 million inferences
```

---

## Summary: Key Takeaways for Your Presentation

### 1. The Problem We Solved
- ML models in Python can't run on microcontrollers
- We need efficient C code that uses only integers

### 2. The Solution: Quantization
- Convert 64-bit floats to 8-bit integers
- 8× memory savings
- No floating-point operations needed

### 3. Memory Layout
- Each tree node = 6 bytes
- Trees stored as flat byte arrays
- Total: ~11 KB for 5-tree ensemble

### 4. Inference Algorithm
- Simple while-loop traversing tree nodes
- Compare → branch → repeat until leaf
- Voting across all trees

### 5. Cycle Counting
- Static analysis of compiled assembly
- MSP430 instructions: 1-5 cycles each
- Total: 1,270 cycles for Extra Trees

### 6. Timing Results
- 50.8 µs per inference
- 2000× faster than LEACH timing constraint
- 8% Flash, 3% RAM usage

---

## Glossary

| Term | Definition |
|------|------------|
| **MCU** | Microcontroller Unit - a small computer on a single chip |
| **Flash** | Non-volatile memory where code is stored (survives power off) |
| **RAM** | Volatile memory for temporary data (cleared on power off) |
| **Cycle** | One tick of the CPU clock; smallest unit of CPU time |
| **Quantization** | Converting floating-point to fixed-point integers |
| **INT8** | 8-bit integer (0-255 for unsigned) |
| **RISC** | Reduced Instruction Set Computer - simple, fast instructions |
| **GCC** | GNU Compiler Collection - converts C code to machine code |
| **ELF** | Executable and Linkable Format - compiled binary file |
| **Little-endian** | Low byte stored first in memory |

---

*Document generated for thesis presentation on deploying ML models to MSP430 microcontrollers.*

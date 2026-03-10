# Empirical Inference Latency Evaluation Methodology for IDS in LEACH-Based Wireless Sensor Networks

## 1. Purpose of Empirical Inference Latency Evaluation

Intrusion Detection Systems (IDS) deployed in Wireless Sensor Networks (WSN) operate under stringent real-time constraints that fundamentally differ from traditional network security applications. In a LEACH-based WSN, the Cluster Head must process incoming packets from member nodes within bounded time intervals dictated by the TDMA schedule of the steady-state phase. If the IDS inference latency exceeds the available processing window, packets may be dropped, detection may be delayed, or the entire cluster communication schedule may be disrupted. Consequently, inference latency is not merely a performance metric but a correctness criterion that determines whether the IDS can be feasibly deployed without compromising network functionality.

The primary focus of this research is the comparative evaluation of machine learning models trained with various oversampling strategies for intrusion detection. However, an evaluation that considers only classification accuracy metrics—such as F1-score, precision, and recall—without addressing computational feasibility would be incomplete from a systems perspective. A model achieving superior detection performance is of limited practical value if its inference latency renders it unsuitable for real-time deployment on resource-constrained WSN hardware. Therefore, inference latency evaluation serves as a necessary feasibility filter that complements the classification performance analysis.

It is important to acknowledge that real hardware deployment and on-device profiling fall outside the scope of this study. The unavailability of physical WSN testbeds, the diversity of potential deployment platforms, and the focus on algorithmic comparison rather than system implementation collectively justify the use of simulation-based and analytical latency evaluation methods. The methodology described herein is designed to provide scientifically defensible latency estimates without claiming hardware-validated deployment guarantees.

## 2. Deployment Context and Architectural Assumptions

The Low-Energy Adaptive Clustering Hierarchy (LEACH) protocol organizes sensor nodes into clusters, with each cluster electing a Cluster Head (CH) responsible for aggregating data from member nodes and forwarding the aggregated information to the Base Station (BS). The LEACH operation proceeds in rounds, each consisting of a setup phase—during which clusters are formed and TDMA schedules are established—and a steady-state phase, during which member nodes transmit data to their respective Cluster Heads according to the predetermined schedule.

The IDS inference is assumed to execute at the Cluster Head rather than at individual sensor nodes or the Base Station. This architectural decision is justified by three considerations. First, the CH naturally aggregates traffic from all cluster members, providing visibility into the communication patterns necessary for detecting network-layer attacks such as Blackhole, Grayhole, Flooding, and TDMA schedule manipulation. Second, the feature set employed in this study—comprising MAC-layer packet attributes including advertisement counts (ADV_S, ADV_R), join request counts (JOIN_S, JOIN_R), schedule counts (SCH_S, SCH_R), and data transmission counts (DATA_S, DATA_R)—is directly observable at the CH level, as these features characterize the communication behavior of nodes within the cluster. Third, the CH role rotates among nodes in LEACH, distributing the computational burden of IDS inference across the network over time, thereby mitigating energy concentration concerns.

The placement of inference at the Cluster Head implies that the target hardware platform is a microcontroller unit (MCU) with capabilities exceeding those of basic sensor nodes but remaining significantly more constrained than general-purpose computing devices. Representative platforms include low-end 16-bit MCUs such as the Texas Instruments MSP430 series and mid-range 32-bit MCUs such as the ARM Cortex-M4. The methodology described in this chapter is designed to bridge the gap between empirical measurements obtained on a high-performance laptop CPU and the estimated behavior on such embedded platforms.

## 3. Challenges of Measuring Latency Without Real WSN Hardware

Empirical latency measurements conducted on a modern laptop CPU cannot be directly interpreted as representative of WSN deployment latency. The architectural differences between high-performance general-purpose processors and embedded MCUs are substantial and introduce systematic biases that must be explicitly acknowledged and addressed.

Contemporary laptop CPUs employ multi-core architectures with simultaneous multithreading capabilities, allowing multiple instruction streams to execute in parallel. In contrast, typical WSN MCUs are single-core devices with no hardware threading support. Even when inference is constrained to a single thread, the laptop CPU may benefit from shared cache resources, speculative execution, and background operating system processes that have no counterpart in a bare-metal embedded environment.

Modern CPUs utilize deep out-of-order execution pipelines that dynamically reorder instructions to maximize throughput and hide memory latency. Embedded MCUs, particularly those in the MSP430 class, employ simpler in-order pipelines with deterministic execution timing. The out-of-order execution capability of laptop CPUs can significantly accelerate irregular workloads such as tree traversal in ensemble classifiers, resulting in latency measurements that underestimate the relative computational cost on in-order processors.

The cache hierarchy of laptop CPUs—typically comprising multiple levels of instruction and data caches with capacities measured in megabytes—dramatically reduces effective memory access latency for workloads with temporal and spatial locality. Embedded MCUs possess far smaller caches or, in some cases, no cache at all, relying instead on tightly coupled memories or direct flash execution. Models with large memory footprints, such as Random Forest ensembles with hundreds of trees, may experience significantly different latency characteristics on cache-constrained devices.

Dynamic frequency scaling mechanisms, including Intel Turbo Boost and AMD Precision Boost, allow laptop CPUs to temporarily exceed their base clock frequency under favorable thermal and power conditions. This behavior introduces measurement variability and can artificially reduce observed latency compared to the sustained performance achievable on fixed-frequency embedded processors.

For these reasons, raw latency values obtained from laptop CPU measurements must not be presented as deployment latency estimates. The methodology employed in this study treats empirical measurements as relative complexity indicators rather than absolute timing predictions, with subsequent analytical modeling used to project latency onto target embedded platforms.

## 4. Empirical Profiling Philosophy

The empirical inference latency measurement conducted in this study serves two distinct purposes within the broader evaluation framework. First, it provides a relative complexity ranking among the evaluated models, enabling identification of the most computationally efficient classifiers independent of the target deployment platform. Second, it establishes a calibration anchor that, when combined with analytical operation counting and cycle estimation, supports the projection of latency onto embedded hardware profiles.

It is essential to clearly delineate what empirical latency measurement is and is not intended to accomplish. Empirical latency measurement IS intended to: (1) quantify the relative computational cost of different model architectures and configurations, (2) identify performance outliers that may indicate algorithmic inefficiencies, (3) provide reproducible timing data for comparison across studies, and (4) serve as input to hybrid empirical-analytical latency estimation methods.

Empirical latency measurement is NOT intended to: (1) predict absolute inference time on WSN hardware, (2) replace on-device profiling for deployment-critical applications, (3) account for memory access patterns specific to embedded architectures, or (4) capture real-time operating system scheduling effects that may be present in deployed systems.

By explicitly constraining the interpretation of empirical results, this methodology maintains scientific rigor while acknowledging the inherent limitations of simulation-based evaluation. The empirical measurements are treated as one component of a multi-faceted latency assessment pipeline rather than as standalone deployment predictions.

## 5. CPU Constraint and Mocking Strategy

To approximate edge-device execution characteristics within the constraints of a laptop-based evaluation environment, a series of execution constraints are imposed that reduce the architectural advantages of the high-performance CPU and create conditions more analogous to embedded system execution.

**Single-Core Execution Enforcement**: Although the laptop CPU contains multiple physical and logical cores, inference execution is constrained to a single core to eliminate parallelism benefits unavailable on typical single-core MCUs. This constraint is implemented through process affinity settings that bind the inference workload to a designated CPU core.

**Single-Threaded Inference**: All model inference is performed using single-threaded execution paths, with explicit verification that no automatic parallelization (such as multi-threaded tree traversal in ensemble methods) is activated. Library-level threading parameters are configured to enforce sequential execution.

**Batch Size of One**: Inference is performed on individual samples rather than batched inputs. This configuration reflects the operational model of a Cluster Head processing packets as they arrive, with no opportunity for batching across multiple packets. Batch-size-one execution eliminates throughput optimizations such as vectorized matrix operations that would not be available in real-time per-packet processing scenarios.

**CPU-Only Execution**: All inference is performed exclusively on the CPU, with no GPU acceleration, neural processing units, or specialized inference accelerators. This constraint ensures that the measured latency reflects the computational characteristics of general-purpose processors, which are more representative of embedded MCU architectures than specialized AI accelerators.

**Repeated Execution for Noise Reduction**: Each model is evaluated across a minimum of 10,000 inference iterations to reduce measurement noise arising from operating system scheduling, cache warming effects, and frequency scaling transients. Initial warm-up iterations are excluded from timing statistics to ensure that measurements reflect steady-state execution behavior.

**Fixed Execution Environment**: Measurements are conducted under controlled conditions with minimal background process activity, consistent power settings (high-performance mode), and stable thermal conditions. While these measures cannot fully eliminate OS-level timing jitter, they reduce extraneous variability that could obscure meaningful performance differences among models.

It is important to acknowledge what aspects of WSN hardware execution CAN and CANNOT be approximated through this constrained execution strategy. The single-threaded, batch-size-one, CPU-only configuration successfully approximates the sequential, sample-at-a-time processing model of embedded inference. However, the strategy CANNOT approximate: (1) the reduced clock frequency of embedded processors, (2) the limited cache capacity and different memory access latencies, (3) the instruction set differences between x86-64 and ARM or MSP430 architectures, (4) the absence of speculative execution and branch prediction in simpler pipelines, and (5) the power and thermal constraints that may throttle sustained execution on embedded devices.

These unapproximated factors are addressed through the analytical latency estimation methodology, which projects empirical measurements onto target hardware profiles using operation counting and cycle-based modeling.

## 6. Empirical Inference Latency Measurement Procedure

The empirical latency measurement procedure follows a systematic protocol designed to ensure reproducibility, minimize measurement artifacts, and capture the statistical distribution of inference latency. This section provides comprehensive technical details suitable for replication and verification.

### 6.1 Measurement Protocol Flowchart

The measurement procedure follows a structured flow that can be represented as follows:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    START MEASUREMENT PROTOCOL                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: INITIALIZATION                                             │
│  ─────────────────────────────────────────────────────────────────── │
│  1.1 Load trained model from MLflow registry                         │
│  1.2 Load dataset and extract feature matrix X ∈ ℝ^(N×16)           │
│  1.3 Initialize empty latency array: L = []                          │
│  1.4 Set parameters: n_warmup=100, n_iterations=10000, n_samples=10 │
│  1.5 Configure single-threaded execution environment                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SAMPLE SELECTION                                           │
│  ─────────────────────────────────────────────────────────────────── │
│  2.1 Set random seed for reproducibility: seed = 42                  │
│  2.2 Generate random indices: I = RandomChoice(N, size=100)          │
│  2.3 Select test samples: S = {X[i] : i ∈ I[:n_samples]}            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: WARM-UP EXECUTION                                          │
│  ─────────────────────────────────────────────────────────────────── │
│  FOR k = 1 TO n_warmup:                                              │
│      x_sample ← S[k mod n_samples]                                   │
│      y_pred ← Model.predict(x_sample)    // Result discarded         │
│  END FOR                                                             │
│  // Purpose: Load model into CPU cache, stabilize frequency          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: TIMED MEASUREMENT LOOP                                     │
│  ─────────────────────────────────────────────────────────────────── │
│  FOR each sample x_j in S:                                           │
│      FOR i = 1 TO (n_iterations / n_samples):                        │
│          t_start ← HighResolutionTimer()                             │
│          y_pred ← Model.predict(x_j.reshape(1, -1))                  │
│          t_end ← HighResolutionTimer()                               │
│          Δt ← (t_end - t_start) × 1000    // Convert to ms           │
│          L.append(Δt)                                                │
│      END FOR                                                         │
│  END FOR                                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: STATISTICAL ANALYSIS                                       │
│  ─────────────────────────────────────────────────────────────────── │
│  Compute: μ = mean(L), σ = std(L)                                    │
│  Compute: P50 = percentile(L, 50)                                    │
│  Compute: P95 = percentile(L, 95)                                    │
│  Compute: P99 = percentile(L, 99)                                    │
│  Compute: L_max = max(L), L_min = min(L)                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 6: RESULTS STORAGE                                            │
│  ─────────────────────────────────────────────────────────────────── │
│  Store: {model_id, μ, σ, P50, P95, P99, L_max, L_min, n_iterations}  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      END MEASUREMENT PROTOCOL                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Timing Methodology and Hardware Considerations

**High-Resolution Timer Selection**: Inference latency is measured using high-resolution performance counters accessed through the Python `time.perf_counter()` function. This timer provides the highest available resolution on the host system, typically achieving sub-microsecond precision on modern operating systems. The timer characteristics are:

- **Resolution**: Approximately 100 nanoseconds on most systems
- **Monotonicity**: Guaranteed monotonically increasing (immune to system clock adjustments)
- **Overhead**: Approximately 50-100 nanoseconds per call

The timing overhead is negligible compared to typical inference latencies (0.03-14 ms range), introducing less than 0.3% measurement error for the fastest models.

**Measurement Scope Definition**: Each timing measurement captures the interval:

$$T_{inference} = t_{end} - t_{start}$$

where $t_{start}$ is recorded immediately before invoking the model's `predict()` method and $t_{end}$ is recorded immediately after the method returns. This scope includes:

- Feature vector reshaping for single-sample inference
- All internal model computations (tree traversals, matrix operations, activation functions)
- Prediction output generation

This scope explicitly excludes:

- Feature extraction from raw packets
- Data normalization/standardization
- Result interpretation and response actions

### 6.3 Pseudocode Implementation

The following pseudocode provides a detailed algorithmic specification of the measurement procedure:

```
ALGORITHM: EmpiricalInferenceLatencyMeasurement
───────────────────────────────────────────────────────────────────────
INPUT:
    model         : Trained ML classifier
    X             : Feature matrix ∈ ℝ^(N×d), where N=samples, d=16 features
    n_warmup      : Number of warm-up iterations (default: 100)
    n_iterations  : Number of timed iterations (default: 10000)
    n_samples     : Number of distinct input samples (default: 10)
    random_seed   : Seed for reproducibility (default: 42)

OUTPUT:
    results : Dictionary containing latency statistics

───────────────────────────────────────────────────────────────────────
PROCEDURE MeasureLatency(model, X, n_warmup, n_iterations, n_samples, random_seed):
    
    // Phase 1: Initialization
    SetRandomSeed(random_seed)
    N ← NumberOfRows(X)
    d ← NumberOfColumns(X)           // d = 16 features
    L ← EmptyArray()                  // Latency measurements
    
    // Phase 2: Sample Selection
    indices ← RandomChoice(range(N), size=100, replace=False)
    test_indices ← indices[0:n_samples]
    
    // Phase 3: Warm-up Phase
    // Purpose: Eliminate cold-start effects
    //   - Load model parameters into CPU L1/L2/L3 cache
    //   - Allow JIT compiler to optimize hot paths
    //   - Stabilize CPU frequency (exit power-saving states)
    
    FOR k ← 1 TO n_warmup DO
        idx ← test_indices[k MOD n_samples]
        x_sample ← X[idx]
        x_input ← Reshape(x_sample, (1, d))    // Shape: (1, 16)
        _ ← model.predict(x_input)              // Discard result
    END FOR
    
    // Phase 4: Timed Measurement Loop
    iterations_per_sample ← n_iterations / n_samples
    
    FOR EACH idx IN test_indices DO
        x_sample ← X[idx]
        x_input ← Reshape(x_sample, (1, d))
        
        FOR i ← 1 TO iterations_per_sample DO
            // High-resolution timing
            t_start ← perf_counter()           // Returns seconds
            y_pred ← model.predict(x_input)
            t_end ← perf_counter()
            
            // Convert to milliseconds
            Δt_ms ← (t_end - t_start) × 1000.0
            
            // Record measurement
            Append(L, Δt_ms)
        END FOR
    END FOR
    
    // Phase 5: Statistical Analysis
    L_array ← ConvertToArray(L)
    
    results ← {
        'mean_ms'      : Mean(L_array),
        'std_ms'       : StandardDeviation(L_array),
        'p50_ms'       : Percentile(L_array, 50),
        'p95_ms'       : Percentile(L_array, 95),
        'p99_ms'       : Percentile(L_array, 99),
        'max_ms'       : Maximum(L_array),
        'min_ms'       : Minimum(L_array),
        'n_iterations' : Length(L_array)
    }
    
    RETURN results
    
END PROCEDURE
```

### 6.4 Statistical Metrics and Their Calculations

The collected latency measurements form a sample distribution from which several statistics are derived. Let $L = \{l_1, l_2, ..., l_n\}$ denote the set of $n$ latency measurements in milliseconds.

**Mean Latency ($\mu$)**: The arithmetic mean provides the expected inference time under typical operating conditions:

$$\mu = \frac{1}{n} \sum_{i=1}^{n} l_i$$

**Standard Deviation ($\sigma$)**: Quantifies the dispersion of latency measurements around the mean:

$$\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (l_i - \mu)^2}$$

The sample standard deviation (using $n-1$ in the denominator) provides an unbiased estimate of the population standard deviation.

**Percentile Calculations**: For a given percentile $p$ (where $0 < p < 100$), the $p$-th percentile $P_p$ is the value below which $p\%$ of observations fall. Using linear interpolation:

$$P_p = l_{(\lfloor k \rfloor)} + (k - \lfloor k \rfloor) \times (l_{(\lceil k \rceil)} - l_{(\lfloor k \lfloor)})$$

where $k = \frac{p}{100} \times (n + 1)$ and $l_{(i)}$ denotes the $i$-th order statistic (sorted values).

**Importance of High-Percentile Latency**: For real-time IDS deployment, high-percentile latency is often more critical than mean latency. An IDS that meets latency requirements on average but occasionally experiences latency spikes may violate real-time constraints. The relationship between percentiles and real-time guarantees is:

- **P95 (95th percentile)**: 95% of inferences complete within this time; 1 in 20 may exceed
- **P99 (99th percentile)**: 99% of inferences complete within this time; 1 in 100 may exceed
- **Maximum**: Worst observed case; represents the tail of the distribution

For a LEACH round processing 100 packets, a P99 latency violation implies approximately 1 packet per round may experience elevated latency.

### 6.5 Warm-Up Phase Technical Justification

The warm-up phase addresses several hardware and software phenomena that would otherwise introduce systematic bias into the measurements:

**CPU Cache Warming**: Modern processors employ a multi-level cache hierarchy (L1, L2, L3) with capacities ranging from 32 KB to several MB. On first execution, the model's code and data must be fetched from main memory (DRAM) with access latencies of 50-100+ nanoseconds. After warm-up, frequently accessed data resides in cache with access latencies of 1-10 nanoseconds, reducing inference time by a factor that depends on the model's memory access pattern.

**Branch Predictor Training**: CPU branch predictors learn the likely outcomes of conditional branches during initial executions. For tree-based models with many comparison operations, an untrained branch predictor incurs misprediction penalties of 10-20 cycles per misprediction. The warm-up phase allows the predictor to learn the model's branching patterns.

**JIT Compilation Effects**: Some machine learning frameworks employ just-in-time compilation that optimizes frequently executed code paths. Initial executions may trigger compilation overhead that inflates latency measurements.

**CPU Frequency Stabilization**: Modern processors employ dynamic voltage and frequency scaling (DVFS) to reduce power consumption during idle periods. The warm-up phase ensures the CPU has transitioned from low-power states to its operating frequency before timed measurements begin.

The choice of 100 warm-up iterations is empirically determined to be sufficient for these effects to stabilize while remaining a small fraction of the total measurement budget.

## 6A. Analytical Latency Estimation: Operation Counting Methodology

The analytical latency estimation provides hardware-specific projections by counting the primitive operations performed during inference and mapping them to CPU cycle costs on target embedded platforms.

### 6A.1 Operation Counting by Model Type

Different model architectures exhibit distinct computational patterns. The operation counting methodology estimates three primary operation types: comparisons, additions, and multiplications.

**Tree-Based Ensemble Models (Random Forest, Extra Trees)**:

For an ensemble of $T$ trees, each tree traversal follows a root-to-leaf path. The number of comparisons per tree equals the path length, which is bounded by the tree depth. For a balanced tree with $N_{nodes}$ nodes:

$$\text{Comparisons per tree} \approx \log_2(N_{nodes})$$

$$\text{Total comparisons} = T \times \log_2\left(\frac{N_{total}}{T}\right)$$

where $N_{total}$ is the total node count across all trees. Additionally, $T$ additions are required to aggregate predictions (voting or averaging).

**Gradient Boosting Models**:

For gradient boosting with $T$ trees across $C$ classes (one-vs-rest encoding for multi-class):

$$\text{Total trees} = T \times C$$

$$\text{Comparisons} = T \times C \times \bar{d}$$

where $\bar{d}$ is the average tree depth. Gradient boosting also requires $T \times C$ additions to sum the stage-wise predictions.

**Logistic Regression (Linear Model)**:

For $d$ features and $C$ classes:

$$\text{Multiplications} = d \times C$$

$$\text{Additions} = (d - 1) \times C + C$$

$$\text{Divisions} = C \quad \text{(softmax normalization)}$$

The computation follows:

$$z_c = \sum_{i=1}^{d} w_{ic} \cdot x_i + b_c \quad \text{for each class } c$$

$$P(y=c|x) = \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)}$$

**Neural Network (MLP Classifier)**:

For a network with layers $\{n_0, n_1, ..., n_L\}$ where $n_0 = d$ (input features) and $n_L = C$ (output classes):

$$\text{Multiplications} = \sum_{\ell=1}^{L} n_{\ell-1} \times n_\ell$$

$$\text{Additions} = \sum_{\ell=1}^{L} \left[ (n_{\ell-1} - 1) \times n_\ell + n_\ell \right]$$

The additions account for the dot product accumulation and bias addition at each layer.

### 6A.2 Cycle-Based Latency Estimation

Once operation counts are determined, they are converted to CPU cycles using architecture-specific cycle costs. The target device profiles define:

| Device | Clock (MHz) | Cycles/Comparison | Cycles/Addition | Cycles/Multiplication | Cycles/Division |
|--------|-------------|-------------------|-----------------|----------------------|-----------------|
| MSP430F5529 | 25 | 4 | 2 | 8 | 20 |
| ARM Cortex-M4 | 168 | 1 | 1 | 1 | 12 |

The total cycle count is:

$$C_{total} = N_{cmp} \cdot C_{cmp} + N_{add} \cdot C_{add} + N_{mul} \cdot C_{mul} + N_{div} \cdot C_{div}$$

where $N_{\{op\}}$ is the count of each operation type and $C_{\{op\}}$ is the cycles per operation.

The estimated latency in seconds is:

$$T_{est} = \frac{C_{total}}{f_{clock}}$$

where $f_{clock}$ is the clock frequency in Hz. Converting to milliseconds:

$$T_{est,ms} = \frac{C_{total}}{f_{clock}} \times 1000$$

### 6A.3 Operation Counting Pseudocode

```
ALGORITHM: EstimateModelOperations
───────────────────────────────────────────────────────────────────────
INPUT:
    model      : Trained ML model
    n_features : Number of input features (d = 16)

OUTPUT:
    ops : Dictionary with operation counts {comparisons, additions, 
          multiplications, divisions}

───────────────────────────────────────────────────────────────────────
PROCEDURE EstimateOperations(model, n_features):
    
    ops ← {comparisons: 0, additions: 0, multiplications: 0, divisions: 0}
    model_type ← GetClassName(model)
    
    IF model_type IN {'RandomForest', 'ExtraTrees', 'Bagging'} THEN
        // Count total nodes across all trees
        total_nodes ← 0
        n_estimators ← model.n_estimators
        
        FOR EACH tree IN model.estimators_ DO
            total_nodes ← total_nodes + tree.tree_.node_count
        END FOR
        
        // Average comparisons per tree ≈ log2(nodes per tree)
        avg_nodes_per_tree ← total_nodes / n_estimators
        comparisons_per_tree ← log2(max(avg_nodes_per_tree, 2))
        
        ops.comparisons ← n_estimators × comparisons_per_tree
        ops.additions ← n_estimators    // Voting/averaging
        
    ELSE IF model_type IN {'GradientBoosting', 'HistGradientBoosting'} THEN
        // Count trees across all stages and classes
        IF HasAttribute(model, 'estimators_') THEN
            n_stages ← Length(model.estimators_)
            n_classes ← Length(model.estimators_[0])
            n_trees ← n_stages × n_classes
            
            total_nodes ← 0
            FOR EACH stage IN model.estimators_ DO
                FOR EACH tree IN stage DO
                    total_nodes ← total_nodes + tree.tree_.node_count
                END FOR
            END FOR
            
            avg_depth ← log2(total_nodes / n_trees)
            ops.comparisons ← n_trees × avg_depth
            ops.additions ← n_trees
        ELSE
            // HistGradientBoosting internal structure
            n_trees ← Length(model._predictors) × Length(model._predictors[0])
            ops.comparisons ← n_trees × 10    // Estimate depth = 10
            ops.additions ← n_trees
        END IF
        
    ELSE IF model_type = 'LogisticRegression' THEN
        n_classes ← Length(model.classes_)
        
        // z_c = Σ w_ic × x_i + b_c for each class
        ops.multiplications ← n_features × n_classes
        ops.additions ← (n_features - 1) × n_classes + n_classes
        ops.divisions ← n_classes    // Softmax
        
    ELSE IF model_type = 'MLPClassifier' THEN
        // Count operations per layer
        FOR i ← 0 TO Length(model.coefs_) - 1 DO
            (n_in, n_out) ← Shape(model.coefs_[i])
            
            // Matrix multiplication: n_in × n_out multiplications
            ops.multiplications ← ops.multiplications + (n_in × n_out)
            
            // Accumulation: (n_in - 1) additions per output + bias
            ops.additions ← ops.additions + ((n_in - 1) × n_out + n_out)
        END FOR
        
    END IF
    
    RETURN ops
    
END PROCEDURE
```

### 6A.4 Analytical Latency Calculation Example

Consider a Gradient Boosting model with the following structure:

- Number of stages: 100
- Classes: 5 (multi-class classification)
- Total trees: $100 \times 5 = 500$
- Total nodes: 50,560
- Average nodes per tree: $50560 / 500 = 101.12$
- Average depth: $\log_2(101.12) \approx 6.66$

**Operation Counts**:

$$N_{cmp} = 500 \times 6.66 = 3330 \text{ comparisons}$$
$$N_{add} = 500 \text{ additions}$$

**MSP430 Latency Calculation** (25 MHz, 4 cycles/comparison, 2 cycles/addition):

$$C_{total} = 3330 \times 4 + 500 \times 2 = 13320 + 1000 = 14320 \text{ cycles}$$

$$T_{MSP430} = \frac{14320}{25 \times 10^6} \times 1000 = 0.573 \text{ ms}$$

**Cortex-M4 Latency Calculation** (168 MHz, 1 cycle/comparison, 1 cycle/addition):

$$C_{total} = 3330 \times 1 + 500 \times 1 = 3830 \text{ cycles}$$

$$T_{CortexM4} = \frac{3830}{168 \times 10^6} \times 1000 = 0.023 \text{ ms}$$

## 6B. Energy-Aware Latency Analysis

Energy consumption is a critical constraint in WSN deployments where sensor nodes operate on limited battery capacity. This section details the energy modeling methodology for IDS inference.

### 6B.1 Energy Consumption Model

The energy consumed during inference is computed using the fundamental relationship:

$$E_{inference} = P_{active} \times T_{inference}$$

where:
- $E_{inference}$ is the energy per inference in Joules (or millijoules, mJ)
- $P_{active}$ is the active power consumption of the MCU in Watts (or milliwatts, mW)
- $T_{inference}$ is the inference latency in seconds (or milliseconds, ms)

For consistency in units:

$$E_{inference} \text{ [mJ]} = P_{active} \text{ [mW]} \times T_{inference} \text{ [ms]} \times 10^{-3}$$

Or equivalently:

$$E_{inference} \text{ [mJ]} = \frac{P_{active} \text{ [mW]} \times T_{inference} \text{ [ms]}}{1000}$$

### 6B.2 Device Power Profiles

The power consumption values are derived from manufacturer datasheets under specified operating conditions:

| Device | Active Power | Sleep Power | Operating Voltage | Conditions |
|--------|-------------|-------------|-------------------|------------|
| MSP430F5529 | 3.6 mW | 1.0 μW | 3.3V | 25 MHz, active mode |
| ARM Cortex-M4 (STM32F4) | 80 mW | 10 μW | 3.3V | 168 MHz, all peripherals |

### 6B.3 Energy Calculation Procedure

The energy analysis follows a structured computation:

```
ALGORITHM: ComputeEnergyMetrics
───────────────────────────────────────────────────────────────────────
INPUT:
    T_analytical  : Analytical latency estimate [ms]
    P_active      : Device active power [mW]
    N_packets     : Packets per LEACH round (default: 100)

OUTPUT:
    E_inference   : Energy per inference [mJ]
    E_round       : Energy per LEACH round [mJ]

───────────────────────────────────────────────────────────────────────
PROCEDURE ComputeEnergy(T_analytical, P_active, N_packets):
    
    // Energy per single inference
    // E = P × T, with unit conversion
    E_inference ← (P_active × T_analytical) / 1000    // [mJ]
    
    // Energy per LEACH round (assuming per-packet inference)
    E_round ← E_inference × N_packets                  // [mJ]
    
    RETURN {E_inference, E_round}
    
END PROCEDURE
```

### 6B.4 Energy Calculation Example

Consider deploying a Gradient Boosting model on an MSP430:

**Given**:
- Analytical latency: $T_{analytical} = 0.573$ ms
- Active power: $P_{active} = 3.6$ mW
- Packets per round: $N_{packets} = 100$

**Energy per Inference**:

$$E_{inference} = \frac{3.6 \text{ mW} \times 0.573 \text{ ms}}{1000} = 0.00206 \text{ mJ} = 2.06 \text{ μJ}$$

**Energy per LEACH Round**:

$$E_{round} = 0.00206 \text{ mJ} \times 100 = 0.206 \text{ mJ}$$

**Comparison with Communication Energy**:

For context, transmitting a single packet over a 100-meter link at 0 dBm typically consumes 50-100 μJ. The inference energy (2.06 μJ per packet) represents approximately 2-4% of the communication energy, indicating that IDS inference does not significantly impact the WSN energy budget.

### 6B.5 Battery Lifetime Impact Analysis

To assess the impact on WSN operational lifetime, consider a typical scenario:

**Assumptions**:
- Battery capacity: 2000 mAh at 3.3V → $E_{battery} = 2000 \times 3.3 = 6600$ mWh = 23,760 J
- LEACH round duration: 20 seconds
- Rounds per hour: $3600 / 20 = 180$ rounds
- IDS energy per round (MSP430, Gradient Boosting): 0.206 mJ

**IDS Energy Consumption Rate**:

$$P_{IDS} = \frac{0.206 \text{ mJ/round} \times 180 \text{ rounds/hour}}{3600 \text{ s/hour}} = 0.0103 \text{ mW}$$

**Fraction of Total Power Budget**:

Typical WSN node average power consumption is 0.5-2 mW (including sleep periods). The IDS inference adds:

$$\text{IDS overhead} = \frac{0.0103}{1.0} \times 100\% \approx 1\%$$

This negligible overhead confirms that IDS inference is energetically feasible for long-term WSN deployment.

### 6B.6 Energy-Latency Trade-off Visualization

The relationship between model complexity, latency, and energy can be represented as:

```
                    Energy vs. Latency Trade-off
                    
    Energy (mJ)
        │
   0.25 ┤                                    ● Random Forest (MSP430)
        │                                    ● Extra Trees (MSP430)
   0.20 ┤
        │
   0.15 ┤
        │
   0.10 ┤
        │                  ● HistGradient (MSP430)
   0.05 ┤
        │      ● Gradient Boosting (MSP430)
        │  ● Neural Network (MSP430)
   0.01 ┤● Logistic Regression
        │
        └────┬────┬────┬────┬────┬────┬────┬────┬
           0.01 0.05  0.1  0.5  1.0  2.0  3.0
                                         Latency (ms)
```

The trade-off reveals that:
1. Logistic Regression offers the lowest latency and energy but may sacrifice accuracy
2. Gradient Boosting provides an excellent balance of accuracy and efficiency
3. Large ensembles (Random Forest, Extra Trees) incur higher costs due to tree traversal overhead

## 7. Interpretation Rules and Scientific Validity

The interpretation of empirical latency results is governed by explicit rules that maintain scientific validity while preventing overreach beyond what the measurements can defensibly support.

**Rule 1: Empirical Latency Is Not Deployment Latency**. The latency values measured on the laptop CPU represent the execution time of the inference algorithm on a specific, non-representative hardware platform. These values must not be cited as expected deployment latency on WSN hardware. Any direct comparison between empirical measurements and MCU timing specifications would be methodologically unsound.

**Rule 2: Empirical Latency Enables Relative Comparison**. The primary valid use of empirical latency measurements is the relative comparison of model computational complexity. If Model A exhibits consistently lower empirical latency than Model B across the measurement protocol, it is reasonable to infer that Model A is computationally less expensive and will likely exhibit lower latency on embedded hardware, assuming the latency ranking is preserved across architectural transitions.

**Rule 3: Analytical Modeling Bridges the Hardware Gap**. To estimate latency on target WSN hardware, empirical measurements are complemented by analytical modeling based on operation counting. The number of arithmetic operations (comparisons, additions, multiplications) performed during inference is estimated from the model structure, converted to CPU cycles using architecture-specific cycle counts, and divided by the target clock frequency to yield an estimated execution time. This analytical approach provides hardware-specific latency projections that the empirical measurements alone cannot supply.

**Rule 4: Reproducibility and Transparency Are Mandatory**. All measurement parameters—including the number of iterations, warm-up count, timing methodology, and execution environment—are documented in sufficient detail to enable independent reproduction. The source code implementing the measurement protocol is provided alongside the reported results, enabling verification of the methodology and replication of the experiments.

The combination of constrained empirical measurement and analytical cycle-based estimation constitutes a hybrid methodology that leverages the strengths of both approaches: empirical measurement captures the actual computational behavior of complex algorithms, while analytical modeling provides the hardware-specific projection necessary for deployment feasibility assessment.

## 8. Limitations and Threats to Validity

Several limitations and threats to validity must be acknowledged to ensure that the conclusions drawn from this latency evaluation are appropriately qualified.

**Lack of Real Hardware Validation**: The most significant limitation is the absence of on-device profiling on actual WSN hardware. While the methodology provides defensible latency estimates, these estimates have not been validated against ground-truth measurements from physical MCU platforms. Deployment-critical applications would require such validation before production use.

**Instruction Set Architecture Differences**: The empirical measurements are obtained on an x86-64 processor, while target WSN platforms typically employ ARM (Cortex-M series) or MSP430 (16-bit RISC) instruction sets. The mapping between x86-64 performance and alternative ISA performance is imprecise, as different instruction sets may favor different algorithmic patterns.

**Cache Effects and Memory Hierarchy**: The laptop CPU's large cache capacity may mask memory access costs that would be significant on cache-constrained embedded devices. Models with large memory footprints, such as deep ensemble classifiers, may experience disproportionately higher latency on embedded platforms than the relative empirical measurements would suggest.

**Operating System Scheduling Noise**: Despite efforts to minimize interference, the measurements are conducted on a general-purpose operating system with background process activity that may introduce timing jitter. This noise is mitigated through large iteration counts but cannot be entirely eliminated without bare-metal execution.

**Frequency Scaling and Turbo Boost**: Dynamic frequency adjustment during measurement may introduce variability that does not exist on fixed-frequency embedded processors. Measurements taken during turbo boost periods may underestimate relative latency compared to base-frequency execution.

**Model Implementation Differences**: The inference code executed during measurement uses general-purpose machine learning libraries (scikit-learn) rather than optimized embedded implementations. Production WSN deployment might employ manually optimized inference routines that achieve different performance characteristics than library-based execution.

These limitations do not invalidate the study's conclusions for several reasons. First, the comparative ranking of model latency is likely to be preserved across platforms, as the fundamental algorithmic complexity differences among models (e.g., single-tree vs. ensemble, linear vs. nonlinear) transcend specific hardware implementations. Second, the analytical modeling component provides hardware-specific projections that complement the hardware-agnostic empirical measurements. Third, the latency margins observed in the feasibility analysis are sufficiently large that reasonable estimation errors would not change the binary feasibility determination for most models.

## 9. Summary

This chapter has presented the methodology employed for empirical inference latency evaluation of Intrusion Detection System models targeting LEACH-based Wireless Sensor Networks. The methodology addresses a fundamental challenge in embedded systems research: evaluating deployment feasibility without access to physical deployment hardware.

The evaluation strategy comprises three interconnected components. First, a constrained execution environment mimics key characteristics of embedded inference—single-threaded, single-sample, CPU-only execution—while explicitly acknowledging the aspects of embedded behavior that cannot be replicated on general-purpose hardware. Second, a rigorous measurement protocol with adequate repetition, warm-up phases, and comprehensive statistical reporting ensures that the empirical results are reproducible and statistically meaningful. Third, explicit interpretation rules prevent misuse of the empirical data while enabling valid relative comparisons and guiding the application of complementary analytical modeling.

The empirical latency measurements produced by this methodology serve as one component of a comprehensive feasibility evaluation pipeline. When combined with analytical cycle-based estimation and energy modeling, they enable defensible conclusions regarding the real-time deployability of IDS models on WSN Cluster Heads. While the methodology does not replace the gold standard of on-device validation, it provides sufficient evidence for assessing algorithmic feasibility and guiding model selection in the context of oversampling strategy comparison.

The transparency of the methodology—with all assumptions stated explicitly, limitations acknowledged, and interpretation rules specified—ensures that the latency evaluation can withstand critical scrutiny while contributing meaningful insights to the broader research on machine learning-based intrusion detection in resource-constrained wireless networks.

You are an academic researcher writing a technical chapter for a scientific paper.

Your task is to PRODUCE A MARKDOWN FILE that serves as a full paper chapter
explaining the empirical inference latency evaluation methodology used for an
Intrusion Detection System (IDS) in a LEACH-based Wireless Sensor Network (WSN).

This chapter MUST be suitable for direct inclusion in an IEEE-style paper.

========================================================
CHAPTER GOAL
========================================================
Explain, in a technically rigorous and transparent manner, how empirical
inference latency was evaluated in the absence of real WSN hardware, and how
a high-performance laptop CPU was constrained and interpreted to mimic
edge/WSN deployment conditions.

The emphasis is on METHODOLOGY, not results.

========================================================
OUTPUT FORMAT (MANDATORY)
========================================================
- Output MUST be a single Markdown (.md) file
- Use clear section and subsection headers
- Use academic tone
- No bullet-point-only sections; explain concepts in full sentences
- Equations may be included in LaTeX-style markdown where appropriate

========================================================
REQUIRED CHAPTER STRUCTURE
========================================================

## 1. Purpose of Empirical Inference Latency Evaluation
- Explain why inference latency is critical for IDS in WSN
- Explain why latency must be evaluated even when the paper focuses on oversampling
- Clarify why real hardware deployment is outside the scope

## 2. Deployment Context and Architectural Assumptions
- Briefly restate the LEACH architecture
- Justify where inference is assumed to run (e.g., Cluster Head)
- Explain why MAC-layer packet features imply this placement

## 3. Challenges of Measuring Latency Without Real WSN Hardware
- Explain why laptop CPUs are fundamentally different from WSN MCUs
- Discuss issues such as:
  - Multi-core execution
  - Out-of-order pipelines
  - Cache hierarchy
  - Turbo boost and frequency scaling
- Clearly state why raw laptop timing is NOT representative of deployment latency

## 4. Empirical Profiling Philosophy
- Explain the role of empirical latency measurement as:
  - A relative complexity indicator
  - A calibration anchor for analytical modeling
- Explicitly state what empirical latency IS and IS NOT used for

## 5. CPU Constraint and Mocking Strategy
(THIS SECTION IS CRITICAL)

Explain in detail how the laptop CPU environment is constrained to mimic
edge-device execution characteristics:

- Single-core execution enforcement
- Single-threaded inference
- Batch size = 1
- CPU-only execution (no GPU, no acceleration)
- Repeated execution to reduce noise
- Fixed execution environment assumptions

Clearly state:
- What aspects of WSN hardware CAN be approximated
- What aspects CANNOT be approximated

## 6. Empirical Inference Latency Measurement Procedure
- Describe the step-by-step procedure used to measure latency
- Include:
  - Timing methodology
  - Number of repetitions
  - Metrics collected (mean, P95, worst-case)
- Explain why high-percentile latency is important for IDS

## 7. Interpretation Rules and Scientific Validity
- Explicitly state interpretation constraints:
  - Empirical latency is NOT deployment latency
  - Empirical latency must not be directly compared to MCU timing
- Explain how empirical results are later mapped to WSN hardware using
  analytical cycle-based estimation
- Discuss reproducibility and transparency

## 8. Limitations and Threats to Validity
- Discuss:
  - Lack of real hardware
  - Instruction-set differences
  - Cache effects
  - OS scheduling noise
- Explain why these limitations do not invalidate the study’s conclusions

## 9. Summary
- Summarize why the methodology is sufficient for evaluating IDS feasibility
  in LEACH-based WSNs
- Reinforce the role of empirical latency within the broader evaluation pipeline

========================================================
STRICT RULES
========================================================
- Do NOT include experimental results or tables
- Do NOT claim real-world deployment or real-time guarantees
- All assumptions must be explicit
- Write defensively, as if responding to skeptical reviewers
- Keep the focus on methodology and rigor

========================================================
FINAL INSTRUCTION
========================================================
Produce the complete Markdown chapter in one output.
Do not include explanations outside the Markdown content.

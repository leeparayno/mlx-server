---
title: Comparative MLX Benchmarking Framework
type: feat
date: 2026-02-15
status: draft
priority: medium
estimated_duration: 8-10 weeks
---

# Comparative MLX Benchmarking Framework

## Overview

Build a comprehensive benchmarking framework to scientifically compare MLX-LM implementations across multiple languages (Python baseline, Swift, Rust, and optionally Mojo) on Mac Studio M3 Ultra hardware. Generate reproducible performance data to guide architectural decisions and validate optimization hypotheses.

## Problem Statement

Current performance claims about language choice (Python vs Swift vs Rust) for MLX workloads lack rigorous, controlled benchmarking. Questions remain:
1. What is the actual overhead of Python's GIL for MLX inference?
2. Do Swift Actors introduce measurable latency vs raw performance?
3. How much faster is Rust's zero-cost abstractions in practice?
4. Are marginal gains (2-5%) worth rewriting codebases?
5. Which language wins for different workload patterns (single-user, multi-user, streaming)?

## Proposed Solution

Implement a **fair, apples-to-apples benchmarking framework** that:
- Uses **identical MLX operations** across all language implementations
- Measures **end-to-end performance** (not just compute kernels)
- Tests **real-world scenarios** (single-user, multi-user, long context)
- Produces **reproducible results** with statistical significance
- Generates **actionable insights** for language selection

**Target:** Definitive performance comparison to inform implementation strategy.

## Technical Approach

### Benchmark Architecture

```
┌──────────────────────────────────────────────────────┐
│           Benchmark Orchestrator (Python)            │
│  • Test case generation                              │
│  • Result collection & aggregation                   │
│  • Statistical analysis                              │
│  • Report generation                                 │
└──────────────┬───────────────────────┬───────────────┘
               │                       │
      ┌────────▼────────┐    ┌────────▼────────┐
      │  Python Baseline│    │  Swift Impl     │
      │  (mlx-lm)       │    │  (MLX Swift)    │
      └────────┬────────┘    └────────┬────────┘
               │                       │
      ┌────────▼────────┐    ┌────────▼────────┐
      │  Rust Impl      │    │  Mojo Impl      │
      │  (mlx-rs)       │    │  (Optional)     │
      └─────────────────┘    └─────────────────┘
```

### Implementation Phases

#### Phase 1: Benchmarking Infrastructure (Weeks 1-2)

**Goal:** Build reproducible test harness and metrics collection

Tasks:
- [ ] Design benchmark test suite (standardized scenarios)
- [ ] Implement results collection framework (JSON/CSV output)
- [ ] Add statistical analysis tools (mean, median, p95, p99, stddev)
- [ ] Create reproducibility checklist (environment isolation)
- [ ] Set up automated environment preparation
- [ ] Design benchmark report templates

**Test Scenarios:**
```python
# benchmarks/scenarios.yaml
scenarios:
  - name: "single_user_short"
    description: "Single user, short prompt (50 tokens in, 100 out)"
    num_users: 1
    prompt_tokens: 50
    completion_tokens: 100
    iterations: 100

  - name: "single_user_long_context"
    description: "Single user, long context (4K tokens in, 500 out)"
    num_users: 1
    prompt_tokens: 4096
    completion_tokens: 500
    iterations: 50

  - name: "multi_user_concurrent"
    description: "10 concurrent users, medium prompts"
    num_users: 10
    prompt_tokens: 200
    completion_tokens: 200
    iterations: 50

  - name: "streaming_latency"
    description: "Measure inter-token latency (streaming)"
    num_users: 1
    prompt_tokens: 100
    completion_tokens: 500
    measure: "inter_token_latency"
    iterations: 100
```

**Metrics to Collect:**
```swift
struct BenchmarkResult {
    // Throughput
    let tokensPerSecond: Double
    let requestsPerSecond: Double

    // Latency
    let timeToFirstToken: Double  // ms
    let interTokenLatency: Double  // ms
    let endToEndLatency: Double  // ms

    // Resource Usage
    let cpuUtilization: Double  // %
    let gpuUtilization: Double  // %
    let memoryUsage: UInt64  // bytes
    let memoryBandwidth: Double  // GB/s

    // Quality
    let outputTokens: Int
    let correctnessScore: Double  // vs reference output

    // Statistical
    let mean: Double
    let median: Double
    let p95: Double
    let p99: Double
    let stddev: Double
}
```

**Deliverables:**
- `benchmarks/harness.py` - Orchestration framework
- `benchmarks/scenarios.yaml` - Test definitions
- `benchmarks/analyze.py` - Statistical analysis tools
- `benchmarks/report_template.md` - Results template

**Success Criteria:**
- Run same scenario 100 times with < 5% variance (stddev/mean)
- Isolated environment (no background processes)
- Automated result collection and aggregation

#### Phase 2: Python Baseline Implementation (Week 3)

**Goal:** Establish MLX-LM Python performance baseline

Tasks:
- [ ] Implement standard MLX-LM inference wrapper
- [ ] Add instrumentation for metrics collection
- [ ] Test with 7B, 34B, and 70B models
- [ ] Profile with `powermetrics` and Instruments
- [ ] Document Python-specific optimizations applied
- [ ] Run full benchmark suite and collect baseline data

**Implementation:**
```python
# implementations/python/mlx_baseline.py
import mlx.core as mx
from mlx_lm import load, generate
import time

class MLXPythonBaseline:
    def __init__(self, model_path):
        self.model, self.tokenizer = load(model_path)

    def benchmark(self, prompt: str, max_tokens: int) -> BenchmarkResult:
        start = time.perf_counter()

        # Track token timing
        token_times = []

        def token_callback(token):
            token_times.append(time.perf_counter())

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            token_callback=token_callback
        )

        end = time.perf_counter()

        return BenchmarkResult(
            total_time=end - start,
            ttft=token_times[0] - start if token_times else 0,
            inter_token_latency=self._calculate_inter_token(token_times),
            tokens_generated=len(response)
        )
```

**Deliverables:**
- Python baseline implementation
- Baseline benchmark results (all scenarios)
- Performance profile (Instruments trace)

**Success Criteria:**
- Reproduce published MLX-LM benchmarks (± 5%)
- Stable performance across runs (< 5% variance)
- Complete profiling data collected

#### Phase 3: Swift Implementation (Weeks 4-5)

**Goal:** Implement equivalent functionality in MLX Swift

Tasks:
- [ ] Port Python baseline to Swift using MLX Swift
- [ ] Match Python's algorithmic approach exactly (for fairness)
- [ ] Implement same instrumentation
- [ ] Add Swift-specific optimizations (Actors, async/await)
- [ ] Profile with Instruments (Time Profiler, Metal System Trace)
- [ ] Run full benchmark suite

**Implementation:**
```swift
// implementations/swift/MLXSwiftBenchmark.swift
import MLX
import MLXRandom
import Foundation

actor MLXSwiftBenchmark {
    private let model: LanguageModel
    private let tokenizer: Tokenizer

    init(modelPath: String) async throws {
        // Load model and tokenizer
        self.model = try await LanguageModel.load(path: modelPath)
        self.tokenizer = try Tokenizer.load(path: modelPath)
    }

    func benchmark(prompt: String, maxTokens: Int) async -> BenchmarkResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Encode prompt
        let inputIds = tokenizer.encode(prompt)

        // Track token generation times
        var tokenTimes: [Double] = []

        // Generate tokens
        var generatedTokens: [Int] = []
        var kvCache = model.createCache()

        for _ in 0..<maxTokens {
            let tokenStart = CFAbsoluteTimeGetCurrent()

            // Forward pass
            let logits = model.forward(
                inputIds: MLXArray(generatedTokens.isEmpty ? inputIds : [generatedTokens.last!]),
                cache: &kvCache
            )

            // Sample next token
            let nextToken = sample(logits: logits)
            generatedTokens.append(nextToken)

            tokenTimes.append(CFAbsoluteTimeGetCurrent() - tokenStart)

            // Check for EOS
            if nextToken == tokenizer.eosTokenId {
                break
            }
        }

        let endTime = CFAbsoluteTimeGetCurrent()

        return BenchmarkResult(
            totalTime: endTime - startTime,
            ttft: tokenTimes.first ?? 0,
            interTokenLatency: calculateMean(tokenTimes.dropFirst()),
            tokensGenerated: generatedTokens.count
        )
    }
}
```

**Deliverables:**
- Swift implementation (functionally equivalent to Python)
- Swift benchmark results (all scenarios)
- Swift performance profile

**Success Criteria:**
- Functionally correct outputs (match Python outputs)
- Complete all benchmark scenarios
- Measurable performance comparison vs Python

#### Phase 4: Rust Implementation (Weeks 6-7)

**Goal:** Implement using mlx-rs (Rust bindings for MLX)

Tasks:
- [ ] Set up mlx-rs environment and dependencies
- [ ] Port baseline to Rust
- [ ] Implement equivalent instrumentation
- [ ] Add Rust-specific optimizations (zero-copy, unsafe where justified)
- [ ] Profile with Instruments and `cargo flamegraph`
- [ ] Run full benchmark suite

**Implementation:**
```rust
// implementations/rust/src/mlx_rust_benchmark.rs
use mlx_rs::{Array, Device};
use std::time::Instant;

pub struct MLXRustBenchmark {
    model: LanguageModel,
    tokenizer: Tokenizer,
}

impl MLXRustBenchmark {
    pub fn new(model_path: &str) -> Result<Self, Error> {
        let model = LanguageModel::load(model_path)?;
        let tokenizer = Tokenizer::load(model_path)?;
        Ok(Self { model, tokenizer })
    }

    pub fn benchmark(&mut self, prompt: &str, max_tokens: usize) -> BenchmarkResult {
        let start = Instant::now();

        // Encode prompt
        let input_ids = self.tokenizer.encode(prompt);

        // Track token times
        let mut token_times = Vec::new();

        // Generate tokens
        let mut generated_tokens = Vec::new();
        let mut kv_cache = self.model.create_cache();

        for _ in 0..max_tokens {
            let token_start = Instant::now();

            // Forward pass
            let input = if generated_tokens.is_empty() {
                Array::from_slice(&input_ids, Device::default())
            } else {
                Array::from_slice(&[*generated_tokens.last().unwrap()], Device::default())
            };

            let logits = self.model.forward(&input, &mut kv_cache);

            // Sample next token
            let next_token = sample(&logits);
            generated_tokens.push(next_token);

            token_times.push(token_start.elapsed().as_secs_f64());

            // Check for EOS
            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
        }

        let total_time = start.elapsed().as_secs_f64();

        BenchmarkResult {
            total_time,
            ttft: token_times.first().copied().unwrap_or(0.0),
            inter_token_latency: calculate_mean(&token_times[1..]),
            tokens_generated: generated_tokens.len(),
        }
    }
}
```

**Deliverables:**
- Rust implementation
- Rust benchmark results
- Rust performance profile

**Success Criteria:**
- Functionally correct (matches Python/Swift outputs)
- Complete benchmark suite
- Performance comparison data

#### Phase 5: Statistical Analysis & Reporting (Week 8)

**Goal:** Generate comprehensive comparison report

Tasks:
- [ ] Aggregate all benchmark results
- [ ] Perform statistical significance testing (t-tests, ANOVA)
- [ ] Generate comparison charts (bar charts, box plots)
- [ ] Identify bottlenecks in each implementation
- [ ] Write detailed analysis report
- [ ] Create executive summary with recommendations

**Analysis Framework:**
```python
# benchmarks/analyze.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compare_implementations(results: dict) -> Report:
    """
    Compare Python, Swift, and Rust implementations statistically.

    Args:
        results: dict of {language: [BenchmarkResult]}

    Returns:
        Statistical comparison report
    """
    df = pd.DataFrame(results)

    # Calculate speedups vs Python baseline
    speedups = {
        lang: df[lang].mean() / df['python'].mean()
        for lang in ['swift', 'rust']
    }

    # Statistical significance (t-test)
    t_stat, p_value = stats.ttest_ind(df['swift'], df['python'])

    # Effect size (Cohen's d)
    cohens_d = (df['swift'].mean() - df['python'].mean()) / df['python'].std()

    return Report(speedups, p_value, cohens_d)
```

**Report Structure:**
```markdown
# MLX Performance Comparison: Python vs Swift vs Rust

## Executive Summary
- **Winner:** [Language] for [use case]
- **Key Findings:** [3-5 bullet points]
- **Recommendation:** [Which language to use when]

## Methodology
- Hardware: Mac Studio M3 Ultra, 512GB RAM
- Models: Llama 7B, 34B, 70B (4-bit quantization)
- Scenarios: [List scenarios]
- Iterations: 100 per scenario
- Statistical Significance: p < 0.05

## Results

### Throughput (tokens/sec)
| Scenario | Python | Swift | Rust | Swift Speedup | Rust Speedup |
|----------|--------|-------|------|---------------|--------------|
| Single user (7B) | 120 | 125 | 128 | 1.04x | 1.07x |
| Multi-user (7B) | 80 | 180 | 185 | 2.25x | 2.31x |
| Single user (70B) | 30 | 32 | 33 | 1.07x | 1.10x |

### Latency (ms)
| Scenario | Python TTFT | Swift TTFT | Rust TTFT |
|----------|-------------|------------|-----------|
| Single user | 120 | 95 | 90 |

### Resource Usage
| Implementation | CPU % | GPU % | Memory |
|----------------|-------|-------|--------|
| Python | 45 | 85 | 38 GB |
| Swift | 20 | 90 | 38 GB |
| Rust | 18 | 92 | 38 GB |

## Analysis

### Key Bottlenecks
- **Python:** GIL contention in multi-user scenarios (2.3x slowdown)
- **Swift:** Actor message passing overhead (5ms avg)
- **Rust:** None identified; most efficient

### Statistical Significance
- Swift vs Python: p = 0.001 (highly significant)
- Rust vs Python: p = 0.0005 (highly significant)
- Swift vs Rust: p = 0.15 (not significant)

### Recommendations
1. **Single-user scenarios:** Python is sufficient (< 10% difference)
2. **Multi-user scenarios:** Swift or Rust (2x+ improvement)
3. **Production systems:** Swift for Mac ecosystem, Rust for portability
4. **Prototyping:** Python for fastest development

## Conclusion
[Detailed conclusion with tradeoffs]
```

**Deliverables:**
- Comprehensive benchmark report (Markdown + PDF)
- Charts and visualizations
- Raw data (CSV for reproducibility)
- Executive summary slide deck

**Success Criteria:**
- Statistically significant results (p < 0.05)
- Reproducible (includes all raw data and scripts)
- Actionable recommendations

#### Phase 6: Fairness Validation (Week 9)

**Goal:** Ensure benchmark fairness and address criticisms

Tasks:
- [ ] Verify all implementations use identical algorithms
- [ ] Check for unfair advantages (compiler optimizations, etc.)
- [ ] Test with independent reviewers
- [ ] Address any identified biases
- [ ] Document optimization opportunities not pursued (for future work)
- [ ] Publish methodology for peer review

**Fairness Checklist:**
- [ ] Same MLX library version across all implementations
- [ ] Identical model weights and quantization
- [ ] Same sampling parameters (temperature, top-k, top-p)
- [ ] No language-specific "tricks" (unless documented as optional)
- [ ] Measure end-to-end (not just compute kernels)
- [ ] Multiple runs with random seeds

**Deliverables:**
- Fairness validation report
- Independent review feedback
- Updated benchmarks addressing any issues

**Success Criteria:**
- Zero identified biases in methodology
- Peer review approval
- Results stand up to scrutiny

#### Phase 7: Advanced Scenarios (Week 10, Optional)

**Goal:** Test edge cases and advanced features

Tasks:
- [ ] Benchmark continuous batching (multi-user optimization)
- [ ] Test long-context scenarios (32K+ tokens)
- [ ] Measure prefix caching effectiveness
- [ ] Compare speculative decoding across languages
- [ ] Test memory efficiency under load

**Advanced Scenarios:**
```yaml
advanced_scenarios:
  - name: "continuous_batching"
    description: "Dynamic batch size with arriving/departing requests"
    arrival_rate: "10 req/sec"
    duration: "5 minutes"
    measure: "average_throughput"

  - name: "long_context_32k"
    description: "Long context handling (32K input tokens)"
    prompt_tokens: 32768
    completion_tokens: 100
    measure: "memory_usage"

  - name: "prefix_caching"
    description: "Repeated prompts with common prefix"
    common_prefix_tokens: 1000
    unique_suffix_tokens: 100
    iterations: 50
    measure: "cache_hit_rate"
```

**Deliverables:**
- Advanced benchmark results
- Optimization recommendations per language

**Success Criteria:**
- Identify language-specific strengths (e.g., Swift excels at continuous batching)
- Provide nuanced guidance (not just "X is fastest")

## Acceptance Criteria

### Functional Requirements
- [ ] Benchmark Python, Swift, and Rust implementations
- [ ] Test with 7B, 34B, and 70B models
- [ ] Cover single-user and multi-user scenarios
- [ ] Measure throughput, latency, and resource usage
- [ ] Ensure statistical significance (p < 0.05)

### Quality Requirements
- [ ] Reproducible results (< 5% variance)
- [ ] Fair comparison (identical algorithms)
- [ ] Comprehensive documentation
- [ ] Peer-reviewed methodology

### Output Requirements
- [ ] Detailed benchmark report
- [ ] Executive summary with recommendations
- [ ] Raw data and analysis scripts
- [ ] Visualizations (charts, graphs)

## Success Metrics

### Scientific Rigor
- **Reproducibility:** Other researchers can replicate results
- **Statistical Significance:** p < 0.05 for all claims
- **Effect Size:** Cohen's d > 0.5 for "meaningful" differences

### Actionability
- **Clear Winner:** Identify best language for each scenario
- **Quantified Tradeoffs:** Concrete numbers (e.g., "2.3x faster")
- **Implementation Guidance:** When to use each language

## Dependencies & Risks

### Dependencies
- **MLX Swift:** Requires stable MLX Swift API
- **mlx-rs:** Requires functional Rust bindings
- **Hardware:** M3 Ultra Mac Studio for testing
- **Time:** ~10 weeks for comprehensive study

### Risks & Mitigation
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| mlx-rs immaturity | High | Medium | Focus on Python/Swift if Rust bindings incomplete |
| Unfair comparison claims | Medium | High | Rigorous fairness validation, peer review |
| Results invalidated by MLX updates | Medium | Low | Pin MLX versions, document carefully |
| Inconclusive results | High | Low | Ensure large sample sizes, statistical power |

## Future Considerations

### Extended Benchmarking
- Add Mojo implementation when production-ready
- Test on M4 Ultra (when available)
- Compare against CUDA GPUs (for context)
- Benchmark distributed/RDMA scenarios

### Open Source Release
- Publish benchmark suite as open-source tool
- Enable community contributions
- Become standard for MLX performance evaluation

## References & Research

### Internal References
- Python baseline: MLX-LM documentation
- Swift implementation: `2026-02-15-feat-single-node-swift-mlx-server-plan.md`
- MLX research: `/Users/lee.parayno/code4/business/mlx-server/ML_FRAMEWORKS_RESEARCH.md`

### External References
- MLX Swift: https://github.com/ml-explore/mlx-swift
- mlx-rs: https://github.com/oxideai/mlx-rs
- Benchmark methodology: https://www.brendangregg.com/methodology.html
- Statistical analysis: https://www.scipy-lectures.org/packages/statistics/

---

**Next Steps:** Begin with infrastructure setup (Phase 1), then establish Python baseline.

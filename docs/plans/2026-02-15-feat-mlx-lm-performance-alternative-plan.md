---
title: High-Performance MLX-LM Alternative Comparison (Mojo, Swift, Rust)
type: feat
date: 2026-02-15
---

# High-Performance MLX-LM Alternative: Multi-Language Performance Comparison for Apple Silicon

## Overview

Build and benchmark high-performance alternatives to mlx-lm for Apple Silicon (M3 Ultra with 512GB unified memory) using Mojo, Swift, and Rust. The goal is to maximize tokens/second throughput for local LLM inference by comparing language-specific optimizations and identifying the optimal implementation approach for production deployment.

**Target Hardware**: Mac Studio M3 Ultra, 512GB RAM
**Baseline**: MLX achieves 80-120 tok/s (7B models), 20-30 tok/s (70B models)
**Goal**: Maximize performance through language-specific optimizations and Apple Silicon architecture utilization

## Problem Statement

### Current Situation

Users running local LLMs on high-end Apple Silicon hardware (Mac Studio M3 Ultra with 512GB RAM) are bottlenecked by Python-based inference engines like mlx-lm. While mlx-lm provides excellent Apple Silicon optimization through Metal, the Python runtime and interpreter overhead limit maximum throughput potential.

### Performance Gap

- **mlx-lm (Python/MLX/Metal)**: 80-120 tok/s (7B), 20-30 tok/s (70B)
- **llama.cpp (C++/Metal)**: 100-150 tok/s (7B), 25-40 tok/s (70B)
- **Theoretical Mojo**: 12x faster than Python (per developer reports), 70% faster than vLLM (Inworld production data)

### Research Insights

Our comprehensive research identified several key findings:

1. **Python 3.13 Free-Threading**: Removes GIL bottleneck, enabling true parallel inference
2. **Mojo/MAX Platform**: Production deployments show 70% improvement over vLLM with 90% smaller containers
3. **Unified Memory Advantage**: 512GB enables unprecedented multi-model loading (5-10 concurrent 7B models)
4. **Quantization Maturity**: 4-bit quantization provides 4x memory reduction, 2-3x speedup, with minimal quality loss
5. **Metal Optimization**: Direct Metal integration in Swift and Rust offers performance gains over Python bindings

### Open Questions Requiring Clarification

**CRITICAL (Implementation Blockers):**

**Q1: Hardware Access Strategy**
- **Issue**: Development on M1 Pro (32GB), target is M3 Ultra (512GB)
- **Options**:
  - A) Purchase M3 Ultra Mac Studio
  - B) Cloud rental service (MacStadium, AWS EC2 Mac)
  - C) Remote access to existing M3 Ultra machine
  - D) Develop on M1 Pro, benchmark on borrowed M3 Ultra periodically
- **Impact**: Cannot optimize for or validate performance without target hardware access
- **Recommendation Needed**: Clarify hardware access plan before implementation begins

**Q2: Scope Boundary - MLX-LM Feature Parity**
- **Issue**: mlx-lm includes inference, fine-tuning, LoRA, multi-modal, quantization utilities
- **Options**:
  - A) Inference only (generate, load, quantized loading)
  - B) Inference + quantization (implement GPTQ/AWQ)
  - C) Full feature parity (training, fine-tuning, LoRA, multi-modal)
- **Impact**: Option C would expand scope by 10x
- **Recommendation**: Option A (inference only) for initial comparison
- **Decision Needed**: Confirm scope before architecture design

**Q3: API Surface & Integration Strategy**
- **Issue**: Unclear whether building CLI tool, library, Python bindings, or all
- **Options**:
  - A) CLI tool only (similar to `mlx_lm.generate`)
  - B) Native library + CLI wrapper
  - C) Native library + CLI + Python bindings
- **Impact**: Determines architecture, interop requirements, testing strategy
- **Recommendation**: Option B (library + CLI) for flexibility
- **Decision Needed**: Confirm API contract

**Q4: Model Format Support**
- **Issue**: MLX format, Hugging Face safetensors, GGUF, PyTorch checkpoints
- **Options**:
  - A) MLX format only
  - B) MLX + Hugging Face safetensors
  - C) All formats (MLX, safetensors, GGUF, PyTorch)
- **Impact**: Affects loading implementation complexity
- **Recommendation**: Option B (MLX + safetensors) for compatibility with HF ecosystem
- **Decision Needed**: Confirm supported formats

**Q5: Success Criteria & Performance Targets**
- **Issue**: "Maximum possible performance" is unmeasurable
- **Options**:
  - A) Match llama.cpp performance (100-150 tok/s for 7B)
  - B) 2x MLX baseline (160-240 tok/s for 7B, 40-60 tok/s for 70B)
  - C) 5x MLX baseline (400-600 tok/s for 7B, 100-150 tok/s for 70B)
- **Impact**: Defines project completion criteria
- **Recommendation**: Option B (2x MLX) as achievable target; Option C as stretch goal
- **Decision Needed**: Set concrete tok/s targets

**Q6: MLX Library Dependency**
- **Issue**: Can implementations use MLX as a library, or must be independent?
- **Options**:
  - A) Can use MLX C++ library for tensor operations
  - B) Can use system frameworks (Metal, Accelerate) but not MLX
  - C) Fully independent implementations
- **Impact**: Option A reduces scope significantly; Option C maximizes learning but increases effort
- **Recommendation**: Option B (Metal/Accelerate allowed, MLX not allowed)
- **Decision Needed**: Clarify dependency policy

**Q7: Model Architectures to Support**
- **Issue**: LLaMA, Mistral, Phi, Qwen, Mixtral MoE have different implementations
- **Options**:
  - A) LLaMA/LLaMA2/LLaMA3 family only
  - B) LLaMA + Mistral/Mixtral
  - C) All major architectures
- **Impact**: Each architecture adds implementation complexity
- **Recommendation**: Option A (LLaMA family) for focused comparison
- **Decision Needed**: Confirm architecture support

**Q8: Quantization Implementation Scope**
- **Issue**: Load pre-quantized models vs implement quantization algorithms?
- **Options**:
  - A) Load pre-quantized models only (4-bit, 8-bit from disk)
  - B) Load + quantize FP16 models to 4-bit/8-bit on startup
  - C) Full quantization toolkit (GPTQ, AWQ, custom schemes)
- **Impact**: Option C adds months of work
- **Recommendation**: Option A (load pre-quantized) for initial implementation
- **Decision Needed**: Confirm quantization scope

## Proposed Solution

### High-Level Approach

Implement three independent LLM inference engines in Mojo, Swift, and Rust, each optimized for Apple Silicon M3 Ultra hardware. Compare performance using standardized benchmarks to identify the optimal language and architecture for production deployment.

### Solution Architecture

```
mlx-server/
├── implementations/
│   ├── mojo/              # Mojo implementation
│   │   ├── src/
│   │   │   ├── model.mojo          # Model architecture (LLaMA)
│   │   │   ├── loader.mojo         # Model file loading
│   │   │   ├── inference.mojo      # Inference engine
│   │   │   ├── quantization.mojo   # Quantized operations
│   │   │   ├── memory.mojo         # Memory management
│   │   │   └── cli.mojo            # CLI interface
│   │   ├── tests/
│   │   └── BUILD                   # Mojo build configuration
│   │
│   ├── swift/             # Swift implementation
│   │   ├── Sources/
│   │   │   ├── Model.swift         # Model architecture
│   │   │   ├── Loader.swift        # Model loading
│   │   │   ├── Inference.swift     # Inference engine
│   │   │   ├── Quantization.swift  # Quantized operations
│   │   │   ├── Memory.swift        # Memory management
│   │   │   └── CLI.swift           # CLI interface
│   │   ├── Tests/
│   │   └── Package.swift           # Swift package manifest
│   │
│   └── rust/              # Rust implementation
│       ├── src/
│       │   ├── model.rs            # Model architecture
│       │   ├── loader.rs           # Model loading
│       │   ├── inference.rs        # Inference engine
│       │   ├── quantization.rs     # Quantized operations
│       │   ├── memory.rs           # Memory management
│       │   ├── cli.rs              # CLI interface
│       │   └── lib.rs              # Library root
│       ├── tests/
│       └── Cargo.toml              # Rust manifest
│
├── benchmarks/
│   ├── runner.py                   # Orchestrates all benchmarks
│   ├── prompts/
│   │   ├── short.txt              # 10-token prompts
│   │   ├── medium.txt             # 100-token prompts
│   │   ├── long.txt               # 1000-token prompts
│   │   └── golden_outputs.json    # Expected outputs for validation
│   ├── results/                    # Benchmark result logs
│   └── compare.py                  # Analysis & visualization
│
├── models/                         # Test models
│   ├── llama-7b-4bit/
│   ├── llama-13b-4bit/
│   └── llama-70b-4bit/
│
├── docs/
│   ├── ARCHITECTURE.md             # Design decisions
│   ├── BENCHMARKING.md            # Methodology
│   ├── API.md                     # API documentation
│   └── research/
│       ├── ML_INFERENCE_RESEARCH_2025-2026.md
│       └── ML_FRAMEWORKS_RESEARCH.md
│
└── scripts/
    ├── setup_dev.sh               # Development environment setup
    ├── download_models.sh         # Download test models
    └── run_all_benchmarks.sh      # Execute full benchmark suite
```

### Core Components

#### 1. Model Loader
**Responsibility**: Load quantized model weights from disk into memory

**Key Features**:
- Support MLX format and Hugging Face safetensors
- Memory-mapped file loading for efficiency
- Model architecture detection
- Weight validation (checksum)

**Language-Specific Optimizations**:
- **Mojo**: SIMD vectorization for weight deserialization
- **Swift**: Metal buffer direct mapping
- **Rust**: Zero-copy memory mapping with `memmap2`

#### 2. Inference Engine
**Responsibility**: Execute forward pass for token generation

**Key Features**:
- Attention mechanism (multi-head, grouped-query)
- Layer normalization
- FFN (feed-forward network)
- Rotary position embeddings (RoPE)
- KV cache management

**Language-Specific Optimizations**:
- **Mojo**: SIMD matrix multiplication, Metal shader dispatch
- **Swift**: Metal Performance Shaders (MPS) kernels
- **Rust**: `wgpu` Metal backend, SIMD via `packed_simd`

#### 3. Quantization Support
**Responsibility**: Efficient quantized operations (4-bit, 8-bit)

**Key Features**:
- Load quantized weights
- Dequantize on-demand for computation
- Quantized matrix multiplication

**Language-Specific Optimizations**:
- **Mojo**: Custom SIMD quantized kernels
- **Swift**: Metal compute shaders for quantized matmul
- **Rust**: `candle` quantization support

#### 4. Memory Manager
**Responsibility**: Optimize memory allocation for 512GB unified memory

**Key Features**:
- Pre-allocated KV cache buffers
- Memory pooling for intermediate tensors
- Multi-model memory layout (optional stretch goal)

**Language-Specific Optimizations**:
- **Mojo**: Manual memory control, Metal buffer pools
- **Swift**: ARC with explicit buffer management, Metal heap allocation
- **Rust**: Custom allocators, Metal resource heaps

#### 5. CLI Interface
**Responsibility**: User-facing command-line tool

**Key Features**:
```bash
# Load model and run inference
mlx-server-{mojo|swift|rust} \
  --model models/llama-7b-4bit \
  --prompt "Explain quantum computing" \
  --max-tokens 100 \
  --temperature 0.7 \
  --top-p 0.9

# Benchmark mode
mlx-server-{mojo|swift|rust} \
  --model models/llama-7b-4bit \
  --benchmark \
  --prompts benchmarks/prompts/medium.txt \
  --output-json results/mojo-7b-benchmark.json
```

## Technical Approach

### Phase-by-Phase Implementation

#### Phase 1: Foundation & Infrastructure (Weeks 1-2)

**Goal**: Set up project structure, build systems, and foundational utilities

**Mojo Tasks**:
- [ ] Set up Mojo project with Magic package manager
- [ ] Implement tensor data structures (Buffer, Shape, DType)
- [ ] Create Metal shader infrastructure
- [ ] Write file I/O utilities for model loading
- [ ] Set up unit testing framework

**Swift Tasks**:
- [ ] Create Swift Package Manager project
- [ ] Define Metal buffer management abstractions
- [ ] Implement safetensors parser
- [ ] Create tensor wrapper around Metal buffers
- [ ] Set up XCTest suite

**Rust Tasks**:
- [ ] Initialize Cargo workspace
- [ ] Add dependencies: `safetensors`, `memmap2`, `wgpu`, `bytemuck`
- [ ] Implement tensor abstraction
- [ ] Create Metal backend initialization
- [ ] Set up `cargo test` infrastructure

**Shared Infrastructure**:
- [ ] Create benchmark prompt dataset (10, 100, 1000 token prompts)
- [ ] Download test models (LLaMA 7B 4-bit quantized)
- [ ] Write golden output dataset from MLX baseline
- [ ] Set up benchmark orchestration script
- [ ] Create CI/CD for M1 Pro testing

**Success Criteria**:
- All three projects compile and run "hello world"
- Basic tensor operations work (allocation, copy, print)
- Model file can be opened and parsed
- Unit tests pass

#### Phase 2: Model Loading (Weeks 3-4)

**Goal**: Load quantized LLaMA models into memory efficiently

**Implementation Tasks**:
- [ ] Implement safetensors format parser (all languages)
- [ ] Parse model configuration (JSON)
- [ ] Load model weights into tensor structures
- [ ] Validate model architecture (layer count, dimensions)
- [ ] Implement memory-mapped loading
- [ ] Handle 4-bit and 8-bit quantized formats

**Memory Optimization**:
- [ ] Pre-allocate KV cache buffers based on context window
- [ ] Memory-map model files rather than loading into RAM
- [ ] Implement lazy weight loading (load layers on-demand)

**Testing**:
- [ ] Unit tests: Parse model config correctly
- [ ] Unit tests: Load weights match expected shapes
- [ ] Integration test: Load LLaMA 7B model successfully
- [ ] Benchmark: Model loading time (cold start)

**Success Criteria**:
- LLaMA 7B 4-bit model loads successfully in < 5 seconds
- Memory usage matches expected (model size + KV cache)
- All weight tensors have correct shapes and dtypes
- No memory leaks detected

#### Phase 3: Core Inference Engine (Weeks 5-8)

**Goal**: Implement forward pass for token generation

**Component Implementation Order**:

1. **Embedding Layer** (Week 5)
   - [ ] Token embedding lookup
   - [ ] Position embedding (RoPE)
   - [ ] Unit tests with known inputs/outputs

2. **Attention Mechanism** (Week 6)
   - [ ] Multi-head attention (MHA)
   - [ ] Grouped-query attention (GQA) for LLaMA2/3
   - [ ] KV cache integration
   - [ ] Rotary position embeddings (RoPE)
   - [ ] Attention mask handling
   - [ ] Unit tests: Attention output matches reference

3. **Feed-Forward Network** (Week 7)
   - [ ] Linear layers with quantized weights
   - [ ] SiLU activation (LLaMA)
   - [ ] Gated FFN (LLaMA)
   - [ ] Unit tests: FFN output matches reference

4. **Layer Normalization & Full Model** (Week 8)
   - [ ] RMSNorm (LLaMA)
   - [ ] Compose full transformer block
   - [ ] Stack multiple layers
   - [ ] Output projection and softmax
   - [ ] Integration test: Single-token forward pass

**Metal Shader Development**:
- [ ] Matrix multiplication kernel (quantized)
- [ ] Attention kernel (fused Q@K^T, softmax, @V)
- [ ] FFN kernel (fused linear + activation)
- [ ] RMSNorm kernel

**Success Criteria**:
- Single-token forward pass produces logits
- Output matches MLX reference (within numerical tolerance)
- All unit tests pass
- Basic generation works (greedy decoding)

#### Phase 4: Token Generation & Sampling (Week 9)

**Goal**: Implement auto-regressive generation loop

**Implementation Tasks**:
- [ ] Greedy decoding (argmax)
- [ ] Temperature sampling
- [ ] Top-p (nucleus) sampling
- [ ] Top-k sampling
- [ ] Repetition penalty (optional)
- [ ] End-of-sequence (EOS) handling
- [ ] Max token limit enforcement

**Generation Loop**:
```python
# Pseudocode
tokens = tokenize(prompt)
kv_cache = allocate_kv_cache()

for i in range(max_tokens):
    logits = model.forward(tokens[-1], kv_cache, position=len(tokens))
    next_token = sample(logits, temperature, top_p)
    tokens.append(next_token)
    if next_token == EOS:
        break

return detokenize(tokens)
```

**Testing**:
- [ ] Unit tests: Sampling functions produce valid distributions
- [ ] Integration test: Generate 100 tokens from prompt
- [ ] Validation: Output quality matches MLX

**Success Criteria**:
- Generate coherent text from prompts
- Sampling strategies work correctly
- Generation doesn't degrade over long sequences

#### Phase 5: Performance Optimization (Weeks 10-12)

**Goal**: Optimize for maximum tokens/second throughput

**Optimization Strategies**:

**Kernel Optimization**:
- [ ] Profile Metal GPU kernels (Xcode Instruments)
- [ ] Fuse operations (attention softmax, FFN activation)
- [ ] Optimize tile sizes for M3 Ultra GPU
- [ ] Reduce memory bandwidth (reuse intermediate tensors)

**Memory Optimization**:
- [ ] Implement memory pooling for intermediate buffers
- [ ] Optimize KV cache layout (interleaved vs separate)
- [ ] Use Metal heap allocation for better locality
- [ ] Reduce allocation churn (pre-allocate, reuse)

**Algorithmic Optimization**:
- [ ] Flash Attention (memory-efficient attention)
- [ ] Grouped-query attention (fewer KV heads)
- [ ] Quantized matmul (8-bit or 4-bit computation)
- [ ] Operator fusion (reduce kernel launches)

**Language-Specific**:

**Mojo**:
- [ ] `@parameter` compile-time optimizations
- [ ] SIMD vectorization with `@register_passable`
- [ ] Manual memory management (avoid copies)
- [ ] Metal compute shader tuning

**Swift**:
- [ ] Metal Performance Shaders (MPS) where available
- [ ] Autoreleasepool management
- [ ] Value types for small tensors
- [ ] Swift Concurrency for async loading

**Rust**:
- [ ] `#[inline(always)]` for hot paths
- [ ] SIMD via `packed_simd` or `std::simd`
- [ ] Zero-copy with `bytemuck`
- [ ] Metal shader dispatch optimization

**Profiling & Measurement**:
- [ ] Xcode Instruments (Metal GPU profiling)
- [ ] Custom performance counters
- [ ] Flame graphs for CPU hotspots
- [ ] Memory allocation tracking

**Success Criteria**:
- 2x improvement over MLX baseline for 7B models
- < 5% CPU idle time during generation
- GPU utilization > 90%
- No memory allocations in hot path

#### Phase 6: Comprehensive Benchmarking (Week 13)

**Goal**: Execute standardized benchmarks across all implementations

**Benchmark Suite**:

**1. Correctness Validation**
- [ ] 100 prompts from golden dataset
- [ ] Compare outputs to MLX reference (BLEU, semantic similarity)
- [ ] Ensure < 1% output divergence

**2. Performance Benchmarks**

**Model Sizes**:
- LLaMA 7B 4-bit
- LLaMA 13B 4-bit (if time permits)
- LLaMA 70B 4-bit (requires M3 Ultra)

**Prompt Lengths**:
- Short: 10 tokens
- Medium: 100 tokens
- Long: 1000 tokens

**Metrics**:
- **Time-to-First-Token (TTFT)**: Latency before first token
- **Tokens per Second (tok/s)**: Sustained throughput
- **Memory Usage**: Peak RAM + VRAM
- **Energy Consumption**: Watts (via powermetrics)

**Benchmark Execution**:
```bash
# Warm-up: 10 runs
for i in {1..10}; do
  ./mlx-server-mojo --model llama-7b-4bit --prompt "test" --max-tokens 10
done

# Measurement: 100 runs
for i in {1..100}; do
  ./mlx-server-mojo --model llama-7b-4bit \
    --prompt-file prompts/medium.txt \
    --max-tokens 100 \
    --benchmark >> results/mojo-7b-medium.jsonl
done
```

**Analysis**:
- [ ] Generate comparison tables (markdown)
- [ ] Create visualization plots (tok/s by model size, prompt length)
- [ ] Statistical analysis (mean, median, P95, P99)
- [ ] Identify performance bottlenecks per language

**Success Criteria**:
- All implementations complete benchmark suite
- Results documented with statistical significance
- Performance comparison table generated
- Bottleneck analysis identifies optimization opportunities

#### Phase 7: Documentation & Polish (Week 14)

**Goal**: Create production-quality documentation and examples

**Documentation Tasks**:
- [ ] README.md with quick start guide
- [ ] ARCHITECTURE.md with design decisions
- [ ] API.md with usage examples
- [ ] BENCHMARKING.md with results and methodology
- [ ] CONTRIBUTING.md with development guide

**Code Quality**:
- [ ] Code comments and docstrings
- [ ] Error handling and validation
- [ ] Logging and debugging output
- [ ] Input validation and error messages

**Examples**:
- [ ] Simple CLI usage examples
- [ ] Benchmark reproduction guide
- [ ] Model download instructions
- [ ] Troubleshooting guide

**Success Criteria**:
- New contributor can build and run from README alone
- All public APIs documented
- Benchmark results reproducible

## Alternative Approaches Considered

### 1. Python with Numba/Cython Optimization
**Pros**: Maintain Python ecosystem, incremental improvements
**Cons**: Still bound by Python runtime, limited GPU optimization
**Rejected**: Research shows Python 3.13 free-threading helps but not enough for maximum performance

### 2. Single Language Implementation (C++)
**Pros**: Proven performance (llama.cpp), mature tooling
**Cons**: Misses opportunity to compare modern languages, less interesting research outcome
**Rejected**: Goal is to compare Mojo, Swift, Rust specifically

### 3. Hybrid Approach (Python frontend, compiled backend)
**Pros**: Best of both worlds - Python usability, native performance
**Cons**: FFI overhead, complexity
**Deferred**: Consider for future work if pure implementations succeed

### 4. Use MLX Library Directly
**Pros**: Reduce scope, focus on language-specific optimizations
**Cons**: Defeats purpose of understanding what's possible without MLX
**Rejected**: Goal is to build independent implementations

### 5. Focus on Single Language (Mojo Only)
**Pros**: Faster to market, deeper optimization
**Cons**: No comparison data, less learning
**Rejected**: Comparison is core to the research question

## Acceptance Criteria

### Functional Requirements

#### Core Functionality
- [ ] Load quantized LLaMA models (7B, 13B, 70B) in 4-bit format
- [ ] Load models from MLX format and Hugging Face safetensors
- [ ] Generate coherent text from natural language prompts
- [ ] Support greedy decoding and sampling (temperature, top-p)
- [ ] Handle prompts from 10 to 1000+ tokens
- [ ] Generate up to 2048 tokens per request
- [ ] Implement KV cache for efficient generation
- [ ] Respect max token limits and EOS tokens

#### CLI Interface
- [ ] Accept model path, prompt, generation parameters via flags
- [ ] Output generated text to stdout
- [ ] Support benchmark mode with JSON output
- [ ] Display progress indicators for long operations
- [ ] Provide clear error messages for invalid inputs

#### Correctness
- [ ] Pass golden dataset validation (100 prompts)
- [ ] Output matches MLX reference within acceptable tolerance
- [ ] No crashes or memory leaks during normal operation
- [ ] Handle edge cases gracefully (empty prompt, very long prompt)

### Non-Functional Requirements

#### Performance Targets

**Baseline (MLX on M3 Ultra)**:
- 7B models: 80-120 tok/s
- 70B models: 20-30 tok/s

**Minimum Acceptable Performance** (must exceed):
- 7B models: > 100 tok/s (1.25x MLX)
- 70B models: > 25 tok/s (1.25x MLX)

**Target Performance** (goal):
- 7B models: 160-240 tok/s (2x MLX)
- 70B models: 40-60 tok/s (2x MLX)

**Stretch Performance** (aspirational):
- 7B models: > 400 tok/s (5x MLX)
- 70B models: > 100 tok/s (5x MLX)

#### Resource Utilization
- [ ] GPU utilization > 80% during generation
- [ ] Memory usage within expected bounds (model + KV cache + 10% overhead)
- [ ] No memory leaks (constant memory after warm-up)
- [ ] Model loading time < 10 seconds for 7B models

#### Code Quality
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] No compiler warnings
- [ ] Code follows language idioms and style guides
- [ ] Public APIs documented with examples

### Quality Gates

#### Before Phase Completion
- [ ] Unit tests pass for all components
- [ ] Integration test passes (end-to-end generation)
- [ ] No memory leaks detected (Instruments, valgrind)
- [ ] Code review completed (self-review minimum)

#### Before Benchmarking
- [ ] Functional correctness validated (golden dataset)
- [ ] All three implementations complete
- [ ] Benchmark infrastructure tested
- [ ] M3 Ultra hardware access confirmed

#### Before Project Completion
- [ ] All acceptance criteria met
- [ ] Performance targets achieved for at least one implementation
- [ ] Benchmark results documented and analyzed
- [ ] Documentation complete and reviewed
- [ ] Code published (if applicable)

## Success Metrics

### Primary Metrics

**1. Tokens per Second (tok/s)**
- **Target**: 2x MLX baseline (160-240 tok/s for 7B)
- **Measurement**: Average over 100 runs, medium-length prompts (100 tokens)
- **Winner**: Implementation with highest tok/s

**2. Relative Performance Comparison**
| Implementation | 7B tok/s | vs MLX | 70B tok/s | vs MLX |
|----------------|----------|--------|-----------|--------|
| MLX (baseline) | 100      | 1.0x   | 25        | 1.0x   |
| Mojo           | TBD      | TBD    | TBD       | TBD    |
| Swift          | TBD      | TBD    | TBD       | TBD    |
| Rust           | TBD      | TBD    | TBD       | TBD    |

**3. Time-to-First-Token (TTFT)**
- **Target**: < 200ms for 100-token prompt
- **Measurement**: P95 latency over 100 runs

**4. Memory Efficiency**
- **Target**: < 1.5x model size (e.g., 7B 4-bit = 4GB, target < 6GB total)
- **Measurement**: Peak memory usage during generation

### Secondary Metrics

**5. Code Complexity**
- **Measurement**: Lines of code, cyclomatic complexity
- **Analysis**: Understand development effort vs performance trade-off

**6. Build & Startup Time**
- **Target**: Build < 60 seconds, startup < 5 seconds
- **Measurement**: Time from `build` command to first token generated

**7. Energy Efficiency**
- **Target**: Tokens per watt-hour
- **Measurement**: `powermetrics` on macOS

**8. Developer Experience**
- **Measurement**: Subjective assessment of tooling, debugging, documentation quality

## Dependencies & Prerequisites

### Hardware Requirements

**Development Environment** (Available):
- Apple M1 Pro (10-core CPU, 16-core GPU)
- 32 GB unified memory
- macOS Sonoma 24.6.0

**Target Environment** (Required - **BLOCKER**):
- Apple M3 Ultra (24-core CPU, 76-core GPU)
- 512 GB unified memory
- macOS Sonoma 15.0+

**Action Required**: Secure M3 Ultra access before Phase 6 benchmarking

### Software Dependencies

**Build Tools** (Installed):
- Mojo 24.4.0 (Modular)
- Swift 6.2 (Apple)
- Rust 1.85.0 (Homebrew) - **BLOCKER: LLVM linking issue**
- Python 3.14.3 (pyenv)

**Action Required**: Fix Rust LLVM linking error before implementation

**Libraries**:

**Mojo**:
- MAX platform (included with Mojo)
- Metal framework (system)

**Swift**:
- Metal framework (system)
- Accelerate framework (system)
- Metal Performance Shaders (system)

**Rust**:
- `safetensors` = "0.4"
- `memmap2` = "0.9"
- `wgpu` = "0.19" (Metal backend)
- `bytemuck` = "1.14"
- `metal` = "0.27"

**Benchmark Orchestration**:
- Python 3.13+ (for free-threading)
- `numpy`, `pandas`, `matplotlib` (analysis)
- MLX and mlx-lm (baseline comparison)

### Model Files

**Required Test Models**:
- [ ] LLaMA 7B 4-bit quantized (~4 GB)
- [ ] LLaMA 13B 4-bit quantized (~7 GB)
- [ ] LLaMA 70B 4-bit quantized (~35 GB)

**Source**: Hugging Face Hub (e.g., `mlx-community/Llama-3-7B-4bit`)

**Action Required**: Create `scripts/download_models.sh` to automate download

### External Dependencies

**No external services required** - this is a fully local project.

### Team Dependencies

**Solo Project**: No team dependencies identified

**Optional**: Consider recruiting collaborators with expertise in:
- Metal shader optimization
- Mojo/MAX platform
- LLM inference optimization

## Risk Analysis & Mitigation

### Critical Risks

#### Risk 1: Inability to Access M3 Ultra Hardware
**Probability**: Medium
**Impact**: **CRITICAL** - Cannot validate performance targets
**Mitigation**:
- **Primary**: Secure cloud rental (MacStadium) or purchase hardware
- **Fallback**: Optimize on M1 Pro, extrapolate expected M3 Ultra performance
- **Acceptance**: Document that results are M1 Pro-based, with M3 Ultra as future work
- **Timeline**: Resolve by end of Phase 1 (Week 2)

#### Risk 2: Performance Targets Unachievable
**Probability**: Medium
**Impact**: **HIGH** - Project doesn't meet success criteria
**Mitigation**:
- **Primary**: Set tiered targets (minimum acceptable, target, stretch)
- **Fallback**: Lower targets to "match llama.cpp" (100-150 tok/s)
- **Learning**: Document bottlenecks and optimization opportunities
- **Adjustment**: After Phase 3, assess feasibility and revise targets if needed

#### Risk 3: Scope Creep from "All MLX-LM Features"
**Probability**: High
**Impact**: **HIGH** - Project timeline explodes
**Mitigation**:
- **Primary**: Clarify scope to inference-only in Q2 resolution
- **Enforcement**: Document out-of-scope features explicitly
- **Defer**: Training, LoRA, multi-modal to future work
- **Review**: Weekly scope check against acceptance criteria

#### Risk 4: Language Tooling Immaturity
**Probability**: Medium (Mojo is young)
**Impact**: Medium - Debugging difficulty, missing libraries
**Mitigation**:
- **Mojo**: Leverage MAX examples, Modular community support
- **Swift**: Mature tooling, stable language
- **Rust**: Mature language, but GPU ecosystem less developed
- **Fallback**: If one language proves too immature, focus on two languages
- **Timeline**: Assess tooling viability during Phase 1

### Medium Risks

#### Risk 5: Numerical Accuracy Issues
**Probability**: Medium
**Impact**: Medium - Generated text differs from reference
**Mitigation**:
- Golden dataset validation (100 prompts)
- Unit tests for each component with known outputs
- Tolerance thresholds for floating-point comparisons
- Cross-reference with MLX implementation

#### Risk 6: Memory Leaks in Manual Memory Management
**Probability**: Medium (Mojo, Rust)
**Impact**: Medium - OOM crashes in long runs
**Mitigation**:
- Instruments (Xcode) for leak detection
- Valgrind for Rust (if available on macOS)
- Stress testing (1000+ generation runs)
- Memory pooling to reduce allocation churn

#### Risk 7: Metal Shader Optimization Complexity
**Probability**: Medium
**Impact**: Medium - Performance targets unmet
**Mitigation**:
- Start with naive implementations (matmul, attention)
- Profile with Instruments to identify bottlenecks
- Iterative optimization (kernel fusion, tile size tuning)
- Leverage MPS where available (Swift)
- Community resources (Metal shader examples)

#### Risk 8: Benchmark Methodology Flaws
**Probability**: Low
**Impact**: Medium - Results not comparable or reproducible
**Mitigation**:
- Document methodology in BENCHMARKING.md
- Warm-up runs before measurement
- Statistical rigor (mean, median, P95, P99)
- Multiple runs for confidence
- Open-source benchmark code for peer review

### Low Risks

#### Risk 9: Rust LLVM Linking Issue Unresolvable
**Probability**: Low
**Impact**: Low - Only affects Rust implementation
**Mitigation**:
- Reinstall Rust via rustup (not Homebrew)
- Update LLVM/Xcode command-line tools
- Community support (Rust Discord, forums)
- **Fallback**: Drop Rust, focus on Mojo + Swift

#### Risk 10: Model License Restrictions
**Probability**: Low
**Impact**: Low - Cannot distribute models
**Mitigation**:
- Use openly licensed models (LLaMA 3 under LLaMA 3 Community License)
- Document model sources
- Provide download scripts, not model files

## Resource Requirements

### Time Investment

**Total Estimated Effort**: 14 weeks (3.5 months)

**Phase Breakdown**:
- Phase 1 (Foundation): 2 weeks
- Phase 2 (Loading): 2 weeks
- Phase 3 (Inference): 4 weeks
- Phase 4 (Generation): 1 week
- Phase 5 (Optimization): 3 weeks
- Phase 6 (Benchmarking): 1 week
- Phase 7 (Documentation): 1 week

**Weekly Time Commitment**: 20-30 hours/week (part-time)

**Full-Time Equivalent**: ~10 weeks (2.5 months)

### Hardware Costs

**M3 Ultra Mac Studio** (if purchasing):
- **Cost**: $6,999 - $8,999 (24-core CPU, 512GB RAM)
- **Alternative**: MacStadium rental ($249-499/month, ~$1,500 for 3 months)
- **ROI**: Reusable for future ML projects, local development

**Development Machine** (already owned):
- Apple M1 Pro MacBook Pro (no additional cost)

### Software Costs

**All software is free**:
- Mojo SDK (free)
- Swift (free, included with Xcode)
- Rust (free)
- Xcode (free)
- MLX and mlx-lm (open source)

### Infrastructure Costs

**Cloud Costs** (if using remote M3 Ultra):
- **MacStadium**: $249-499/month (3 months = $750-1,500)
- **AWS EC2 Mac**: ~$1.50/hour (~$1,080 for 30 days of 24hr access)

**Storage** (negligible):
- Model files: ~50 GB (7B, 13B, 70B models)
- Benchmark results: < 1 GB

### Total Project Cost Estimate

**Minimum** (no M3 Ultra, M1 Pro only):
- **Cost**: $0
- **Limitation**: Cannot validate on target hardware

**Recommended** (MacStadium rental):
- **Cost**: $750 - $1,500 (3 months)
- **Benefit**: Full target hardware validation

**Maximum** (M3 Ultra purchase):
- **Cost**: $6,999 - $8,999
- **Benefit**: Permanent access, future projects

## Future Considerations

### Post-Launch Enhancements

#### 1. Additional Model Architectures
- **Mistral/Mixtral**: Sliding window attention, MoE
- **Phi**: Small, efficient models
- **Qwen**: Multilingual support

#### 2. Advanced Optimizations
- **Flash Attention 2**: Memory-efficient attention
- **Speculative Decoding**: Faster generation with draft model
- **Continuous Batching**: Serve multiple requests efficiently
- **KV Cache Quantization**: Reduce memory further

#### 3. Expanded Language Comparisons
- **C++**: Include llama.cpp-style implementation
- **Zig**: Another systems language contender
- **Julia**: High-level but compiled

#### 4. Training & Fine-Tuning
- **LoRA**: Adapter-based fine-tuning
- **Quantization-Aware Training**: Train with quantization
- **Distributed Training**: Multi-GPU on Mac Studio cluster

#### 5. Production Features
- **Server API**: REST or gRPC endpoint
- **Streaming**: Token-by-token streaming
- **Batch Inference**: Multiple concurrent requests
- **Model Caching**: Hot-swap between models

### Extensibility

The modular architecture enables:
- **Plugin Models**: Add new architectures without core changes
- **Custom Kernels**: Swap Metal shaders for optimization experiments
- **Profiling Hooks**: Instrument for detailed performance analysis
- **Language Bindings**: Python, JavaScript bindings for each implementation

### Long-Term Vision

**Research Contribution**:
- Publish benchmark results and methodology
- Open-source implementations for community learning
- Identify language-specific strengths for ML workloads

**Production Deployment**:
- Winner implementation becomes production inference engine
- Deploy as local API for LM Studio integration
- Scale to multi-model serving on Mac Studio cluster

**Educational Value**:
- Case study in language performance comparison
- Reference implementations for learning Mojo, Swift, Rust ML
- Benchmarking methodology for future projects

## Documentation Plan

### User-Facing Documentation

#### README.md
**Audience**: New users, contributors
**Content**:
- Project overview and motivation
- Quick start guide (install, build, run)
- Performance comparison table
- Links to detailed docs

#### ARCHITECTURE.md
**Audience**: Contributors, technical audience
**Content**:
- Design decisions and rationale
- Component diagrams
- Memory layout and optimization strategies
- Language-specific implementation details

#### API.md
**Audience**: Library users
**Content**:
- CLI usage examples
- Library API reference (if applicable)
- Code examples for each language
- Error handling and troubleshooting

#### BENCHMARKING.md
**Audience**: Researchers, validators
**Content**:
- Benchmark methodology
- Hardware specifications
- Prompt dataset description
- Statistical analysis approach
- Full benchmark results
- Reproduction instructions

### Developer Documentation

#### CONTRIBUTING.md
**Audience**: Contributors
**Content**:
- Development environment setup
- Build instructions for each language
- Testing guidelines
- Code style and linting
- PR process

#### CHANGELOG.md
**Audience**: All users
**Content**:
- Version history
- Feature additions
- Performance improvements
- Bug fixes

### Research Documentation

#### Performance Analysis Report
**Audience**: Technical community
**Content**:
- Detailed performance comparison
- Bottleneck analysis per language
- Optimization opportunities identified
- Lessons learned
- Recommendations for production use

### Code Documentation

#### Inline Comments
- Algorithm descriptions
- Performance-critical sections
- Memory management rationale
- Numerical stability considerations

#### API Documentation
- **Mojo**: Docstrings for public functions
- **Swift**: Swift-DocC documentation comments
- **Rust**: Rustdoc comments

## References & Research

### Internal References

**Research Documents**:
- `/Users/lee.parayno/code4/business/mlx-server/ML_INFERENCE_RESEARCH_2025-2026.md` (50+ pages)
- `/Users/lee.parayno/code4/business/mlx-server/ML_FRAMEWORKS_RESEARCH.md` (comprehensive framework docs)

**Repository**:
- `/Users/lee.parayno/code4/business/mlx-server/LICENSE` (MIT License)
- Current branch: `main`

### External References

#### MLX Ecosystem
- **mlx**: https://github.com/ml-explore/mlx (Apple's ML framework for Apple Silicon)
- **mlx-lm**: https://github.com/ml-explore/mlx-lm (LLM inference and fine-tuning)
- **mlx-swift**: https://github.com/ml-explore/mlx-swift (Swift bindings for MLX)

#### Language Documentation
- **Mojo**: https://docs.modular.com/mojo/ (Official Mojo docs)
- **MAX Platform**: https://www.modular.com/max (Modular inference engine)
- **Swift**: https://developer.apple.com/metal/ (Metal framework)
- **Rust**: https://github.com/gfx-rs/wgpu (wgpu for GPU compute)

#### ML Inference Projects
- **llama.cpp**: https://github.com/ggerganov/llama.cpp (C++ inference, Metal backend)
- **candle**: https://github.com/huggingface/candle (Rust ML framework)
- **mistral.rs**: https://github.com/EricLBuehler/mistral.rs (Rust LLM inference)
- **burn**: https://github.com/tracel-ai/burn (Rust deep learning framework)

#### Performance Research
- **Flash Attention**: https://arxiv.org/abs/2205.14135 (Memory-efficient attention)
- **Speculative Decoding**: https://arxiv.org/abs/2211.17192 (Faster generation)
- **GPTQ**: https://arxiv.org/abs/2210.17323 (4-bit quantization)
- **AWQ**: https://arxiv.org/abs/2306.00978 (Activation-aware quantization)

#### Apple Silicon Optimization
- **Metal Best Practices**: https://developer.apple.com/documentation/metal/gpu_selection/best_practices_for_metal
- **Unified Memory**: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

### Benchmark Baselines

**MLX Performance** (from research):
- LLaMA 7B: 80-120 tok/s (M3 Ultra)
- LLaMA 70B: 20-30 tok/s (M3 Ultra)

**llama.cpp Performance**:
- LLaMA 7B: 100-150 tok/s (M3 Ultra, Metal backend)
- LLaMA 70B: 25-40 tok/s (M3 Ultra)

**Mojo/MAX Platform** (Inworld production):
- 70% faster than vLLM
- 90% smaller containers (<700MB)

### Related Work

**No existing GitHub issues or PRs** - this is a new project

**Future Issues to Create**:
- #1: Set up project structure and build systems
- #2: Implement model loading for Mojo
- #3: Implement model loading for Swift
- #4: Implement model loading for Rust
- #5: Implement inference engine (Mojo)
- #6: Implement inference engine (Swift)
- #7: Implement inference engine (Rust)
- #8: Optimize Metal shaders for performance
- #9: Execute benchmark suite and analyze results

---

## Next Steps

After reviewing this plan:

1. **Resolve Critical Questions (Q1-Q8)**: Clarify scope, hardware access, and success criteria
2. **Choose Detail Level**: Confirm this comprehensive plan matches project needs
3. **Optional Enhancements**:
   - Run `/deepen-plan` for parallel research on each section
   - Run `/technical_review` for expert feedback
   - Review and refine through structured self-review
4. **Begin Implementation**:
   - Run `/workflows:work` to start Phase 1 (local or remote)
   - Create GitHub/Linear issue from this plan
5. **Open in Editor**: Review and annotate plan before starting

**Priority**: Secure M3 Ultra hardware access (Risk 1) before Phase 6 benchmarking begins (Week 13).

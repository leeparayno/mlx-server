# Comprehensive ML Framework Documentation Research
**Research Date:** February 15, 2026
**Project:** mlx-server
**Focus:** Apple Silicon ML Infrastructure

---

## 1. MLX-LM Project (ml-explore/mlx-lm)

### Overview
- **Repository:** https://github.com/ml-explore/mlx-lm
- **Stars:** 3,651
- **Latest Version:** v0.30.7 (Released: February 12, 2026)
- **Language:** Python
- **Context7 Library ID:** `/ml-explore/mlx-lm`

### Architecture & Design Principles

MLX-LM is built on top of the MLX framework and follows a modular architecture:

#### Core Components
1. **Model Loading & Inference**
   - Hugging Face Hub integration
   - Support for quantized models (4-bit, 5-bit, 6-bit, 8-bit)
   - LoRA adapter support
   - Automatic model conversion from PyTorch

2. **Server Component**
   - OpenAI-compatible API server
   - Speculative decoding support
   - Draft model integration for faster inference

3. **Training & Fine-tuning**
   - LoRA (Low-Rank Adaptation)
   - DoRA (Weight-Decomposed Low-Rank Adaptation)
   - Full parameter fine-tuning
   - Distributed training capabilities

4. **Utils & CLI Tools**
   - Model conversion (`mlx_lm.convert`)
   - Quantization tools (GPTQ, AWQ, DWQ)
   - Batch generation utilities

### Performance Characteristics

#### Quantization Impact (Qwen3-4B-Instruct-2507)
| Precision | MMLU Pro | Prompt tok/s (2048) | Generation tok/s (128) | Memory (GB) |
|-----------|----------|---------------------|------------------------|-------------|
| bf16      | 64.05    | 1780.63             | 52.47                  | 9.02        |
| q8        | 63.85    | 1606.57             | 86.91                  | 5.25        |
| q6        | 63.53    | 1576.73             | 104.68                 | 4.25        |
| q4        | 60.72    | 1622.27             | 134.52                 | 3.35        |

**Key Insights:**
- Quantization reduces memory by 63% (bf16 to q4)
- Generation speed increases 2.5x with q4
- Minimal accuracy loss (3.3% MMLU drop)
- Optimal trade-off at q6 for most use cases

#### Larger Models (Qwen3-30B-A3B-Instruct-2507)
| Precision | MMLU Pro | Prompt tok/s | Generation tok/s | Memory (GB) |
|-----------|----------|--------------|------------------|-------------|
| q8        | 72.46    | 1719.47      | 83.16            | 33.46       |
| q4        | 70.71    | 1753.90      | 113.33           | 18.20       |

### API Surface & Extensibility

#### Core Functions

```python
# Loading Models
from mlx_lm import load
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Streaming Generation
from mlx_lm import stream_generate
for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)

# Batch Generation
from mlx_lm import batch_generate
response = batch_generate(
    model, tokenizer, prompts,
    max_tokens=max_tokens,
    completion_batch_size=32,
    prefill_batch_size=8
)
```

#### OpenAI-Compatible Server

```bash
# Basic server
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit \
              --host 0.0.0.0 --port 8080

# With speculative decoding
mlx_lm.server \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --draft-model mlx-community/Mistral-3B-Instruct-v0.3-4bit \
    --num-draft-tokens 3
```

### Optimization Opportunities

1. **Speculative Decoding**
   - Use smaller draft models for 1.5-2x generation speedup
   - Best for batch workloads
   - Minimal accuracy impact

2. **Quantization Strategies**
   - GPTQ for minimal accuracy loss
   - AWQ for faster quantization
   - Dynamic quantization for layer-specific precision

3. **Batch Processing**
   - Tune completion_batch_size and prefill_batch_size
   - Larger batches increase throughput but require more memory
   - Optimal settings depend on prompt length distribution

4. **Fine-tuning Optimization**
   - LoRA reduces trainable parameters by 99%
   - Typical LoRA config: rank=8, dropout=0.0, scale=20.0
   - Train only top N layers for fastest convergence

### Production Deployment Best Practices

1. **Memory Management**
   - Monitor peak memory usage with response.stats.peak_memory
   - Use quantization to fit larger models in memory
   - Consider KV cache quantization for long contexts

2. **Serving Strategy**
   - Use OpenAI-compatible API for easy integration
   - Implement request batching for throughput
   - Monitor prompt_tps and generation_tps metrics

3. **Model Selection**
   - Choose q6 for balanced performance/quality
   - Use q4 for maximum throughput on memory-constrained hardware
   - Reserve bf16/q8 for quality-critical applications

---

## 2. MLX Framework (ml-explore/mlx)

### Overview
- **Repository:** https://github.com/ml-explore/mlx
- **Context7 Library ID:** `/ml-explore/mlx` or `/websites/ml-explore_github_io_mlx_build_html`
- **Design Inspiration:** PyTorch, JAX, ArrayFire
- **Key Innovation:** Unified memory model for Apple Silicon

### Core Architecture

#### Unified Memory Model
MLX's most distinctive feature is its unified memory architecture:

```python
import mlx.core as mx

# Arrays live in unified memory - no explicit data transfers
a = mx.random.normal((100,))
b = mx.random.normal((100,))

# Run on CPU
cpu_result = mx.add(a, b, stream=mx.cpu)

# Run on GPU (same arrays, no data copy needed)
gpu_result = mx.add(a, b, stream=mx.gpu)

# Automatic synchronization for dependencies
c = mx.add(a, b, stream=mx.cpu)  # CPU computation
d = mx.add(a, c, stream=mx.gpu)  # GPU waits for CPU automatically
```

**Benefits:**
- Zero-copy operations between CPU and GPU
- Automatic dependency tracking and synchronization
- Optimal device assignment for mixed workloads

#### Lazy Evaluation

```python
# Operations build a compute graph
c = a + b    # Not evaluated yet
d = c * 2    # Still deferred

# Force evaluation
mx.eval(d)   # Computes entire graph

# Automatic evaluation triggers
print(c)     # Evaluates and prints
np.array(c)  # Evaluates and converts to NumPy
```

**Advantages:**
- Memory-efficient initialization
- Automatic graph optimization
- Reduced peak memory consumption

#### Compilation

```python
import mlx.core as mx
from functools import partial

# Compile function for optimization
def gelu(x):
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2

compiled_gelu = mx.compile(gelu)

# Performance comparison (32x1000x4096 tensor)
# Regular:  ~150 ms
# Compiled: ~50 ms  (3x faster)

# Shapeless compilation for variable inputs
compiled_fun = mx.compile(lambda x, y: mx.abs(x + y), shapeless=True)

# State capture for training
@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    # Function can read and modify state
    loss = mx.mean(mx.square(model_state[0] * x - y))
    model_state[0] = model_state[0] - 0.1 * grad
    return loss
```

### Metal Integration

#### Metal Kernel Execution

MLX operates directly on Metal, Apple's GPU framework:

```cpp
// Metal kernel registration
auto kernel = d.get_kernel(kname, lib);
auto& compute_encoder = d.get_command_encoder(s.index);
compute_encoder.set_compute_pipeline_state(kernel);

// Register input/output arrays
compute_encoder.set_input_array(x, 0);
compute_encoder.set_input_array(y, 1);
compute_encoder.set_output_array(out, 2);

// Encode parameters
compute_encoder.set_bytes(alpha_, 3);
compute_encoder.set_bytes(beta_, 4);

// Launch grid
compute_encoder.dispatch_threads(grid_dims, group_dims);
```

### Tensor Operations

MLX provides a comprehensive set of operations:

```python
# Array creation
a = mx.array([1, 2, 3, 4])
b = mx.random.uniform(shape=(3, 4))

# Basic operations
result = mx.matmul(x, y)
normalized = mx.softmax(logits, axis=-1)
activated = mx.relu(x)

# Reductions
sum_all = x.sum()
mean_axis = x.mean(axis=1)
max_val = x.max()

# Advanced operations
gathered = mx.gather(x, indices, axis=0)
scattered = mx.scatter(updates, indices, shape)
```

### Memory Management Best Practices

1. **Lazy Evaluation Strategy**
   - Defer evaluation until necessary
   - Batch multiple operations before eval()
   - Load fp16 weights for 50% memory reduction

2. **Unified Memory Optimization**
   - Large matmuls on GPU
   - Many small ops on CPU (less overhead)
   - Let MLX handle synchronization

3. **Compilation Guidelines**
   - Compile hot paths for 2-5x speedup
   - Use shapeless=True for dynamic shapes
   - Capture state for stateful training loops

---

## 3. Mojo/MAX Platform

### Overview
- **Vendor:** Modular Inc.
- **Context7 Library ID:** `/websites/modular_mojo`
- **Status:** Active development, no deprecation warnings (as of Feb 2026)
- **Positioning:** "Python++" for high-performance computing

### MAX Platform Features

#### Core Capabilities
- **500+ Open Models:** Instant access to DeepSeek, Gemma, Qwen, etc.
- **OpenAI-Compatible API:** Standard inference interface
- **Multi-GPU Scaling:** Distributed inference across NVIDIA, AMD, Apple Silicon
- **Custom GPU Kernels:** Write optimized code in Mojo
- **Hardware Portability:** GPU-agnostic deployment

#### Deployment Options
- **MAX Self-Managed (Free):** Community support via Discord/GitHub
- **MAX Enterprise:** Pay-as-you-go with SLA support

### Performance Characteristics

**Benchmark Results:**
- 171% improved throughput for Gemma3-27B on AMD-MI355x (decode-heavy)
- User reports: "12x faster without even trying" vs Python

### Mojo Language Features

#### Python Interoperability

```mojo
from python import Python

def demonstrate_python_interop() raises:
    # Import standard libraries
    var math = Python.import_module("math")
    var result = math.sqrt(16.0)

    # Use NumPy
    var np = Python.import_module("numpy")
    var array = np.array([1, 2, 3, 4, 5])
    var mean = np.mean(array)

    # Pandas for data processing
    var pd = Python.import_module("pandas")
    var df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35]
    })

    # Create nested PythonObjects
    var a = np.array([1, 2, 3])
    var b = np.array([4, 5, 6])
    var arrays = PythonObject([a, b])
    var stacked = np.hstack((a, b))
```

**Interop Features (Python 3.12+):**
- Direct module imports
- Seamless type conversion
- NumPy array indexing and slicing
- Nested PythonObject creation

#### SIMD Vectorization

```mojo
from builtin import SIMD

# Process 4 floats simultaneously
var vec1 = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
var vec2 = SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0)

# Element-wise operations
var sum = vec1 + vec2
var product = vec1 * vec2
var scaled = vec1 * 2.0

# Automatic vectorization
from algorithm.functional import vectorize

alias simd_width = simd_width_of[DType.int32]()

@parameter
fn closure[width: Int](i: Int):
    p.store[width=width](i, i)

vectorize[closure, simd_width](size)
```

#### GPU Compute

```mojo
from gpu import thread_idx, block_idx, block_dim
from max.tensor import ManagedTensorSlice

fn gpu_add_kernel(out: ManagedTensorSlice, x: ManagedTensorSlice):
    tid_x = thread_idx.x + block_dim.x * block_idx.x
    tid_y = thread_idx.y + block_dim.y * block_idx.y
    if tid_x < x.dim_size(0) and tid_y < x.dim_size(1):
        out[tid_x, tid_y] = x[tid_x, tid_y] + 1
```

### Performance Optimization Techniques

1. **SIMD Utilization**
   - Use simd_width_of for optimal vector width
   - Leverage vectorize() for automatic optimization
   - Process 4-8 elements per operation on typical hardware

2. **GPU Kernel Design**
   - Calculate global thread indices correctly
   - Use shared memory for tile-based algorithms
   - Ensure proper thread synchronization

3. **Memory Management**
   - Mojo uses value semantics by default
   - borrowed for read-only references (no copy)
   - inout for mutable references
   - owned for move semantics

4. **Zero-Cost Abstractions**
   - Compile-time parameter evaluation with @parameter
   - Inline functions automatically
   - No runtime overhead for type conversions

---

## 4. Swift for ML

### MLX Swift Overview
- **Repository:** https://github.com/ml-explore/mlx-swift
- **Stars:** 1,576
- **Language:** C++ (core) with Swift bindings
- **Examples Repo:** https://github.com/ml-explore/mlx-swift-examples (2,416 stars)
- **Context7 Library ID:** `/ml-explore/mlx-swift`

### Architecture

MLX Swift provides native Swift bindings to the MLX C++ framework, designed specifically for Apple Silicon.

### Core API

#### Tensor Operations

```swift
import MLX

// Array creation
let x = MLXArray(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
let y = x * 2.0

// Element-wise operations (extensive library)
abs(_:stream:)
add(_:_:stream:)
multiply(_:_:stream:)
softmax(_:axes:precise:stream:)
sqrt(_:stream:)

// Matrix operations
let result = matmul(a, b, stream: .gpu)
let inner_prod = inner(a, b)
let outer_prod = outer(a, b)
let tensor_dot = tensordot(a, b, axes: [[0, 1], [1, 0]])

// Quantized operations
quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)
gatherQuantizedMatmul(_:_:scales:biases:lhsIndices:rhsIndices:...)
```

#### Device Control

```swift
// Specify execution device
add(a, b, stream: .cpu)
add(a, b, stream: .gpu)

// Arrays live in unified memory
let a = MLXArray([1, 2, 3])
let b = a + 1  // Lazy evaluation
let result = b.eval()  // Force computation
```

### MLX Swift LM (Language Models)

**Repository:** https://github.com/ml-explore/mlx-swift-lm
**Context7 Library ID:** `/ml-explore/mlx-swift-lm`

#### Model Loading & Inference

```swift
import MLXLMCommon

let modelConfiguration = ModelConfiguration(
    id: "mlx-community/quantized-gemma-2b-it"
)

// Load model (downloads from Hugging Face)
let container = try await MLXModelFactory.shared.loadContainer(
    configuration: modelConfiguration
)

// Prepare input
let generateParameters = GenerateParameters()
let input = UserInput(prompt: "Are cherries sweet?")

// Run inference
let result = try await container.perform { [input] context in
    let input = try context.processor.prepare(input: input)

    return generate(
        input: input,
        parameters: generateParameters,
        context: context
    ) { tokens in
        if tokens.count >= 20 {
            return .stop
        } else {
            return .more
        }
    }
}

print(result.output)
```

### Production Deployment

1. **Model Packaging**
   - Bundle quantized models (4-bit recommended)
   - Use memory mapping for large models
   - Preload models on app launch

2. **Memory Management**
   - Monitor KV cache growth
   - Use quantized KV cache for long contexts
   - Clear cache between sessions

3. **Performance Tuning**
   - Tune prefillStepSize for prompt processing
   - Adjust batch sizes based on device
   - Use temperature=0.0 for deterministic output

---

## 5. Rust for ML

### Overview of Rust ML Ecosystem

Three major frameworks dominate:

1. **Burn** - Comprehensive, flexible, multi-backend
2. **Candle** - Minimalist, PyTorch-like, inference-focused
3. **tch-rs** - PyTorch C++ bindings, full torch integration

### Burn Framework

**Repository:** https://github.com/tracel-ai/burn
**Stars:** 14,356
**Context7 Library ID:** `/tracel-ai/burn`

#### Architecture

Burn is a full-featured deep learning framework with:

- **Dynamic Graphs:** PyTorch-style flexibility
- **Static Performance:** Optimized execution
- **Multi-Backend Support:** NdArray, LibTorch, Wgpu, Candle
- **Type Safety:** Compile-time guarantees

#### Backend Abstraction

```rust
use burn::prelude::*;

// Choose backend at compile time
// type Backend = Candle<f32, i64>;
// type Backend = LibTorch<f32>;
// type Backend = NdArray<f32>;
type Backend = Wgpu;  // WebGPU for cross-platform

let device = Default::default();

// Same code works with any backend
let tensor_1 = Tensor::<Backend, 2>::from_data(
    [[2., 3.], [4., 5.]],
    &device
);
let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);
let result = tensor_1 + tensor_2;
```

### Candle Framework

**Repository:** https://github.com/huggingface/candle
**Stars:** 19,388
**Language:** Rust
**Last Updated:** February 15, 2026

#### Design Philosophy

Candle prioritizes:
- **Minimalism:** Small binaries, no Python runtime
- **Serverless Inference:** Optimized for production deployment
- **PyTorch-like Syntax:** Familiar API for Python developers

#### Core Features

```rust
use candle_core::{Tensor, Device, DType};

let device = Device::cuda_if_available(0)?;

// Tensor creation
let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &device)?;

// Operations
let c = a.add(&b)?;
let d = a.matmul(&b.reshape((4, 1))?)?;

// Neural network layers
use candle_nn::{linear, Linear, VarBuilder};

let vb = VarBuilder::zeros(DType::F32, &device);
let linear = linear(784, 128, vb.pp("layer1"))?;
let output = linear.forward(&input)?;
```

#### GPU Support

**CUDA:**
- Full CUDA backend with custom kernels
- NCCL for multi-GPU distribution
- Flash Attention v2 support

**Metal (Apple Silicon):**
- Native Metal backend
- Accelerate framework integration
- Optimized for M-series chips

### tch-rs (PyTorch Bindings)

**Repository:** https://github.com/LaurentMazare/tch-rs
**Stars:** 5,279
**Context7 Library ID:** `/laurentmazare/tch-rs`

#### Architecture

tch-rs provides thin Rust wrappers around PyTorch's C++ API (libtorch).

#### Comparison

| Feature         | Candle | Burn | tch-rs |
|-----------------|--------|------|--------|
| Binary Size     | Small  | Medium | Large (includes libtorch) |
| Training        | Yes    | Yes  | Yes    |
| Inference Focus | High   | Medium | High   |
| Backend Options | CPU/CUDA/Metal | 4+ backends | PyTorch only |
| Learning Curve  | Easy   | Medium | Easy (if know PyTorch) |

---

## Summary & Recommendations

### Framework Selection Matrix

| Use Case | Recommended Framework | Rationale |
|----------|----------------------|-----------|
| Apple Silicon LLM Inference | **MLX-LM (Python)** or **MLX Swift** | Native unified memory, excellent performance, OpenAI-compatible API |
| Apple Silicon Research | **MLX (Python)** | Composable transforms, lazy evaluation, easy experimentation |
| iOS/macOS Native Apps | **MLX Swift** | Native Swift integration, on-device inference, memory mapped models |
| Cross-Platform Inference | **Candle (Rust)** | Small binaries, no Python, serverless-ready |
| Production Training | **Burn (Rust)** or **PyTorch (tch-rs)** | Type safety, multi-backend, or full PyTorch ecosystem |
| Custom GPU Kernels (Apple) | **Mojo/MAX** or **MLX Swift** | Direct Metal access, high-level abstractions |
| Embedded ML | **Burn (Rust)** with no-std | Memory-safe, efficient, no runtime dependencies |

### Version Information (Feb 2026)

- **MLX-LM:** v0.30.7
- **MLX:** Active development
- **Mojo/MAX:** Active, no deprecation warnings
- **MLX Swift:** Active, follows MLX releases
- **Burn:** Active (14K+ stars)
- **Candle:** Active (19K+ stars, Feb 15, 2026 update)
- **tch-rs:** Active, PyTorch 2.x compatible

---

## References

### Official Documentation
- MLX: https://ml-explore.github.io/mlx/
- MLX-LM: https://github.com/ml-explore/mlx-lm
- MLX Swift: https://ml-explore.github.io/mlx-swift/
- Mojo: https://docs.modular.com/mojo/
- MAX: https://www.modular.com/max
- Burn: https://burn.dev/
- Candle: https://github.com/huggingface/candle
- tch-rs: https://github.com/LaurentMazare/tch-rs

### Context7 Library IDs
- `/ml-explore/mlx-lm`
- `/ml-explore/mlx`
- `/websites/ml-explore_github_io_mlx_build_html`
- `/ml-explore/mlx-swift`
- `/ml-explore/mlx-swift-lm`
- `/websites/modular_mojo`
- `/tracel-ai/burn`
- `/laurentmazare/tch-rs`

---

**Report Generated:** February 15, 2026
**Research Agent:** Claude Code (Sonnet 4.5)
**Project Context:** mlx-server infrastructure planning

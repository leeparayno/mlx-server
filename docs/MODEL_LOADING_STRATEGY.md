# Model Loading Strategy: Swift vs Python MLX-LM

## The Question

**Your concern:** The Phase 1 plan says "use MLX Swift's model loading APIs" but MLX-LM (Python) does something special with lazy loading for models that don't fit in memory. What's our strategy?

## TL;DR Answer

**You're right to question this.** We have three realistic options:

1. **Use mlx-swift-lm** (separate package) - Takes advantage of existing Swift LLM infrastructure
2. **Reimplement MLX-LM patterns in Swift** - More control, more work
3. **Python interop for loading only** - Hybrid approach

**Recommendation:** Start with **Option 1** (mlx-swift-lm) and extend as needed.

## Understanding the Landscape

### MLX Core vs MLX-LM vs MLX Swift

```
┌─────────────────────────────────────────────────┐
│              Application Layer                   │
│  (Your inference server, chat apps, etc.)        │
└───────────────┬────────────────┬────────────────┘
                │                │
     ┌──────────▼─────────┐  ┌───▼────────────┐
     │   MLX-LM (Python)  │  │ mlx-swift-lm   │
     │  • HF Hub          │  │ • HF Hub       │
     │  • Lazy loading    │  │ • Model archs  │
     │  • Model archs     │  │ • Tokenizers   │
     │  • Tokenizers      │  │ • LoRA        │
     │  • Quantization    │  │                │
     └──────────┬─────────┘  └───┬────────────┘
                │                │
     ┌──────────▼────────────────▼────────────┐
     │        MLX Core Framework               │
     │  (C++ + Python/Swift bindings)          │
     │  • Arrays, ops, Metal kernels           │
     └─────────────────────────────────────────┘
```

**Key Insight:** Neither MLX Swift nor mlx-swift-lm are in our current `Package.swift`!

### What Python MLX-LM Does for Large Models

When you load a 70B model with MLX-LM in Python:

```python
from mlx_lm import load

# This is "smart"
model, tokenizer = load("mlx-community/Llama-70B-4bit")
```

**MLX-LM's lazy loading strategy:**

1. **Safetensors Memory Mapping:**
   - Models are stored in `.safetensors` format
   - MLX-LM memory-maps the file (doesn't load entire model into RAM)
   - Only loads weights into RAM when they're actually accessed

2. **Sharded Loading:**
   - Large models are split into multiple `.safetensors` files
   - Example: `model-00001-of-00015.safetensors`, `model-00002-of-00015.safetensors`, etc.
   - MLX-LM loads shards on-demand

3. **MLX Array Lazy Evaluation:**
   - MLX itself has lazy evaluation
   - Arrays aren't materialized until computation requires them

**Example structure of a 70B model on disk:**
```
models/Llama-70B-4bit/
├── config.json                    # Model architecture config
├── tokenizer.json                 # Tokenizer
├── model-00001-of-00015.safetensors  # ~2-3GB per shard
├── model-00002-of-00015.safetensors
├── ...
└── model-00015-of-00015.safetensors
```

**Why this matters for 512GB RAM:**
- A 70B 4-bit quantized model is ~35GB
- With 512GB RAM, you can fit it entirely
- But for 200B+ models, you'd need lazy loading
- Also helps when running multiple models simultaneously

## Current State: What Exists in Swift

### mlx-swift-lm (Separate Package)

Repository: https://github.com/ml-explore/mlx-swift-lm

**What it provides:**
- ✅ Hugging Face Hub integration
- ✅ Model architectures (Llama, Mistral, Qwen, Phi, etc.)
- ✅ Tokenizer loading
- ✅ LoRA support
- ✅ Text generation utilities
- ❓ **Lazy loading/memory mapping: UNCLEAR** (not documented)

**Example API (from mlx-swift-examples):**
```swift
import MLXLLM

// Registry-based loading
let modelConfiguration = LLMRegistry.qwen3_8b_4bit

// Or custom Hugging Face model
let modelConfiguration = ModelConfiguration(
    id: "mlx-community/Llama-70B-4bit"
)

// Load model
let model = try await MLXLLM.load(configuration: modelConfiguration)

// Generate
let result = await model.generate(prompt: "Hello", maxTokens: 100)
```

### What We Don't Know About mlx-swift-lm

**Critical questions:**
1. ❓ Does it memory-map safetensors files?
2. ❓ Does it load shards lazily?
3. ❓ Can it handle 200B+ models on 512GB RAM?
4. ❓ Is it production-ready for high-concurrency servers?

**We need to investigate the source code to answer these.**

## Three Implementation Options

### Option 1: Use mlx-swift-lm (Recommended Starting Point)

**Add to Package.swift:**
```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.30.3"),
    // ... existing dependencies
],
targets: [
    .target(
        name: "Core",
        dependencies: [
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLX", package: "mlx-swift"),
            // ...
        ]
    )
]
```

**Update ModelLoader.swift:**
```swift
import MLXLLM

public actor ModelLoader {
    public func load(modelPath: String) async throws -> (model: LanguageModel, tokenizer: Tokenizer) {
        // Use mlx-swift-lm's loading
        let config = ModelConfiguration(id: modelPath)
        let llmModel = try await MLXLLM.load(configuration: config)

        // Wrap in our protocol
        return (ModelAdapter(llmModel), TokenizerAdapter(llmModel.tokenizer))
    }
}

// Adapter to wrap mlx-swift-lm's model in our protocol
struct ModelAdapter: LanguageModel {
    let wrapped: MLXLLM.Model

    func forward(inputIds: MLXArray, cache: inout KVCache?) -> MLXArray {
        // Delegate to wrapped model
        return wrapped.forward(inputIds, cache: &cache)
    }
    // ...
}
```

**Pros:**
- ✅ Quick to implement (days, not weeks)
- ✅ HF Hub integration already done
- ✅ Model architectures already implemented
- ✅ Actively maintained by Apple ML team
- ✅ Follows Swift conventions

**Cons:**
- ❌ Unknown lazy loading capabilities
- ❌ May not be optimized for server workloads
- ❌ Less control over loading strategy
- ❌ Adds dependency

**When to use:**
- Phase 1-2: Get something working quickly
- Validate architecture with known-good models
- Learn from their implementation

**When to switch:**
- If we hit memory issues with large models
- If we need custom loading optimizations
- If server-specific features are missing

### Option 2: Reimplement MLX-LM Patterns in Swift

**Implement from scratch:**

```swift
import MLX
import Foundation

public actor ModelLoader {
    public func load(modelPath: String) async throws -> (model: LanguageModel, tokenizer: Tokenizer) {
        // 1. Download from HF Hub (implement our own HF client)
        let localPath = try await downloadFromHuggingFace(modelPath)

        // 2. Load config.json
        let config = try loadConfig(from: localPath)

        // 3. Memory-map safetensors files (lazy loading)
        let weights = try memoryMapWeights(from: localPath, config: config)

        // 4. Initialize model architecture
        let model = try initializeModel(config: config, weights: weights)

        // 5. Load tokenizer
        let tokenizer = try loadTokenizer(from: localPath)

        return (model, tokenizer)
    }

    private func memoryMapWeights(from path: String, config: ModelConfig) throws -> ModelWeights {
        // Use mmap() to memory-map safetensors files
        // Only loads into RAM when accessed

        let shardPaths = try findShardFiles(at: path)
        var weightDict: [String: MLXArray] = [:]

        for shardPath in shardPaths {
            // Memory map this shard
            let mappedRegion = try mmap(shardPath)

            // Parse safetensors header to find weight tensors
            let tensors = try parseSafetensorsHeader(mappedRegion)

            for (name, tensorInfo) in tensors {
                // Create MLXArray that references mapped memory
                // Lazy: doesn't copy until computation needs it
                weightDict[name] = MLXArray(
                    unsafePointer: mappedRegion + tensorInfo.offset,
                    shape: tensorInfo.shape,
                    dtype: tensorInfo.dtype
                )
            }
        }

        return ModelWeights(weights: weightDict)
    }
}
```

**Pros:**
- ✅ Full control over loading strategy
- ✅ Can optimize for server workload
- ✅ No dependency on mlx-swift-lm
- ✅ Can implement custom optimizations
- ✅ Learn internals deeply

**Cons:**
- ❌ **Weeks of work** (4-6 weeks for full implementation)
- ❌ Have to implement HF Hub API client
- ❌ Have to implement safetensors parsing
- ❌ Have to implement model architectures (Llama, Mistral, etc.)
- ❌ Have to implement tokenizer loading
- ❌ Maintenance burden

**When to use:**
- If mlx-swift-lm doesn't meet our needs
- If we need maximum performance
- If we're building a product (not just research)

### Option 3: Python Interop for Loading Only

**Hybrid approach:**

```swift
import PythonKit

public actor ModelLoader {
    public func load(modelPath: String) async throws -> (model: LanguageModel, tokenizer: Tokenizer) {
        // 1. Use Python MLX-LM for loading
        let mlx_lm = Python.import("mlx_lm")
        let (pyModel, pyTokenizer) = mlx_lm.load(modelPath)

        // 2. Extract weights and convert to Swift MLXArrays
        let weights = try convertPythonWeightsToSwift(pyModel)

        // 3. Initialize Swift model with loaded weights
        let swiftModel = try LlamaModel(config: config, weights: weights)

        // 4. Wrap tokenizer
        let swiftTokenizer = PythonTokenizerWrapper(pyTokenizer)

        return (swiftModel, swiftTokenizer)
    }
}
```

**Pros:**
- ✅ Leverage battle-tested MLX-LM loading
- ✅ Get lazy loading for free
- ✅ Quick to implement
- ✅ Can still use Swift for inference (where performance matters)

**Cons:**
- ❌ Python runtime required
- ❌ PythonKit complexity
- ❌ Potential serialization overhead
- ❌ Less "pure Swift" solution
- ❌ Deployment complexity (ship Python + Swift)

**When to use:**
- Prototyping phase
- If loading happens infrequently
- If we want MLX-LM's exact behavior

## Recommendation

### Phase 1 (Now): Start with Option 1 (mlx-swift-lm)

**Action items:**
1. Add mlx-swift-lm to Package.swift
2. Implement ModelLoader using MLXLLM
3. Test with 7B and 13B models
4. Document any limitations we encounter

**Rationale:**
- Gets us running quickly (Phase 1 goal)
- Validates overall architecture
- We can always switch later
- Likely "good enough" for 512GB RAM

### Phase 1.5: Investigate mlx-swift-lm Memory Behavior

**Research tasks:**
1. Load a 70B model and monitor memory usage
2. Check if memory usage grows gradually (lazy) or all at once (eager)
3. Read mlx-swift-lm source code for mmap usage
4. Test with multiple simultaneous models

**Decision point:**
- If mlx-swift-lm handles memory well → stick with it
- If we hit issues → move to Option 2 or 3

### Phase 6+: Consider Option 2 if Needed

**Only if:**
- mlx-swift-lm can't handle 200B+ models
- We need custom optimizations
- Server-specific features are missing
- We're productionizing (not researching)

## Immediate Next Step

**Update Package.swift to include mlx-swift-lm:**

```swift
dependencies: [
    // Add this
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.30.3"),

    // Existing
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),
    .package(url: "https://github.com/vapor/vapor.git", from: "4.99.0"),
    // ...
]
```

Then update ModelLoader.swift to use MLXLLM instead of implementing from scratch.

## Questions to Answer Through Testing

1. **Memory mapping:** Does mlx-swift-lm use mmap for safetensors?
2. **Shard loading:** How does it handle multi-shard models?
3. **Peak memory:** What's peak RAM usage loading 70B 4-bit? (Should be ~35GB, not 70GB)
4. **Concurrent models:** Can we load 5x 13B models simultaneously?
5. **Load time:** How long to load 70B model? (With mmap: seconds, without: minutes)

## Summary

**Your instinct was correct** - the plan glossed over a significant architectural decision.

**The reality:**
- MLX Swift (core) doesn't include model loading utilities
- mlx-swift-lm (separate package) provides this
- It likely has smart loading, but we need to verify
- For Phase 1, use mlx-swift-lm and extend if needed

**Next action:**
1. Add mlx-swift-lm dependency
2. Update ModelLoader to use MLXLLM
3. Test and document memory behavior
4. Reassess after Phase 1 completion

This is the pragmatic path: start with what exists, validate it works, then optimize or rewrite only if necessary.

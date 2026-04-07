# MLX Swift & Vapor Quick Reference Guide

**Last Updated:** February 15, 2026
**Status:** All frameworks active, no deprecation warnings

---

## Framework Status Summary

| Framework | Status | Version | Deprecation Risk |
|-----------|--------|---------|------------------|
| MLX Swift | Active | 0.10.0+ | Low |
| MLX Swift LM | Active | Latest | Low |
| Vapor | Active | 4.x | Low |
| SwiftNIO | Active | Latest | Low (Apple-maintained) |
| MLX Distributed | Development | N/A | Not yet released |

---

## Quick Start: MLX Swift Model Loading

```swift
import MLXLMCommon

// Load quantized model (recommended for production)
let config = ModelConfiguration(id: "mlx-community/gemma-2-2b-it-4bit")
let container = try await LLMModelFactory.shared.loadContainer(configuration: config)

// Generate text
let result = try await container.perform { context in
    let input = try await context.processor.prepare(
        input: UserInput(prompt: "Hello, world!")
    )
    return try generate(
        input: input,
        parameters: GenerateParameters(maxTokens: 100),
        context: context
    ) { _ in .more }
}

print(result.output)
```

---

## Quick Start: Vapor Server with MLX

```swift
import Vapor
import MLXLMCommon

// Actor for thread-safe model management
actor ModelManager {
    private let model: ModelContainer

    init(modelId: String) async throws {
        let config = ModelConfiguration(id: modelId)
        self.model = try await LLMModelFactory.shared.loadContainer(configuration: config)
    }

    func generate(prompt: String, maxTokens: Int) async throws -> String {
        try await model.perform { context in
            let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
            let result = try generate(
                input: input,
                parameters: GenerateParameters(maxTokens: maxTokens),
                context: context
            ) { _ in .more }
            return result.output
        }
    }
}

// Vapor application
let app = Application()
let modelManager = try await ModelManager(modelId: "mlx-community/gemma-2-2b-it-4bit")

app.post("generate") { req async throws -> Response in
    struct Request: Content {
        let prompt: String
        let maxTokens: Int?
    }

    let request = try req.content.decode(Request.self)
    let output = try await modelManager.generate(
        prompt: request.prompt,
        maxTokens: request.maxTokens ?? 100
    )

    return Response(status: .ok, body: .init(string: output))
}

try await app.execute()
```

---

## Performance Benchmarks (M3 Max)

### Qwen3-4B Model Performance

| Quantization | Memory | Prompt tok/s | Gen tok/s | MMLU Drop |
|--------------|--------|--------------|-----------|-----------|
| bf16         | 9.02 GB | 1780         | 52        | Baseline  |
| q8           | 5.25 GB | 1606         | 87        | 0.2%      |
| q6           | 4.25 GB | 1576         | 105       | 0.5%      |
| q4           | 3.35 GB | 1622         | 135       | 3.3%      |

**Recommendation:** Use q6 for best quality/speed balance, q4 for maximum throughput.

---

## Memory Optimization Cheat Sheet

### Model Quantization

```swift
// 4-bit quantization (75% memory reduction)
let config = ModelConfiguration(id: "mlx-community/MODEL-NAME-4bit")

// Available quantizations:
// - bf16: Full precision (baseline)
// - q8: 8-bit (42% memory savings)
// - q6: 6-bit (53% memory savings)
// - q4: 4-bit (63% memory savings)
```

### KV Cache Quantization

```swift
let parameters = GenerateParameters(
    maxTokens: 2048,
    maxKVSize: 4096,        // Maximum cache size
    kvBits: 4,              // 4-bit quantization
    kvGroupSize: 64,
    quantizedKVStart: 256   // Start quantizing after 256 tokens
)

// Memory savings for 4K context:
// fp16: ~2GB → 4-bit: ~512MB (75% reduction)
```

### TurboQuant KV Cache (mlx-server)

**Enable TurboQuant KV cache:**
```bash
export MLX_KV_QUANT=true
export MLX_KV_QUANT_IMPL=turbo     # mlx | turbo
export MLX_KV_QUANT_MODE=mse       # mse | prod
export MLX_KV_QUANT_BITS=3
export MLX_KV_GROUP_SIZE=64
export MLX_KV_QUANT_START=0
export MLX_KV_ROTATION=true
export MLX_KV_ROTATION_SEED=1337
export MLX_KV_QJL_SEED=4242
```

**Batch quantize path (fast):**
```bash
export MLX_KV_QUANT=true
export MLX_KV_QUANT_IMPL=turbo
export MLX_KV_QUANT_MODE=mse
export MLX_KV_ROTATION=false  # required for batch path
```

Notes:
- TurboQuant implementation is **correct but not yet fully optimized** (dequantizes per update).
- Batch path is only available for **MSE + rotation disabled**.

---

## Device Assignment Best Practices

```swift
import MLX

// GPU for large operations
let result = matmul(largeMatrix1, largeMatrix2, stream: .gpu)

// CPU for many small operations (less overhead)
for i in 0..<1000 {
    let result = add(smallArray1, smallArray2, stream: .cpu)
}

// MLX automatically handles synchronization between CPU and GPU
```

---

## Vapor Async/Await Patterns

### Basic Route Handler

```swift
app.get("hello") { req async throws -> String in
    return "Hello, World!"
}
```

### Database Query

```swift
app.get("users") { req async throws -> [User] in
    try await User.query(on: req.db).all()
}
```

### Concurrent Operations

```swift
await withThrowingTaskGroup(of: Void.self) { group in
    models.forEach { model in
        group.addTask {
            try await model.save(on: database)
        }
    }
}
```

### Blocking Operations

```swift
// Use thread pool for blocking work (e.g., file I/O, sleep)
app.get("blocking") { req async throws -> String in
    return await withCheckedContinuation { continuation in
        req.application.threadPool.runIfActive(eventLoop: req.eventLoop) {
            sleep(5)  // Blocks only the background thread
            continuation.resume(returning: "Done")
        }
    }
}
```

---

## SwiftNIO Configuration

### Event Loop Group

```swift
import NIOPosix

// Scale to CPU cores
let eventLoopGroup = MultiThreadedEventLoopGroup(
    numberOfThreads: System.coreCount
)

// For I/O-bound workloads
let eventLoopGroup = MultiThreadedEventLoopGroup(
    numberOfThreads: System.coreCount * 2
)
```

### NIOAsyncChannel Back Pressure

```swift
let config = NIOAsyncChannel.Configuration(
    backPressureStrategy: .init(
        lowWatermark: 4,   // Resume reading at 4 items
        highWatermark: 16  // Stop reading at 16 items
    )
)
```

---

## Common Issues & Solutions

### Issue: Metal Shaders Not Found

**Solution:** Use `mlx-run` wrapper or set `DYLD_FRAMEWORK_PATH`:

```bash
./mlx-run your-command
# or
export DYLD_FRAMEWORK_PATH=/path/to/mlx-swift/build
```

### Issue: Memory Pressure

**Solutions:**
1. Use 4-bit quantized models
2. Enable KV cache quantization
3. Reduce batch sizes
4. Monitor peak memory usage:

```swift
let result = try await model.perform { context in
    // ... generation code
}
print("Peak memory: \(result.stats.peakMemory / 1_000_000_000) GB")
```

### Issue: Blocking Event Loop

**Solution:** Use thread pool for blocking operations:

```swift
return req.application.threadPool.runIfActive(eventLoop: req.eventLoop) {
    // Blocking code here
}
```

---

## Production Deployment Checklist

### Pre-deployment

- [ ] Use quantized models (q4 or q6)
- [ ] Preload models on startup
- [ ] Configure KV cache quantization
- [ ] Set up actor-based model containers
- [ ] Implement rate limiting
- [ ] Add health check endpoint
- [ ] Configure logging and metrics

### Runtime Monitoring

- [ ] Track memory usage
- [ ] Monitor tokens/second
- [ ] Log request latencies
- [ ] Set up alerting for errors
- [ ] Expose Prometheus metrics

### Configuration

```swift
// Optimal settings for M3 Ultra (192GB)
let config = ModelConfiguration(id: "mlx-community/Qwen3-30B-A3B-4bit")

let parameters = GenerateParameters(
    maxTokens: 2048,
    maxKVSize: 8192,
    kvBits: 4,
    completionBatchSize: 64,
    prefillBatchSize: 16
)

// Expected capacity:
// - Model: 18GB
// - KV cache: ~2GB per request
// - Can serve ~8 concurrent long-context requests
// - Or 32+ short-context requests
```

---

## Useful Context7 Library IDs

- `/ml-explore/mlx-swift` - MLX Swift framework
- `/ml-explore/mlx-swift-lm` - Language models
- `/websites/vapor_codes` - Vapor documentation
- `/apple/swift-nio` - SwiftNIO networking

---

## Key Resources

- **MLX Swift Docs:** https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx
- **MLX Examples:** https://github.com/ml-explore/mlx-swift-examples
- **Vapor Docs:** https://docs.vapor.codes
- **SwiftNIO Docs:** https://swiftpackageindex.com/apple/swift-nio

---

**For comprehensive documentation, see:** `/Users/lee.parayno/code4/business/mlx-server/docs/COMPREHENSIVE_MLX_SWIFT_DOCUMENTATION.md`

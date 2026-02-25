# Phase 7: Advanced Optimizations - Implementation Plan

**Date:** February 24, 2026
**Status:** Planning
**Prerequisites:** Phase 6 Complete (Authentication, real models, deployment)

## Overview

Phase 7 implements advanced optimizations to dramatically improve inference performance: speculative decoding for 2-3x speedup, prefix caching for shared prompts, quantized KV cache for 4x memory savings, and dynamic sequence optimization. These techniques transform the server from a functional service to a high-performance inference platform.

**Key Outcomes:**
- 2-3x throughput improvement via speculative decoding
- 50-70% latency reduction for repeated prompts via prefix caching
- 4x memory savings via quantized KV cache
- Support for 4-8 bit model quantization
- Dynamic sequence length optimization

## Current State (Phase 6)

### What Works
- ✅ Real MLX model loading and inference
- ✅ Continuous batching with slot management
- ✅ PagedKVCache for memory efficiency
- ✅ API key authentication and rate limiting
- ✅ Docker/Kubernetes deployment
- ✅ ~90 tests passing

### Performance Baseline
- Throughput: ~10-20 tok/s (single small model)
- Memory: ~50-100MB per request (PagedKVCache)
- Latency: <100ms TTFT, ~20ms inter-token

### What Can Be Optimized
1. **Token Generation:** Auto-regressive decoding is sequential (blocked on each token)
2. **Repeated Prompts:** System prompts re-computed for every request
3. **KV Cache Memory:** Full precision (fp16) uses significant memory
4. **Model Weights:** Full precision models are large
5. **Sequence Padding:** Fixed-length batches waste compute

## Phase 7 Implementation

### Phase 7.1: Speculative Decoding ⏳

**Goal:** Achieve 2-3x throughput improvement through speculative execution

**Concept:**
Use a small "draft" model to speculatively generate multiple tokens in parallel, then verify with the target model in a single forward pass. Accepted tokens are kept; rejected tokens trigger fallback.

**Architecture:**
```
┌──────────────────────────────────────────────────┐
│           Request with Prompt                     │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│    Draft Model (Small, Fast)                     │
│    - Generate K=5 tokens speculatively           │
│    - Takes ~5ms @ 1000 tok/s                     │
└────────────────┬─────────────────────────────────┘
                 │
                 │  Draft tokens: [t1, t2, t3, t4, t5]
                 │
┌────────────────▼─────────────────────────────────┐
│    Target Model (Large, Accurate)                │
│    - Verify all K tokens in single pass          │
│    - Takes ~20ms (same as 1 token normally)      │
└────────────────┬─────────────────────────────────┘
                 │
                 │  Verification: [✓, ✓, ✓, ✗, ✗]
                 │
┌────────────────▼─────────────────────────────────┐
│    Accept t1, t2, t3 → Emit to user              │
│    Reject t4, t5 → Generate from t3              │
└──────────────────────────────────────────────────┘

Speedup: Accept 3/5 tokens per verification
         → 3 tokens in 25ms vs 60ms (3x20ms)
         → 2.4x speedup
```

**Tasks:**

1. **Create Draft Model Actor**
   ```swift
   // Sources/Optimization/SpeculativeDecoding.swift
   actor DraftModel {
       private var modelContainer: ModelContainer?

       func loadModel(path: String) async throws {
           // Load small draft model (e.g., 0.5B parameters)
           let config = ModelConfiguration(
               id: path,
               quantization: .q4_0
           )
           modelContainer = try await ModelContainer(config: config)
       }

       func generateDraft(
           prompt: [Int],
           kvCache: KVCache,
           k: Int = 5
       ) async throws -> [Int] {
           guard let container = modelContainer else {
               throw SpeculativeError.draftModelNotLoaded
           }

           var tokens: [Int] = []
           var cache = kvCache

           // Generate K tokens auto-regressively with draft model
           for _ in 0..<k {
               let logits = try await container.model.forward(
                   MLXArray([tokens.last ?? prompt.last!]),
                   cache: cache
               )
               let nextToken = sample(logits)
               tokens.append(nextToken)
               cache = updateCache(cache, with: nextToken)
           }

           return tokens
       }
   }
   ```

2. **Implement Verification Logic**
   ```swift
   // Sources/Optimization/SpeculativeDecoding.swift
   actor SpeculativeVerifier {
       private let targetModel: InferenceEngine
       private let draftModel: DraftModel

       func verifyAndAccept(
           draftTokens: [Int],
           kvCache: KVCache
       ) async throws -> (accepted: [Int], rejectedIndex: Int?) {
           // Verify all draft tokens in single target model pass
           let logits = try await targetModel.forwardBatch(
               BatchInput(
                   tokens: draftTokens,
                   kvCacheBlockIds: kvCache.blockIds,
                   temperature: 0.7
               )
           )

           // Check each draft token against target distribution
           var accepted: [Int] = []
           for (i, draftToken) in draftTokens.enumerated() {
               let targetProbs = softmax(logits[i])
               let draftProb = targetProbs[draftToken]

               // Accept if probability above threshold
               if draftProb > 0.1 {
                   accepted.append(draftToken)
               } else {
                   // Reject this and all subsequent tokens
                   return (accepted: accepted, rejectedIndex: i)
               }
           }

           return (accepted: accepted, rejectedIndex: nil)
       }
   }
   ```

3. **Integrate with ContinuousBatcher**
   ```swift
   // Sources/Scheduler/ContinuousBatcher.swift (modifications)

   private let speculativeDecoder: SpeculativeVerifier?

   func step() async throws {
       guard hasActiveSlots else { return }

       if let speculative = speculativeDecoder {
           // Speculative decoding path
           for slot in activeSlots {
               // Generate draft tokens
               let draftTokens = try await speculative.generateDraft(
                   prompt: slot.generatedTokens,
                   kvCache: slot.kvCache,
                   k: 5
               )

               // Verify in single pass
               let (accepted, _) = try await speculative.verifyAndAccept(
                   draftTokens: draftTokens,
                   kvCache: slot.kvCache
               )

               // Emit accepted tokens
               for token in accepted {
                   await slot.stream.yield(token)
               }
           }
       } else {
           // Standard decoding path (1 token per step)
           // ... existing logic ...
       }
   }
   ```

4. **Write Speculative Decoding Tests**
   - Test: Draft model loads successfully
   - Test: Draft generates K tokens
   - Test: Verification accepts matching tokens
   - Test: Verification rejects diverging tokens
   - Test: Speedup measured (2-3x improvement)
   - Test: Output quality matches non-speculative

**Success Criteria:**
- 2-3x throughput improvement measured
- Output quality identical to standard decoding
- No additional memory overhead per request
- Works with continuous batching

**Deliverables:**
- `Sources/Optimization/SpeculativeDecoding.swift` (~300 lines)
- Updated `Sources/Scheduler/ContinuousBatcher.swift` (+80 lines)
- `Tests/OptimizationTests/SpeculativeDecodingTests.swift` (~250 lines)
- `docs/SPECULATIVE_DECODING.md` (technical documentation)

### Phase 7.2: Prefix Caching ⏳

**Goal:** Reduce latency by 50-70% for requests with shared prompt prefixes

**Concept:**
Cache the KV states for common prompt prefixes (e.g., system messages, few-shot examples). Subsequent requests reuse these cached states instead of re-computing.

**Architecture:**
```
Request 1: System prompt (1000 tokens) + User query (50 tokens)
           ↓
           Compute KV for 1000 tokens → Cache with key=hash(system_prompt)
           ↓
           Generate response

Request 2: Same system prompt + Different user query
           ↓
           Cache HIT → Reuse KV for first 1000 tokens (0ms compute)
           ↓
           Only compute KV for new 50 tokens
           ↓
           Generate response (50-70% faster TTFT)
```

**Tasks:**

1. **Create Prefix Cache**
   ```swift
   // Sources/Memory/PrefixCache.swift
   actor PrefixCache {
       private struct CacheEntry {
           let kvCache: KVCache
           let tokenCount: Int
           let lastAccessed: Date
       }

       private var cache: [String: CacheEntry] = [:]
       private let maxEntries: Int = 100

       func lookup(prefix: [Int]) async -> KVCache? {
           let key = hashPrefix(prefix)

           if let entry = cache[key] {
               // Update LRU
               cache[key] = CacheEntry(
                   kvCache: entry.kvCache,
                   tokenCount: entry.tokenCount,
                   lastAccessed: Date()
               )
               return entry.kvCache
           }

           return nil
       }

       func store(prefix: [Int], kvCache: KVCache) async {
           let key = hashPrefix(prefix)

           // Evict if cache full (LRU)
           if cache.count >= maxEntries {
               let oldestKey = cache.min(by: { $0.value.lastAccessed < $1.value.lastAccessed })!.key
               cache.removeValue(forKey: oldestKey)
           }

           cache[key] = CacheEntry(
               kvCache: kvCache,
               tokenCount: prefix.count,
               lastAccessed: Date()
           )
       }

       private func hashPrefix(_ tokens: [Int]) -> String {
           // Hash first N tokens for cache key
           return tokens.prefix(1000).map(String.init).joined(separator: ",").sha256()
       }
   }
   ```

2. **Integrate with InferenceEngine**
   ```swift
   // Sources/Core/InferenceEngine.swift (modifications)

   private let prefixCache: PrefixCache

   func processRequest(
       tokens: [Int],
       maxTokens: Int
   ) async throws -> AsyncStream<Int> {
       // Check for cached prefix
       let prefixLength = min(1000, tokens.count)
       let prefix = Array(tokens.prefix(prefixLength))

       var kvCache: KVCache
       var startIndex: Int

       if let cachedKV = await prefixCache.lookup(prefix: prefix) {
           // Cache hit: reuse KV for prefix
           kvCache = cachedKV
           startIndex = prefixLength
           logger.info("Prefix cache HIT: reusing \(prefixLength) tokens")
       } else {
           // Cache miss: compute from scratch
           kvCache = KVCache()
           startIndex = 0
           logger.info("Prefix cache MISS: computing all tokens")
       }

       // Process remaining tokens
       for i in startIndex..<tokens.count {
           let logits = try await forward(MLXArray([tokens[i]]), cache: kvCache)
           updateKVCache(kvCache, with: logits)
       }

       // Cache the prefix for future requests
       if startIndex == 0 {
           await prefixCache.store(prefix: prefix, kvCache: kvCache)
       }

       // Generate new tokens
       return generateTokens(from: kvCache, maxTokens: maxTokens)
   }
   ```

3. **Add Prefix Detection Heuristics**
   ```swift
   // Sources/Optimization/PrefixDetector.swift
   struct PrefixDetector {
       func detectCommonPrefix(in requests: [InferenceRequest]) -> [Int]? {
           // Find longest common prefix across requests
           guard let first = requests.first else { return nil }

           var commonPrefix: [Int] = []

           for (i, token) in first.prompt.enumerated() {
               let allMatch = requests.allSatisfy { req in
                   i < req.prompt.count && req.prompt[i] == token
               }

               if allMatch {
                   commonPrefix.append(token)
               } else {
                   break
               }
           }

           // Only cache if prefix is substantial (>100 tokens)
           return commonPrefix.count > 100 ? commonPrefix : nil
       }
   }
   ```

4. **Write Prefix Caching Tests**
   - Test: Cache stores and retrieves prefixes
   - Test: Cache hit skips computation
   - Test: Cache miss computes full prompt
   - Test: LRU eviction works correctly
   - Test: Latency improvement measured (50-70%)
   - Test: Output identical with/without cache

**Success Criteria:**
- 50-70% TTFT reduction for cached prefixes
- Cache hit rate >80% in production workloads
- LRU eviction prevents unbounded growth
- No quality degradation

**Deliverables:**
- `Sources/Memory/PrefixCache.swift` (~200 lines)
- `Sources/Optimization/PrefixDetector.swift` (~100 lines)
- Updated `Sources/Core/InferenceEngine.swift` (+60 lines)
- `Tests/MemoryTests/PrefixCacheTests.swift` (~200 lines)

### Phase 7.3: Quantized KV Cache ⏳

**Goal:** Reduce KV cache memory by 4x through quantization

**Concept:**
Store KV cache in 4-bit or 8-bit quantized format instead of fp16. Dequantize on-the-fly during forward pass. This enables 4x more concurrent requests.

**Memory Savings:**
```
Current (fp16):  16 bits per value × 2 (K+V) × hidden_dim × seq_len
                 = 32 × 4096 × 2048 = 256 MB per request

Quantized (4-bit): 4 bits per value × 2 × 4096 × 2048 = 64 MB per request
                   → 4x reduction, 4x more concurrent requests
```

**Tasks:**

1. **Implement Quantization Functions**
   ```swift
   // Sources/Memory/QuantizedKVCache.swift
   struct QuantizedKVCache {
       struct QuantizedBlock {
           let data: MLXArray  // int4 or int8
           let scale: Float
           let zeroPoint: Float
       }

       private var blocks: [Int: QuantizedBlock] = [:]

       func quantize(_ kvTensor: MLXArray, bits: Int = 4) -> QuantizedBlock {
           let min = kvTensor.min().item(Float.self)
           let max = kvTensor.max().item(Float.self)

           let scale = (max - min) / Float((1 << bits) - 1)
           let zeroPoint = min

           let quantized = ((kvTensor - zeroPoint) / scale).asType(.int8)

           return QuantizedBlock(
               data: quantized,
               scale: scale,
               zeroPoint: zeroPoint
           )
       }

       func dequantize(_ block: QuantizedBlock) -> MLXArray {
           return block.data.asType(.float32) * block.scale + block.zeroPoint
       }

       mutating func storeBlock(id: Int, kv: MLXArray) {
           blocks[id] = quantize(kv, bits: 4)
       }

       func retrieveBlock(id: Int) -> MLXArray? {
           guard let block = blocks[id] else { return nil }
           return dequantize(block)
       }
   }
   ```

2. **Integrate with PagedKVCache**
   ```swift
   // Sources/Memory/PagedKVCache.swift (modifications)

   private let quantizationBits: Int  // 4, 8, or 16 (no quant)
   private var quantizedBlocks: QuantizedKVCache?

   func storeKVBlock(_ blockId: Int, kv: MLXArray) async {
       if quantizationBits < 16 {
           // Quantized storage
           await quantizedBlocks?.storeBlock(id: blockId, kv: kv)
       } else {
           // Full precision storage
           blocks[blockId] = kv
       }
   }

   func retrieveKVBlock(_ blockId: Int) async -> MLXArray? {
       if quantizationBits < 16 {
           // Dequantize on retrieval
           return await quantizedBlocks?.retrieveBlock(id: blockId)
       } else {
           return blocks[blockId]
       }
   }
   ```

3. **Add Quality vs Memory Trade-off Config**
   ```swift
   // Sources/Core/InferenceConfig.swift
   struct InferenceConfig: Codable {
       enum KVCacheQuantization: String, Codable {
           case fp16 = "fp16"      // No quantization, best quality
           case int8 = "int8"      // 2x memory savings, minimal quality loss
           case int4 = "int4"      // 4x memory savings, slight quality loss
       }

       let kvCacheQuantization: KVCacheQuantization
       let maxConcurrentRequests: Int

       static let highQuality = InferenceConfig(
           kvCacheQuantization: .fp16,
           maxConcurrentRequests: 100
       )

       static let highThroughput = InferenceConfig(
           kvCacheQuantization: .int4,
           maxConcurrentRequests: 400
       )
   }
   ```

4. **Write Quantization Tests**
   - Test: 4-bit quantization reduces memory by 4x
   - Test: 8-bit quantization reduces memory by 2x
   - Test: Dequantization recovers reasonable values
   - Test: Quality degradation is minimal (<5% perplexity increase)
   - Test: Quantized cache works with batching
   - Test: Performance overhead is acceptable (<10%)

**Success Criteria:**
- 4x memory reduction with int4 quantization
- Quality degradation <5% (measured via perplexity)
- Quantization/dequantization overhead <10% latency
- Supports 4x more concurrent requests

**Deliverables:**
- `Sources/Memory/QuantizedKVCache.swift` (~250 lines)
- Updated `Sources/Memory/PagedKVCache.swift` (+100 lines)
- `Sources/Core/InferenceConfig.swift` (~60 lines)
- `Tests/MemoryTests/QuantizedKVCacheTests.swift` (~200 lines)

### Phase 7.4: Dynamic Sequence Optimization ⏳

**Goal:** Eliminate wasted computation from padding

**Tasks:**

1. **Implement Variable-Length Batching**
   ```swift
   // Sources/Scheduler/VariableLengthBatcher.swift
   struct VariableLengthBatch {
       let tokens: [Int]           // Flat array of all tokens
       let sequenceLengths: [Int]  // Length of each sequence
       let positions: [Int]        // Position within each sequence

       func toMLXArray() -> (tokens: MLXArray, positions: MLXArray) {
           // No padding, just concatenated sequences
           return (
               tokens: MLXArray(tokens),
               positions: MLXArray(positions)
           )
       }
   }
   ```

2. **Add Flash Attention Support**
   - Integrate MLX's flash attention for variable lengths
   - Optimize memory layout for Metal

3. **Write Dynamic Batching Tests**
   - Test: Variable-length batching works
   - Test: No padding overhead
   - Test: Throughput improvement measured

**Success Criteria:**
- 10-20% throughput improvement from eliminating padding
- Works with all optimizations (speculative, prefix cache, quantized)

**Deliverables:**
- `Sources/Scheduler/VariableLengthBatcher.swift` (~200 lines)
- `Tests/SchedulerTests/VariableLengthBatchingTests.swift` (~150 lines)

## Success Criteria

### Performance Requirements
- [ ] 2-3x throughput improvement (speculative decoding)
- [ ] 50-70% TTFT reduction (prefix caching)
- [ ] 4x memory savings (quantized KV cache)
- [ ] 10-20% additional speedup (dynamic batching)
- [ ] **Combined:** 5-10x overall improvement from Phase 5 baseline

### Quality Requirements
- [ ] Output quality degradation <5%
- [ ] All existing tests pass (90+ tests)
- [ ] New optimization tests (30+ tests)
- [ ] Zero data races (Swift 6)

### Operational Requirements
- [ ] Optimizations configurable (on/off)
- [ ] Metrics track optimization effectiveness
- [ ] Documentation for tuning trade-offs

## Timeline

**Week 1:**
- Days 1-3: Speculative Decoding (Phase 7.1)
- Days 4-5: Prefix Caching (Phase 7.2)

**Week 2:**
- Days 1-3: Quantized KV Cache (Phase 7.3)
- Days 4-5: Dynamic Sequence Optimization (Phase 7.4)

**Week 3:**
- Days 1-2: Integration testing and benchmarking
- Days 3-5: Documentation and tuning

**Total: 3 weeks**

## Dependencies

**Existing:**
- ✅ Phase 6 complete (real models, deployment)
- ✅ MLX Swift 0.30.6 (quantization support)
- ✅ ContinuousBatcher (batching foundation)

**New:**
- MLX flash attention APIs
- Quantization primitives (MLX.quantize, MLX.dequantize)

## Risks and Mitigations

### Risk 1: Speculative Decoding Acceptance Rate
**Risk:** Draft model acceptance rate <50%, negating speedup
**Mitigation:**
- Tune draft model selection (size vs quality)
- Adjust acceptance threshold dynamically
- Fallback to standard decoding if ineffective

### Risk 2: Quantization Quality Loss
**Risk:** 4-bit quantization degrades output too much
**Mitigation:**
- Support 8-bit as middle ground
- Make quantization configurable
- Measure quality metrics (perplexity)

### Risk 3: Complexity Interactions
**Risk:** Optimizations interfere with each other
**Mitigation:**
- Comprehensive integration tests
- Gradual rollout (one optimization at a time)
- Monitor metrics closely

## References

**Internal:**
- Phase 6 Plan: `docs/plans/2026-02-24-feat-phase6-production-hardening-plan.md`
- PagedKVCache: `Sources/Memory/PagedKVCache.swift`
- ContinuousBatcher: `Sources/Scheduler/ContinuousBatcher.swift`

**External:**
- Speculative Decoding: https://arxiv.org/abs/2302.01318
- Prefix Caching: https://arxiv.org/abs/2304.04487
- KV Cache Quantization: https://arxiv.org/abs/2308.16237
- Flash Attention: https://arxiv.org/abs/2205.14135

---

**Created:** 2026-02-24
**Author:** Claude Code + Lee Parayno
**Status:** Planning - Ready After Phase 6

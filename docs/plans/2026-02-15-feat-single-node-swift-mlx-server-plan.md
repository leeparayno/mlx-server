---
title: Single-Node Swift MLX Inference Server
type: feat
date: 2026-02-15
status: draft
priority: high
estimated_duration: 6-8 weeks
---

# Single-Node Swift MLX Inference Server

## Overview

Build a high-performance, single-node MLX inference server in Swift that maximizes utilization of Mac Studio M3 Ultra's 512GB unified memory through continuous batching and Actor-based concurrency. This serves as the foundation for future multi-node clustering.

## Problem Statement

Current MLX-LM Python implementation achieves ~30 tok/s for 70B models in single-user scenarios, leaving the M3 Ultra's 800 GB/s memory bandwidth largely unutilized. The Python GIL and asyncio overhead prevent efficient multi-user serving, resulting in GPU idle time and poor hardware utilization.

## Proposed Solution

Implement a Swift-based inference server using:
- **Swift Actors** for thread-safe request queuing and batch management
- **Continuous batching** to keep GPU saturated with concurrent requests
- **PagedAttention** for efficient KV cache management across multiple users
- **Vapor 4.x** for async/await HTTP/WebSocket API layer
- **MLX Swift** for native Metal GPU acceleration

**Target Performance:** 200+ tok/s (70B model, multi-user), 6-7x improvement over baseline.

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────┐
│              Vapor API Layer                     │
│  (HTTP/WebSocket, Async Request Handling)        │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         Scheduler Actor (The Brain)              │
│  • Request Queue Management                      │
│  • Continuous Batch Formation                    │
│  • KV Cache Allocation (PagedAttention)          │
│  • Response Streaming                            │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│       MLX Engine Actor (The Muscle)              │
│  • Model Loading & Management                    │
│  • Forward Pass Execution                        │
│  • Token Sampling & Generation                   │
│  • Metal Kernel Optimization                     │
└──────────────────────────────────────────────────┘
```

### Implementation Phases

#### Phase 1: Foundation (Weeks 1-2)

**Goal:** Set up project structure and basic MLX Swift integration

Tasks:
- [ ] Initialize Swift Package Manager project (`Package.swift`)
- [ ] Add dependencies: MLX Swift, Vapor 4.x, Swift Argument Parser
- [ ] Implement model loading from Hugging Face Hub
- [ ] Create basic CLI for model testing
- [ ] Verify 4-bit quantized model loading (test with Llama 7B)
- [ ] Set up memory wiring: `sudo sysctl iogpu.wired_limit_mb=450000`

**Deliverables:**
- `Sources/Core/ModelLoader.swift` - Hugging Face model loading
- `Sources/Core/Tokenizer.swift` - Token encoding/decoding
- Basic CLI that loads and runs a single inference

**Success Criteria:**
- Successfully load and run inference on Llama 7B 4-bit model
- Verify unified memory usage (~4GB for 7B model)
- Baseline single-token latency measurement

#### Phase 2: Actor-Based Request Handling (Weeks 3-4)

**Goal:** Implement thread-safe concurrent request management

Tasks:
- [ ] Create `InferenceEngine` Actor for model execution isolation
- [ ] Create `Scheduler` Actor for request queue management
- [ ] Implement basic request batching (static batching first)
- [ ] Add request prioritization and timeout handling
- [ ] Implement streaming token responses via AsyncSequence

**Deliverables:**
- `Sources/Engine/InferenceEngine.swift`
- `Sources/Scheduler/RequestScheduler.swift`
- `Sources/Models/Request.swift` and `Response.swift`

**Success Criteria:**
- Process 10 concurrent requests without race conditions
- Demonstrate Actor isolation preventing data races
- Measure throughput improvement vs sequential processing

#### Phase 3: Continuous Batching (Weeks 5-6)

**Goal:** Implement dynamic continuous batching for maximum GPU utilization

Tasks:
- [ ] Replace static batching with continuous batching loop
- [ ] Implement "slot management" - add/remove requests mid-batch
- [ ] Add EOS token detection and request completion handling
- [ ] Optimize batch concatenation operations
- [ ] Profile GPU utilization with `sudo powermetrics`

**Deliverables:**
- `Sources/Scheduler/ContinuousBatcher.swift`
- `Sources/Scheduler/BatchState.swift`

**Code Example:**
```swift
actor ContinuousBatcher {
    private var activeSlots: [Int: Request] = [:]
    private var pendingQueue: [Request] = []

    func step() async throws -> BatchResult {
        // 1. Collect tokens from all active requests
        let batchTokens = activeSlots.values.map { $0.lastToken }
        let input = MLXArray(batchTokens) // Shape: [BatchSize, 1]

        // 2. Single forward pass for entire batch
        let logits = await engine.forward(input)

        // 3. Sample next tokens for all requests
        let nextTokens = sampler.sample(logits)

        // 4. Update slots: remove finished, add pending
        updateSlots(with: nextTokens)

        return BatchResult(completed: finishedRequests)
    }
}
```

**Success Criteria:**
- GPU utilization > 90% under concurrent load
- Zero GPU idle time between token generations
- Smooth request ingestion/completion without batch stalls

#### Phase 4: PagedAttention KV Cache (Weeks 7-8)

**Goal:** Implement memory-efficient KV cache management

Tasks:
- [ ] Design page-based KV cache structure (16-token blocks)
- [ ] Implement block allocation/deallocation pool
- [ ] Add cache eviction policy (LRU for long contexts)
- [ ] Optimize memory layout for Metal access patterns
- [ ] Measure memory fragmentation and utilization

**Deliverables:**
- `Sources/Memory/PagedKVCache.swift`
- `Sources/Memory/BlockAllocator.swift`

**Architecture:**
```swift
struct KVCachePage {
    let blockID: Int
    let tokens: MLXArray  // Shape: [16, hidden_dim]
    var inUse: Bool
}

actor PagedKVCache {
    private var blockPool: [KVCachePage]
    private var requestBlocks: [RequestID: [Int]]

    func allocateBlocks(for request: RequestID, count: Int) -> [Int]
    func releaseBlocks(for request: RequestID)
}
```

**Success Criteria:**
- Support 100+ concurrent requests with 2K context each
- Memory utilization > 95% (minimal fragmentation)
- No out-of-memory crashes under sustained load

#### Phase 5: Vapor API Layer (Week 9)

**Goal:** Expose inference server via HTTP and WebSocket APIs

Tasks:
- [ ] Create Vapor routes for `/v1/completions` and `/v1/chat/completions`
- [ ] Implement Server-Sent Events (SSE) streaming
- [ ] Add WebSocket support for bidirectional streaming
- [ ] Implement OpenAI-compatible API format
- [ ] Add health check and metrics endpoints

**Deliverables:**
- `Sources/API/Routes.swift`
- `Sources/API/OpenAICompatibility.swift`
- API documentation (OpenAPI/Swagger)

**Example API:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b-4bit",
    "prompt": "Explain quantum computing",
    "max_tokens": 500,
    "stream": true
  }'
```

**Success Criteria:**
- OpenAI-compatible API responses
- SSE streaming with < 50ms latency per token
- Handle 1000 req/min without errors

#### Phase 6: Performance Optimization (Week 10)

**Goal:** Profile and optimize critical paths

Tasks:
- [ ] Profile with Instruments (Time Profiler, Metal System Trace)
- [ ] Optimize token sampling (top-k, top-p implementations)
- [ ] Reduce Actor message passing overhead
- [ ] Optimize MLXArray concatenation operations
- [ ] Tune batch size and scheduling parameters

**Deliverables:**
- Performance profiling report
- Optimized hot-path implementations
- Benchmarking suite

**Success Criteria:**
- Achieve 200+ tok/s for 70B model (multi-user)
- < 100ms time-to-first-token (TTFT)
- < 5ms inter-token latency

#### Phase 7: Production Readiness (Week 11-12)

**Goal:** Add production features and documentation

Tasks:
- [ ] Implement graceful shutdown and request draining
- [ ] Add comprehensive logging (structured logging)
- [ ] Implement metrics (Prometheus format)
- [ ] Add load testing suite (K6 or Locust)
- [ ] Write deployment documentation
- [ ] Create Docker/OCI container image

**Deliverables:**
- `Sources/Monitoring/Metrics.swift`
- Load testing scripts
- Production deployment guide
- Performance tuning guide

**Success Criteria:**
- Zero-downtime restart capability
- Observable metrics (throughput, latency, memory)
- Successfully handle 10K requests under load test

## Acceptance Criteria

### Functional Requirements
- [ ] Load and serve MLX models (7B to 70B parameters)
- [ ] Support 4-bit and 6-bit quantization
- [ ] Handle 100+ concurrent requests
- [ ] OpenAI-compatible API
- [ ] Streaming and non-streaming responses

### Performance Requirements
- [ ] 200+ tok/s for 70B model (multi-user continuous batching)
- [ ] < 100ms time-to-first-token (TTFT)
- [ ] < 5ms inter-token latency
- [ ] GPU utilization > 90% under load
- [ ] Memory utilization > 95% (PagedAttention)

### Quality Requirements
- [ ] Zero data races (verified with Thread Sanitizer)
- [ ] No memory leaks (verified with Leaks instrument)
- [ ] Graceful degradation under overload
- [ ] Comprehensive error handling
- [ ] Structured logging for debugging

## Technical Considerations

### Memory Management
- **Unified Memory Wiring:** Must configure `iogpu.wired_limit_mb` before loading large models
- **KV Cache Size:** 70B model with 2K context = ~35GB per user. Plan for 10-15 concurrent users max on 512GB system
- **Model Loading:** Pre-load models at startup to avoid lazy loading latency

### Concurrency Model
- **Actor Isolation:** Prevents data races but adds message-passing overhead
- **TaskGroups:** Use for parallel token sampling across batch
- **Async/Await:** Maintain throughout stack for backpressure propagation

### Security
- [ ] Input validation (max token limits, prompt sanitization)
- [ ] Rate limiting per client
- [ ] Model file SHA256 verification
- [ ] No arbitrary code execution in sampling

## Success Metrics

### Performance KPIs
- **Throughput:** 200+ tok/s (70B model, 10 concurrent users)
- **Latency:** < 100ms TTFT, < 5ms inter-token
- **Utilization:** > 90% GPU, > 95% memory
- **Availability:** 99.9% uptime

### Development Velocity
- **Code Coverage:** > 80% unit test coverage
- **Build Time:** < 2 minutes clean build
- **Hot Reload:** < 5 seconds for code changes

## Dependencies & Risks

### Dependencies
- **MLX Swift:** Version 0.30.6+ (check for API stability)
- **Vapor:** 4.x with async/await support
- **Swift:** 5.10+ (required for complete concurrency support)
- **macOS:** 15.0+ (for memory wiring sysctl)

### Risks & Mitigation
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MLX Swift API changes | High | Medium | Pin to specific version, monitor releases |
| Memory wiring limits | High | Low | Document sysctl configuration, test on target hardware |
| Actor message overhead | Medium | Medium | Profile early, consider shared memory if needed |
| GPU thermal throttling | Medium | Low | Monitor temps, reduce batch size if needed |

## Future Considerations

### Multi-Node Clustering (Phase 2 Project)
Once single-node performance is optimized, extend to multi-node with:
- JACCL backend for RDMA over Thunderbolt
- Tensor parallelism for model sharding
- Global scheduler for load balancing

### Advanced Features
- Speculative decoding with draft models
- LoRA adapter support for fine-tuned models
- Multi-modal support (vision, audio)
- Prefix caching for repeated prompts

## References & Research

### Internal References
- Project charter: `/Users/lee.parayno/code4/business/mlx-server/CLAUDE.md`
- MLX research: `/Users/lee.parayno/code4/business/mlx-server/ML_FRAMEWORKS_RESEARCH.md`
- Swift architecture: `/Users/lee.parayno/code4/business/mlx-server/gemini-research/MLX_SERVER.md`
- Distributed research: `/Users/lee.parayno/code4/business/mlx-server/DISTRIBUTED_ML_INFERENCE_BEST_PRACTICES_2025-2026.md`

### External References
- MLX Swift: https://github.com/ml-explore/mlx-swift
- Vapor Framework: https://vapor.codes
- vLLM Continuous Batching: https://docs.vllm.ai/en/latest/
- PagedAttention Paper: https://arxiv.org/abs/2309.06180

### Related Work
- Existing plan: `2026-02-15-feat-mlx-lm-performance-alternative-plan.md` (14-week multi-language comparison)
- This plan focuses on Swift-only, single-node optimization first

---

**Next Steps:** Review plan, then use `/workflows:work` to begin Phase 1 implementation.

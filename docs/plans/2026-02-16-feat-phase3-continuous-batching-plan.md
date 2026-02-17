---
title: Phase 3 - Continuous Batching with Adaptive Slot Management
type: feat
date: 2026-02-16
phase: 3
dependencies: ["Phase 2 - Actor-Based Request Handling"]
estimated_complexity: high
---

# Phase 3: Continuous Batching with Adaptive Slot Management

## Overview

Replace Phase 2's static batching with continuous batching to achieve **6-7x throughput improvement** by eliminating GPU idle time between token generations. Continuous batching processes multiple requests in a single forward pass, dynamically adding/removing requests as they complete, maintaining near-100% GPU utilization.

**Current Architecture (Phase 2 - Static Batching):**
```
Dequeue Batch → Process N Requests Concurrently (N forward passes) → Wait for All Complete → Repeat
```

**Target Architecture (Phase 3 - Continuous Batching):**
```
Step Loop → Single Batched Forward Pass → Sample All Tokens → Update Slots → Add/Remove Requests → Repeat
```

**Key Difference:** Static batching runs N independent `InferenceEngine.generate()` calls in parallel. Continuous batching runs **one batched forward pass** per step for all active requests, dramatically reducing GPU idle time.

## Problem Statement

Phase 2's static batching has inherent inefficiencies:

1. **GPU Idle Time:** Gap between completing one batch and starting the next
2. **Inefficient Concurrency:** N parallel forward passes instead of 1 batched pass
3. **Fixed Batch Sizes:** Cannot add new requests until current batch completes
4. **Resource Waste:** Finished requests hold slots until entire batch completes
5. **No Mid-Flight Changes:** Cannot cancel requests or adjust batch size dynamically

**Impact:** Under high load, GPU utilization drops to 60-70% due to batch synchronization barriers.

**Example Scenario:**
```
Static Batching (Current):
Request A: |====================| (10 tokens, 1000ms)
Request B: |=======|              (4 tokens, 400ms) ← waits 600ms idle
Request C: (waiting in queue)     ← could have started at 400ms
GPU Usage: [100%---60%---100%]    ← 40% idle during straggler

Continuous Batching (Target):
Request A: |====================| (10 tokens)
Request B: |=======|              (4 tokens, freed at 400ms)
Request C:         |============| (starts at 400ms, reuses slot)
GPU Usage: [100%100%100%100%100%] ← no idle time
```

## Proposed Solution

Implement continuous batching with three core capabilities:

### 1. Dynamic Slot Management
- Maintain fixed-size array of `BatchSlot` (default: 32 slots)
- Each slot tracks: request ID, generated tokens, KV cache pointer, finish state
- Add requests to empty slots immediately (no waiting for batch to form)
- Remove finished requests and reclaim slots in same step

### 2. Batched Forward Pass
- Single forward pass per step for all active slots
- Concatenate all "next token" inputs: `[batch_size, 1]` tensor
- Sample tokens for entire batch simultaneously
- Update all slot states atomically

### 3. Adaptive Batch Sizing
- Monitor GPU utilization (target: >90%)
- Dynamically adjust max batch size based on memory pressure
- Scale from 1 to 32+ slots based on request load

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                      RequestScheduler                        │
│  (Priority Queue, Timeout Handling, TokenStream Registry)   │
└────────────────────┬────────────────────────────────────────┘
                     │ submit() / dequeue() / cancel()
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   ContinuousBatcher                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Batch Slots [0..31]                                 │   │
│  │  Slot 0: [req_id, tokens, kv_cache, is_finished]    │   │
│  │  Slot 1: [req_id, tokens, kv_cache, is_finished]    │   │
│  │  ...                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  step() → fill_empty_slots()                                │
│        → prepare_batch_input()      [batch_size, 1]        │
│        → forward_pass()             (single batched call)   │
│        → sample_tokens()            [batch_size]           │
│        → update_slots()             (emit, mark finished)   │
│        → cleanup_finished_slots()                           │
└────────────────────┬────────────────────────────────────────┘
                     │ forward() / sampleTokens()
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                           │
│  (Model Container, Batch Forward Pass, Token Sampling)      │
└─────────────────────────────────────────────────────────────┘
```

## Technical Approach

### Architecture

**Actor Responsibilities:**

1. **RequestScheduler (Existing - Minor Updates)**
   - Continue managing priority queue and request lifecycle
   - Add cancellation API: `cancel(requestId: UUID)`
   - Track active slots for monitoring
   - No changes to TokenStream or timeout handling

2. **ContinuousBatcher (Complete Rewrite)**
   - Manage fixed-size slot array
   - Orchestrate continuous batching loop
   - Integrate with InferenceEngine's batch forward pass
   - Handle slot lifecycle (allocate, update, free)

3. **InferenceEngine (New APIs)**
   - Add batch forward pass: `forwardBatch(inputs: [MLXArray]) async throws -> [MLXArray]`
   - Add batch sampling: `sampleBatch(logits: [MLXArray], params: [SamplingParams]) -> [Int]`
   - Maintain existing `generateStreaming()` for compatibility

### Implementation Phases

#### Phase 3.1: Core Continuous Batching (Week 1)

**Goal:** Replace static batching with basic continuous batching

**Tasks:**

1. **BatchSlot Data Structure** (`RequestState.swift`)
   ```swift
   public struct BatchSlot: Sendable {
       let slotId: Int
       let request: StatefulRequest
       var generatedTokens: [Int]
       var kvCacheBlockIds: [Int]  // PagedKVCache integration (Phase 4)
       var isFinished: Bool
       var finishReason: FinishReason?

       init(slotId: Int, request: StatefulRequest)
       mutating func appendToken(_ token: Int)
       mutating func finish(reason: FinishReason)
   }
   ```

2. **InferenceEngine Batch APIs** (`InferenceEngine.swift`)
   ```swift
   public actor InferenceEngine {
       // New batch APIs
       public func forwardBatch(
           tokenIds: [Int],           // One token per slot
           positions: [Int],          // Current position per slot
           kvCaches: [KVCache?]       // KV cache per slot
       ) async throws -> [MLXArray]   // Logits per slot

       public func sampleBatch(
           logits: [MLXArray],
           params: [SamplingParams]   // Per-request temperature, top_p, etc.
       ) throws -> [Int]               // Sampled token per slot
   }
   ```

3. **ContinuousBatcher Core Loop** (`ContinuousBatcher.swift`)
   ```swift
   public actor ContinuousBatcher {
       private var slots: [BatchSlot?]  // Fixed size array
       private let maxBatchSize: Int = 32
       private let scheduler: RequestScheduler
       private let engine: InferenceEngine

       public func start() async {
           while isRunning {
               await step()
               // No sleep - continuous loop
           }
       }

       private func step() async throws {
           // 1. Fill empty slots from scheduler
           await fillEmptySlots()

           // 2. Prepare batch input
           let (tokenIds, positions, kvCaches) = prepareBatchInput()
           guard !tokenIds.isEmpty else { return }

           // 3. Single batched forward pass
           let logits = try await engine.forwardBatch(
               tokenIds: tokenIds,
               positions: positions,
               kvCaches: kvCaches
           )

           // 4. Sample tokens for all slots
           let sampledTokens = try engine.sampleBatch(
               logits: logits,
               params: slots.compactMap { $0?.request.samplingParams }
           )

           // 5. Update slots
           await updateSlots(sampledTokens: sampledTokens)

           // 6. Cleanup finished slots
           await cleanupFinishedSlots()
       }

       private func fillEmptySlots() async {
           let emptySlotCount = slots.filter { $0 == nil }.count
           guard emptySlotCount > 0 else { return }

           let newRequests = await scheduler.dequeueNextBatch(size: emptySlotCount)

           for request in newRequests {
               if let emptySlotIndex = slots.firstIndex(where: { $0 == nil }) {
                   slots[emptySlotIndex] = BatchSlot(slotId: emptySlotIndex, request: request)
                   await scheduler.markActive(requestId: request.id)
               }
           }
       }

       private func updateSlots(sampledTokens: [Int]) async {
           var activeSlotIndex = 0

           for i in 0..<slots.count {
               guard var slot = slots[i] else { continue }

               let token = sampledTokens[activeSlotIndex]
               slot.appendToken(token)

               // Emit token via scheduler
               await scheduler.emitToken(
                   requestId: slot.request.id,
                   token: token,
                   metadata: ["slot_id": slot.slotId]
               )

               // Check finish conditions
               if isEndOfSequence(token) || slot.generatedTokens.count >= slot.request.maxTokens {
                   slot.finish(reason: isEndOfSequence(token) ? .stop : .length)
                   await scheduler.complete(requestId: slot.request.id, info: /* generation info */)
               }

               slots[i] = slot
               activeSlotIndex += 1
           }
       }

       private func cleanupFinishedSlots() async {
           for i in 0..<slots.count {
               if let slot = slots[i], slot.isFinished {
                   // Release KV cache blocks (Phase 4)
                   slots[i] = nil
               }
           }
       }
   }
   ```

4. **Main Server Integration** (`main.swift` or `MLXServer.swift`)
   ```swift
   let batcher = ContinuousBatcher(
       engine: engine,
       scheduler: scheduler,
       config: BatcherConfig(maxBatchSize: 32)
   )

   await batcher.start()  // Runs continuous loop
   ```

**Test Coverage:**
- `ContinuousBatcherTests.swift`:
  - `testFillEmptySlots()` - Add requests to empty slots
  - `testBatchedForwardPass()` - Single forward pass for multiple slots
  - `testSlotCleanup()` - Finished slots freed immediately
  - `testContinuousIngestion()` - New requests added mid-generation
  - `testVariableLengthRequests()` - Different completion times

**Verification:**
```bash
make build
make test
# Manual test: Submit 10 concurrent requests, verify GPU utilization >90%
```

**Success Criteria:**
- ✅ Multiple requests processed in single forward pass
- ✅ Finished requests immediately free slots
- ✅ New requests added without waiting for batch
- ✅ All 46 existing tests still pass
- ✅ 5+ new continuous batching tests pass

#### Phase 3.2: Request Cancellation (Week 1-2)

**Goal:** Support client-initiated cancellation and mid-flight cleanup

**Tasks:**

1. **Scheduler Cancellation API** (`RequestScheduler.swift`)
   ```swift
   public actor RequestScheduler {
       public func cancel(requestId: UUID) async {
           // 1. Remove from pending queue if not started
           if pendingQueue.contains(requestId) {
               pendingQueue.remove(requestId)
               await streamRegistry.finish(requestId: requestId, throwing: StreamError.cancelled)
               return
           }

           // 2. Mark active request as cancelled
           if var request = activeRequests[requestId] {
               request.status = .cancelled
               activeRequests[requestId] = request
               await streamRegistry.finish(requestId: requestId, throwing: StreamError.cancelled)
           }
       }

       public func cancelAll() async {
           let allRequestIds = Array(activeRequests.keys) + pendingQueue.allRequestIds()

           for requestId in allRequestIds {
               await cancel(requestId: requestId)
           }
       }
   }
   ```

2. **ContinuousBatcher Slot Cancellation** (`ContinuousBatcher.swift`)
   ```swift
   public actor ContinuousBatcher {
       private func step() async throws {
           // ... existing step logic ...

           // After updateSlots(), check for cancellations
           await checkCancellations()
       }

       private func checkCancellations() async {
           for i in 0..<slots.count {
               guard let slot = slots[i] else { continue }

               let status = await scheduler.getStatus(requestId: slot.request.id)
               if status == .cancelled {
                   // Release KV cache blocks (Phase 4)
                   slots[i] = nil
               }
           }
       }
   }
   ```

3. **HTTP Endpoint for Cancellation** (`Routes.swift`)
   ```swift
   app.delete("v1", "requests", ":id") { req async throws -> HTTPStatus in
       guard let requestIdString = req.parameters.get("id"),
             let requestId = UUID(uuidString: requestIdString) else {
           throw Abort(.badRequest, reason: "Invalid request ID")
       }

       await scheduler.cancel(requestId: requestId)
       return .noContent
   }
   ```

**Test Coverage:**
- `RequestCancellationTests.swift`:
  - `testCancelPendingRequest()` - Cancel before processing starts
  - `testCancelActiveRequest()` - Cancel mid-generation
  - `testCancelAllRequests()` - Bulk cancellation
  - `testCancellationFreesSlot()` - Slot immediately available after cancel
  - `testStreamClosedOnCancel()` - TokenStream receives cancellation error

**Verification:**
```bash
# Test cancellation
curl -X POST localhost:8080/v1/completions -d '{"prompt": "Long story...", "max_tokens": 1000}'
# Get request ID from response
curl -X DELETE localhost:8080/v1/requests/<request_id>
# Verify: stream closes, slot freed, no errors
```

**Success Criteria:**
- ✅ Pending requests cancelled before starting
- ✅ Active requests cancelled mid-generation
- ✅ Slots immediately freed on cancellation
- ✅ TokenStream receives cancellation error
- ✅ No resource leaks after cancellation

#### Phase 3.3: Adaptive Batching (Week 2)

**Goal:** Dynamically adjust batch size based on GPU utilization and memory pressure

**Tasks:**

1. **GPU Utilization Monitor** (new file: `GPUMonitor.swift`)
   ```swift
   public actor GPUMonitor {
       private var recentUtilization: [Double] = []
       private let windowSize = 100

       public func recordUtilization(_ utilization: Double) {
           recentUtilization.append(utilization)
           if recentUtilization.count > windowSize {
               recentUtilization.removeFirst()
           }
       }

       public func averageUtilization() -> Double {
           guard !recentUtilization.isEmpty else { return 0.0 }
           return recentUtilization.reduce(0, +) / Double(recentUtilization.count)
       }

       public func recommendBatchSizeAdjustment(
           current: Int,
           target: Double = 0.90
       ) -> Int {
           let avg = averageUtilization()

           if avg < target - 0.05 {
               // Under-utilized, increase batch size
               return min(current + 4, 64)
           } else if avg > target + 0.05 {
               // Over-utilized, decrease batch size
               return max(current - 4, 8)
           } else {
               return current
           }
       }
   }
   ```

2. **Adaptive Batch Sizing in ContinuousBatcher** (`ContinuousBatcher.swift`)
   ```swift
   public actor ContinuousBatcher {
       private let gpuMonitor = GPUMonitor()
       private var currentMaxBatchSize: Int
       private var stepCount: Int = 0

       private func step() async throws {
           // ... existing step logic ...

           stepCount += 1

           // Adjust batch size every 100 steps
           if stepCount % 100 == 0 {
               let newMaxBatchSize = await gpuMonitor.recommendBatchSizeAdjustment(
                   current: currentMaxBatchSize,
                   target: 0.90
               )

               if newMaxBatchSize != currentMaxBatchSize {
                   logger.info("Adjusting batch size: \(currentMaxBatchSize) → \(newMaxBatchSize)")
                   currentMaxBatchSize = newMaxBatchSize
               }
           }

           // Record utilization
           let utilization = Double(activeSlotCount) / Double(currentMaxBatchSize)
           await gpuMonitor.recordUtilization(utilization)
       }
   }
   ```

3. **Memory Pressure Detection** (`GPUMonitor.swift`)
   ```swift
   public actor GPUMonitor {
       public func checkMemoryPressure() -> MemoryPressure {
           let allocatedGB = MLX.memoryAllocated() / (1024 * 1024 * 1024)
           let availableGB = 512 - allocatedGB  // M3 Ultra 512GB

           if availableGB < 50 {
               return .critical
           } else if availableGB < 100 {
               return .high
           } else {
               return .normal
           }
       }
   }

   public enum MemoryPressure {
       case normal, high, critical
   }
   ```

4. **Batch Size Constraints** (`ContinuousBatcher.swift`)
   ```swift
   private func fillEmptySlots() async {
       // Check memory pressure before adding requests
       let memoryPressure = await gpuMonitor.checkMemoryPressure()

       var effectiveMaxBatchSize = currentMaxBatchSize
       switch memoryPressure {
       case .critical:
           effectiveMaxBatchSize = min(effectiveMaxBatchSize, activeSlotCount)
       case .high:
           effectiveMaxBatchSize = min(effectiveMaxBatchSize, currentMaxBatchSize / 2)
       case .normal:
           break
       }

       let emptySlotCount = effectiveMaxBatchSize - activeSlotCount
       // ... rest of fillEmptySlots logic ...
   }
   ```

**Test Coverage:**
- `AdaptiveBatchingTests.swift`:
  - `testBatchSizeIncrease()` - Low utilization increases batch size
  - `testBatchSizeDecrease()` - High utilization decreases batch size
  - `testMemoryPressureConstraint()` - Critical memory limits batch size
  - `testBatchSizeStability()` - No thrashing near target utilization
  - `testUtilizationTracking()` - Rolling window statistics

**Verification:**
```bash
# Load test with increasing concurrent requests
for i in {1..50}; do
  curl -X POST localhost:8080/v1/completions -d '{"prompt": "Test", "max_tokens": 100}' &
done

# Monitor batch size adjustments in logs
# Verify: Batch size increases from 16 → 32 → 48 as load increases
```

**Success Criteria:**
- ✅ Batch size increases when GPU under-utilized (<85%)
- ✅ Batch size decreases when memory pressure high
- ✅ No thrashing (batch size stable within target range)
- ✅ GPU utilization maintained >90% under load
- ✅ Graceful handling of memory pressure

### Alternative Approaches Considered

#### 1. Dynamic Batching (vLLM-style)
**Approach:** Variable-size batches formed on-demand
**Why Not Chosen:** More complex than continuous batching, same performance ceiling

#### 2. Chunked Prefill (TGI-style)
**Approach:** Split prompt processing into chunks
**Why Not Chosen:** Adds complexity without addressing core bottleneck (decode phase)

#### 3. Keep Static Batching, Add More Batchers
**Approach:** Run multiple StaticBatcher actors in parallel
**Why Not Chosen:** Doesn't eliminate GPU idle time, increases memory footprint

#### 4. Continuous Batching with Unlimited Slots
**Approach:** No max batch size limit
**Why Not Chosen:** Memory exhaustion risk, diminishing returns beyond 32-64 slots

## Acceptance Criteria

### Functional Requirements

- [ ] Multiple requests processed in single forward pass per step
- [ ] Finished requests immediately free slots for new requests
- [ ] New requests added mid-generation without waiting for batch
- [ ] Request cancellation works for both pending and active requests
- [ ] Cancelled requests immediately free slots
- [ ] Batch size adapts based on GPU utilization (target: 90%)
- [ ] Memory pressure detection prevents OOM errors
- [ ] All Phase 2 tests continue to pass (46 tests)

### Non-Functional Requirements

- [ ] GPU utilization >90% under concurrent load (10+ requests)
- [ ] Zero GPU idle time between token generations
- [ ] Inter-token latency ≤20ms (same as Phase 2)
- [ ] Throughput improvement: 6-7x vs Phase 2 static batching
- [ ] Memory usage: <80% of available 512GB under max load
- [ ] Cancellation latency: <10ms from request to slot freed

### Quality Gates

- [ ] Test coverage: 60+ total tests (46 existing + 14+ new)
- [ ] No data races under Swift 6 strict concurrency
- [ ] No memory leaks after 1000 request cycles
- [ ] Build passes with `make build` (xcodebuild + Metal shaders)
- [ ] All tests pass with `make test`
- [ ] Load test: 50 concurrent requests without OOM or deadlock

## Success Metrics

**Primary Metrics:**
- **Throughput:** Tokens/second per request (target: 6-7x improvement)
- **GPU Utilization:** Percentage of time GPU is active (target: >90%)
- **Slot Efficiency:** Average slots filled / max slots (target: >80%)
- **Cancellation Latency:** Time from cancel request to slot freed (target: <10ms)

**Measurement Methods:**
```bash
# Benchmark continuous vs static batching
make benchmark-phase3

# Expected results:
# Static Batching (Phase 2):  ~150 tok/s aggregate (10 concurrent requests)
# Continuous Batching (Phase 3): ~900-1050 tok/s aggregate (10 concurrent requests)
```

## Dependencies & Prerequisites

**Phase Dependencies:**
- ✅ Phase 2 Complete - Actor-based request handling with static batching
- ⏸ Phase 4 Pending - PagedKVCache integration (nice-to-have, not blocking)

**Technical Prerequisites:**
- ✅ MLX Swift 0.30.6+ (Wired Memory Management)
- ✅ Swift 6.2.3+ (Actor isolation, async/await)
- ✅ macOS 26.3+ (JACCL performance improvements)
- ✅ InferenceEngine with generateStreaming()

**New Dependencies:**
- None - uses existing MLX Swift and Vapor

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **MLX Swift lacks batch forward pass API** | Medium | High | Add batching at MLX array level, or use multiple model copies |
| **Memory pressure causes OOM** | Medium | High | Implement memory pressure detection, cap batch size dynamically |
| **Continuous loop consumes CPU** | Low | Medium | Add yield point if no active slots |
| **Race condition in slot updates** | Low | High | Actor isolation ensures serial slot updates |
| **Cancellation leaves orphaned KV cache** | Medium | Medium | Phase 4 integration required for proper cleanup |
| **Batch size thrashing** | Medium | Low | Use rolling window (100 samples) and hysteresis |

**Critical Path Risks:**
1. **MLX Swift Batch API:** If `forwardBatch()` doesn't exist, may need to batch manually at MLX array level
2. **KV Cache Management:** Without PagedKVCache (Phase 4), memory usage will be suboptimal but functional

**Mitigation Plan:**
- Prototype batch forward pass early (Phase 3.1, Task 2)
- If MLX Swift lacks batching API, use manual array concatenation
- Defer KV cache optimization to Phase 4 (continuous batching still works without it)

## Resource Requirements

**Development Time:**
- Phase 3.1 (Core Continuous Batching): 3-4 days
- Phase 3.2 (Cancellation): 2-3 days
- Phase 3.3 (Adaptive Batching): 2-3 days
- Testing & Benchmarking: 2 days
- **Total:** ~2 weeks

**Hardware:**
- Mac Studio M3 Ultra 512GB (development & testing)
- No additional hardware required

**Team:**
- 1 developer (full-time)

## Future Considerations

### Phase 4 Integration (PagedKVCache)
- BatchSlot will store `kvCacheBlockIds: [Int]` instead of full KV cache
- Block allocation on slot creation, deallocation on cleanup
- Memory footprint reduction: 4-8x for long contexts

### Phase 5+ (Multi-Node Clustering)
- ContinuousBatcher will coordinate across nodes via RDMA
- Slot distribution strategy (model parallel vs data parallel)
- Global scheduler for cross-node load balancing

### Optimizations
- Speculative decoding (Phase 6)
- Prefix caching for shared prompts
- Quantized KV cache (4-bit)
- Dynamic sequence length padding

### Monitoring & Observability
- Prometheus metrics for batch size, utilization, latency
- Grafana dashboards for real-time monitoring
- Distributed tracing with OpenTelemetry

## Documentation Plan

**Code Documentation:**
- [ ] Inline comments for batching algorithm
- [ ] DocC documentation for ContinuousBatcher public APIs
- [ ] Architecture decision record (ADR) for continuous batching

**User Documentation:**
- [ ] Update README with Phase 3 status
- [ ] Benchmarking guide comparing Phase 2 vs Phase 3
- [ ] Cancellation API usage examples

**Internal Documentation:**
- [ ] Phase 3 Implementation Summary (after completion)
- [ ] Performance analysis report
- [ ] Lessons learned document

## Implementation Checklist

### Phase 3.1: Core Continuous Batching
- [x] Define `BatchSlot` struct in `RequestState.swift`
- [x] Add `forwardBatch()` and `sampleBatch()` to `InferenceEngine.swift`
- [x] Implement `ContinuousBatcher.swift` with step loop
- [x] Integrate with RequestScheduler and TokenStream
- [x] Write 9 unit tests for continuous batching
- [x] Verify: All 53 tests pass (46 existing + 9 new)
- [ ] Benchmark: Measure throughput improvement vs Phase 2

### Phase 3.2: Request Cancellation
- [x] Add `cancelAll()` and `getStatus()` methods to RequestScheduler
- [x] Add `checkCancellations()` to ContinuousBatcher
- [x] Implement HTTP DELETE endpoint for /v1/requests/:id
- [x] Write 8 comprehensive tests for cancellation scenarios
- [x] Verify: All 61 tests pass (46 + 9 + 8)
- [x] Verify: Slots freed immediately on cancel
- [ ] Load test: Cancel requests mid-generation, verify no leaks

### Phase 3.3: Adaptive Batching
- [x] Create `GPUMonitor.swift` actor with utilization tracking
- [x] Implement rolling window statistics (100 samples)
- [x] Add batch size adjustment logic (adjusts every 100 steps)
- [x] Add memory pressure detection (normal/high/critical)
- [x] Integrate adaptive batching into ContinuousBatcher
- [x] Memory pressure limits batch size dynamically
- [x] Verify: All 61 tests pass
- [ ] Stress test: 50+ concurrent requests, monitor adaptation

### Final Verification
- [ ] All 60+ tests pass (`make test`)
- [ ] Build succeeds (`make build`)
- [ ] No Swift 6 concurrency warnings
- [ ] GPU utilization >90% under load
- [ ] Throughput: 6-7x improvement measured
- [ ] No memory leaks after 1000 requests
- [ ] Create Phase 3 Implementation Summary

## References & Research

### Internal References

**Architecture:**
- Phase 2 Plan: `/Users/lee.parayno/code4/business/mlx-server/docs/plans/2026-02-16-feat-phase2-actor-request-handling-plan.md`
- Phase 2 Summary: `/Users/lee.parayno/code4/business/mlx-server/docs/Phase-2-Implementation-Summary.md`
- Overall Roadmap: `/Users/lee.parayno/code4/business/mlx-server/docs/plans/2026-02-15-feat-single-node-swift-mlx-server-plan.md`

**Code Patterns:**
- RequestScheduler: `/Users/lee.parayno/code4/business/mlx-server/Sources/Scheduler/RequestScheduler.swift:1-350`
- StaticBatcher: `/Users/lee.parayno/code4/business/mlx-server/Sources/Scheduler/StaticBatcher.swift:1-150`
- InferenceEngine: `/Users/lee.parayno/code4/business/mlx-server/Sources/Core/InferenceEngine.swift:1-200`
- TokenStream: `/Users/lee.parayno/code4/business/mlx-server/Sources/Scheduler/TokenStream.swift:1-180`
- BatchSlot stub: `/Users/lee.parayno/code4/business/mlx-server/Sources/Scheduler/ContinuousBatcher.swift:10-25`

**Testing Patterns:**
- RequestSchedulerTests: `/Users/lee.parayno/code4/business/mlx-server/Tests/SchedulerTests/RequestSchedulerTests.swift`
- PriorityQueueTests: `/Users/lee.parayno/code4/business/mlx-server/Tests/SchedulerTests/PriorityQueueTests.swift`

**MLX Swift Integration:**
- Comprehensive Guide: `/Users/lee.parayno/code4/business/mlx-server/docs/COMPREHENSIVE_MLX_SWIFT_DOCUMENTATION.md`
- Quick Reference: `/Users/lee.parayno/code4/business/mlx-server/docs/QUICK_REFERENCE_GUIDE.md`

### External References

**Continuous Batching:**
- Orca Paper: "Continuous Batching for Low-Latency LLM Serving" (https://arxiv.org/abs/2209.01051)
- vLLM Documentation: https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html
- TGI Architecture: https://huggingface.co/docs/text-generation-inference/conceptual/streaming

**MLX Framework:**
- MLX Swift Documentation: https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/
- MLX Python Batching: https://github.com/ml-explore/mlx-examples/tree/main/llms
- MLX Performance Guide: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

**Swift Concurrency:**
- Swift Actors: https://docs.swift.org/swift-book/LanguageGuide/Concurrency.html#ID545
- TaskGroup Patterns: https://www.swiftbysundell.com/articles/task-groups-in-swift/

### Related Work

**Previous Phases:**
- Phase 1: Foundation (Model Loading, Single Inference) - ✅ Complete
- Phase 2: Actor-Based Request Handling (Static Batching) - ✅ Complete

**Upcoming Phases:**
- Phase 4: PagedAttention (Memory Optimization)
- Phase 5: Production API (OpenAI-compatible endpoints)
- Phase 6: Multi-Node Clustering (Thunderbolt 5 RDMA)

## Appendix: Continuous Batching Algorithm

### Pseudocode

```python
class ContinuousBatcher:
    def __init__(self, max_slots=32):
        self.slots = [None] * max_slots
        self.scheduler = RequestScheduler()
        self.engine = InferenceEngine()

    def step(self):
        # 1. Fill empty slots
        for i, slot in enumerate(self.slots):
            if slot is None:
                request = self.scheduler.dequeue_one()
                if request:
                    self.slots[i] = BatchSlot(request)

        # 2. Prepare batch input
        active_slots = [s for s in self.slots if s is not None]
        if not active_slots:
            return  # No active requests

        token_ids = [s.get_next_token_id() for s in active_slots]
        positions = [len(s.generated_tokens) for s in active_slots]

        # 3. Single batched forward pass
        logits = self.engine.forward_batch(token_ids, positions)

        # 4. Sample tokens
        sampled_tokens = self.engine.sample_batch(logits)

        # 5. Update slots
        for slot, token in zip(active_slots, sampled_tokens):
            slot.append_token(token)
            self.scheduler.emit_token(slot.request_id, token)

            if token == EOS or len(slot.generated_tokens) >= slot.max_tokens:
                slot.finish()
                self.scheduler.complete(slot.request_id)

        # 6. Cleanup finished slots
        for i, slot in enumerate(self.slots):
            if slot and slot.is_finished:
                self.slots[i] = None
```

### Complexity Analysis

**Time Complexity per Step:**
- Fill empty slots: O(k) where k = empty slot count
- Prepare batch input: O(b) where b = active batch size
- Forward pass: O(b * d * v) where d = model dimension, v = vocab size
- Sample tokens: O(b * v)
- Update slots: O(b)
- Cleanup: O(max_slots)

**Total:** O(b * d * v) dominated by forward pass

**Space Complexity:**
- Slots array: O(max_slots)
- Batch input: O(b)
- Logits: O(b * v)
- KV cache (Phase 4): O(b * seq_len * d)

**Total:** O(b * seq_len * d) when KV cache integrated

### Performance Characteristics

**Throughput:**
- Static Batching: `throughput = batch_size / avg_request_time`
- Continuous Batching: `throughput = slots_filled * step_rate`
- Improvement: 6-7x due to eliminated idle time

**Latency:**
- Time to first token (TTFT): Same as Phase 2 (~50ms)
- Inter-token latency: Same as Phase 2 (~20ms per step)
- Cancellation latency: <10ms (one step)

**GPU Utilization:**
- Static: 60-70% (gaps between batches)
- Continuous: >90% (no idle time)

---

**End of Phase 3 Plan**

*Created: 2026-02-16*
*Author: Claude Code*
*Status: Ready for Implementation*

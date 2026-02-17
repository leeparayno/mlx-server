---
title: Phase 2 - Actor-Based Request Handling
type: feat
date: 2026-02-16
status: planned
priority: high
estimated_duration: 3-4 weeks
phase: 2
dependencies: Phase 1 (Foundation - Complete)
---

# Phase 2: Actor-Based Request Handling - Implementation Plan

## Context

Phase 1 (Foundation) is complete with working model loading, single-request inference, and Metal shader compilation. Phase 2 builds on this foundation to enable **concurrent request handling with static batching**, laying the groundwork for Phase 3's continuous batching.

**Why this change is needed:**
- Current InferenceEngine processes requests serially (one at a time)
- GPU sits idle between requests, wasting M3 Ultra's 800 GB/s memory bandwidth
- No request prioritization or timeout handling
- Callback-based streaming is hard to compose with HTTP/SSE
- Target: 10+ concurrent requests with 3-5x throughput improvement

**Current state:**
- InferenceEngine actor exists with single-request methods
- RequestScheduler has basic submit() but minimal state tracking
- ContinuousBatcher is a stub (Phase 3 will complete it)
- PagedKVCache exists but unused (Phase 4 will integrate it)

## Goals

1. **Request Lifecycle Management**: Track full state (pending → active → streaming → completed/failed/timeout)
2. **Priority Queue**: Fair scheduling with FIFO within priority levels (critical > high > normal > low)
3. **Timeout Handling**: Prevent resource leaks from abandoned requests
4. **AsyncSequence Streaming**: Replace callbacks with composable AsyncSequence for token streaming
5. **Static Batching**: Process multiple requests concurrently (simpler than continuous batching)

**Success Criteria:**
- Process 10+ concurrent requests without data races
- Priority ordering respected (high priority served first)
- Requests timeout after configured threshold
- Stream tokens via AsyncSequence without drops
- 3-5x throughput improvement vs sequential processing

## Implementation Approach

### 1. Request State Management

**New File:** `Sources/Scheduler/RequestState.swift`

Define request lifecycle states and priority:

```swift
public enum RequestStatus: String, Sendable {
    case pending, active, streaming, completed, failed, timeout, cancelled
}

public enum RequestPriority: Int, Comparable, Sendable {
    case low = 0, normal = 1, high = 2, critical = 3
}

public struct StatefulRequest: Sendable {
    let id: UUID
    let request: InferenceRequest
    var status: RequestStatus
    let priority: RequestPriority
    let submittedAt: Date
    var startedAt: Date?
    var completedAt: Date?

    var age: TimeInterval { Date().timeIntervalSince(submittedAt) }
    var isActive: Bool { [.active, .streaming].contains(status) }
    var isTerminal: Bool { [.completed, .failed, .timeout, .cancelled].contains(status) }
}
```

**Rationale:** Explicit state machine with timestamps enables timeout detection and observability.

### 2. Token Streaming with AsyncSequence

**New File:** `Sources/Scheduler/TokenStream.swift`

Replace callback pattern with AsyncSequence:

```swift
public struct TokenChunk: Sendable {
    let requestId: UUID
    let token: String
    let index: Int
    let timestamp: Date
}

public struct GenerationInfo: Sendable {
    let requestId: UUID
    let totalTokens: Int
    let duration: TimeInterval
    let tokensPerSecond: Double
    let finishReason: FinishReason  // .stop, .length, .timeout, .error
}

// AsyncThrowingStream wrapper for token streaming
public struct TokenStream: AsyncSequence {
    typealias Element = TokenChunk
    // Internal: AsyncThrowingStream with continuation
    // Methods: yield(_:), finish(with:), finish(throwing:)
}

actor TokenStreamRegistry {
    // Manages active token streams by request ID
    // Methods: register(requestId:), yield(requestId:chunk:), finish(requestId:)
}
```

**Rationale:** AsyncSequence provides backpressure-aware streaming, composability, and clean cancellation semantics.

### 3. Priority Queue

**New File:** `Sources/Scheduler/PriorityQueue.swift`

Efficient priority-based queue with FIFO within levels:

```swift
struct PriorityRequestQueue {
    private var queues: [RequestPriority: [StatefulRequest]]

    mutating func enqueue(_ request: StatefulRequest)
    mutating func dequeue() -> StatefulRequest?
    mutating func dequeue(count: Int) -> [StatefulRequest]

    var count: Int
    var countsByPriority: [RequestPriority: Int]
}
```

**Rationale:** Simple array-based implementation per priority level. O(1) enqueue, O(1) amortized dequeue.

### 4. Enhanced Request Scheduler

**Modify:** `Sources/Scheduler/RequestScheduler.swift`

Upgrade to handle priority queue, timeouts, and stream management:

**Key changes:**
- Replace simple arrays with `PriorityRequestQueue`
- Add `TokenStreamRegistry` for stream lifecycle management
- Add `SchedulerConfig` with `maxConcurrentRequests`, `requestTimeout`, `timeoutCheckInterval`
- Track statistics: total/completed/failed/timeout requests

**New methods:**
```swift
// Submit request with priority, returns UUID + TokenStream
func submit(_ request: InferenceRequest, priority: RequestPriority = .normal) async -> (UUID, TokenStream)

// Get next batch for processing (up to batchSize)
func dequeueNextBatch(size: Int) -> [StatefulRequest]

// Update request state
func markStreaming(requestId: UUID)
func emitToken(requestId: UUID, token: String, index: Int) async
func complete(requestId: UUID, info: GenerationInfo) async
func fail(requestId: UUID, error: Error) async

// Timeout handling
func handleTimeouts() async

// Observability
var stats: SchedulerStats
```

**Rationale:** Central coordinator for request lifecycle. Actor isolation ensures thread safety. Stream registry manages AsyncSequence lifecycle across actor boundaries.

### 5. Static Batch Processor

**New File:** `Sources/Scheduler/StaticBatcher.swift`

Simple static batching loop:

```swift
public actor StaticBatcher {
    private let engine: InferenceEngine
    private let scheduler: RequestScheduler
    private let config: BatchConfig

    func start() async  // Start batching loop
    func stop()         // Stop loop

    private func runBatchingLoop() async {
        while isRunning {
            // 1. Get next batch from scheduler
            let batch = await scheduler.dequeueNextBatch(size: config.maxBatchSize)

            // 2. Process batch concurrently
            await processBatch(batch)
        }
    }

    private func processBatch(_ batch: [StatefulRequest]) async {
        // Process requests concurrently using TaskGroup
        await withTaskGroup(of: Void.self) { group in
            for request in batch {
                group.addTask { await processRequest(request) }
            }
        }
    }

    private func processRequest(_ request: StatefulRequest) async {
        // Generate with streaming callback
        // Emit tokens via scheduler.emitToken()
        // Complete/fail via scheduler.complete()/fail()
    }
}
```

**Rationale:** Static batching is simpler than continuous batching - collect batch, process concurrently, repeat. Uses Swift's TaskGroup for structured concurrency. Provides foundation for Phase 3.

### 6. Update InferenceRequest

**Modify:** `Sources/Scheduler/RequestScheduler.swift` (InferenceRequest struct)

Add fields:
- `topP: Double = 1.0` - Sampling parameter
- Change `stream: Bool` default to `true`

Maintain `Sendable` conformance for actor isolation.

## Integration Flow

```
Client Request
    ↓
RequestScheduler.submit(priority) → (UUID, TokenStream)
    ↓
PriorityQueue.enqueue(StatefulRequest)
    ↓
StaticBatcher.dequeueNextBatch()
    ↓
Process batch with TaskGroup
    ↓
InferenceEngine.generateStreaming() → tokens via callback
    ↓
Scheduler.emitToken() → TokenStream.yield()
    ↓
Client receives AsyncSequence<TokenChunk>
    ↓
Scheduler.complete(info) → TokenStream.finish()
```

**Timeout checker runs in parallel:**
```
Timer (every 10s)
    ↓
Scheduler.handleTimeouts()
    ↓
Check age of active requests
    ↓
Timeout old requests → TokenStream.finish(throwing:)
```

## Critical Files

1. **Sources/Scheduler/RequestState.swift** - NEW: Request lifecycle state machine
2. **Sources/Scheduler/TokenStream.swift** - NEW: AsyncSequence streaming infrastructure
3. **Sources/Scheduler/PriorityQueue.swift** - NEW: Priority queue implementation
4. **Sources/Scheduler/RequestScheduler.swift** - MAJOR MODIFY: Priority, timeout, streams
5. **Sources/Scheduler/StaticBatcher.swift** - NEW: Static batch processor
6. **Sources/Core/InferenceEngine.swift** - MINOR: Use existing streaming callback pattern

## Testing Strategy

### Unit Tests

1. **RequestStateTests.swift** - State transitions, priority comparison
2. **PriorityQueueTests.swift** - FIFO within priority, dequeue batch
3. **TokenStreamTests.swift** - AsyncSequence emission, completion
4. **RequestSchedulerTests.swift** - Submit, dequeue, timeout, statistics

### Integration Tests

1. **ConcurrentRequestTests.swift** - 10+ concurrent requests, all complete
2. **TimeoutTests.swift** - Requests timeout after threshold
3. **PriorityTests.swift** - High priority served before normal

### Load Tests

1. **benchmarks/LoadTest.swift** - N concurrent clients, M requests each
2. Measure: throughput, latency, memory usage
3. Target: 3-5x improvement vs sequential

## Build Verification

```bash
# Build with Metal shaders
make build

# Run unit tests
make test

# Run load test
make benchmark
```

**Verification checklist:**
- [ ] All targets build without errors
- [ ] Metal shaders compile successfully
- [ ] Unit tests pass (100%)
- [ ] Integration tests pass
- [ ] Load test completes without crashes
- [ ] No data race warnings (Swift 6)
- [ ] Memory usage stable under load

## Timeline

**Total: 3-4 weeks**

**Week 1: Core Infrastructure**
- RequestState.swift with state enum and StatefulRequest
- PriorityQueue.swift with tests
- TokenStream.swift with AsyncSequence
- Unit tests for all

**Week 2: Scheduler Enhancement**
- Integrate priority queue into RequestScheduler
- Add timeout handling logic
- Add stream registry
- Update tests

**Week 3: Static Batching**
- Implement StaticBatcher actor
- Connect scheduler → batcher → engine
- Integration tests
- Load testing

**Week 4: Polish & Optimization**
- Fix any race conditions found in testing
- Optimize performance bottlenecks
- Documentation
- Final verification

## Success Metrics

**Functional:**
- ✅ Process 10+ concurrent requests without races
- ✅ Priority ordering respected
- ✅ Timeouts work correctly
- ✅ AsyncSequence streaming without drops
- ✅ Error handling doesn't crash system

**Performance:**
- ✅ 3-5x throughput vs sequential
- ✅ Average queue latency < 5ms
- ✅ Memory stable (no leaks)
- ✅ CPU usage < 80% during batching

## Dependencies

**External:**
- mlx-swift v0.30.6 (Metal shaders)
- mlx-swift-lm v2.30.3 (ModelContainer)
- swift-async-algorithms v1.1.2 (AsyncSequence utilities)
- swift-log v1.9.1 (Logger)

**Internal:**
- Core module (InferenceEngine, ModelLoader)
- Scheduler module (existing foundation)

## Risks & Mitigation

**Risk 1: AsyncSequence Backpressure**
- Use AsyncThrowingStream with buffering
- Monitor stream lag
- Add flow control if needed

**Risk 2: Timeout Check Performance**
- Batch checks (every 10s, not per-request)
- Use efficient Date comparisons
- Profile under load

**Risk 3: Priority Starvation**
- Monitor queue depths by priority
- Consider priority aging in Phase 3 if needed

**Risk 4: Metal Compilation Issues**
- Use xcodebuild exclusively (never swift build)
- Test on clean builds
- Follow build verification checklist

## Future Enhancements (Phase 3+)

Phase 3 will build on this foundation to add:
1. **Continuous batching** - Dynamic slot management instead of static batches
2. **KV cache integration** - Connect PagedKVCache for memory efficiency
3. **Request cancellation** - Allow clients to cancel in-flight requests
4. **Adaptive batching** - Dynamically adjust batch size based on load

## References

- Original Plan: `docs/plans/2026-02-15-feat-single-node-swift-mlx-server-plan.md` (Phase 2, lines 82-101)
- InferenceEngine: `Sources/Core/InferenceEngine.swift`
- RequestScheduler: `Sources/Scheduler/RequestScheduler.swift`
- ContinuousBatcher (stub): `Sources/Scheduler/ContinuousBatcher.swift`

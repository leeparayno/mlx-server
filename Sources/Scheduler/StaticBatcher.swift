import Foundation
import Core
import Logging

// MARK: - Batch Configuration

/// Configuration for static batching
public struct BatchConfig: Sendable {
    public let maxBatchSize: Int
    public let batchInterval: TimeInterval

    public init(
        maxBatchSize: Int = 10,
        batchInterval: TimeInterval = 0.1  // 100ms
    ) {
        self.maxBatchSize = maxBatchSize
        self.batchInterval = batchInterval
    }
}

// MARK: - Static Batcher

/// Implements static batching for concurrent request processing
public actor StaticBatcher {
    private let engine: InferenceEngine
    private let scheduler: RequestScheduler
    private let config: BatchConfig
    private let logger = Logger(label: "static-batcher")

    private var isRunning: Bool = false
    private var batchTask: Task<Void, Never>?

    public init(
        engine: InferenceEngine,
        scheduler: RequestScheduler,
        config: BatchConfig = BatchConfig()
    ) {
        self.engine = engine
        self.scheduler = scheduler
        self.config = config
    }

    // MARK: - Lifecycle

    /// Start the batching loop
    public func start() {
        guard !isRunning else {
            logger.warning("Batcher already running")
            return
        }

        isRunning = true
        batchTask = Task { [weak self] in
            guard let self = self else { return }
            await self.runBatchingLoop()
        }

        logger.info("Static batcher started", metadata: [
            "max_batch_size": "\(config.maxBatchSize)",
            "batch_interval": "\(config.batchInterval)s"
        ])
    }

    /// Stop the batching loop
    public func stop() {
        isRunning = false
        batchTask?.cancel()
        batchTask = nil
        logger.info("Static batcher stopped")
    }

    // MARK: - Batching Loop

    /// Main batching loop - continuously process batches
    private func runBatchingLoop() async {
        logger.debug("Batching loop started")

        while isRunning {
            do {
                // Get next batch from scheduler
                let batch = await scheduler.dequeueNextBatch(size: config.maxBatchSize)

                if !batch.isEmpty {
                    // Process batch concurrently
                    await processBatch(batch)
                } else {
                    // No requests available, wait before checking again
                    try await Task.sleep(for: .milliseconds(Int(config.batchInterval * 1000)))
                }
            } catch {
                if error is CancellationError {
                    logger.debug("Batching loop cancelled")
                    break
                }
                logger.error("Error in batching loop", metadata: [
                    "error": "\(error.localizedDescription)"
                ])
            }
        }

        logger.debug("Batching loop exited")
    }

    /// Process a batch of requests concurrently
    private func processBatch(_ batch: [StatefulRequest]) async {
        logger.debug("Processing batch", metadata: [
            "batch_size": "\(batch.count)"
        ])

        let startTime = Date()

        // Process requests concurrently using TaskGroup
        await withTaskGroup(of: Void.self) { group in
            for request in batch {
                group.addTask {
                    await self.processRequest(request)
                }
            }
        }

        let duration = Date().timeIntervalSince(startTime)
        logger.debug("Batch completed", metadata: [
            "batch_size": "\(batch.count)",
            "duration": "\(String(format: "%.3f", duration))s"
        ])
    }

    /// Process a single request
    private func processRequest(_ statefulRequest: StatefulRequest) async {
        let requestId = statefulRequest.id
        let request = statefulRequest.request

        logger.trace("Processing request", metadata: [
            "request_id": "\(requestId)",
            "prompt_length": "\(request.prompt.count)",
            "max_tokens": "\(request.maxTokens)"
        ])

        let startTime = Date()

        do {
            // Mark as streaming
            await scheduler.markStreaming(requestId: requestId)

            // Use actor-isolated state for token counting
            final class TokenCounter: @unchecked Sendable {
                private var _count = 0
                private let lock = NSLock()

                var count: Int {
                    lock.lock()
                    defer { lock.unlock() }
                    return _count
                }

                func increment() {
                    lock.lock()
                    defer { lock.unlock() }
                    _count += 1
                }
            }

            let counter = TokenCounter()

            // Generate with streaming callback
            _ = try await engine.generateStreaming(
                prompt: request.prompt,
                maxTokens: request.maxTokens,
                temperature: request.temperature
            ) { @Sendable token in
                // Get current count and increment
                let currentIndex = counter.count
                counter.increment()

                // Emit token to stream
                await self.scheduler.emitToken(
                    requestId: requestId,
                    token: token,
                    index: currentIndex
                )
            }

            // Calculate completion info
            let duration = Date().timeIntervalSince(startTime)
            let tokenCount = counter.count
            let info = GenerationInfo(
                requestId: requestId,
                totalTokens: tokenCount,
                duration: duration,
                finishReason: tokenCount >= request.maxTokens ? .length : .stop
            )

            // Mark as completed
            await scheduler.complete(requestId: requestId, info: info)

        } catch {
            logger.error("Request processing failed", metadata: [
                "request_id": "\(requestId)",
                "error": "\(error.localizedDescription)"
            ])

            // Mark as failed
            await scheduler.fail(requestId: requestId, error: error)
        }
    }

    // MARK: - Status

    /// Check if batcher is running
    public var running: Bool {
        isRunning
    }

    /// Get current statistics
    public func getStats() async -> BatcherStats {
        let schedulerStats = await scheduler.stats
        return BatcherStats(
            isRunning: isRunning,
            maxBatchSize: config.maxBatchSize,
            schedulerStats: schedulerStats
        )
    }
}

// MARK: - Batcher Statistics

/// Statistics about batcher performance
public struct BatcherStats: Sendable {
    public let isRunning: Bool
    public let maxBatchSize: Int
    public let schedulerStats: SchedulerStats

    public init(
        isRunning: Bool,
        maxBatchSize: Int,
        schedulerStats: SchedulerStats
    ) {
        self.isRunning = isRunning
        self.maxBatchSize = maxBatchSize
        self.schedulerStats = schedulerStats
    }
}

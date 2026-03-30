import Foundation
import Core
import MLXLMCommon
import Memory
import Logging

/// Implements continuous batching for maximum GPU utilization
public actor ContinuousBatcher {
    private let logger = Logger(label: "continuous-batcher")

    // Slot management
    private var slots: [BatchSlot?]
    private var currentMaxBatchSize: Int
    private let initialMaxBatchSize: Int

    // Dependencies
    private let scheduler: RequestScheduler
    private let engine: InferenceEngine
    private let gpuMonitor: GPUMonitor
    private let kvCache: PagedKVCache

    // State
    private var isRunning: Bool = false
    private var stepCount: Int = 0

    // Configuration
    public struct Config: Sendable {
        public let maxBatchSize: Int
        public let eosTokenId: Int
        public let kvQuantization: QuantizationConfig

        public init(maxBatchSize: Int = 32, eosTokenId: Int = 2, kvQuantization: QuantizationConfig = QuantizationConfig()) {
            self.maxBatchSize = maxBatchSize
            self.eosTokenId = eosTokenId
            self.kvQuantization = kvQuantization
        }
    }

    private let config: Config

    public init(
        scheduler: RequestScheduler,
        engine: InferenceEngine,
        config: Config = Config()
    ) {
        self.scheduler = scheduler
        self.engine = engine
        self.config = config
        self.initialMaxBatchSize = config.maxBatchSize
        self.currentMaxBatchSize = config.maxBatchSize
        self.slots = Array(repeating: nil, count: config.maxBatchSize)
        self.gpuMonitor = GPUMonitor(config: GPUMonitor.Config(
            maxBatchSize: config.maxBatchSize
        ))
        self.kvCache = PagedKVCache(blockSize: 16, numBlocks: 1024, quantization: config.kvQuantization)
    }

    // MARK: - Main Loop

    /// Start the continuous batching loop
    public func start() async {
        guard !isRunning else {
            logger.warning("Batcher already running")
            return
        }

        // Attach KV cache to inference engine (scaffolding for Phase 4.3)
        await engine.attachKVCache(kvCache)

        isRunning = true
        logger.info("Starting continuous batcher", metadata: [
            "initial_max_batch_size": "\(initialMaxBatchSize)"
        ])

        while isRunning {
            do {
                try await step()
            } catch {
                logger.error("Batching step failed", metadata: [
                    "error": "\(error.localizedDescription)"
                ])
            }

            // Small yield to prevent tight loop if no active slots
            if activeSlotCount == 0 {
                try? await Task.sleep(for: .milliseconds(10))
            }
        }

        logger.info("Continuous batcher stopped")
    }

    /// Stop the continuous batching loop
    public func stop() async {
        isRunning = false

        // Final cleanup pass: check for cancellations and free slots
        await checkCancellations()
    }

    /// Execute one batching step
    internal func step() async throws {
        stepCount += 1

        // 1. Check for cancellations first (before early return)
        await checkCancellations()

        // 2. Fill empty slots from scheduler
        await fillEmptySlots()

        // 3. Prepare batch input
        guard let batchInput = prepareBatchInput() else {
            return  // No active slots
        }

        // 4. Batched forward pass (Phase 4.2: with KV cache and sampling params)
        let nextTokens = try await engine.forwardBatch(
            tokenIds: batchInput.tokenIds,
            positions: batchInput.positions,
            prompts: batchInput.prompts,
            kvCacheBlockIds: batchInput.kvCacheBlockIds,
            temperatures: Array(repeating: 0.7, count: batchInput.tokenIds.count),  // TODO: from request config
            topP: Array(repeating: 0.95, count: batchInput.tokenIds.count)  // TODO: from request config
        )

        // 5. Update slots with new tokens
        await updateSlots(nextTokens: nextTokens, activeIndices: batchInput.activeIndices)

        // 6. Cleanup finished slots
        await cleanupFinishedSlots()

        // 7. Adjust batch size every 100 steps (Phase 3.3)
        if stepCount % 100 == 0 {
            await adjustBatchSize()
        }

        // 8. Record utilization (Phase 4.3: combine slot and memory utilization)
        let slotUtilization = Double(activeSlotCount) / Double(currentMaxBatchSize)

        // Get memory utilization from KV cache stats
        let kvStats = await kvCache.stats
        let memoryUtilization = kvStats.utilizationPercent / 100.0

        // Use max of slot and memory utilization for adaptive sizing
        let combinedUtilization = max(slotUtilization, memoryUtilization)
        await gpuMonitor.recordUtilization(combinedUtilization)

        let queueLen = await scheduler.queueLength
        logger.trace("Batching step completed", metadata: [
            "step": "\(stepCount)",
            "active_slots": "\(activeSlotCount)",
            "pending_queue": "\(queueLen)",
            "slot_utilization": "\(String(format: "%.2f", slotUtilization))",
            "memory_utilization": "\(String(format: "%.2f", memoryUtilization))",
            "combined_utilization": "\(String(format: "%.2f", combinedUtilization))"
        ])
    }

    // MARK: - Slot Management

    /// Fill empty slots with pending requests
    private func fillEmptySlots() async {
        // Check memory pressure (Phase 4.3: with real KV cache stats)
        let kvStats = await kvCache.stats
        let memoryPressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)

        // Calculate effective max batch size based on memory pressure
        var effectiveMaxBatchSize = currentMaxBatchSize
        switch memoryPressure {
        case .critical:
            // Don't add new requests if memory critical
            effectiveMaxBatchSize = min(effectiveMaxBatchSize, activeSlotCount)
            logger.warning("Critical memory pressure, limiting batch size", metadata: [
                "current_active": "\(activeSlotCount)",
                "max_batch_size": "\(currentMaxBatchSize)"
            ])
        case .high:
            // Reduce to half if memory high
            effectiveMaxBatchSize = min(effectiveMaxBatchSize, currentMaxBatchSize / 2)
            logger.info("High memory pressure, reducing batch size", metadata: [
                "effective_max": "\(effectiveMaxBatchSize)"
            ])
        case .normal:
            // Phase 4.3: Limit by available KV cache blocks
            // Calculate max slots based on available blocks
            // Assume 512 tokens per request on average (reasonable for most LLM workloads)
            let avgTokensPerRequest = 512
            let blocksPerRequest = (avgTokensPerRequest + 15) / 16  // 16 tokens per block (PagedKVCache default)
            let maxSlotsByBlocks = kvStats.freeBlocks / blocksPerRequest
            effectiveMaxBatchSize = min(effectiveMaxBatchSize, maxSlotsByBlocks)

            if maxSlotsByBlocks < currentMaxBatchSize {
                logger.info("Limiting batch size by available KV cache blocks", metadata: [
                    "free_blocks": "\(kvStats.freeBlocks)",
                    "blocks_per_request": "\(blocksPerRequest)",
                    "max_slots_by_blocks": "\(maxSlotsByBlocks)",
                    "effective_max": "\(effectiveMaxBatchSize)"
                ])
            }
        }

        var emptySlotCount = 0
        for slot in slots where slot == nil {
            emptySlotCount += 1
        }

        // Limit empty slot count by effective max batch size
        let targetEmptySlots = min(emptySlotCount, effectiveMaxBatchSize - activeSlotCount)
        guard targetEmptySlots > 0 else { return }

        // Dequeue requests from scheduler
        let newRequests = await scheduler.dequeueNextBatch(size: targetEmptySlots)

        guard !newRequests.isEmpty else { return }

        // Assign to empty slots
        var requestIndex = 0
        for i in 0..<slots.count {
            guard requestIndex < newRequests.count else { break }

            if slots[i] == nil {
                let request = newRequests[requestIndex]

                // Allocate KV cache blocks for this request
                do {
                    let kvBlockIds = try await kvCache.allocate(
                        for: request.id,
                        numTokens: request.request.maxTokens
                    )

                    slots[i] = BatchSlot(
                        slotId: i,
                        request: request,
                        kvCacheBlockIds: kvBlockIds
                    )

                    // Mark as streaming
                    await scheduler.markStreaming(requestId: request.id)

                    logger.debug("Request assigned to slot", metadata: [
                        "slot_id": "\(i)",
                        "request_id": "\(request.id)",
                        "kv_blocks": "\(kvBlockIds.count)"
                    ])

                    requestIndex += 1
                } catch {
                    // Failed to allocate KV cache blocks (memory pressure)
                    logger.warning("Failed to allocate KV cache blocks, requeueing request", metadata: [
                        "request_id": "\(request.id)",
                        "error": "\(error.localizedDescription)"
                    ])

                    // Enqueue the request back to try again later
                    await scheduler.enqueue(request)

                    requestIndex += 1
                }
            }
        }
    }

    /// Prepare input for batched forward pass
    private func prepareBatchInput() -> BatchInput? {
        var tokenIds: [Int] = []
        var positions: [Int] = []
        var prompts: [String] = []
        var kvCacheBlockIds: [[Int]] = []
        var activeIndices: [Int] = []

        for (i, slot) in slots.enumerated() {
            guard let slot = slot, !slot.isFinished else { continue }

            // For first token, use prompt; otherwise use last generated token
            let tokenId = slot.generatedTokens.last ?? 0  // TODO: Get from tokenizer

            tokenIds.append(tokenId)
            positions.append(slot.nextPosition)
            prompts.append(slot.request.request.prompt)
            kvCacheBlockIds.append(slot.kvCacheBlockIds)
            activeIndices.append(i)
        }

        guard !tokenIds.isEmpty else { return nil }

        return BatchInput(
            tokenIds: tokenIds,
            positions: positions,
            prompts: prompts,
            kvCacheBlockIds: kvCacheBlockIds,
            activeIndices: activeIndices
        )
    }

    /// Update slots with newly generated tokens
    private func updateSlots(nextTokens: [Int], activeIndices: [Int]) async {
        guard nextTokens.count == activeIndices.count else {
            logger.error("Token count mismatch", metadata: [
                "tokens": "\(nextTokens.count)",
                "indices": "\(activeIndices.count)"
            ])
            return
        }

        for (tokenIdx, slotIdx) in activeIndices.enumerated() {
            guard var slot = slots[slotIdx] else { continue }

            let token = nextTokens[tokenIdx]
            slot.appendToken(token)

            // Emit token to stream
            await scheduler.emitToken(
                requestId: slot.request.id,
                token: String(token),  // TODO: Decode with tokenizer
                index: slot.totalTokens - 1
            )

            // Check finish conditions
            if isEndOfSequence(token) {
                slot.finish(reason: .stop)

                let info = GenerationInfo(
                    requestId: slot.request.id,
                    totalTokens: slot.totalTokens,
                    duration: slot.request.processingTime ?? 0,
                    finishReason: .stop
                )
                await scheduler.complete(requestId: slot.request.id, info: info)

                logger.debug("Request completed (EOS)", metadata: [
                    "slot_id": "\(slotIdx)",
                    "request_id": "\(slot.request.id)",
                    "tokens": "\(slot.totalTokens)"
                ])
            } else if slot.hasReachedMaxTokens() {
                slot.finish(reason: .length)

                let info = GenerationInfo(
                    requestId: slot.request.id,
                    totalTokens: slot.totalTokens,
                    duration: slot.request.processingTime ?? 0,
                    finishReason: .length
                )
                await scheduler.complete(requestId: slot.request.id, info: info)

                logger.debug("Request completed (max tokens)", metadata: [
                    "slot_id": "\(slotIdx)",
                    "request_id": "\(slot.request.id)",
                    "tokens": "\(slot.totalTokens)"
                ])
            }

            slots[slotIdx] = slot
        }
    }

    /// Remove finished slots
    private func cleanupFinishedSlots() async {
        for i in 0..<slots.count {
            if let slot = slots[i], slot.isFinished {
                // Release KV cache blocks
                await kvCache.release(for: slot.request.id)

                slots[i] = nil
                logger.trace("Slot freed", metadata: [
                    "slot_id": "\(i)",
                    "request_id": "\(slot.request.id)"
                ])
            }
        }
    }

    /// Check for cancelled requests and free their slots
    private func checkCancellations() async {
        for i in 0..<slots.count {
            guard let slot = slots[i] else { continue }

            // Check if request was cancelled
            if let status = await scheduler.getStatus(requestId: slot.request.id),
               status == .cancelled {
                // Release KV cache blocks
                await kvCache.release(for: slot.request.id)

                // Free the slot immediately
                slots[i] = nil

                logger.debug("Slot freed due to cancellation", metadata: [
                    "slot_id": "\(i)",
                    "request_id": "\(slot.request.id)"
                ])
            }
        }
    }

    /// Check if token is end-of-sequence
    private func isEndOfSequence(_ tokenId: Int) -> Bool {
        return tokenId == config.eosTokenId
    }

    /// Adjust batch size based on utilization (Phase 3.3)
    private func adjustBatchSize() async {
        let newMaxBatchSize = await gpuMonitor.recommendBatchSizeAdjustment(
            current: currentMaxBatchSize
        )

        if newMaxBatchSize != currentMaxBatchSize {
            let avgUtil = await gpuMonitor.averageUtilization()
            logger.info("Adjusting batch size", metadata: [
                "old_size": "\(currentMaxBatchSize)",
                "new_size": "\(newMaxBatchSize)",
                "utilization": "\(String(format: "%.2f", avgUtil))"
            ])
            currentMaxBatchSize = newMaxBatchSize
        }
    }

    // MARK: - Observability

    /// Get current batch utilization
    public var utilization: Double {
        let active = activeSlotCount
        return Double(active) / Double(currentMaxBatchSize)
    }

    /// Count of active (non-finished) slots
    private var activeSlotCount: Int {
        slots.compactMap { $0 }.filter { !$0.isFinished }.count
    }

    /// Get current statistics
    public func getStats() async -> (activeSlots: Int, totalSlots: Int, utilization: Double, stepCount: Int) {
        return (
            activeSlots: activeSlotCount,
            totalSlots: currentMaxBatchSize,
            utilization: utilization,
            stepCount: stepCount
        )
    }

    /// Get GPU monitor statistics
    public func getGPUStats() async -> (averageUtilization: Double, currentUtilization: Double, sampleCount: Int) {
        return await gpuMonitor.stats()
    }

    /// Check if batcher is running
    public var running: Bool {
        return isRunning
    }
}

// MARK: - Supporting Types

/// Input for a batched forward pass
private struct BatchInput {
    let tokenIds: [Int]
    let positions: [Int]
    let prompts: [String]
    let kvCacheBlockIds: [[Int]]  // KV cache block IDs for each slot
    let activeIndices: [Int]
}

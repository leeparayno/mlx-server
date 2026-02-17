import Foundation
import Core
import MLXLMCommon
import Logging

/// Implements continuous batching for maximum GPU utilization
public actor ContinuousBatcher {
    private let logger = Logger(label: "continuous-batcher")

    // Slot management
    private var slots: [BatchSlot?]
    private let maxBatchSize: Int

    // Dependencies
    private let scheduler: RequestScheduler
    private let engine: InferenceEngine

    // State
    private var isRunning: Bool = false
    private var stepCount: Int = 0

    // Configuration
    public struct Config: Sendable {
        public let maxBatchSize: Int
        public let eosTokenId: Int

        public init(maxBatchSize: Int = 32, eosTokenId: Int = 2) {
            self.maxBatchSize = maxBatchSize
            self.eosTokenId = eosTokenId
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
        self.maxBatchSize = config.maxBatchSize
        self.slots = Array(repeating: nil, count: config.maxBatchSize)
    }

    // MARK: - Main Loop

    /// Start the continuous batching loop
    public func start() async {
        guard !isRunning else {
            logger.warning("Batcher already running")
            return
        }

        isRunning = true
        logger.info("Starting continuous batcher", metadata: [
            "max_batch_size": "\(maxBatchSize)"
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
    public func stop() {
        isRunning = false
    }

    /// Execute one batching step
    internal func step() async throws {
        stepCount += 1

        // 1. Fill empty slots from scheduler
        await fillEmptySlots()

        // 2. Prepare batch input
        guard let batchInput = prepareBatchInput() else {
            return  // No active slots
        }

        // 3. Batched forward pass
        let nextTokens = try await engine.forwardBatch(
            tokenIds: batchInput.tokenIds,
            positions: batchInput.positions,
            prompts: batchInput.prompts
        )

        // 4. Update slots with new tokens
        await updateSlots(nextTokens: nextTokens, activeIndices: batchInput.activeIndices)

        // 5. Cleanup finished slots
        cleanupFinishedSlots()

        let queueLen = await scheduler.queueLength
        logger.trace("Batching step completed", metadata: [
            "step": "\(stepCount)",
            "active_slots": "\(activeSlotCount)",
            "pending_queue": "\(queueLen)"
        ])
    }

    // MARK: - Slot Management

    /// Fill empty slots with pending requests
    private func fillEmptySlots() async {
        var emptySlotCount = 0
        for slot in slots where slot == nil {
            emptySlotCount += 1
        }

        guard emptySlotCount > 0 else { return }

        // Dequeue requests from scheduler
        let newRequests = await scheduler.dequeueNextBatch(size: emptySlotCount)

        guard !newRequests.isEmpty else { return }

        // Assign to empty slots
        var requestIndex = 0
        for i in 0..<slots.count {
            guard requestIndex < newRequests.count else { break }

            if slots[i] == nil {
                let request = newRequests[requestIndex]
                slots[i] = BatchSlot(slotId: i, request: request)

                // Mark as streaming
                await scheduler.markStreaming(requestId: request.id)

                logger.debug("Request assigned to slot", metadata: [
                    "slot_id": "\(i)",
                    "request_id": "\(request.id)"
                ])

                requestIndex += 1
            }
        }
    }

    /// Prepare input for batched forward pass
    private func prepareBatchInput() -> BatchInput? {
        var tokenIds: [Int] = []
        var positions: [Int] = []
        var prompts: [String] = []
        var activeIndices: [Int] = []

        for (i, slot) in slots.enumerated() {
            guard let slot = slot, !slot.isFinished else { continue }

            // For first token, use prompt; otherwise use last generated token
            let tokenId = slot.generatedTokens.last ?? 0  // TODO: Get from tokenizer

            tokenIds.append(tokenId)
            positions.append(slot.nextPosition)
            prompts.append(slot.request.request.prompt)
            activeIndices.append(i)
        }

        guard !tokenIds.isEmpty else { return nil }

        return BatchInput(
            tokenIds: tokenIds,
            positions: positions,
            prompts: prompts,
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
    private func cleanupFinishedSlots() {
        for i in 0..<slots.count {
            if let slot = slots[i], slot.isFinished {
                slots[i] = nil
                logger.trace("Slot freed", metadata: ["slot_id": "\(i)"])
            }
        }
    }

    /// Check if token is end-of-sequence
    private func isEndOfSequence(_ tokenId: Int) -> Bool {
        return tokenId == config.eosTokenId
    }

    // MARK: - Observability

    /// Get current batch utilization
    public var utilization: Double {
        let active = activeSlotCount
        return Double(active) / Double(maxBatchSize)
    }

    /// Count of active (non-finished) slots
    private var activeSlotCount: Int {
        slots.compactMap { $0 }.filter { !$0.isFinished }.count
    }

    /// Get current statistics
    public func getStats() async -> (activeSlots: Int, totalSlots: Int, utilization: Double, stepCount: Int) {
        return (
            activeSlots: activeSlotCount,
            totalSlots: maxBatchSize,
            utilization: utilization,
            stepCount: stepCount
        )
    }
}

// MARK: - Supporting Types

/// Input for a batched forward pass
private struct BatchInput {
    let tokenIds: [Int]
    let positions: [Int]
    let prompts: [String]
    let activeIndices: [Int]
}

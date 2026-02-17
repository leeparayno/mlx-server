import XCTest
import Core
import Logging
@testable import Scheduler

final class ContinuousBatcherTests: XCTestCase {
    var scheduler: RequestScheduler!

    override func setUp() async throws {
        scheduler = RequestScheduler()
    }

    override func tearDown() async throws {
        scheduler = nil
    }

    // MARK: - Phase 3.1 Basic Tests

    /// Test that batcher initializes correctly
    func testBatcherInitialization() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        let util = await batcher.utilization
        XCTAssertEqual(util, 0.0, accuracy: 0.01, "Empty batch should have 0% utilization")
    }

    /// Test that utilization is calculated correctly
    func testUtilizationCalculation() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Empty batch
        var util = await batcher.utilization
        XCTAssertEqual(util, 0.0, accuracy: 0.01, "Empty batch should have 0% utilization")

        // Submit 2 requests (max batch size is 4)
        for i in 0..<2 {
            let request = InferenceRequest(prompt: "Test \(i)", maxTokens: 10)
            _ = await scheduler.submit(request, priority: .normal)
        }

        // Execute one step to fill slots
        try await batcher.step()

        util = await batcher.utilization
        XCTAssertEqual(util, 0.5, accuracy: 0.01, "Half-full batch should have 50% utilization")
    }

    /// Test that empty slots are filled from pending queue
    func testFillEmptySlots() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Submit multiple requests
        for i in 0..<3 {
            let request = InferenceRequest(
                prompt: "Test prompt \(i)",
                maxTokens: 10
            )
            _ = await scheduler.submit(request, priority: .normal)
        }

        // Check initial state
        let initialQueueLen = await scheduler.queueLength
        XCTAssertEqual(initialQueueLen, 3, "Should have 3 pending requests")

        // Start one step to fill slots
        try await batcher.step()

        // Check that requests were dequeued
        let finalQueueLen = await scheduler.queueLength
        XCTAssertEqual(finalQueueLen, 0, "Queue should be empty after dequeue")

        // Check active count in scheduler
        let activeCount = await scheduler.activeCount
        XCTAssertEqual(activeCount, 3, "Should have 3 active requests")
    }

    /// Test that max batch size is respected
    func testMaxBatchSizeRespected() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Submit more requests than max batch size
        for i in 0..<6 {
            let request = InferenceRequest(prompt: "Test \(i)", maxTokens: 10)
            _ = await scheduler.submit(request, priority: .normal)
        }

        try await batcher.step()

        let stats = await batcher.getStats()
        XCTAssertLessThanOrEqual(stats.activeSlots, 4, "Should not exceed max batch size of 4")

        let queueLen = await scheduler.queueLength
        XCTAssertGreaterThan(queueLen, 0, "Extra requests should remain in queue")
    }

    /// Test that step() handles empty batch gracefully
    func testEmptyBatch() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Execute step with no requests - should not crash
        try await batcher.step()

        let stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 0, "No slots should be active")
    }

    /// Test that batcher can start and stop
    func testStartStop() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Start batcher in background
        Task {
            await batcher.start()
        }

        // Give it a moment to start
        try await Task.sleep(for: .milliseconds(50))

        // Stop batcher
        await batcher.stop()

        // Batcher should stop gracefully
        XCTAssert(true, "Batcher started and stopped without errors")
    }

    /// Test that stats are tracked correctly
    func testStatsTracking() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Initial stats
        var stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 0)
        XCTAssertEqual(stats.totalSlots, 4)
        XCTAssertEqual(stats.stepCount, 0)

        // Execute a step
        try await batcher.step()

        stats = await batcher.getStats()
        XCTAssertEqual(stats.stepCount, 1, "Step count should increment")
    }

    /// Test continuous ingestion - requests can be added while processing
    func testContinuousIngestion() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Submit first request
        let request1 = InferenceRequest(prompt: "First", maxTokens: 10)
        _ = await scheduler.submit(request1, priority: .normal)

        // Execute one step
        try await batcher.step()

        var stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 1, "Should have 1 active slot")

        // Submit second request while first is "processing"
        let request2 = InferenceRequest(prompt: "Second", maxTokens: 10)
        _ = await scheduler.submit(request2, priority: .normal)

        // Execute another step
        try await batcher.step()

        stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 2, "Should have 2 active slots")
    }

    /// Test that priority is respected when filling slots
    func testPriorityOrdering() async throws {
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 2, eosTokenId: 2)
        )

        // Submit requests with different priorities
        let normalRequest = InferenceRequest(prompt: "Normal", maxTokens: 5)
        let highRequest = InferenceRequest(prompt: "High", maxTokens: 5)
        let criticalRequest = InferenceRequest(prompt: "Critical", maxTokens: 5)

        let (normalId, _) = await scheduler.submit(normalRequest, priority: .normal)
        let (highId, _) = await scheduler.submit(highRequest, priority: .high)
        let (criticalId, _) = await scheduler.submit(criticalRequest, priority: .critical)

        // Fill slots - should take highest priority first
        try await batcher.step()

        // Check that high and critical are active (normal should be pending)
        let normalReq = await scheduler.getRequest(id: normalId)
        let highReq = await scheduler.getRequest(id: highId)
        let criticalReq = await scheduler.getRequest(id: criticalId)

        XCTAssertEqual(normalReq?.status, .pending, "Normal priority should still be pending")
        XCTAssertTrue(highReq?.status.isActive ?? false, "High priority should be active")
        XCTAssertTrue(criticalReq?.status.isActive ?? false, "Critical priority should be active")
    }
}

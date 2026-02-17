import XCTest
@testable import Scheduler
@testable import Memory
@testable import Core
import MLX
import Logging

/// Tests for Phase 4.3: Real Memory Tracking & Adaptive Limits
final class MemoryPressureTests: XCTestCase {

    var gpuMonitor: GPUMonitor!
    var kvCache: PagedKVCache!

    override func setUp() async throws {
        gpuMonitor = GPUMonitor(config: GPUMonitor.Config(
            windowSize: 10,
            maxBatchSize: 32,
            totalMemoryGB: 512
        ))
        kvCache = PagedKVCache(blockSize: 16, numBlocks: 1024)
    }

    override func tearDown() async throws {
        gpuMonitor = nil
        kvCache = nil
    }

    // MARK: - Real MLX Memory Tracking Tests

    func testRealMLXMemoryTracking() async throws {
        // Get KV cache stats
        let kvStats = await kvCache.stats

        // Check memory pressure with real MLX tracking
        let pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)

        // With no blocks allocated, pressure should be normal
        XCTAssertEqual(pressure, .normal, "Empty cache should have normal pressure")
        XCTAssertEqual(kvStats.usedBlocks, 0, "No blocks should be used initially")
        XCTAssertEqual(kvStats.freeBlocks, 1024, "All blocks should be free initially")
    }

    func testKVCacheUtilizationTracking() async throws {
        // Allocate some blocks
        let requestId = UUID()
        let blockIds = try await kvCache.allocate(for: requestId, numTokens: 512)

        // Check stats reflect allocation
        let kvStats = await kvCache.stats
        XCTAssertEqual(blockIds.count, 32, "Should allocate 32 blocks for 512 tokens")
        XCTAssertEqual(kvStats.usedBlocks, 32, "Stats should show 32 used blocks")
        XCTAssertEqual(kvStats.freeBlocks, 992, "Stats should show 992 free blocks")

        let expectedUtilization = Double(32) / Double(1024) * 100
        XCTAssertEqual(kvStats.utilizationPercent, expectedUtilization, accuracy: 0.01,
                      "Utilization should match used blocks / total blocks")

        // Release blocks
        await kvCache.release(for: requestId)

        let finalStats = await kvCache.stats
        XCTAssertEqual(finalStats.usedBlocks, 0, "All blocks should be released")
        XCTAssertEqual(finalStats.utilizationPercent, 0, "Utilization should be zero after release")
    }

    // MARK: - Memory Pressure Detection Tests

    func testNormalMemoryPressure() async throws {
        // Allocate 50% of blocks (512 out of 1024)
        for _ in 0..<16 {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
        }

        let kvStats = await kvCache.stats
        let pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)

        XCTAssertEqual(pressure, .normal, "50% utilization should be normal pressure")
        XCTAssertLessThan(kvStats.utilizationPercent, 80, "Should be under 80% utilization")
    }

    func testHighMemoryPressure() async throws {
        // Allocate 85% of blocks
        let numRequests = Int(Double(1024) * 0.85 / 32)  // Each request uses 32 blocks
        for _ in 0..<numRequests {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
        }

        let kvStats = await kvCache.stats
        let pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)

        XCTAssertEqual(pressure, .high, "85% utilization should be high pressure")
        XCTAssertGreaterThan(kvStats.utilizationPercent, 80, "Should exceed 80% utilization")
        XCTAssertLessThan(kvStats.utilizationPercent, 90, "Should be under 90% utilization")
    }

    func testCriticalMemoryPressure() async throws {
        // Allocate 95% of blocks
        let numRequests = Int(Double(1024) * 0.95 / 32)
        for _ in 0..<numRequests {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
        }

        let kvStats = await kvCache.stats
        let pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)

        XCTAssertEqual(pressure, .critical, "95% utilization should be critical pressure")
        XCTAssertGreaterThan(kvStats.utilizationPercent, 90, "Should exceed 90% utilization")
    }

    // MARK: - Batch Size Limiting Tests

    func testBatchSizeLimitByAvailableBlocks() async throws {
        // Create continuous batcher
        let scheduler = RequestScheduler()
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 32, eosTokenId: 2)
        )

        // Allocate most blocks (leaving only enough for 2 requests)
        let blocksToAllocate = 1024 - 64  // Leave 64 blocks = 2 requests at 512 tokens each
        let numRequests = blocksToAllocate / 32
        for _ in 0..<numRequests {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
        }

        let kvStats = await kvCache.stats
        XCTAssertEqual(kvStats.freeBlocks, 64, "Should have 64 free blocks")

        // Calculate max slots by blocks (as ContinuousBatcher does)
        let avgTokensPerRequest = 512
        let blocksPerRequest = (avgTokensPerRequest + 15) / 16
        let maxSlotsByBlocks = kvStats.freeBlocks / blocksPerRequest

        XCTAssertEqual(maxSlotsByBlocks, 2, "Should limit to 2 slots based on available blocks")

        await batcher.stop()
    }

    // MARK: - Utilization Tracking Tests

    func testCombinedUtilizationTracking() async throws {
        // Record utilization measurements
        for i in 0..<10 {
            let utilization = Double(i) / 10.0  // 0.0, 0.1, ..., 0.9
            await gpuMonitor.recordUtilization(utilization)
        }

        let avgUtil = await gpuMonitor.averageUtilization()
        XCTAssertEqual(avgUtil, 0.45, accuracy: 0.01, "Average should be 0.45")

        let currentUtil = await gpuMonitor.currentUtilization()
        XCTAssertEqual(currentUtil, 0.9, accuracy: 0.01, "Current should be last recorded value")

        let stats = await gpuMonitor.stats()
        XCTAssertEqual(stats.sampleCount, 10, "Should have 10 samples")
    }

    func testMemoryUtilizationInBatcher() async throws {
        let scheduler = RequestScheduler()
        let engine = InferenceEngine()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Submit requests
        for i in 0..<2 {
            let request = InferenceRequest(prompt: "Test \(i)", maxTokens: 100)
            _ = await scheduler.submit(request, priority: .normal)
        }

        // Execute step to allocate KV cache blocks
        try await batcher.step()

        // Get GPU stats - should have recorded utilization
        let gpuStats = await batcher.getGPUStats()
        XCTAssertGreaterThanOrEqual(gpuStats.sampleCount, 1, "Should have at least one utilization sample")

        await batcher.stop()
    }

    // MARK: - Pressure Transition Tests

    func testPressureTransitionNormalToHigh() async throws {
        // Start with normal pressure
        var kvStats = await kvCache.stats
        var pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)
        XCTAssertEqual(pressure, .normal)

        // Allocate to high pressure threshold (85%)
        let numRequests = Int(Double(1024) * 0.85 / 32)
        for _ in 0..<numRequests {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
        }

        kvStats = await kvCache.stats
        pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)
        XCTAssertEqual(pressure, .high, "Should transition to high pressure")
    }

    func testGracefulRecoveryFromHighPressure() async throws {
        // Allocate to high pressure
        var requestIds: [UUID] = []
        let numRequests = Int(Double(1024) * 0.85 / 32)
        for _ in 0..<numRequests {
            let requestId = UUID()
            _ = try await kvCache.allocate(for: requestId, numTokens: 512)
            requestIds.append(requestId)
        }

        var kvStats = await kvCache.stats
        var pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)
        XCTAssertEqual(pressure, .high, "Should start with high pressure")

        // Release half the blocks
        for i in 0..<(requestIds.count / 2) {
            await kvCache.release(for: requestIds[i])
        }

        kvStats = await kvCache.stats
        pressure = await gpuMonitor.checkMemoryPressure(kvCacheStats: kvStats)
        XCTAssertEqual(pressure, .normal, "Should recover to normal pressure")
        XCTAssertLessThan(kvStats.utilizationPercent, 80, "Utilization should drop below 80%")
    }
}

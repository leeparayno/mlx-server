import XCTest
import Core
import Logging
@testable import Scheduler

final class RequestCancellationTests: XCTestCase {
    var scheduler: RequestScheduler!
    var engine: InferenceEngine!
    var batcher: ContinuousBatcher!

    override func setUp() async throws {
        scheduler = RequestScheduler()
        engine = InferenceEngine()
        batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )
    }

    override func tearDown() async throws {
        await batcher.stop()
        scheduler = nil
        engine = nil
        batcher = nil
    }

    // MARK: - Phase 3.2 Cancellation Tests

    /// Test that pending requests can be cancelled before processing starts
    func testCancelPendingRequest() async throws {
        // Submit a request
        let request = InferenceRequest(prompt: "Test", maxTokens: 10)
        let (requestId, _) = await scheduler.submit(request, priority: .normal)

        // Verify it's pending
        let status = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(status, .pending, "Request should be pending")

        // Cancel it
        await scheduler.cancel(requestId: requestId)

        // Verify it's cancelled
        let cancelledStatus = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(cancelledStatus, .cancelled, "Request should be cancelled")

        // Verify it's not in the queue anymore
        let queueLen = await scheduler.queueLength
        XCTAssertEqual(queueLen, 0, "Queue should be empty")
    }

    /// Test that active requests can be cancelled mid-generation
    func testCancelActiveRequest() async throws {
        // Submit and start processing a request
        let request = InferenceRequest(prompt: "Test", maxTokens: 10)
        let (requestId, _) = await scheduler.submit(request, priority: .normal)

        // Start processing (makes it active)
        try await batcher.step()

        // Verify it's active
        let status = await scheduler.getStatus(requestId: requestId)
        XCTAssertTrue(status?.isActive ?? false, "Request should be active")

        // Cancel it
        await scheduler.cancel(requestId: requestId)

        // Verify it's cancelled
        let cancelledStatus = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(cancelledStatus, .cancelled, "Request should be cancelled")
    }

    /// Test bulk cancellation of all requests
    func testCancelAllRequests() async throws {
        // Submit multiple requests
        var requestIds: [UUID] = []
        for i in 0..<5 {
            let request = InferenceRequest(prompt: "Test \(i)", maxTokens: 10)
            let (id, _) = await scheduler.submit(request, priority: .normal)
            requestIds.append(id)
        }

        // Verify all are pending
        let initialQueueLen = await scheduler.queueLength
        XCTAssertEqual(initialQueueLen, 5, "Should have 5 pending requests")

        // Cancel all
        await scheduler.cancelAll()

        // Verify all are cancelled
        let finalQueueLen = await scheduler.queueLength
        XCTAssertEqual(finalQueueLen, 0, "Queue should be empty")

        for requestId in requestIds {
            let status = await scheduler.getStatus(requestId: requestId)
            XCTAssertEqual(status, .cancelled, "All requests should be cancelled")
        }
    }

    /// Test that cancellation immediately frees slots in the batcher
    func testCancellationFreesSlot() async throws {
        // Submit requests to fill some slots
        let request1 = InferenceRequest(prompt: "First", maxTokens: 10)
        let request2 = InferenceRequest(prompt: "Second", maxTokens: 10)

        let (id1, _) = await scheduler.submit(request1, priority: .normal)
        let (id2, _) = await scheduler.submit(request2, priority: .normal)

        // Start processing (fills 2 slots)
        try await batcher.step()

        var stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 2, "Should have 2 active slots")

        // Cancel first request
        await scheduler.cancel(requestId: id1)

        // Run another step to process cancellations
        try await batcher.step()

        stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 1, "Should have 1 active slot after cancellation")

        // Cancel second request
        await scheduler.cancel(requestId: id2)

        // Run another step
        try await batcher.step()

        stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 0, "All slots should be freed")
    }

    /// Test that TokenStream receives cancellation error
    func testStreamClosedOnCancel() async throws {
        // Submit a request
        let request = InferenceRequest(prompt: "Test", maxTokens: 10)
        let (requestId, stream) = await scheduler.submit(request, priority: .normal)

        // Start consuming stream in background
        let streamTask = Task {
            var receivedError: Error?
            do {
                for try await _ in stream {
                    // Should never reach here after cancel
                }
            } catch {
                receivedError = error
            }
            return receivedError
        }

        // Give stream time to start
        try await Task.sleep(for: .milliseconds(10))

        // Cancel the request
        await scheduler.cancel(requestId: requestId)

        // Check that stream received an error
        let error = await streamTask.value
        XCTAssertNotNil(error, "Stream should receive an error")

        if let streamError = error as? StreamError {
            XCTAssertEqual(streamError, StreamError.cancelled, "Should be cancellation error")
        }
    }

    /// Test cancelling request that doesn't exist
    func testCancelNonexistentRequest() async throws {
        let nonexistentId = UUID()

        // Should not crash
        await scheduler.cancel(requestId: nonexistentId)

        // Verify no side effects
        let queueLen = await scheduler.queueLength
        XCTAssertEqual(queueLen, 0, "Queue should still be empty")
    }

    /// Test getStatus() method
    func testGetStatus() async throws {
        // Test nonexistent request
        let nonexistentId = UUID()
        let status1 = await scheduler.getStatus(requestId: nonexistentId)
        XCTAssertNil(status1, "Nonexistent request should return nil")

        // Test pending request
        let request = InferenceRequest(prompt: "Test", maxTokens: 10)
        let (requestId, _) = await scheduler.submit(request, priority: .normal)

        let status2 = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(status2, .pending, "New request should be pending")

        // Test active request
        try await batcher.step()

        let status3 = await scheduler.getStatus(requestId: requestId)
        XCTAssertTrue(status3?.isActive ?? false, "Processing request should be active")

        // Test cancelled request
        await scheduler.cancel(requestId: requestId)

        let status4 = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(status4, .cancelled, "Cancelled request should show cancelled status")
    }

    /// Test cancellation statistics tracking
    func testCancellationStatistics() async throws {
        // Submit requests
        let request1 = InferenceRequest(prompt: "First", maxTokens: 10)
        let request2 = InferenceRequest(prompt: "Second", maxTokens: 10)

        let (id1, _) = await scheduler.submit(request1, priority: .normal)
        let (id2, _) = await scheduler.submit(request2, priority: .normal)

        // Cancel them
        await scheduler.cancel(requestId: id1)
        await scheduler.cancel(requestId: id2)

        // Check statistics
        let stats = await scheduler.stats
        XCTAssertEqual(stats.totalCancelled, 2, "Should have 2 cancelled requests")
        XCTAssertEqual(stats.totalSubmitted, 2, "Should have 2 submitted requests")
    }
}

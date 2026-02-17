import XCTest
@testable import Scheduler
@testable import Core

final class RequestSchedulerTests: XCTestCase {
    func testSchedulerInitialization() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)
        let queueLength = await scheduler.queueLength
        XCTAssertEqual(queueLength, 0, "Queue should be empty initially")
    }

    func testSubmitRequest() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)
        let request = InferenceRequest(prompt: "Test prompt")

        let (requestId, stream) = await scheduler.submit(request)
        XCTAssertNotNil(requestId, "Should return request ID")
        XCTAssertNotNil(stream, "Should return token stream")

        let queueLength = await scheduler.queueLength
        XCTAssertEqual(queueLength, 1, "Should have one pending request")
    }

    func testSubmitWithPriority() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        let normalRequest = InferenceRequest(prompt: "Normal")
        let highRequest = InferenceRequest(prompt: "High")

        let (normalId, _) = await scheduler.submit(normalRequest, priority: .normal)
        let (highId, _) = await scheduler.submit(highRequest, priority: .high)

        XCTAssertNotEqual(normalId, highId)

        // High priority should be dequeued first
        let batch = await scheduler.dequeueNextBatch(size: 2)
        XCTAssertEqual(batch.count, 2)
        XCTAssertEqual(batch[0].id, highId, "High priority should be first")
        XCTAssertEqual(batch[1].id, normalId, "Normal priority should be second")
    }

    func testDequeueBatch() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        // Submit 5 requests
        for i in 0..<5 {
            let request = InferenceRequest(prompt: "Request \(i)")
            _ = await scheduler.submit(request, priority: .normal)
        }

        // Dequeue batch of 3
        let batch = await scheduler.dequeueNextBatch(size: 3)
        XCTAssertEqual(batch.count, 3)

        // Should have 2 remaining in queue
        let queueLength = await scheduler.queueLength
        XCTAssertEqual(queueLength, 2)

        // Should have 3 active
        let activeCount = await scheduler.activeCount
        XCTAssertEqual(activeCount, 3)
    }

    func testCompleteRequest() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        let request = InferenceRequest(prompt: "Test")
        let (requestId, _) = await scheduler.submit(request)

        // Dequeue and mark as streaming
        let batch = await scheduler.dequeueNextBatch(size: 1)
        XCTAssertEqual(batch.count, 1)

        await scheduler.markStreaming(requestId: requestId)

        // Complete the request
        let info = GenerationInfo(
            requestId: requestId,
            totalTokens: 10,
            duration: 1.0,
            finishReason: .stop
        )
        await scheduler.complete(requestId: requestId, info: info)

        // Should no longer be active
        let activeCount = await scheduler.activeCount
        XCTAssertEqual(activeCount, 0)

        // Check stats
        let stats = await scheduler.stats
        XCTAssertEqual(stats.totalCompleted, 1)
    }

    func testFailRequest() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        let request = InferenceRequest(prompt: "Test")
        let (requestId, _) = await scheduler.submit(request)

        let batch = await scheduler.dequeueNextBatch(size: 1)
        XCTAssertEqual(batch.count, 1)

        // Fail the request
        await scheduler.fail(requestId: requestId, error: TestError.testFailure)

        // Check stats
        let stats = await scheduler.stats
        XCTAssertEqual(stats.totalFailed, 1)
    }

    func testCancelPendingRequest() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        let request = InferenceRequest(prompt: "Test")
        let (requestId, _) = await scheduler.submit(request)

        // Cancel while still pending
        await scheduler.cancel(requestId: requestId)

        // Should not be in queue
        let queueLength = await scheduler.queueLength
        XCTAssertEqual(queueLength, 0)

        let stats = await scheduler.stats
        XCTAssertEqual(stats.totalCancelled, 1)
    }

    func testStatistics() async throws {
        let config = SchedulerConfig(maxConcurrentRequests: 10)
        let scheduler = RequestScheduler(config: config)

        // Submit some requests
        for i in 0..<3 {
            let request = InferenceRequest(prompt: "Request \(i)")
            _ = await scheduler.submit(request)
        }

        let stats = await scheduler.stats
        XCTAssertEqual(stats.totalSubmitted, 3)
        XCTAssertEqual(stats.currentPending, 3)
        XCTAssertEqual(stats.currentActive, 0)
    }
}

// MARK: - Test Errors

enum TestError: Error {
    case testFailure
}

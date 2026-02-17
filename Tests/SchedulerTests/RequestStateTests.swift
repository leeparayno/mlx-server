import XCTest
@testable import Scheduler

final class RequestStateTests: XCTestCase {
    func testRequestStatusIsTerminal() {
        XCTAssertTrue(RequestStatus.completed.isTerminal)
        XCTAssertTrue(RequestStatus.failed.isTerminal)
        XCTAssertTrue(RequestStatus.timeout.isTerminal)
        XCTAssertTrue(RequestStatus.cancelled.isTerminal)

        XCTAssertFalse(RequestStatus.pending.isTerminal)
        XCTAssertFalse(RequestStatus.active.isTerminal)
        XCTAssertFalse(RequestStatus.streaming.isTerminal)
    }

    func testRequestStatusIsActive() {
        XCTAssertTrue(RequestStatus.active.isActive)
        XCTAssertTrue(RequestStatus.streaming.isActive)

        XCTAssertFalse(RequestStatus.pending.isActive)
        XCTAssertFalse(RequestStatus.completed.isActive)
        XCTAssertFalse(RequestStatus.failed.isActive)
        XCTAssertFalse(RequestStatus.timeout.isActive)
        XCTAssertFalse(RequestStatus.cancelled.isActive)
    }

    func testRequestPriorityComparison() {
        XCTAssertLessThan(RequestPriority.low, .normal)
        XCTAssertLessThan(RequestPriority.normal, .high)
        XCTAssertLessThan(RequestPriority.high, .critical)

        XCTAssertGreaterThan(RequestPriority.critical, .high)
        XCTAssertGreaterThan(RequestPriority.high, .normal)
        XCTAssertGreaterThan(RequestPriority.normal, .low)
    }

    func testStatefulRequestInitialization() {
        let request = InferenceRequest(prompt: "Test")
        let stateful = StatefulRequest(request: request, priority: .high)

        XCTAssertEqual(stateful.status, .pending)
        XCTAssertEqual(stateful.priority, .high)
        XCTAssertNil(stateful.startedAt)
        XCTAssertNil(stateful.completedAt)
        XCTAssertFalse(stateful.isActive)
        XCTAssertFalse(stateful.isTerminal)
    }

    func testStatefulRequestAge() {
        let request = InferenceRequest(prompt: "Test")
        let stateful = StatefulRequest(request: request)

        // Age should be very small (just created)
        XCTAssertLessThan(stateful.age, 0.1)

        // Age should increase
        Thread.sleep(forTimeInterval: 0.1)
        XCTAssertGreaterThan(stateful.age, 0.05)
    }

    func testMarkActive() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markActive()

        XCTAssertEqual(stateful.status, .active)
        XCTAssertNotNil(stateful.startedAt)
        XCTAssertTrue(stateful.isActive)
        XCTAssertFalse(stateful.isTerminal)
    }

    func testMarkStreaming() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markStreaming()

        XCTAssertEqual(stateful.status, .streaming)
        XCTAssertNotNil(stateful.startedAt)
        XCTAssertTrue(stateful.isActive)
    }

    func testMarkCompleted() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markActive()
        Thread.sleep(forTimeInterval: 0.05)
        stateful.markCompleted()

        XCTAssertEqual(stateful.status, .completed)
        XCTAssertNotNil(stateful.completedAt)
        XCTAssertTrue(stateful.isTerminal)
        XCTAssertFalse(stateful.isActive)

        // Processing time should be available
        let processingTime = stateful.processingTime
        XCTAssertNotNil(processingTime)
        XCTAssertGreaterThan(processingTime!, 0)
    }

    func testMarkFailed() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markFailed()

        XCTAssertEqual(stateful.status, .failed)
        XCTAssertNotNil(stateful.completedAt)
        XCTAssertTrue(stateful.isTerminal)
    }

    func testMarkTimeout() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markTimeout()

        XCTAssertEqual(stateful.status, .timeout)
        XCTAssertNotNil(stateful.completedAt)
        XCTAssertTrue(stateful.isTerminal)
    }

    func testMarkCancelled() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markCancelled()

        XCTAssertEqual(stateful.status, .cancelled)
        XCTAssertNotNil(stateful.completedAt)
        XCTAssertTrue(stateful.isTerminal)
    }

    func testProcessingTimeBeforeStart() {
        let request = InferenceRequest(prompt: "Test")
        let stateful = StatefulRequest(request: request)

        XCTAssertNil(stateful.processingTime)
    }

    func testProcessingTimeWhileActive() {
        let request = InferenceRequest(prompt: "Test")
        var stateful = StatefulRequest(request: request)

        stateful.markActive()
        Thread.sleep(forTimeInterval: 0.1)

        let processingTime = stateful.processingTime
        XCTAssertNotNil(processingTime)
        XCTAssertGreaterThan(processingTime!, 0.05)
    }
}

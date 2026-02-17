import XCTest
@testable import Scheduler

final class PriorityQueueTests: XCTestCase {
    func testInitiallyEmpty() {
        var queue = PriorityRequestQueue()

        XCTAssertEqual(queue.count, 0)
        XCTAssertTrue(queue.isEmpty)
        XCTAssertNil(queue.peek())
        XCTAssertNil(queue.dequeue())
    }

    func testEnqueueDequeue() {
        var queue = PriorityRequestQueue()

        let request = InferenceRequest(prompt: "Test")
        let stateful = StatefulRequest(request: request, priority: .normal)

        queue.enqueue(stateful)

        XCTAssertEqual(queue.count, 1)
        XCTAssertFalse(queue.isEmpty)

        let dequeued = queue.dequeue()
        XCTAssertNotNil(dequeued)
        XCTAssertEqual(dequeued?.id, stateful.id)

        XCTAssertEqual(queue.count, 0)
        XCTAssertTrue(queue.isEmpty)
    }

    func testPriorityOrdering() {
        var queue = PriorityRequestQueue()

        // Add requests in reverse priority order
        let lowReq = StatefulRequest(
            request: InferenceRequest(prompt: "Low"),
            priority: .low
        )
        let normalReq = StatefulRequest(
            request: InferenceRequest(prompt: "Normal"),
            priority: .normal
        )
        let highReq = StatefulRequest(
            request: InferenceRequest(prompt: "High"),
            priority: .high
        )
        let criticalReq = StatefulRequest(
            request: InferenceRequest(prompt: "Critical"),
            priority: .critical
        )

        queue.enqueue(lowReq)
        queue.enqueue(normalReq)
        queue.enqueue(highReq)
        queue.enqueue(criticalReq)

        XCTAssertEqual(queue.count, 4)

        // Should dequeue in priority order
        XCTAssertEqual(queue.dequeue()?.id, criticalReq.id)
        XCTAssertEqual(queue.dequeue()?.id, highReq.id)
        XCTAssertEqual(queue.dequeue()?.id, normalReq.id)
        XCTAssertEqual(queue.dequeue()?.id, lowReq.id)

        XCTAssertTrue(queue.isEmpty)
    }

    func testFIFOWithinPriority() {
        var queue = PriorityRequestQueue()

        // Add multiple requests at same priority
        let req1 = StatefulRequest(
            request: InferenceRequest(prompt: "First"),
            priority: .normal
        )
        let req2 = StatefulRequest(
            request: InferenceRequest(prompt: "Second"),
            priority: .normal
        )
        let req3 = StatefulRequest(
            request: InferenceRequest(prompt: "Third"),
            priority: .normal
        )

        queue.enqueue(req1)
        queue.enqueue(req2)
        queue.enqueue(req3)

        // Should dequeue in FIFO order
        XCTAssertEqual(queue.dequeue()?.id, req1.id)
        XCTAssertEqual(queue.dequeue()?.id, req2.id)
        XCTAssertEqual(queue.dequeue()?.id, req3.id)
    }

    func testDequeueBatch() {
        var queue = PriorityRequestQueue()

        // Add 5 requests
        for i in 0..<5 {
            let req = StatefulRequest(
                request: InferenceRequest(prompt: "Request \(i)"),
                priority: .normal
            )
            queue.enqueue(req)
        }

        // Dequeue batch of 3
        let batch = queue.dequeue(count: 3)
        XCTAssertEqual(batch.count, 3)
        XCTAssertEqual(queue.count, 2)

        // Dequeue remaining
        let remaining = queue.dequeue(count: 5)
        XCTAssertEqual(remaining.count, 2)
        XCTAssertTrue(queue.isEmpty)
    }

    func testDequeueBatchWithPriorities() {
        var queue = PriorityRequestQueue()

        // Add mixed priorities
        let low1 = StatefulRequest(
            request: InferenceRequest(prompt: "Low 1"),
            priority: .low
        )
        let high1 = StatefulRequest(
            request: InferenceRequest(prompt: "High 1"),
            priority: .high
        )
        let normal1 = StatefulRequest(
            request: InferenceRequest(prompt: "Normal 1"),
            priority: .normal
        )

        queue.enqueue(low1)
        queue.enqueue(high1)
        queue.enqueue(normal1)

        // Batch should respect priority
        let batch = queue.dequeue(count: 3)
        XCTAssertEqual(batch.count, 3)
        XCTAssertEqual(batch[0].id, high1.id)
        XCTAssertEqual(batch[1].id, normal1.id)
        XCTAssertEqual(batch[2].id, low1.id)
    }

    func testCountsByPriority() {
        var queue = PriorityRequestQueue()

        // Add requests at different priorities
        for _ in 0..<3 {
            queue.enqueue(StatefulRequest(
                request: InferenceRequest(prompt: "Low"),
                priority: .low
            ))
        }
        for _ in 0..<2 {
            queue.enqueue(StatefulRequest(
                request: InferenceRequest(prompt: "High"),
                priority: .high
            ))
        }

        let counts = queue.countsByPriority
        XCTAssertEqual(counts[.low], 3)
        XCTAssertEqual(counts[.normal], 0)
        XCTAssertEqual(counts[.high], 2)
        XCTAssertEqual(counts[.critical], 0)

        XCTAssertEqual(queue.count, 5)
    }

    func testPeek() {
        var queue = PriorityRequestQueue()

        let req1 = StatefulRequest(
            request: InferenceRequest(prompt: "Normal"),
            priority: .normal
        )
        let req2 = StatefulRequest(
            request: InferenceRequest(prompt: "High"),
            priority: .high
        )

        queue.enqueue(req1)
        queue.enqueue(req2)

        // Peek should return highest priority without removing
        let peeked = queue.peek()
        XCTAssertEqual(peeked?.id, req2.id)
        XCTAssertEqual(queue.count, 2)  // Not removed
    }

    func testRemoveById() {
        var queue = PriorityRequestQueue()

        let req1 = StatefulRequest(
            request: InferenceRequest(prompt: "First"),
            priority: .normal
        )
        let req2 = StatefulRequest(
            request: InferenceRequest(prompt: "Second"),
            priority: .normal
        )
        let req3 = StatefulRequest(
            request: InferenceRequest(prompt: "Third"),
            priority: .high
        )

        queue.enqueue(req1)
        queue.enqueue(req2)
        queue.enqueue(req3)

        // Remove middle request
        let removed = queue.remove(id: req2.id)
        XCTAssertNotNil(removed)
        XCTAssertEqual(removed?.id, req2.id)
        XCTAssertEqual(queue.count, 2)

        // Verify remaining order
        XCTAssertEqual(queue.dequeue()?.id, req3.id)
        XCTAssertEqual(queue.dequeue()?.id, req1.id)
    }

    func testFindById() {
        var queue = PriorityRequestQueue()

        let req = StatefulRequest(
            request: InferenceRequest(prompt: "Test"),
            priority: .normal
        )

        queue.enqueue(req)

        let found = queue.find(id: req.id)
        XCTAssertNotNil(found)
        XCTAssertEqual(found?.id, req.id)
        XCTAssertEqual(queue.count, 1)  // Not removed

        let notFound = queue.find(id: UUID())
        XCTAssertNil(notFound)
    }

    func testRemoveAll() {
        var queue = PriorityRequestQueue()

        for i in 0..<5 {
            queue.enqueue(StatefulRequest(
                request: InferenceRequest(prompt: "Request \(i)"),
                priority: .normal
            ))
        }

        XCTAssertEqual(queue.count, 5)

        queue.removeAll()

        XCTAssertEqual(queue.count, 0)
        XCTAssertTrue(queue.isEmpty)
    }

    func testAverageAge() {
        var queue = PriorityRequestQueue()

        let req1 = StatefulRequest(
            request: InferenceRequest(prompt: "Old"),
            priority: .normal
        )

        queue.enqueue(req1)

        // Wait a bit
        Thread.sleep(forTimeInterval: 0.1)

        let req2 = StatefulRequest(
            request: InferenceRequest(prompt: "New"),
            priority: .normal
        )

        queue.enqueue(req2)

        let avgAge = queue.averageAge
        XCTAssertGreaterThan(avgAge, 0)
        XCTAssertLessThan(avgAge, 1.0)  // Should be less than 1 second
    }

    func testMaxAge() {
        var queue = PriorityRequestQueue()

        let oldReq = StatefulRequest(
            request: InferenceRequest(prompt: "Old"),
            priority: .normal
        )

        queue.enqueue(oldReq)

        Thread.sleep(forTimeInterval: 0.1)

        let newReq = StatefulRequest(
            request: InferenceRequest(prompt: "New"),
            priority: .normal
        )

        queue.enqueue(newReq)

        let maxAge = queue.maxAge
        XCTAssertGreaterThan(maxAge, 0.08)  // Should be close to 0.1s
    }
}

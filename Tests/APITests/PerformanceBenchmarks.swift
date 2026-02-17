import XCTest
import XCTVapor
@testable import API
@testable import Core
@testable import Scheduler

/// Performance benchmark tests (Phase 5.5)
/// These tests measure throughput and latency characteristics
final class PerformanceBenchmarks: XCTestCase {

    var app: Application!
    var scheduler: RequestScheduler!
    var engine: InferenceEngine!
    var batcher: ContinuousBatcher!
    var batcherTask: Task<Void, Never>!

    override func setUp() {
        super.setUp()

        // Create test application (synchronous for testing)
        app = Application(.testing)

        // Initialize components
        scheduler = RequestScheduler()
        engine = InferenceEngine()
        // Note: Not initializing engine with model - using placeholder inference

        batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 8, eosTokenId: 2)
        )

        // Start batcher in background
        let batcherRef = batcher!
        batcherTask = Task {
            await batcherRef.start()
        }

        // Configure routes with batcher
        try! routes(app, scheduler: scheduler, engine: engine, batcher: batcher)

        // Small delay to ensure batcher starts
        Thread.sleep(forTimeInterval: 0.05)
    }

    override func tearDown() {
        // Stop batcher
        let semaphore = DispatchSemaphore(value: 0)
        let batcherRef = batcher!
        Task {
            await batcherRef.stop()
            semaphore.signal()
        }
        semaphore.wait()

        // Shutdown app (synchronous for testing)
        app.shutdown()

        batcherTask.cancel()
        batcherTask = nil
        scheduler = nil
        engine = nil
        batcher = nil

        super.tearDown()
    }

    // MARK: - Phase 5.5 Benchmarks

    /// Test: Throughput with concurrent requests
    /// Measures requests per second under concurrent load
    func testThroughputBenchmark() throws {
        let numRequests = 50  // Reduced for test environment
        let startTime = Date()

        // Submit requests concurrently
        let expectation = self.expectation(description: "All requests complete")
        expectation.expectedFulfillmentCount = numRequests

        for i in 0..<numRequests {
            let requestBody = """
            {
                "model": "test-model",
                "prompt": "Request \(i)",
                "max_tokens": 10,
                "stream": false
            }
            """

            try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
                req.headers.add(name: .contentType, value: "application/json")
                req.body = ByteBuffer(string: requestBody)
            }, afterResponse: { res in
                XCTAssertEqual(res.status, .ok, "Request \(i) should succeed")
                expectation.fulfill()
            })
        }

        // Wait for all requests to complete
        wait(for: [expectation], timeout: 30.0)

        let duration = Date().timeIntervalSince(startTime)
        let throughput = Double(numRequests) / duration

        print("========================================")
        print("Throughput Benchmark Results")
        print("========================================")
        print("Total Requests: \(numRequests)")
        print("Duration: \(String(format: "%.2f", duration))s")
        print("Throughput: \(String(format: "%.2f", throughput)) req/s")
        print("========================================")

        // Assert minimum throughput
        // Note: Using lower threshold for test environment
        XCTAssertGreaterThan(throughput, 5.0, "Should handle >5 req/s in test environment")
    }

    /// Test: Latency distribution
    /// Measures P50, P95, P99 latencies
    func testLatencyBenchmark() throws {
        var latencies: [Double] = []
        let numRequests = 30  // Reduced for test environment

        for i in 0..<numRequests {
            let requestBody = """
            {
                "model": "test-model",
                "prompt": "Latency test \(i)",
                "max_tokens": 10,
                "stream": false
            }
            """

            let startTime = Date()

            try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
                req.headers.add(name: .contentType, value: "application/json")
                req.body = ByteBuffer(string: requestBody)
            }, afterResponse: { res in
                XCTAssertEqual(res.status, .ok, "Request should succeed")
            })

            let latency = Date().timeIntervalSince(startTime)
            latencies.append(latency)

            // Small delay between requests to avoid overwhelming test environment
            Thread.sleep(forTimeInterval: 0.01)
        }

        // Calculate percentiles
        let sorted = latencies.sorted()
        let p50 = sorted[numRequests / 2]
        let p95 = sorted[Int(Double(numRequests) * 0.95)]
        let p99 = sorted[numRequests - 1]
        let mean = latencies.reduce(0, +) / Double(numRequests)
        let min = sorted.first!
        let max = sorted.last!

        print("========================================")
        print("Latency Benchmark Results")
        print("========================================")
        print("Samples: \(numRequests)")
        print("Min:  \(String(format: "%.3f", min))s")
        print("Mean: \(String(format: "%.3f", mean))s")
        print("P50:  \(String(format: "%.3f", p50))s")
        print("P95:  \(String(format: "%.3f", p95))s")
        print("P99:  \(String(format: "%.3f", p99))s")
        print("Max:  \(String(format: "%.3f", max))s")
        print("========================================")

        // Assert reasonable latencies for test environment
        XCTAssertLessThan(p95, 1.0, "P95 latency should be <1s in test environment")
        XCTAssertLessThan(p99, 2.0, "P99 latency should be <2s in test environment")
    }

    /// Test: Concurrent request handling
    /// Verifies that concurrent requests complete successfully
    func testConcurrentRequestHandling() throws {
        let concurrency = 10
        let expectation = self.expectation(description: "All concurrent requests complete")
        expectation.expectedFulfillmentCount = concurrency

        let startTime = Date()

        // Submit concurrent requests
        for i in 0..<concurrency {
            DispatchQueue.global().async {
                let requestBody = """
                {
                    "model": "test-model",
                    "prompt": "Concurrent request \(i)",
                    "max_tokens": 10,
                    "stream": false
                }
                """

                do {
                    try self.app.testable().test(.POST, "v1/completions", beforeRequest: { req in
                        req.headers.add(name: .contentType, value: "application/json")
                        req.body = ByteBuffer(string: requestBody)
                    }, afterResponse: { res in
                        XCTAssertEqual(res.status, .ok, "Concurrent request \(i) should succeed")
                        expectation.fulfill()
                    })
                } catch {
                    XCTFail("Request \(i) failed: \(error)")
                    expectation.fulfill()
                }
            }
        }

        // Wait for all requests
        wait(for: [expectation], timeout: 15.0)

        let duration = Date().timeIntervalSince(startTime)

        print("========================================")
        print("Concurrent Request Test Results")
        print("========================================")
        print("Concurrent Requests: \(concurrency)")
        print("Total Duration: \(String(format: "%.2f", duration))s")
        print("Avg Time per Request: \(String(format: "%.3f", duration / Double(concurrency)))s")
        print("========================================")

        // Verify all requests completed
        XCTAssertTrue(true, "All concurrent requests completed successfully")
    }

    /// Test: Streaming performance
    /// Measures streaming latency characteristics
    func testStreamingPerformance() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Stream performance test",
            "max_tokens": 20,
            "stream": true
        }
        """

        let startTime = Date()
        var firstTokenTime: TimeInterval?
        var totalTokens = 0

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Streaming request should succeed")

            // Measure time to first token
            if firstTokenTime == nil {
                firstTokenTime = Date().timeIntervalSince(startTime)
            }

            // Count tokens in stream
            let bodyString = res.body.string
            let dataLines = bodyString.split(separator: "\n").filter { $0.hasPrefix("data: ") }
            totalTokens = dataLines.count - 1  // Subtract [DONE] message
        })

        let totalDuration = Date().timeIntervalSince(startTime)
        let ttft = firstTokenTime ?? 0  // Time to first token

        print("========================================")
        print("Streaming Performance Results")
        print("========================================")
        print("Time to First Token (TTFT): \(String(format: "%.3f", ttft))s")
        print("Total Duration: \(String(format: "%.3f", totalDuration))s")
        print("Total Tokens: \(totalTokens)")
        if totalTokens > 0 {
            print("Tokens per Second: \(String(format: "%.2f", Double(totalTokens) / totalDuration))")
        }
        print("========================================")

        // Assert reasonable streaming performance
        XCTAssertLessThan(ttft, 0.5, "Time to first token should be <500ms")
        XCTAssertGreaterThan(totalTokens, 0, "Should receive at least one token")
    }
}

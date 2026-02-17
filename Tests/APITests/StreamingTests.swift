import XCTest
import XCTVapor
import NIOCore
@testable import API
@testable import Core
@testable import Scheduler

/// Integration tests for SSE streaming (Phase 5.2)
final class StreamingTests: XCTestCase {

    var app: Application!
    var scheduler: RequestScheduler!
    var engine: InferenceEngine!
    var batcher: ContinuousBatcher!
    var batcherTask: Task<Void, Never>!

    override func setUp() {
        super.setUp()

        // Create test application
        app = Application(.testing)

        // Initialize components
        scheduler = RequestScheduler()
        engine = InferenceEngine()

        batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 4, eosTokenId: 2)
        )

        // Start batcher in background
        let batcherRef = batcher!
        batcherTask = Task {
            await batcherRef.start()
        }

        // Configure routes
        try! routes(app, scheduler: scheduler, engine: engine)

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

        app.shutdown()

        batcherTask.cancel()
        batcherTask = nil
        scheduler = nil
        engine = nil
        batcher = nil

        super.tearDown()
    }

    // MARK: - Phase 5.2 Tests

    /// Test: SSE headers are correct
    func testSSEHeaders() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 10,
            "stream": true
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            // Verify SSE headers
            XCTAssertEqual(
                res.headers.first(name: "Content-Type"),
                "text/event-stream",
                "Content-Type should be text/event-stream"
            )
            XCTAssertEqual(
                res.headers.first(name: "Cache-Control"),
                "no-cache",
                "Cache-Control should be no-cache"
            )
            XCTAssertEqual(
                res.headers.first(name: "Connection"),
                "keep-alive",
                "Connection should be keep-alive"
            )
        })
    }

    /// Test: Tokens arrive incrementally in SSE format
    func testTokensArriveIncrementally() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Count to 5",
            "max_tokens": 20,
            "stream": true
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            // Read body as string
            let bodyString = res.body.string
            XCTAssertFalse(bodyString.isEmpty, "Body should not be empty")

            // Verify SSE format (data: prefix on each chunk)
            let lines = bodyString.split(separator: "\n")
            let dataLines = lines.filter { $0.hasPrefix("data: ") }

            XCTAssertGreaterThan(dataLines.count, 0, "Should have multiple data chunks")

            // Verify [DONE] message is present
            XCTAssertTrue(
                bodyString.contains("data: [DONE]"),
                "Should contain [DONE] message"
            )
        })
    }

    /// Test: [DONE] message sent at end
    func testDoneMessageSent() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Test",
            "max_tokens": 5,
            "stream": true
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let bodyString = res.body.string

            // Verify [DONE] is the last data message
            XCTAssertTrue(
                bodyString.contains("data: [DONE]"),
                "[DONE] message should be present"
            )

            // Verify [DONE] comes after the stream chunks
            let doneRange = bodyString.range(of: "data: [DONE]")
            XCTAssertNotNil(doneRange, "[DONE] should exist in response")

            if let doneRange = doneRange {
                let beforeDone = String(bodyString[..<doneRange.lowerBound])
                XCTAssertTrue(
                    beforeDone.contains("data: {"),
                    "Should have JSON chunks before [DONE]"
                )
            }
        })
    }

    /// Test: Stream chunks have correct JSON structure
    func testStreamChunkStructure() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": true
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let bodyString = res.body.string

            // Extract first JSON chunk (skip [DONE])
            let lines = bodyString.split(separator: "\n")
            let dataLines = lines.filter { $0.hasPrefix("data: ") && !$0.contains("[DONE]") }

            guard let firstDataLine = dataLines.first else {
                XCTFail("Should have at least one data line")
                return
            }

            // Parse JSON
            let jsonString = String(firstDataLine.dropFirst("data: ".count))
            guard let jsonData = jsonString.data(using: .utf8) else {
                XCTFail("Should be valid UTF-8")
                return
            }

            do {
                let chunk = try JSONDecoder().decode(CompletionStreamChunk.self, from: jsonData)

                // Verify structure
                XCTAssertEqual(chunk.object, "text_completion.chunk", "Object type should be text_completion.chunk")
                XCTAssertEqual(chunk.model, "test-model", "Model should match request")
                XCTAssertEqual(chunk.choices.count, 1, "Should have one choice")
                XCTAssertEqual(chunk.choices[0].index, 0, "Choice index should be 0")
                XCTAssertFalse(chunk.id.isEmpty, "Should have request ID")
                XCTAssertGreaterThan(chunk.created, 0, "Should have valid timestamp")
            } catch {
                XCTFail("Failed to decode JSON: \(error)")
            }
        })
    }

    /// Test: Multiple concurrent streaming requests
    func testConcurrentStreamingRequests() throws {
        let numRequests = 3
        let group = DispatchGroup()
        var errors: [Error] = []

        for i in 0..<numRequests {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }

                let requestBody = """
                {
                    "model": "test-model",
                    "prompt": "Stream \(i)",
                    "max_tokens": 10,
                    "stream": true
                }
                """

                do {
                    try self.app.testable().test(.POST, "v1/completions", beforeRequest: { req in
                        req.headers.add(name: .contentType, value: "application/json")
                        req.body = ByteBuffer(string: requestBody)
                    }, afterResponse: { res in
                        XCTAssertEqual(res.status, .ok, "Concurrent stream \(i) should succeed")

                        let bodyString = res.body.string
                        XCTAssertFalse(bodyString.isEmpty, "Stream \(i) should have content")
                        XCTAssertTrue(
                            bodyString.contains("data: [DONE]"),
                            "Stream \(i) should have [DONE]"
                        )
                    })
                } catch {
                    errors.append(error)
                }
            }
        }

        group.wait()

        if !errors.isEmpty {
            XCTFail("Concurrent streaming had errors: \(errors)")
        }
    }
}

// MARK: - Response Models (for test decoding)

struct CompletionStreamChunk: Decodable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionStreamChoice]
}

struct CompletionStreamChoice: Decodable {
    let text: String
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text, index
        case finishReason = "finish_reason"
    }
}

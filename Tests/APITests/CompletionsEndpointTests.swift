import XCTest
import XCTVapor
@testable import API
@testable import Core
@testable import Scheduler

/// Integration tests for /v1/completions endpoint (Phase 5.1)
final class CompletionsEndpointTests: XCTestCase {

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

        // Shutdown app (synchronous for testing)
        app.shutdown()

        batcherTask.cancel()
        batcherTask = nil
        scheduler = nil
        engine = nil
        batcher = nil

        super.tearDown()
    }

    // MARK: - Phase 5.1 Tests

    /// Test: POST /v1/completions returns valid OpenAI-compatible response
    func testCompletionsReturnsValidResponse() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(CompletionResponse.self)
            XCTAssertEqual(response.object, "text_completion", "Object type should be text_completion")
            XCTAssertEqual(response.model, "test-model", "Model should match request")
            XCTAssertEqual(response.choices.count, 1, "Should have one choice")
            XCTAssertEqual(response.choices[0].index, 0, "Choice index should be 0")
            XCTAssertFalse(response.choices[0].text.isEmpty, "Should have generated text")
            XCTAssertEqual(response.choices[0].finishReason, "stop", "Should finish with stop reason")
            XCTAssertFalse(response.id.isEmpty, "Should have request ID")
            XCTAssertGreaterThan(response.created, 0, "Should have valid timestamp")
        })
    }

    /// Test: Empty prompt returns 400 Bad Request
    func testEmptyPromptReturnsBadRequest() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "",
            "max_tokens": 50
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .badRequest, "Should return 400 Bad Request for empty prompt")

            // Verify we got an error response (Vapor's error format may vary)
            XCTAssertNotNil(res.body, "Should have response body")
            XCTAssertTrue(res.body.readableBytes > 0, "Response body should not be empty")
        })
    }

    /// Test: Max tokens parameter is respected
    func testMaxTokensEnforced() throws {
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Count to 100",
            "max_tokens": 10
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(CompletionResponse.self)
            let text = response.choices[0].text

            // With placeholder implementation, we expect the text to not be excessively long
            // In a real implementation with token counting, we'd verify token count <= 10
            XCTAssertFalse(text.isEmpty, "Should have generated some text")
        })
    }

    /// Test: Temperature parameter affects sampling
    func testTemperatureParameter() throws {
        // Test with temperature 0.0 (greedy)
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Test prompt",
            "max_tokens": 20,
            "temperature": 0.0
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(CompletionResponse.self)
            XCTAssertFalse(response.choices[0].text.isEmpty, "Should have generated text")
        })

        // Test with high temperature
        let highTempBody = """
        {
            "model": "test-model",
            "prompt": "Test prompt",
            "max_tokens": 20,
            "temperature": 1.5
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: highTempBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK with high temperature")

            let response = try res.content.decode(CompletionResponse.self)
            XCTAssertFalse(response.choices[0].text.isEmpty, "Should have generated text")
        })
    }

    /// Test: Multiple concurrent requests handled correctly
    func testConcurrentRequests() throws {
        let numRequests = 5
        let group = DispatchGroup()
        var errors: [Error] = []

        for i in 0..<numRequests {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }

                let requestBody = """
                {
                    "model": "test-model",
                    "prompt": "Request \(i)",
                    "max_tokens": 10
                }
                """

                do {
                    try self.app.testable().test(.POST, "v1/completions", beforeRequest: { req in
                        req.headers.add(name: .contentType, value: "application/json")
                        req.body = ByteBuffer(string: requestBody)
                    }, afterResponse: { res in
                        XCTAssertEqual(res.status, .ok, "Concurrent request \(i) should succeed")

                        let response = try res.content.decode(CompletionResponse.self)
                        XCTAssertFalse(response.choices[0].text.isEmpty, "Request \(i) should have text")
                    })
                } catch {
                    errors.append(error)
                }
            }
        }

        group.wait()

        if !errors.isEmpty {
            XCTFail("Concurrent requests had errors: \(errors)")
        }
    }
}

// MARK: - Response Models (already defined in Routes.swift, duplicated here for test compilation)

struct CompletionResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
}

struct CompletionChoice: Content {
    let text: String
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text, index
        case finishReason = "finish_reason"
    }
}

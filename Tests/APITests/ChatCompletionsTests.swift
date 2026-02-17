import XCTest
import XCTVapor
@testable import API
@testable import Core
@testable import Scheduler

/// Integration tests for /v1/chat/completions endpoint (Phase 5.3)
final class ChatCompletionsTests: XCTestCase {

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

    // MARK: - Phase 5.3 Tests

    /// Test: Chat messages converted to prompt correctly
    func testChatMessagesConvertedToPrompt() throws {
        let requestBody = """
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        """

        try app.testable().test(.POST, "v1/chat/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(ChatCompletionResponse.self)
            XCTAssertEqual(response.object, "chat.completion", "Object type should be chat.completion")
            XCTAssertEqual(response.model, "test-model", "Model should match request")
            XCTAssertEqual(response.choices.count, 1, "Should have one choice")
            XCTAssertEqual(response.choices[0].index, 0, "Choice index should be 0")
            XCTAssertEqual(response.choices[0].message.role, "assistant", "Role should be assistant")
            XCTAssertFalse(response.choices[0].message.content.isEmpty, "Should have generated content")
            XCTAssertEqual(response.choices[0].finishReason, "stop", "Should finish with stop reason")
            XCTAssertFalse(response.id.isEmpty, "Should have request ID")
            XCTAssertGreaterThan(response.created, 0, "Should have valid timestamp")
        })
    }

    /// Test: System message included in prompt
    func testSystemMessageIncluded() throws {
        let requestBody = """
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 50
        }
        """

        try app.testable().test(.POST, "v1/chat/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(ChatCompletionResponse.self)
            XCTAssertEqual(response.choices[0].message.role, "assistant", "Role should be assistant")
            XCTAssertFalse(response.choices[0].message.content.isEmpty, "Should have generated content")
        })
    }

    /// Test: Multi-turn conversation handled correctly
    func testMultiTurnConversation() throws {
        let requestBody = """
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What about 3+3?"}
            ],
            "max_tokens": 50
        }
        """

        try app.testable().test(.POST, "v1/chat/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(ChatCompletionResponse.self)
            XCTAssertEqual(response.choices.count, 1, "Should have one choice")
            XCTAssertEqual(response.choices[0].message.role, "assistant", "Role should be assistant")
            XCTAssertFalse(response.choices[0].message.content.isEmpty, "Should have generated content")
            XCTAssertEqual(response.choices[0].finishReason, "stop", "Should finish with stop reason")
        })
    }

    /// Test: Streaming chat responses work correctly
    func testStreamingChatResponses() throws {
        let requestBody = """
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 50,
            "stream": true
        }
        """

        try app.testable().test(.POST, "v1/chat/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            // Verify SSE headers
            XCTAssertEqual(res.headers.first(name: "Content-Type"), "text/event-stream", "Should have SSE content type")
            XCTAssertEqual(res.headers.first(name: "Cache-Control"), "no-cache", "Should have no-cache header")
            XCTAssertEqual(res.headers.first(name: "Connection"), "keep-alive", "Should have keep-alive header")

            // Verify SSE body format
            let bodyString = res.body.string
            XCTAssertTrue(bodyString.contains("data: "), "Should contain SSE data lines")
            XCTAssertTrue(bodyString.contains("[DONE]"), "Should end with [DONE] message")

            // Verify chunk structure by parsing first data line
            let lines = bodyString.split(separator: "\n")
            let dataLines = lines.filter { $0.hasPrefix("data: ") }
            XCTAssertGreaterThan(dataLines.count, 0, "Should have at least one data line")

            // Parse first chunk (skip [DONE])
            let firstDataLine = dataLines.first { !$0.contains("[DONE]") }
            if let firstData = firstDataLine {
                let jsonData = String(firstData.dropFirst(6)) // Remove "data: " prefix
                let data = jsonData.data(using: .utf8)!
                let chunk = try? JSONDecoder().decode(ChatCompletionStreamChunk.self, from: data)
                XCTAssertNotNil(chunk, "Should decode ChatCompletionStreamChunk")
                XCTAssertEqual(chunk?.object, "chat.completion.chunk", "Object type should be chat.completion.chunk")
                XCTAssertEqual(chunk?.choices.count, 1, "Should have one choice")
                XCTAssertEqual(chunk?.choices[0].delta.role, "assistant", "Delta role should be assistant")
            }
        })
    }

    /// Test: Empty messages rejected
    func testEmptyMessagesRejected() throws {
        let requestBody = """
        {
            "model": "test-model",
            "messages": [],
            "max_tokens": 50
        }
        """

        try app.testable().test(.POST, "v1/chat/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .badRequest, "Should return 400 Bad Request for empty messages")
        })
    }
}

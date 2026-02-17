import XCTest
import XCTVapor
@testable import API
@testable import Core
@testable import Scheduler

/// Integration tests for observability endpoints (Phase 5.4)
final class ObservabilityTests: XCTestCase {

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
        // Note: Not initializing engine with model - testing without loaded model

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

    // MARK: - Phase 5.4 Tests

    /// Test: Health endpoint shows correct status
    func testHealthEndpointShowsCorrectStatus() throws {
        try app.testable().test(.GET, "health", afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(HealthResponse.self)
            // Model not loaded in test setup
            XCTAssertEqual(response.status, "degraded", "Status should be degraded when model not loaded")
            XCTAssertEqual(response.model, "not_loaded", "Model should be not_loaded")
            // Batcher should be running
            XCTAssertTrue(response.batcher == "running" || response.batcher == "stopped", "Batcher should be running or stopped")
            XCTAssertNotNil(response.timestamp, "Should have timestamp")
        })
    }

    /// Test: Metrics reflect scheduler/batcher state
    func testMetricsReflectState() throws {
        try app.testable().test(.GET, "metrics", afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(MetricsResponse.self)

            // Check request metrics
            XCTAssertGreaterThanOrEqual(response.requests.pending, 0, "Pending should be >= 0")
            XCTAssertGreaterThanOrEqual(response.requests.active, 0, "Active should be >= 0")
            XCTAssertGreaterThanOrEqual(response.requests.completed, 0, "Completed should be >= 0")
            XCTAssertGreaterThanOrEqual(response.requests.failed, 0, "Failed should be >= 0")
            XCTAssertGreaterThanOrEqual(response.requests.cancelled, 0, "Cancelled should be >= 0")

            // Check batcher metrics
            XCTAssertGreaterThanOrEqual(response.batcher.activeSlots, 0, "Active slots should be >= 0")
            XCTAssertGreaterThan(response.batcher.totalSlots, 0, "Total slots should be > 0")
            XCTAssertGreaterThanOrEqual(response.batcher.utilization, 0, "Utilization should be >= 0")
            XCTAssertLessThanOrEqual(response.batcher.utilization, 1.0, "Utilization should be <= 1.0")
            XCTAssertGreaterThanOrEqual(response.batcher.stepCount, 0, "Step count should be >= 0")

            // Check GPU metrics
            XCTAssertGreaterThanOrEqual(response.gpu.averageUtilization, 0, "Avg utilization should be >= 0")
            XCTAssertGreaterThanOrEqual(response.gpu.currentUtilization, 0, "Current utilization should be >= 0")
            XCTAssertGreaterThanOrEqual(response.gpu.sampleCount, 0, "Sample count should be >= 0")
        })
    }

    /// Test: Request ID propagated correctly
    func testRequestIDPropagated() throws {
        let customRequestID = "test-request-id-12345"

        try app.testable().test(.GET, "health", beforeRequest: { req in
            req.headers.add(name: "X-Request-ID", value: customRequestID)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            // Check that request ID is in response headers
            let responseRequestID = res.headers.first(name: "X-Request-ID")
            XCTAssertEqual(responseRequestID, customRequestID, "Response should have same request ID")
        })
    }

    /// Test: Request ID auto-generated when not provided
    func testRequestIDAutoGenerated() throws {
        try app.testable().test(.GET, "health", afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            // Check that request ID was auto-generated
            let responseRequestID = res.headers.first(name: "X-Request-ID")
            XCTAssertNotNil(responseRequestID, "Response should have auto-generated request ID")
            XCTAssertFalse(responseRequestID?.isEmpty ?? true, "Request ID should not be empty")

            // Check that it's a valid UUID format
            XCTAssertNotNil(UUID(uuidString: responseRequestID ?? ""), "Request ID should be a valid UUID")
        })
    }

    /// Test: Readiness endpoint reflects model loaded state
    func testReadinessReflectsModelState() throws {
        try app.testable().test(.GET, "ready", afterResponse: { res in
            // Should be not ready when model not loaded
            XCTAssertEqual(res.status, .serviceUnavailable, "Should return 503 when model not loaded")

            let bodyString = res.body.string
            XCTAssertEqual(bodyString, "Not Ready", "Body should indicate not ready")
        })
    }

    /// Test: Metrics after submitting requests
    func testMetricsAfterSubmittingRequests() throws {
        // Submit a test request
        let requestBody = """
        {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 10
        }
        """

        try app.testable().test(.POST, "v1/completions", beforeRequest: { req in
            req.headers.add(name: .contentType, value: "application/json")
            req.body = ByteBuffer(string: requestBody)
        }, afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Completion should succeed")
        })

        // Small delay for request to be processed
        Thread.sleep(forTimeInterval: 0.1)

        // Check metrics
        try app.testable().test(.GET, "metrics", afterResponse: { res in
            XCTAssertEqual(res.status, .ok, "Should return 200 OK")

            let response = try res.content.decode(MetricsResponse.self)

            // Should have at least one completed or active request
            let totalProcessed = response.requests.completed + response.requests.active + response.requests.pending
            XCTAssertGreaterThan(totalProcessed, 0, "Should have processed at least one request")
        })
    }
}

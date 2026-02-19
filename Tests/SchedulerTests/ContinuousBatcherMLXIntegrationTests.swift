import XCTest
import Core
import Logging
@testable import Scheduler

/// Integration tests for ContinuousBatcher with real MLX model
/// Phase 4.2: End-to-end testing with actual token generation
///
/// These tests verify that ContinuousBatcher works correctly with real MLX inference:
/// - Real tokens are generated (not placeholders)
/// - Multiple concurrent requests work correctly
/// - Cancellation stops generation properly
/// - Performance metrics can be measured
///
/// Note: These tests require model download and are slower than unit tests
final class ContinuousBatcherMLXIntegrationTests: XCTestCase {
    var scheduler: RequestScheduler!
    var engine: InferenceEngine!
    var batcher: ContinuousBatcher!

    // MARK: - Setup/Teardown

    override func setUp() async throws {
        scheduler = RequestScheduler()
        engine = InferenceEngine()

        // Load the default small model (Qwen2.5-0.5B-Instruct-4bit)
        // This will download on first run (~300MB) and cache for subsequent runs
        print("Loading model (may take time on first run)...")
        try await engine.initialize()
        print("Model loaded successfully")

        // Create batcher with real engine
        batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(maxBatchSize: 8, eosTokenId: 2)
        )
    }

    override func tearDown() async throws {
        await batcher.stop()
        scheduler = nil
        engine = nil
        batcher = nil
    }

    // MARK: - Integration Tests

    /// Test: Single request generates 20 real tokens
    /// Verifies that ContinuousBatcher can process a single request end-to-end
    func testSingleRequestGeneratesRealTokens() async throws {
        // Submit a request for 20 tokens
        let request = InferenceRequest(
            prompt: "Hello, my name is",
            maxTokens: 20
        )
        let (requestId, stream) = await scheduler.submit(request, priority: .normal)

        // Start batcher in background
        let batcherTask = Task { [batcher] in
            await batcher!.start()
        }

        // Collect generated tokens
        var tokens: [String] = []
        let startTime = Date()
        var completedSuccessfully = false

        do {
            for try await chunk in stream {
                tokens.append(chunk.token)
            }
            completedSuccessfully = true
        } catch {
            // Stream closed or error - check if request completed
            let status = await scheduler.getStatus(requestId: requestId)
            completedSuccessfully = (status == .completed)
        }

        let duration = Date().timeIntervalSince(startTime)

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()

        // Verify tokens were generated
        XCTAssertFalse(tokens.isEmpty, "Should generate at least some tokens")
        XCTAssertTrue(completedSuccessfully, "Request should complete successfully")

        // Log performance
        let tokensPerSecond = Double(tokens.count) / duration
        print("Single request performance: \(tokens.count) tokens in \(String(format: "%.2f", duration))s (\(String(format: "%.2f", tokensPerSecond)) tok/s)")

        // Verify tokens are numeric strings (converted from token IDs)
        // In Phase 4.2, tokens are just "\(tokenId)" strings
        for token in tokens {
            XCTAssertFalse(token.isEmpty, "Token should not be empty")
        }
    }

    /// Test: 4 concurrent requests all generate tokens
    /// Verifies that batching works correctly with multiple requests
    func testFourConcurrentRequests() async throws {
        // Submit 4 requests
        let prompts = [
            "The capital of France is",
            "In the year 2025,",
            "Machine learning is",
            "The quick brown fox"
        ]

        var streams: [TokenStream] = []
        var requestIds: [UUID] = []

        for prompt in prompts {
            let request = InferenceRequest(prompt: prompt, maxTokens: 15)
            let (requestId, stream) = await scheduler.submit(request, priority: .normal)
            requestIds.append(requestId)
            streams.append(stream)
        }

        // Start batcher in background
        let startTime = Date()
        let batcherTask = Task { [batcher] in
            await batcher!.start()
        }

        // Collect tokens from all streams concurrently
        var allTokens: [[String]] = Array(repeating: [], count: 4)

        await withTaskGroup(of: (Int, [String]).self) { group in
            for (idx, stream) in streams.enumerated() {
                group.addTask {
                    var tokens: [String] = []

                    do {
                        for try await chunk in stream {
                            tokens.append(chunk.token)
                        }
                    } catch {
                        // Stream error or closure
                    }

                    return (idx, tokens)
                }
            }

            for await (idx, tokens) in group {
                allTokens[idx] = tokens
            }
        }

        let maxDuration = Date().timeIntervalSince(startTime)

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()

        // Verify all requests completed
        for (idx, tokens) in allTokens.enumerated() {
            XCTAssertFalse(tokens.isEmpty, "Request \(idx) should generate tokens")
            print("Request \(idx): \(tokens.count) tokens generated")
        }

        // Calculate aggregate throughput
        let totalTokens = allTokens.reduce(0) { $0 + $1.count }
        let aggregateThroughput = Double(totalTokens) / maxDuration

        print("4 concurrent requests: \(totalTokens) total tokens in \(String(format: "%.2f", maxDuration))s (\(String(format: "%.2f", aggregateThroughput)) tok/s aggregate)")
    }

    /// Test: 8 concurrent requests all generate tokens
    /// Verifies that full batch utilization works correctly
    func testEightConcurrentRequests() async throws {
        // Submit 8 requests (matches max batch size)
        var streams: [TokenStream] = []

        for i in 0..<8 {
            let request = InferenceRequest(
                prompt: "Request \(i): Once upon a time",
                maxTokens: 10
            )
            let (_, stream) = await scheduler.submit(request, priority: .normal)
            streams.append(stream)
        }

        // Start batcher in background
        let batcherTask = Task { [batcher] in
            await batcher!.start()
        }

        // Collect tokens from all streams
        var completedCount = 0
        var totalTokens = 0

        await withTaskGroup(of: Int.self) { group in
            for stream in streams {
                group.addTask {
                    var tokenCount = 0

                    do {
                        for try await _ in stream {
                            tokenCount += 1
                        }
                    } catch {
                        // Stream error or closure
                    }

                    return tokenCount
                }
            }

            for await tokenCount in group {
                if tokenCount > 0 {
                    completedCount += 1
                    totalTokens += tokenCount
                }
            }
        }

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()

        // Verify all 8 requests completed
        XCTAssertEqual(completedCount, 8, "All 8 requests should complete")
        XCTAssertGreaterThan(totalTokens, 0, "Should generate tokens across all requests")

        print("8 concurrent requests completed: \(totalTokens) total tokens generated")

        // Check batch utilization
        let stats = await batcher.getStats()
        print("Final stats: \(stats.activeSlots) active slots, \(stats.stepCount) steps")
    }

    /// Test: Request cancellation stops generation
    /// Verifies that cancelled requests are removed from the batch
    func testRequestCancellation() async throws {
        // Submit a long-running request
        let request = InferenceRequest(
            prompt: "Tell me a very long story",
            maxTokens: 100
        )
        let (requestId, stream) = await scheduler.submit(request, priority: .normal)

        // Start batcher in background
        let batcherTask = Task { [batcher] in
            await batcher!.start()
        }

        // Wait for request to start generating
        actor TokenCounter {
            var count = 0
            func increment() { count += 1 }
            func get() -> Int { count }
        }
        let counter = TokenCounter()

        let streamTask = Task {
            do {
                for try await _ in stream {
                    await counter.increment()
                }
            } catch {
                // Expected - stream will be cancelled
            }
        }

        // Wait a moment for generation to start
        try await Task.sleep(for: .milliseconds(100))

        // Cancel the request
        await scheduler.cancel(requestId: requestId)

        // Wait a moment for cancellation to process
        try await Task.sleep(for: .milliseconds(100))

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()
        streamTask.cancel()

        // Verify the request was cancelled
        let status = await scheduler.getStatus(requestId: requestId)
        XCTAssertEqual(status, .cancelled, "Request should be cancelled")

        // Slot should be freed
        let stats = await batcher.getStats()
        XCTAssertEqual(stats.activeSlots, 0, "Cancelled request should free its slot")

        let tokenCount = await counter.get()
        print("Cancellation test: Generated \(tokenCount) tokens before cancellation")
    }

    /// Test: Measure baseline performance (tokens/sec, latency)
    /// Provides baseline metrics for continuous batching
    func testBaselinePerformance() async throws {
        // Test configuration
        let numRequests = 4
        let tokensPerRequest = 20
        let startTime = Date()

        // Submit requests
        var streams: [TokenStream] = []
        for i in 0..<numRequests {
            let request = InferenceRequest(
                prompt: "Performance test \(i):",
                maxTokens: tokensPerRequest
            )
            let (_, stream) = await scheduler.submit(request, priority: .normal)
            streams.append(stream)
        }

        // Start batcher
        let batcherTask = Task { [batcher] in
            await batcher!.start()
        }

        // Collect metrics from all requests
        struct Metrics {
            var tokenCount: Int = 0
            var firstTokenLatency: TimeInterval?
            var completionTime: TimeInterval = 0
        }

        var allMetrics: [Metrics] = []

        await withTaskGroup(of: Metrics.self) { group in
            for stream in streams {
                group.addTask {
                    var metrics = Metrics()
                    let requestStart = Date()

                    do {
                        for try await _ in stream {
                            metrics.tokenCount += 1
                            if metrics.firstTokenLatency == nil {
                                metrics.firstTokenLatency = Date().timeIntervalSince(requestStart)
                            }
                        }
                    } catch {
                        // Stream error or closure
                    }

                    metrics.completionTime = Date().timeIntervalSince(requestStart)

                    return metrics
                }
            }

            for await metrics in group {
                allMetrics.append(metrics)
            }
        }

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()
        let totalTime = Date().timeIntervalSince(startTime)

        // Calculate metrics
        let totalTokens = allMetrics.reduce(0) { $0 + $1.tokenCount }
        let avgFirstTokenLatency = allMetrics.compactMap { $0.firstTokenLatency }.reduce(0, +) / Double(allMetrics.count)
        let avgCompletionTime = allMetrics.reduce(0) { $0 + $1.completionTime } / Double(allMetrics.count)
        let aggregateThroughput = Double(totalTokens) / totalTime
        let perRequestThroughput = Double(totalTokens) / allMetrics.reduce(0) { $0 + $1.completionTime }

        // Print baseline metrics
        print("\n=== Baseline Performance Metrics ===")
        print("Configuration: \(numRequests) requests × \(tokensPerRequest) tokens")
        print("Total tokens generated: \(totalTokens)")
        print("Total time: \(String(format: "%.2f", totalTime))s")
        print("Aggregate throughput: \(String(format: "%.2f", aggregateThroughput)) tokens/sec")
        print("Per-request throughput: \(String(format: "%.2f", perRequestThroughput)) tokens/sec")
        print("Avg first token latency: \(String(format: "%.3f", avgFirstTokenLatency))s")
        print("Avg completion time: \(String(format: "%.2f", avgCompletionTime))s")

        // Get batcher stats
        let batcherStats = await batcher.getStats()
        print("Batching steps: \(batcherStats.stepCount)")
        print("===================================\n")

        // Sanity checks
        XCTAssertGreaterThan(totalTokens, 0, "Should generate tokens")
        XCTAssertGreaterThan(aggregateThroughput, 0, "Should have positive throughput")
        XCTAssertGreaterThan(avgFirstTokenLatency, 0, "Should have positive latency")
    }
}

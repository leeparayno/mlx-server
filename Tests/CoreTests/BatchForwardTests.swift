import XCTest
@testable import Core
import MLX

/// Comprehensive tests for Phase 4.2: True MLX Batched Forward Pass
final class BatchForwardTests: XCTestCase {

    var engine: InferenceEngine!

    override func setUp() async throws {
        engine = InferenceEngine()
        // Note: Not initializing with model - testing sampling logic independently
    }

    override func tearDown() async throws {
        engine = nil
    }

    // MARK: - Temperature Scaling Tests

    func testGreedySampling() async throws {
        // Temperature 0.0 should always pick highest probability token
        let logits = [
            [Float(1.0), Float(5.0), Float(2.0), Float(3.0)],  // Token 1 has highest logit
            [Float(3.0), Float(1.0), Float(4.0), Float(2.0)],  // Token 2 has highest logit
            [Float(2.0), Float(3.0), Float(1.0), Float(5.0)]   // Token 3 has highest logit
        ]

        let temperatures: [Float] = [0.0, 0.0, 0.0]
        let topP: [Float] = [1.0, 1.0, 1.0]

        let samples = try await engine.sampleBatch(
            logits: logits,
            temperatures: temperatures,
            topP: topP
        )

        XCTAssertEqual(samples.count, 3)
        XCTAssertEqual(samples[0], 1, "Should pick token 1 (highest logit)")
        XCTAssertEqual(samples[1], 2, "Should pick token 2 (highest logit)")
        XCTAssertEqual(samples[2], 3, "Should pick token 3 (highest logit)")
    }

    func testTemperatureScaling() async throws {
        // Higher temperature should produce more diverse samples
        let logits = Array(repeating: [Float(10.0), Float(9.0), Float(8.0), Float(7.0)], count: 100)

        // Low temperature (close to greedy)
        let lowTempSamples = try await engine.sampleBatch(
            logits: logits,
            temperatures: Array(repeating: 0.1, count: 100),
            topP: Array(repeating: 1.0, count: 100)
        )

        // High temperature (more diverse)
        let highTempSamples = try await engine.sampleBatch(
            logits: logits,
            temperatures: Array(repeating: 2.0, count: 100),
            topP: Array(repeating: 1.0, count: 100)
        )

        // Low temperature should heavily favor token 0
        let lowTempToken0Count = lowTempSamples.filter { $0 == 0 }.count
        let highTempToken0Count = highTempSamples.filter { $0 == 0 }.count

        XCTAssertGreaterThan(lowTempToken0Count, highTempToken0Count,
                            "Low temperature should favor highest logit token more than high temperature")
        XCTAssertGreaterThan(lowTempToken0Count, 80,
                            "Low temperature should pick highest token >80% of the time")
    }

    // MARK: - Top-P Sampling Tests

    func testTopPFiltering() async throws {
        // Test that top-p cuts off low probability tokens
        let logits = [
            [Float(10.0), Float(5.0), Float(1.0), Float(0.1)]  // After softmax: ~0.95, ~0.047, ~0.002, ~0.001
        ]

        // Top-p = 0.95 should only consider first token
        let strictTopPSamples = try await engine.sampleBatch(
            logits: logits,
            temperatures: [1.0],
            topP: [0.95]
        )

        // Top-p = 0.999 should consider first two tokens
        let relaxedTopPSamples = try await engine.sampleBatch(
            logits: Array(repeating: logits[0], count: 100),
            temperatures: Array(repeating: 1.0, count: 100),
            topP: Array(repeating: 0.999, count: 100)
        )

        XCTAssertEqual(strictTopPSamples[0], 0,
                      "Top-p=0.95 should always pick token 0 with these logits")

        let tokensUsed = Set(relaxedTopPSamples)
        XCTAssertLessThanOrEqual(tokensUsed.count, 2,
                                "Top-p=0.999 should only use first 2 tokens")
    }

    func testTopPVsNoTopP() async throws {
        // Top-p should reduce diversity compared to full distribution
        let logits = Array(repeating: [Float(5.0), Float(4.8), Float(4.6), Float(4.4), Float(4.2), Float(1.0), Float(0.5)], count: 100)

        let withTopP = try await engine.sampleBatch(
            logits: logits,
            temperatures: Array(repeating: 1.0, count: 100),
            topP: Array(repeating: 0.9, count: 100)
        )

        let withoutTopP = try await engine.sampleBatch(
            logits: logits,
            temperatures: Array(repeating: 1.0, count: 100),
            topP: Array(repeating: 1.0, count: 100)
        )

        let topPUniqueTokens = Set(withTopP)
        let fullUniqueTokens = Set(withoutTopP)

        XCTAssertLessThanOrEqual(topPUniqueTokens.count, fullUniqueTokens.count,
                                "Top-p should produce fewer unique tokens than full distribution")
    }

    // MARK: - Batch Consistency Tests

    func testBatchSizeConsistency() async throws {
        let batchSizes = [1, 5, 10, 20]

        for batchSize in batchSizes {
            let logits = Array(repeating: [Float(1.0), Float(2.0), Float(3.0)], count: batchSize)
            let temps = Array(repeating: Float(0.7), count: batchSize)
            let topP = Array(repeating: Float(0.9), count: batchSize)

            let samples = try await engine.sampleBatch(
                logits: logits,
                temperatures: temps,
                topP: topP
            )

            XCTAssertEqual(samples.count, batchSize,
                          "Output batch size should match input for batch size \(batchSize)")
        }
    }

    func testBatchParameterMismatch() async throws {
        let logits = [[Float(1.0), Float(2.0)], [Float(3.0), Float(4.0)]]
        let temps: [Float] = [0.7]  // Wrong size
        let topP: [Float] = [0.9, 0.9]

        await XCTAssertThrowsErrorAsync(
            try await engine.sampleBatch(logits: logits, temperatures: temps, topP: topP)
        ) { error in
            if case InferenceError.invalidParameters(let msg) = error {
                XCTAssertTrue(msg.contains("mismatch"), "Should report batch size mismatch")
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }
    }

    func testMixedParametersInBatch() async throws {
        // Test that different temperatures/top-p work in same batch
        let logits = [
            [Float(10.0), Float(5.0), Float(1.0)],  // Slot 0: greedy
            [Float(10.0), Float(5.0), Float(1.0)],  // Slot 1: high temp
            [Float(10.0), Float(5.0), Float(1.0)]   // Slot 2: strict top-p
        ]

        let temps: [Float] = [0.0, 2.0, 1.0]
        let topP: [Float] = [1.0, 1.0, 0.95]

        // Sample multiple times to test diversity
        var samples: [[Int]] = []
        for _ in 0..<50 {
            let batch = try await engine.sampleBatch(
                logits: logits,
                temperatures: temps,
                topP: topP
            )
            samples.append(batch)
        }

        // Slot 0 (greedy) should always be the same
        let slot0Samples = samples.map { $0[0] }
        XCTAssertEqual(Set(slot0Samples).count, 1, "Greedy sampling should be deterministic")
        XCTAssertEqual(slot0Samples[0], 0, "Greedy should pick highest logit")

        // Slot 1 (high temp) should be more diverse
        let slot1Samples = samples.map { $0[1] }
        XCTAssertGreaterThan(Set(slot1Samples).count, 1, "High temperature should produce diversity")
    }

    // MARK: - Edge Cases

    func testEmptyLogits() async throws {
        let logits: [[Float]] = [[]]  // Empty array is fine
        let temps: [Float] = [0.7]
        let topP: [Float] = [0.9]

        let samples = try await engine.sampleBatch(logits: logits, temperatures: temps, topP: topP)

        XCTAssertEqual(samples.count, 1)
        XCTAssertEqual(samples[0], 0, "Empty logits should return fallback token 0")
    }

    func testUniformLogits() async throws {
        // All logits equal - should sample uniformly
        let logits = Array(repeating: [Float(1.0), Float(1.0), Float(1.0), Float(1.0)], count: 100)

        let samples = try await engine.sampleBatch(
            logits: logits,
            temperatures: Array(repeating: 1.0, count: 100),
            topP: Array(repeating: 1.0, count: 100)
        )

        let uniqueTokens = Set(samples)
        XCTAssertGreaterThan(uniqueTokens.count, 2,
                            "Uniform logits should produce diverse samples")

        // Check roughly uniform distribution (chi-square test approximation)
        let tokenCounts = (0..<4).map { token in
            samples.filter { $0 == token }.count
        }

        for count in tokenCounts {
            XCTAssertGreaterThan(count, 10, "Each token should appear at least 10 times")
            XCTAssertLessThan(count, 40, "Each token shouldn't dominate (expect ~25/100)")
        }
    }

    func testExtremeLogits() async throws {
        // Test with very large/small logit values
        let logits: [[Float]] = [
            [Float(1000.0), Float(-1000.0), Float(0.0)],  // Very extreme
            [Float.leastNormalMagnitude, Float.leastNormalMagnitude, Float.leastNormalMagnitude]  // Very small
        ]

        let samples = try await engine.sampleBatch(
            logits: logits,
            temperatures: [1.0, 1.0],
            topP: [1.0, 1.0]
        )

        XCTAssertEqual(samples.count, 2)
        XCTAssertEqual(samples[0], 0, "Extreme logits should pick highest")
        // Second sample should not crash - just produce some valid token
        XCTAssertTrue(samples[1] >= 0 && samples[1] < 3)
    }

    // MARK: - Forward Batch API Tests

    func testForwardBatchPlaceholder() async throws {
        // Test the forward batch API without model initialization
        let tokenIds = [1, 2, 3]
        let positions = [0, 1, 2]
        let prompts = ["test1", "test2", "test3"]

        let nextTokens = try await engine.forwardBatch(
            tokenIds: tokenIds,
            positions: positions,
            prompts: prompts
        )

        XCTAssertEqual(nextTokens.count, 3, "Should return one token per slot")
        // Placeholder should return non-zero tokens
        for token in nextTokens {
            XCTAssertGreaterThan(token, 0, "Placeholder tokens should be > 0")
        }
    }

    func testForwardBatchWithKVCache() async throws {
        // Test forward batch API with KV cache block IDs
        let tokenIds = [1, 2]
        let positions = [0, 1]
        let prompts = ["test1", "test2"]
        let kvCacheBlockIds = [[0, 1], [2, 3]]
        let temps: [Float] = [0.7, 0.9]
        let topP: [Float] = [0.95, 0.95]

        let nextTokens = try await engine.forwardBatch(
            tokenIds: tokenIds,
            positions: positions,
            prompts: prompts,
            kvCacheBlockIds: kvCacheBlockIds,
            temperatures: temps,
            topP: topP
        )

        XCTAssertEqual(nextTokens.count, 2)
    }

    func testForwardBatchSizeValidation() async throws {
        let tokenIds = [1, 2]
        let positions = [0]  // Mismatched size
        let prompts = ["test1", "test2"]

        await XCTAssertThrowsErrorAsync(
            try await engine.forwardBatch(
                tokenIds: tokenIds,
                positions: positions,
                prompts: prompts
            )
        ) { error in
            if case InferenceError.invalidParameters(let msg) = error {
                XCTAssertTrue(msg.contains("Mismatched"), "Should report size mismatch")
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }
    }
}

// MARK: - Test Helpers

extension XCTestCase {
    func XCTAssertThrowsErrorAsync<T>(
        _ expression: @autoclosure () async throws -> T,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line,
        _ errorHandler: (_ error: Error) -> Void = { _ in }
    ) async {
        do {
            _ = try await expression()
            XCTFail("Expected error to be thrown", file: file, line: line)
        } catch {
            errorHandler(error)
        }
    }
}

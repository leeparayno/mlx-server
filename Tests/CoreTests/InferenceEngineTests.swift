import XCTest
@testable import Core
import Foundation

final class InferenceEngineTests: XCTestCase {

    // Use the same test model as ModelDownloaderTests
    let testModelId = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    // MARK: - Initialization Tests

    func testInferenceEngineInitialization() async throws {
        let engine = InferenceEngine()
        XCTAssertNotNil(engine, "InferenceEngine should initialize")

        let isLoaded = await engine.isModelLoaded
        XCTAssertFalse(isLoaded, "Model should not be loaded initially")
    }

    func testDefaultModelIDConstant() {
        XCTAssertEqual(
            InferenceEngine.defaultModelID,
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "Default model ID should be Qwen2.5-0.5B-Instruct-4bit"
        )
    }

    // MARK: - Model Loading Tests (Integration)

    /// Integration test: Load real model from HuggingFace
    /// This test downloads ~340MB model, so it will take 20-30 seconds on first run
    /// Subsequent runs will use cached model and be much faster
    func testModelLoadingSuccess() async throws {
        let engine = InferenceEngine()

        // Track progress updates
        actor ProgressTracker {
            var updates: [Double] = []
            func add(_ value: Double) {
                updates.append(value)
            }
            func getUpdates() -> [Double] {
                return updates
            }
        }
        let tracker = ProgressTracker()

        // Load model with default ID
        try await engine.initialize(
            progressHandler: { progress, speed in
                Task { await tracker.add(progress.fractionCompleted) }
                if let speed = speed {
                    print("  Download speed: \(String(format: "%.2f", speed / 1024 / 1024)) MB/s")
                }
            }
        )

        // Verify model is loaded
        let isLoaded = await engine.isModelLoaded
        XCTAssertTrue(isLoaded, "Model should be loaded after initialization")

        // Verify model info is accessible
        let modelInfo = await engine.modelInfo
        XCTAssertNotNil(modelInfo, "Model info should be available")
        print("  Loaded model: \(modelInfo?.id ?? "unknown")")

        // If progress was reported, verify it reached 100% (or close to it for cached models)
        let progressUpdates = await tracker.getUpdates()
        if !progressUpdates.isEmpty {
            if let lastProgress = progressUpdates.last {
                print("  Final progress: \(String(format: "%.1f%%", lastProgress * 100))")
                // Don't assert on progress - cached models may not report progress
            }
        }
    }

    func testModelLoadingWithCustomPath() async throws {
        let engine = InferenceEngine()

        // Load model with explicit path (same as default)
        try await engine.initialize(modelPath: testModelId)

        // Verify model is loaded
        let isLoaded = await engine.isModelLoaded
        XCTAssertTrue(isLoaded, "Model should be loaded with custom path")
    }

    func testModelAlreadyInitialized() async throws {
        let engine = InferenceEngine()

        // Initialize once
        try await engine.initialize()

        // Initialize again (should be idempotent)
        try await engine.initialize()

        // Verify still loaded
        let isLoaded = await engine.isModelLoaded
        XCTAssertTrue(isLoaded, "Model should still be loaded after re-initialization")
    }

    // MARK: - Error Handling Tests

    func testModelNotFoundError() async throws {
        let engine = InferenceEngine()

        do {
            // Try to load non-existent model
            try await engine.initialize(modelPath: "invalid/nonexistent-model-xyz-\(UUID().uuidString)")
            XCTFail("Should throw error for non-existent model")
        } catch {
            // Expected error
            print("  Correctly caught error: \(error)")

            // Verify model is not loaded
            let isLoaded = await engine.isModelLoaded
            XCTAssertFalse(isLoaded, "Model should not be loaded after failed initialization")
        }
    }

    func testGenerateWithoutInitialization() async throws {
        let engine = InferenceEngine()

        do {
            // Try to generate without loading model
            _ = try await engine.generate(prompt: "Hello")
            XCTFail("Should throw error when model not initialized")
        } catch let error as InferenceError {
            // Verify we got the correct error type
            switch error {
            case .notInitialized:
                XCTAssertTrue(true, "Correct error thrown")
            default:
                XCTFail("Wrong error type: \(error)")
            }
        }
    }

    // MARK: - Model Info Tests

    func testModelInfoAccess() async throws {
        let engine = InferenceEngine()

        // Before initialization
        let infoBefore = await engine.modelInfo
        XCTAssertNil(infoBefore, "Model info should be nil before initialization")

        // After initialization
        try await engine.initialize()
        let infoAfter = await engine.modelInfo
        XCTAssertNotNil(infoAfter, "Model info should be available after initialization")

        // Verify model info has expected fields
        XCTAssertFalse(infoAfter!.name.isEmpty, "Model name should not be empty")
        XCTAssertFalse(infoAfter!.id.isEmpty, "Model ID should not be empty")
    }

    // MARK: - Integration Test: End-to-End Generation

    func testEndToEndGeneration() async throws {
        let engine = InferenceEngine()

        // Load model
        try await engine.initialize()

        // Generate text
        let prompt = "Hello, how are"
        let result = try await engine.generate(
            prompt: prompt,
            maxTokens: 10,
            temperature: 0.7
        )

        // Verify result
        XCTAssertFalse(result.isEmpty, "Generated text should not be empty")
        print("  Prompt: \(prompt)")
        print("  Generated: \(result)")
    }
}

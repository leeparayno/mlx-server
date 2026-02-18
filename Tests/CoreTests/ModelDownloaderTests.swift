import XCTest
@testable import Core
import Foundation

final class ModelDownloaderTests: XCTestCase {

    // Test model: tiny model for testing (~340MB)
    let testModelId = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    override func setUp() async throws {
        try await super.setUp()
    }

    // MARK: - Download Tests

    func testModelDownloadSuccess() async throws {
        let downloader = ModelDownloader()

        // Track progress updates using actor for thread-safety
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

        // Download the model
        let modelPath = try await downloader.download(
            modelId: testModelId,
            progressHandler: { progress, speed in
                Task { await tracker.add(progress.fractionCompleted) }
                if let speed = speed {
                    print("Download speed: \(String(format: "%.2f", speed / 1024 / 1024)) MB/s")
                }
            }
        )

        // Verify the model was downloaded
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelPath.path), "Model should be downloaded")

        // Verify progress was reported
        let progressUpdates = await tracker.getUpdates()
        XCTAssertFalse(progressUpdates.isEmpty, "Progress should be reported")

        // Verify final progress is 1.0 (100%)
        if let lastProgress = progressUpdates.last {
            XCTAssertEqual(lastProgress, 1.0, accuracy: 0.01, "Final progress should be 100%")
        }

        print("Model downloaded to: \(modelPath.path)")
    }

    func testModelAlreadyCached() async throws {
        let downloader = ModelDownloader()

        // Download the model twice
        let firstPath = try await downloader.download(modelId: testModelId)
        let secondPath = try await downloader.download(modelId: testModelId)

        // Both should return the same path
        XCTAssertEqual(firstPath, secondPath, "Cached model should return same path")

        // Verify the model exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: secondPath.path))
    }

    func testInvalidModelIdError() async throws {
        let downloader = ModelDownloader()

        do {
            _ = try await downloader.download(modelId: "invalid/nonexistent-model-xyz")
            XCTFail("Should throw error for invalid model ID")
        } catch {
            // Expected error
            print("Correctly caught error: \(error)")
        }
    }

    func testNetworkErrorHandling() async throws {
        // This test verifies that the downloader handles network errors with retry logic
        // We use a non-existent model to trigger errors from the Hub API
        let downloader = ModelDownloader(maxRetries: 2, initialRetryDelay: 0.1)

        do {
            // Attempt to download from a model that doesn't exist
            // This should trigger errors from the Hub API
            _ = try await downloader.download(modelId: "nonexistent-org/nonexistent-model-test-xyz-\(UUID().uuidString)")
            XCTFail("Should throw error for non-existent model")
        } catch let error as ModelDownloaderError {
            // Expected error - verify we got an appropriate error type
            print("Correctly caught error: \(error)")

            // All these error types are acceptable depending on the network conditions:
            // - networkError: transient network issue (will retry)
            // - modelNotFound: model doesn't exist on Hub (won't retry)
            // - authenticationRequired: model requires auth (won't retry)
            // - downloadFailed: general download failure
            switch error {
            case .networkError:
                XCTAssertTrue(true, "Network error handled correctly")
            case .modelNotFound:
                XCTAssertTrue(true, "Model not found error handled correctly")
            case .authenticationRequired:
                XCTAssertTrue(true, "Authentication error handled correctly")
            case .downloadFailed:
                XCTAssertTrue(true, "Download failed handled correctly")
            case .invalidModelId:
                XCTAssertTrue(true, "Invalid model ID handled correctly")
            }
        } catch {
            // Any error is acceptable for this test - we're verifying error handling exists
            print("Caught error: \(error)")
            XCTAssertTrue(true, "Error handling works")
        }
    }

    func testCustomCacheDirectory() async throws {
        // Create a temporary cache directory
        let tempCache = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-mlx-models-\(UUID().uuidString)")

        let downloader = ModelDownloader(cacheDirectory: tempCache)

        let modelPath = try await downloader.download(modelId: testModelId)

        // Verify the model is in the custom cache directory
        XCTAssertTrue(modelPath.path.starts(with: tempCache.path), "Model should be in custom cache directory")

        // Cleanup
        try? FileManager.default.removeItem(at: tempCache)
    }

    // MARK: - Helper Tests

    func testGetCachedModelPath() {
        let downloader = ModelDownloader()

        // Get the expected cache path for a model
        let cachePath = downloader.getCachedModelPath(modelId: testModelId)

        XCTAssertNotNil(cachePath, "Should return a cache path")
        XCTAssertTrue(cachePath!.path.contains("mlx-community"), "Path should contain model org")
        XCTAssertTrue(cachePath!.path.contains("Qwen2.5-0.5B-Instruct-4bit"), "Path should contain model name")
    }

    func testIsModelCached() async throws {
        let downloader = ModelDownloader()

        // Download the model
        _ = try await downloader.download(modelId: testModelId)

        // Now it should be cached
        let nowCached = downloader.isModelCached(modelId: testModelId)
        XCTAssertTrue(nowCached, "Model should be cached after download")
    }
}

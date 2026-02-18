import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Logging

/// Handles loading and initialization of MLX models from disk or Hugging Face
/// Currently uses mlx-swift-lm (Option 1) - will transition to custom implementation (Option 2) later
public actor ModelLoader {
    private let logger = Logger(label: "model-loader")
    private let downloader: ModelDownloader

    public init(downloader: ModelDownloader? = nil) {
        self.downloader = downloader ?? ModelDownloader()
    }

    /// Load a model from Hugging Face Hub or local path
    /// - Parameters:
    ///   - modelPath: HF model ID (e.g., "mlx-community/Llama-3.2-1B-Instruct-4bit") or local path
    ///   - progressHandler: Optional callback for download progress (if model needs to be downloaded)
    /// - Returns: Loaded model container
    public func load(
        modelPath: String,
        progressHandler: (@Sendable (Progress, Double?) -> Void)? = nil
    ) async throws -> ModelContainer {
        logger.info("Loading model from: \(modelPath)")

        let startTime = Date()

        // Load using mlx-swift-lm's loadModelContainer function
        // Note: loadModelContainer internally handles downloading via Hub API
        logger.info("Initializing model with mlx-swift-lm...")
        let modelContainer = try await MLXLMCommon.loadModelContainer(
            id: modelPath,
            progressHandler: { progress in
                // Convert Progress to our format with optional speed
                progressHandler?(progress, nil)
            }
        )

        let loadTime = Date().timeIntervalSince(startTime)
        let configuration = await modelContainer.configuration
        logger.info("Model loaded successfully", metadata: [
            "load_time_seconds": "\(String(format: "%.2f", loadTime))",
            "model_id": "\(configuration.id)"
        ])

        // Log memory usage
        logMemoryUsage()

        return modelContainer
    }

    /// Download a model without loading it (useful for pre-caching)
    /// - Parameters:
    ///   - modelId: HF model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ///   - progressHandler: Optional callback for download progress (progress, speed in bytes/sec)
    /// - Returns: URL to the downloaded model directory
    public func download(
        modelId: String,
        progressHandler: (@Sendable (Progress, Double?) -> Void)? = nil
    ) async throws -> URL {
        let localDownloader = downloader
        return try await localDownloader.download(modelId: modelId, progressHandler: progressHandler)
    }

    /// Check if a model is already cached locally
    /// - Parameter modelId: HF model ID
    /// - Returns: true if the model is cached
    public func isModelCached(modelId: String) -> Bool {
        return downloader.isModelCached(modelId: modelId)
    }

    /// Log current memory usage
    private func logMemoryUsage() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) { infoPtr in
                withUnsafeMutablePointer(to: &count) {
                    $0.withMemoryRebound(to: mach_msg_type_number_t.self, capacity: 1) { countPtr in
                        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), infoPtr, countPtr)
                    }
                }
            }
        }

        if kerr == KERN_SUCCESS {
            let memoryMB = Double(info.resident_size) / 1024.0 / 1024.0
            logger.info("Memory usage", metadata: [
                "resident_mb": "\(String(format: "%.2f", memoryMB))"
            ])
        }
    }
}

// MARK: - Errors

public enum ModelLoaderError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidConfiguration(String)
    case loadFailed(String)
    case unsupportedFormat(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path): return "Model not found at: \(path)"
        case .invalidConfiguration(let msg): return "Invalid configuration: \(msg)"
        case .loadFailed(let reason): return "Failed to load model: \(reason)"
        case .unsupportedFormat(let format): return "Unsupported model format: \(format)"
        }
    }
}

// MARK: - Transition Notes for Option 2

/*
 When transitioning to Option 2 (custom implementation), replace the following:

 1. ModelLoader.load():
    - Implement custom Hugging Face Hub client
    - Implement safetensors memory mapping
    - Implement shard-by-shard loading
    - Implement model architecture initialization

 2. ModelContainer:
    - Replace MLXLMCommon's ModelContainer with custom model protocol
    - Implement custom forward pass
    - Implement custom tokenization

 3. Memory Management:
    - Add PagedAttention integration
    - Add explicit memory mapping control
    - Add shard loading policies

 The current API surface (ModelLoader.load) should remain stable,
 making the transition transparent to the rest of the codebase.
 */

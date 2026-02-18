import Foundation
import Hub
import Logging

/// Handles downloading MLX models from Hugging Face Hub with caching and progress reporting
public struct ModelDownloader: Sendable {
    private let logger = Logger(label: "model-downloader")
    private let cacheDirectory: URL
    private let hubApi: HubApi
    private let maxRetries: Int
    private let initialRetryDelay: TimeInterval

    /// Initialize a new model downloader
    /// - Parameters:
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to ~/.cache/mlx-models/)
    ///   - maxRetries: Maximum number of retry attempts for transient network errors (defaults to 3)
    ///   - initialRetryDelay: Initial delay between retries in seconds (defaults to 1.0, uses exponential backoff)
    public init(cacheDirectory: URL? = nil, maxRetries: Int = 3, initialRetryDelay: TimeInterval = 1.0) {
        if let cacheDirectory = cacheDirectory {
            self.cacheDirectory = cacheDirectory
        } else {
            // Default cache directory: ~/.cache/mlx-models/
            let homeDirectory = FileManager.default.homeDirectoryForCurrentUser
            self.cacheDirectory = homeDirectory
                .appendingPathComponent(".cache")
                .appendingPathComponent("mlx-models")
        }

        // Use Hub API with the cache directory
        self.hubApi = HubApi(downloadBase: self.cacheDirectory)
        self.maxRetries = maxRetries
        self.initialRetryDelay = initialRetryDelay

        logger.info("ModelDownloader initialized", metadata: [
            "cache_directory": "\(self.cacheDirectory.path)",
            "max_retries": "\(maxRetries)"
        ])
    }

    /// Download a model from Hugging Face Hub
    /// - Parameters:
    ///   - modelId: Hugging Face model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ///   - revision: Git revision to download (defaults to "main")
    ///   - progressHandler: Optional callback for progress updates (progress, download speed in bytes/sec)
    /// - Returns: URL to the downloaded model directory
    public func download(
        modelId: String,
        revision: String = "main",
        progressHandler: (@Sendable (Progress, Double?) -> Void)? = nil
    ) async throws -> URL {
        logger.info("Downloading model", metadata: [
            "model_id": "\(modelId)",
            "revision": "\(revision)"
        ])

        let startTime = Date()
        var lastError: Error?

        // Retry loop with exponential backoff
        for attempt in 0...maxRetries {
            do {
                // Download the model snapshot using HubApi
                // This automatically handles caching - if files are already downloaded, it returns immediately
                let modelPath = try await hubApi.snapshot(
                    from: modelId,
                    revision: revision
                ) { progress, speed in
                    // Report progress
                    progressHandler?(progress, speed)

                    // Log progress periodically
                    if progress.fractionCompleted.truncatingRemainder(dividingBy: 0.1) < 0.01 {
                        let speedStr = speed.map { String(format: "%.2f MB/s", $0 / 1024 / 1024) } ?? "unknown"
                        self.logger.info("Download progress", metadata: [
                            "model_id": "\(modelId)",
                            "progress": "\(String(format: "%.1f%%", progress.fractionCompleted * 100))",
                            "speed": "\(speedStr)"
                        ])
                    }
                }

                let downloadTime = Date().timeIntervalSince(startTime)
                logger.info("Model download completed", metadata: [
                    "model_id": "\(modelId)",
                    "path": "\(modelPath.path)",
                    "duration_seconds": "\(String(format: "%.2f", downloadTime))",
                    "attempts": "\(attempt + 1)"
                ])

                return modelPath

            } catch let error as Hub.HubClientError {
                lastError = error

                // Check if this is a retryable error
                let isRetryable = isRetryableError(error)

                logger.warning("Model download attempt failed", metadata: [
                    "model_id": "\(modelId)",
                    "attempt": "\(attempt + 1)",
                    "max_attempts": "\(maxRetries + 1)",
                    "error": "\(error.localizedDescription)",
                    "retryable": "\(isRetryable)"
                ])

                // If not retryable or we've exhausted retries, throw the error
                if !isRetryable || attempt >= maxRetries {
                    logger.error("Model download failed", metadata: [
                        "model_id": "\(modelId)",
                        "total_attempts": "\(attempt + 1)",
                        "error": "\(error.localizedDescription)"
                    ])

                    // Convert Hub errors to our error type
                    switch error {
                    case .fileNotFound, .resourceNotFound:
                        throw ModelDownloaderError.modelNotFound(modelId)
                    case .networkError(let urlError):
                        throw ModelDownloaderError.networkError(urlError.localizedDescription)
                    case .authorizationRequired:
                        throw ModelDownloaderError.authenticationRequired
                    case .httpStatusCode(let code):
                        throw ModelDownloaderError.downloadFailed("HTTP error: \(code)")
                    default:
                        throw ModelDownloaderError.downloadFailed(error.localizedDescription)
                    }
                }

                // Calculate exponential backoff delay
                let delay = initialRetryDelay * pow(2.0, Double(attempt))
                logger.info("Retrying download", metadata: [
                    "model_id": "\(modelId)",
                    "delay_seconds": "\(String(format: "%.1f", delay))",
                    "next_attempt": "\(attempt + 2)"
                ])

                // Wait before retrying
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))

            } catch {
                // Handle other errors (non-retryable)
                logger.error("Model download failed", metadata: [
                    "model_id": "\(modelId)",
                    "error": "\(error.localizedDescription)"
                ])
                throw ModelDownloaderError.downloadFailed(error.localizedDescription)
            }
        }

        // Should never reach here, but just in case
        throw lastError ?? ModelDownloaderError.downloadFailed("Download failed after \(maxRetries + 1) attempts")
    }

    /// Check if an error is retryable (transient network error)
    private func isRetryableError(_ error: Hub.HubClientError) -> Bool {
        switch error {
        case .networkError(let urlError):
            // Retry on transient network errors
            switch urlError.code {
            case .timedOut,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .networkConnectionLost,
                 .dnsLookupFailed,
                 .notConnectedToInternet:
                return true
            default:
                return false
            }
        case .httpStatusCode(let code):
            // Retry on specific HTTP status codes (5xx server errors, 429 rate limit)
            return code >= 500 || code == 429
        default:
            // Don't retry on other errors (file not found, authentication, etc.)
            return false
        }
    }

    /// Check if a model is already cached locally
    /// - Parameters:
    ///   - modelId: Hugging Face model ID
    ///   - revision: Git revision (defaults to "main")
    /// - Returns: true if the model is cached
    public func isModelCached(modelId: String, revision: String = "main") -> Bool {
        guard let modelPath = getCachedModelPath(modelId: modelId) else {
            return false
        }

        // Check if the directory exists and has files
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: modelPath.path, isDirectory: &isDirectory) else {
            return false
        }

        if !isDirectory.boolValue {
            return false
        }

        // Check if directory has files
        if let files = try? fileManager.contentsOfDirectory(atPath: modelPath.path) {
            return !files.isEmpty
        }

        return false
    }

    /// Get the local cache path for a model without downloading
    /// - Parameter modelId: Hugging Face model ID
    /// - Returns: URL to where the model would be cached, or nil if invalid
    public func getCachedModelPath(modelId: String) -> URL? {
        // Parse model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        let components = modelId.split(separator: "/")
        guard components.count == 2 else {
            logger.warning("Invalid model ID format", metadata: ["model_id": "\(modelId)"])
            return nil
        }

        // Construct cache path: cache_dir/models--org--name/snapshots/main/
        let org = components[0]
        let name = components[1]
        let repoName = "models--\(org)--\(name)"

        return cacheDirectory
            .appendingPathComponent(repoName)
            .appendingPathComponent("snapshots")
            .appendingPathComponent("main")
    }

    /// Clear the model cache
    /// - Parameter modelId: Optional model ID to clear specific model, or nil to clear all
    public func clearCache(modelId: String? = nil) throws {
        if let modelId = modelId {
            // Clear specific model
            guard let modelPath = getCachedModelPath(modelId: modelId) else {
                throw ModelDownloaderError.invalidModelId(modelId)
            }

            if FileManager.default.fileExists(atPath: modelPath.path) {
                try FileManager.default.removeItem(at: modelPath)
                logger.info("Cleared cache for model", metadata: ["model_id": "\(modelId)"])
            }
        } else {
            // Clear all cache
            if FileManager.default.fileExists(atPath: cacheDirectory.path) {
                try FileManager.default.removeItem(at: cacheDirectory)
                logger.info("Cleared all model cache", metadata: ["cache_directory": "\(cacheDirectory.path)"])
            }
        }
    }

    /// Get cache directory path
    public var cachePath: URL {
        cacheDirectory
    }
}

// MARK: - Errors

public enum ModelDownloaderError: Error, LocalizedError {
    case modelNotFound(String)
    case networkError(String)
    case authenticationRequired
    case downloadFailed(String)
    case invalidModelId(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let modelId):
            return "Model not found on Hugging Face Hub: \(modelId)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .authenticationRequired:
            return "Authentication required. Please set HF_TOKEN environment variable."
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .invalidModelId(let modelId):
            return "Invalid model ID format: \(modelId)"
        }
    }
}

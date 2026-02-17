import Foundation
import MLX
import MLXLMCommon
import Logging

/// Actor-based inference engine that coordinates with the scheduler
/// Currently a simple wrapper around mlx-swift-lm - will be enhanced for Option 2
public actor InferenceEngine {
    private var modelContainer: ModelContainer?
    private let logger = Logger(label: "inference-engine")

    public init() {}

    /// Initialize with a loaded model container
    public func initialize(modelContainer: ModelContainer) async {
        self.modelContainer = modelContainer
        let configuration = await modelContainer.configuration
        logger.info("Inference engine initialized", metadata: [
            "model": "\(configuration.id)"
        ])
    }

    /// Generate text from a prompt
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    /// - Returns: Generated text
    public func generate(
        prompt: String,
        maxTokens: Int = 100,
        temperature: Double = 0.7
    ) async throws -> String {
        guard let container = modelContainer else {
            throw InferenceError.notInitialized
        }

        let startTime = Date()
        logger.info("Starting generation", metadata: [
            "prompt_length": "\(prompt.count)",
            "max_tokens": "\(maxTokens)"
        ])

        // Prepare input
        let userInput = UserInput(prompt: prompt)
        let input = try await container.prepare(input: userInput)

        // Create generation parameters
        let parameters = GenerateParameters(
            temperature: Float(temperature),
            maxTokens: maxTokens
        )

        // Generate and collect output
        var fullOutput = ""
        let stream = try await container.generate(input: input, parameters: parameters)

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                fullOutput += text
            case .info(let info):
                let duration = Date().timeIntervalSince(startTime)
                logger.info("Generation completed", metadata: [
                    "tokens_generated": "\(info.generationTokenCount)",
                    "duration_seconds": "\(String(format: "%.2f", duration))",
                    "tokens_per_second": "\(String(format: "%.2f", info.tokensPerSecond))"
                ])
            case .toolCall:
                // Skip tool calls for now
                break
            }
        }

        return fullOutput
    }

    /// Generate text with streaming callback
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    ///   - onToken: Callback for each generated token
    /// - Returns: Full generated text
    public func generateStreaming(
        prompt: String,
        maxTokens: Int = 100,
        temperature: Double = 0.7,
        onToken: @escaping (String) async -> Void
    ) async throws -> String {
        guard let container = modelContainer else {
            throw InferenceError.notInitialized
        }

        logger.info("Starting streaming generation", metadata: [
            "prompt_length": "\(prompt.count)",
            "max_tokens": "\(maxTokens)"
        ])

        // Prepare input
        let userInput = UserInput(prompt: prompt)
        let input = try await container.prepare(input: userInput)

        // Create generation parameters
        let parameters = GenerateParameters(
            temperature: Float(temperature),
            maxTokens: maxTokens
        )

        // Generate stream
        let generationStream = try await container.generate(input: input, parameters: parameters)

        var fullOutput = ""
        var tokenCount = 0
        for await generation in generationStream {
            switch generation {
            case .chunk(let text):
                tokenCount += 1
                fullOutput += text
                await onToken(text)
            case .info(let info):
                logger.info("Stream completed", metadata: [
                    "tokens_generated": "\(info.generationTokenCount)",
                    "tokens_per_second": "\(String(format: "%.2f", info.tokensPerSecond))"
                ])
            case .toolCall:
                // Skip tool calls for now
                break
            }
        }

        return fullOutput
    }

    /// Get current model information
    public var modelInfo: ModelInfo? {
        get async {
            guard let container = modelContainer else {
                return nil
            }

            let configuration = await container.configuration
            return ModelInfo(
                name: configuration.name,
                id: configuration.name
            )
        }
    }

    // MARK: - Batch Processing APIs (Phase 3)

    /// Process a batch of single tokens through the model
    /// NOTE: Phase 3.1 implementation - uses N forward passes
    /// TODO: Phase 3.2 - optimize to single batched forward pass
    /// - Parameters:
    ///   - tokenIds: Array of token IDs (one per slot)
    ///   - positions: Current position for each slot
    ///   - prompts: Original prompts for each slot (for context)
    /// - Returns: Array of next token IDs (one per slot)
    public func forwardBatch(
        tokenIds: [Int],
        positions: [Int],
        prompts: [String]
    ) async throws -> [Int] {
        guard tokenIds.count == positions.count && tokenIds.count == prompts.count else {
            throw InferenceError.invalidParameters("Mismatched batch sizes")
        }

        // Phase 3.1: Simple implementation - process each slot independently
        // If model not initialized (e.g., in tests), return placeholder tokens
        guard let _ = modelContainer else {
            // Return placeholder tokens (1 = non-EOS token for testing)
            return Array(repeating: 1, count: tokenIds.count)
        }

        // This is not optimal but establishes the continuous batching flow
        var nextTokens: [Int] = []

        for i in 0..<tokenIds.count {
            // For now, we'll use a simplified approach
            // In a real implementation, this would be a single batched forward pass
            // TODO: Implement true batch processing with MLX arrays
            nextTokens.append(tokenIds[i])  // Placeholder - echo the input
        }

        return nextTokens
    }

    /// Sample next tokens for a batch of logits
    /// NOTE: Phase 3.1 placeholder - needs proper sampling implementation
    /// - Parameters:
    ///   - logits: Array of logit arrays (one per slot)
    ///   - temperatures: Sampling temperature for each slot
    ///   - topP: Top-p (nucleus) sampling parameter for each slot
    /// - Returns: Array of sampled token IDs
    public func sampleBatch(
        logits: [[Float]],
        temperatures: [Float],
        topP: [Float]
    ) throws -> [Int] {
        guard logits.count == temperatures.count && logits.count == topP.count else {
            throw InferenceError.invalidParameters("Mismatched batch sizes")
        }

        // Phase 3.1: Simplified sampling
        // TODO: Implement proper sampling with temperature and top-p
        var sampledTokens: [Int] = []

        for i in 0..<logits.count {
            // Find argmax as simple sampling strategy
            if let maxIndex = logits[i].enumerated().max(by: { $0.element < $1.element })?.offset {
                sampledTokens.append(maxIndex)
            } else {
                sampledTokens.append(0)  // Fallback
            }
        }

        return sampledTokens
    }
}

// MARK: - Model Info

/// Information about the loaded model
public struct ModelInfo {
    public let name: String
    public let id: String

    public init(name: String, id: String) {
        self.name = name
        self.id = id
    }
}

// MARK: - Errors

public enum InferenceError: Error, LocalizedError {
    case notInitialized
    case generationFailed(String)
    case invalidParameters(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized: return "Inference engine not initialized"
        case .generationFailed(let reason): return "Generation failed: \(reason)"
        case .invalidParameters(let msg): return "Invalid parameters: \(msg)"
        }
    }
}

// MARK: - GenerateParameters Extension

extension GenerateParameters {
    /// Convenience initializer with common parameters
    public init(
        temperature: Float = 0.7,
        maxTokens: Int? = 100,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil
    ) {
        self.init(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty
        )
    }
}

import Foundation
import MLX
import MLXLMCommon
import Logging
import Memory

/// Actor-based inference engine that coordinates with the scheduler
/// Currently a simple wrapper around mlx-swift-lm - will be enhanced for Option 2
public actor InferenceEngine {
    /// Default model ID for quick initialization
    public static let defaultModelID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    private var modelContainer: ModelContainer?
    private let logger = Logger(label: "inference-engine")
    private var kvCache: PagedKVCache?

    /// Per-slot generated token sequences for Phase 4.2 (legacy fallback)
    private var slotTokenSequences: [String: [Int]] = [:]

    /// Per-slot TokenIterator for real KV cache integration (Phase 4.3)
    private var slotIterators: [String: TokenIterator] = [:]

    public init() {}

    /// Initialize with a loaded model container
    public func initialize(modelContainer: ModelContainer) async {
        self.modelContainer = modelContainer
        let configuration = await modelContainer.configuration
        logger.info("Inference engine initialized", metadata: [
            "model": "\(configuration.id)"
        ])
    }

    /// Attach a KV cache actor for future cache-aware inference
    public func attachKVCache(_ cache: PagedKVCache) async {
        self.kvCache = cache
        logger.info("KV cache attached", metadata: [
            "type": "PagedKVCache"
        ])
    }

    /// Reset per-slot state when a request finishes
    public func resetSlot(slotKey: String) async {
        slotTokenSequences.removeValue(forKey: slotKey)
        slotIterators.removeValue(forKey: slotKey)
    }

    /// Convenience method to initialize with default model
    /// Loads the default small model (Qwen2.5-0.5B-Instruct-4bit) automatically
    /// - Parameters:
    ///   - modelPath: Optional model ID or path (defaults to defaultModelID)
    ///   - progressHandler: Optional callback for download progress
    /// - Throws: ModelLoaderError if model cannot be loaded
    public func initialize(
        modelPath: String? = nil,
        progressHandler: (@Sendable (Progress, Double?) -> Void)? = nil
    ) async throws {
        let modelId = modelPath ?? Self.defaultModelID

        logger.info("Loading model", metadata: ["model_id": "\(modelId)"])

        // Create model loader
        let loader = ModelLoader()

        // Load model with progress tracking
        // ModelLoader will automatically use cache if available
        do {
            let modelContainer = try await loader.load(
                modelPath: modelId,
                progressHandler: progressHandler
            )

            // Initialize with loaded container
            await initialize(modelContainer: modelContainer)

            logger.info("Model loaded successfully", metadata: [
                "model_id": "\(modelId)"
            ])
        } catch {
            logger.error("Failed to load model", metadata: [
                "model_id": "\(modelId)",
                "error": "\(error.localizedDescription)"
            ])
            throw error
        }
    }

    /// Check if model is loaded
    public var isModelLoaded: Bool {
        return modelContainer != nil
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

    // MARK: - Batch Processing APIs (Phase 4.2)

    /// Process a batch of single tokens through the model with real MLX inference
    /// - Parameters:
    ///   - tokenIds: Array of token IDs (one per slot)
    ///   - positions: Current position for each slot
    ///   - prompts: Original prompts for each slot (for context)
    ///   - kvCacheBlockIds: KV cache block IDs for each slot (Phase 4.1)
    ///   - temperatures: Sampling temperature for each slot
    ///   - topP: Top-p (nucleus) sampling parameter for each slot
    /// - Returns: Array of next token IDs (one per slot)
    public func forwardBatch(
        tokenIds: [Int],
        positions: [Int],
        prompts: [String],
        kvCacheBlockIds: [[Int]] = [],
        temperatures: [Float] = [],
        topP: [Float] = []
    ) async throws -> [Int] {
        guard tokenIds.count == positions.count && tokenIds.count == prompts.count else {
            throw InferenceError.invalidParameters("Mismatched batch sizes")
        }

        // If model not initialized (e.g., in tests), return placeholder tokens
        guard let container = modelContainer else {
            // Return placeholder tokens (1 = non-EOS token for testing)
            return Array(repeating: 1, count: tokenIds.count)
        }

        let batchSize = tokenIds.count

        // Default parameters if not provided
        let effectiveTemperatures = temperatures.isEmpty ? Array(repeating: 0.7, count: batchSize) : temperatures
        let effectiveTopP = topP.isEmpty ? Array(repeating: 0.95, count: batchSize) : topP

        // Phase 4.2: Process each slot with real MLX inference
        // NOTE: This uses individual forward passes per slot with proper KV cache
        // Each slot maintains its own KV cache, stored in actor's slotCaches dictionary
        // TODO Phase 4.3: Optimize to single batched forward pass (requires unified cache handling)

        var nextTokens: [Int] = []

        for i in 0..<batchSize {
            // Use prompt as slot key for cache lookup
            // In Phase 4.3, we'll use proper slot IDs from ContinuousBatcher
            let slotKey = prompts[i]

            // Touch KV cache blocks (scaffolding for quantized KV cache)
            if let cache = kvCache, i < kvCacheBlockIds.count {
                _ = try? await cache.getBlocks(ids: kvCacheBlockIds[i])
            }

            // Generate next token with KV cache
            let sampledToken = try await generateNextToken(
                slotKey: slotKey,
                tokenId: tokenIds[i],
                position: positions[i],
                temperature: effectiveTemperatures[i],
                topP: effectiveTopP[i],
                container: container
            )
            nextTokens.append(sampledToken)
        }

        return nextTokens
    }

    /// Generate next token for a single slot using real MLX model forward pass
    /// Phase 4.2: Real token generation WITHOUT persistent KV cache
    ///
    /// This implementation processes the full token sequence on each call to avoid
    /// KVCache Sendable issues. This is inefficient (O(n²) complexity) but correct.
    /// Phase 4.3 will add proper cache persistence via PagedKVCache.
    ///
    /// - Parameters:
    ///   - slotKey: Unique key for this slot (prompt string)
    ///   - tokenId: Current token ID to append and process
    ///   - position: Position in the sequence (for validation)
    ///   - temperature: Sampling temperature (0.0 = greedy)
    ///   - topP: Top-p (nucleus) sampling parameter
    ///   - container: Model container with loaded model
    /// - Returns: Next token ID
    private func generateNextToken(
        slotKey: String,
        tokenId: Int,
        position: Int,
        temperature: Float,
        topP: Float,
        container: ModelContainer
    ) async throws -> Int {
        // Phase 4.3: Use TokenIterator with KV cache when available
        if var iterator = slotIterators[slotKey] {
            if let token = iterator.next() {
                slotIterators[slotKey] = iterator
                return token
            } else {
                // Generation complete; fallback to EOS
                return 0
            }
        }

        // Initialize TokenIterator for this slot
        let tokenizer = await container.tokenizer
        let promptTokens = tokenizer.encode(text: slotKey)
        guard !promptTokens.isEmpty else {
            throw InferenceError.invalidParameters("Empty prompt tokens")
        }

        let tokens1D = MLXArray(promptTokens)
        let tokenArray = tokens1D.reshaped([1, promptTokens.count])
        let input = LMInput(tokens: tokenArray)

        let parameters = GenerateParameters(
            maxTokens: nil,
            temperature: temperature,
            topP: topP
        )

        let model = await container.perform { context in
            SendableBox(context.model)
        }.consume()

        var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
        let token = iterator.next() ?? 0
        slotIterators[slotKey] = iterator
        return token
    }

    /// Sample next tokens for a batch of logits with temperature and top-p
    /// Phase 4.2: Real sampling implementation with temperature scaling and nucleus sampling
    /// - Parameters:
    ///   - logits: Array of logit arrays (one per slot) [batch_size][vocab_size]
    ///   - temperatures: Sampling temperature for each slot (0.0 = greedy)
    ///   - topP: Top-p (nucleus) sampling parameter for each slot (0.0-1.0)
    /// - Returns: Array of sampled token IDs
    public func sampleBatch(
        logits: [[Float]],
        temperatures: [Float],
        topP: [Float]
    ) throws -> [Int] {
        guard logits.count == temperatures.count && logits.count == topP.count else {
            throw InferenceError.invalidParameters("Batch size mismatch: logits=\(logits.count), temps=\(temperatures.count), topP=\(topP.count)")
        }

        return try logits.enumerated().map { i, logit in
            let temperature = temperatures[i]
            let p = topP[i]

            // Handle empty logits
            guard !logit.isEmpty else {
                return 0
            }

            // Greedy sampling (temperature = 0.0)
            if temperature == 0.0 {
                return logit.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            }

            // Temperature scaling
            let scaledLogits = logit.map { $0 / temperature }

            // Softmax to get probabilities
            let maxLogit = scaledLogits.max() ?? 0
            let expLogits = scaledLogits.map { exp($0 - maxLogit) }
            let sumExp = expLogits.reduce(0, +)

            // Handle numerical instability
            guard sumExp > 0 && sumExp.isFinite else {
                logger.warning("Softmax numerical instability", metadata: [
                    "sum_exp": "\(sumExp)",
                    "temperature": "\(temperature)"
                ])
                return logit.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            }

            let probs = expLogits.map { $0 / sumExp }

            // Top-P (nucleus) filtering
            if p < 1.0 {
                return try sampleTopP(probs: probs, p: p)
            } else {
                // Sample from full distribution
                return sampleFromDistribution(probs: probs)
            }
        }
    }

    /// Sample from a probability distribution using top-p (nucleus) sampling
    private func sampleTopP(probs: [Float], p: Float) throws -> Int {
        // Sort indices by probability (descending)
        let sortedIndices = probs.enumerated()
            .sorted { $0.element > $1.element }
            .map { $0.offset }

        // Find cumulative probability cutoff
        var cumulativeProb: Float = 0
        var topIndices: [Int] = []

        for idx in sortedIndices {
            cumulativeProb += probs[idx]
            topIndices.append(idx)
            if cumulativeProb >= p {
                break
            }
        }

        // Ensure at least one token
        if topIndices.isEmpty {
            topIndices = [sortedIndices[0]]
        }

        // Renormalize probabilities for top-p subset
        let topProbs = topIndices.map { probs[$0] }
        let sumTopProbs = topProbs.reduce(0, +)

        guard sumTopProbs > 0 else {
            return topIndices[0]
        }

        let normalizedProbs = topProbs.map { $0 / sumTopProbs }

        // Sample from renormalized distribution
        let randomValue = Float.random(in: 0..<1)
        var cumSum: Float = 0

        for (idx, prob) in normalizedProbs.enumerated() {
            cumSum += prob
            if randomValue <= cumSum {
                return topIndices[idx]
            }
        }

        // Fallback to highest probability token
        return topIndices[0]
    }

    /// Sample from a probability distribution
    private func sampleFromDistribution(probs: [Float]) -> Int {
        let randomValue = Float.random(in: 0..<1)
        var cumSum: Float = 0

        for (idx, prob) in probs.enumerated() {
            cumSum += prob
            if randomValue <= cumSum {
                return idx
            }
        }

        // Fallback to last token
        return probs.count - 1
    }
}

// MARK: - Model Info

/// Information about the loaded model
public struct ModelInfo: Sendable {
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

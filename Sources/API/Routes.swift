import Vapor
import Core
import Scheduler
import Authentication

/// Configure API routes
public func routes(_ app: Application, scheduler: RequestScheduler? = nil, engine: InferenceEngine? = nil, batcher: ContinuousBatcher? = nil, apiKeyStore: APIKeyStore? = nil, rateLimiter: RateLimiter? = nil) throws {
    // Add Request ID middleware (Phase 5.4)
    app.middleware.use(RequestIDMiddleware())

    // Health check endpoint (Phase 5.4)
    app.get("health") { req async -> HealthResponse in
        let modelLoaded = await engine?.isModelLoaded ?? false
        let batcherRunning = await batcher?.running ?? false

        return HealthResponse(
            status: modelLoaded && batcherRunning ? "healthy" : "degraded",
            model: modelLoaded ? "loaded" : "not_loaded",
            batcher: batcherRunning ? "running" : "stopped",
            timestamp: Date()
        )
    }

    // Readiness check endpoint
    app.get("ready") { req async -> Response in
        // Check if model is loaded and batcher is running
        let modelLoaded = await engine?.isModelLoaded ?? false
        let batcherRunning = await batcher?.running ?? false
        let ready = modelLoaded && batcherRunning

        return Response(status: ready ? .ok : .serviceUnavailable, body: .init(string: ready ? "Ready" : "Not Ready"))
    }

    // Metrics endpoint (Phase 5.4)
    app.get("metrics") { req async throws -> MetricsResponse in
        guard let scheduler = scheduler, let batcher = batcher else {
            throw Abort(.internalServerError, reason: "Scheduler or batcher not initialized")
        }

        let schedulerStats = await scheduler.stats
        let batcherStats = await batcher.getStats()
        let gpuStats = await batcher.getGPUStats()

        return MetricsResponse(
            requests: RequestMetrics(
                pending: schedulerStats.currentPending,
                active: schedulerStats.currentActive,
                completed: schedulerStats.totalCompleted,
                failed: schedulerStats.totalFailed,
                cancelled: schedulerStats.totalCancelled
            ),
            batcher: BatcherMetrics(
                activeSlots: batcherStats.activeSlots,
                totalSlots: batcherStats.totalSlots,
                utilization: batcherStats.utilization,
                stepCount: batcherStats.stepCount
            ),
            gpu: GPUMetrics(
                averageUtilization: gpuStats.averageUtilization,
                currentUtilization: gpuStats.currentUtilization,
                sampleCount: gpuStats.sampleCount
            )
        )
    }

    // Phase 6.1: Protected routes (require authentication)
    let protected: RoutesBuilder
    if let keyStore = apiKeyStore, let limiter = rateLimiter {
        // Authentication enabled: protect v1 API endpoints
        let authMiddleware = APIKeyMiddleware(keyStore: keyStore, rateLimiter: limiter)
        protected = app.grouped(authMiddleware)
    } else {
        // No authentication: use app directly (for backward compatibility/testing)
        protected = app
    }

    // Cancel request endpoint (Phase 3.2)
    protected.delete("v1", "requests", ":id") { req async throws -> HTTPStatus in
        guard let requestIdString = req.parameters.get("id"),
              let requestId = UUID(uuidString: requestIdString) else {
            throw Abort(.badRequest, reason: "Invalid request ID format")
        }

        guard let scheduler = scheduler else {
            throw Abort(.internalServerError, reason: "Scheduler not initialized")
        }

        await scheduler.cancel(requestId: requestId)
        return .noContent
    }

    // OpenAI-compatible completions endpoint (Phase 5.1 + 5.2)
    protected.post("v1", "completions") { req async throws -> Response in
        let request = try req.content.decode(CompletionRequest.self)

        // Validate scheduler is available
        guard let scheduler = scheduler else {
            throw Abort(.internalServerError, reason: "Scheduler not initialized")
        }

        // Validate request
        guard !request.prompt.isEmpty else {
            throw Abort(.badRequest, reason: "Prompt cannot be empty")
        }

        // Create inference request
        let inferenceRequest = InferenceRequest(
            prompt: request.prompt,
            maxTokens: request.maxTokens ?? 100,
            temperature: request.temperature ?? 0.7
        )

        // Submit to scheduler
        let (requestId, stream) = await scheduler.submit(
            inferenceRequest,
            priority: .normal
        )

        // Check if streaming is requested
        if request.stream == true {
            // SSE Streaming mode (Phase 5.2)
            return Response.sse(req) { writer in
                let created = Int(Date().timeIntervalSince1970)

                do {
                    // Stream tokens as they arrive
                    for try await chunk in stream {
                        let streamChunk = CompletionStreamChunk(
                            id: requestId.uuidString,
                            created: created,
                            model: request.model,
                            choices: [
                                CompletionStreamChoice(
                                    text: chunk.token,
                                    index: 0,
                                    finishReason: nil
                                )
                            ]
                        )

                        let json = try JSONEncoder().encode(streamChunk)
                        let jsonString = String(data: json, encoding: .utf8)!
                        try await writer.send(data: jsonString)
                    }

                    // Send final chunk with finish_reason
                    let finalChunk = CompletionStreamChunk(
                        id: requestId.uuidString,
                        created: created,
                        model: request.model,
                        choices: [
                            CompletionStreamChoice(
                                text: "",
                                index: 0,
                                finishReason: "stop"
                            )
                        ]
                    )
                    let finalJson = try JSONEncoder().encode(finalChunk)
                    let finalJsonString = String(data: finalJson, encoding: .utf8)!
                    try await writer.send(data: finalJsonString)

                    // Send [DONE] message
                    try await writer.send(data: "[DONE]")
                } catch {
                    req.logger.error("Streaming error: \(error)")
                    throw error
                }
            }
        } else {
            // Non-streaming mode (Phase 5.1)
            var fullText = ""
            do {
                for try await chunk in stream {
                    fullText += chunk.token
                }
            } catch {
                throw Abort(.internalServerError, reason: "Generation failed: \(error.localizedDescription)")
            }

            // Return OpenAI-compatible response
            let response = CompletionResponse(
                id: requestId.uuidString,
                created: Int(Date().timeIntervalSince1970),
                model: request.model,
                choices: [
                    CompletionChoice(
                        text: fullText,
                        index: 0,
                        finishReason: "stop"
                    )
                ]
            )

            return try await response.encodeResponse(for: req)
        }
    }

    // OpenAI-compatible chat completions endpoint (Phase 5.3)
    protected.post("v1", "chat", "completions") { req async throws -> Response in
        let request = try req.content.decode(ChatCompletionRequest.self)

        // Validate scheduler is available
        guard let scheduler = scheduler else {
            throw Abort(.internalServerError, reason: "Scheduler not initialized")
        }

        // Validate request
        guard !request.messages.isEmpty else {
            throw Abort(.badRequest, reason: "Messages cannot be empty")
        }

        // Convert chat messages to prompt
        let formatter = ChatTemplateFormatter()
        let prompt = formatter.format(messages: request.messages)

        // Create inference request
        let inferenceRequest = InferenceRequest(
            prompt: prompt,
            maxTokens: request.maxTokens ?? 100,
            temperature: request.temperature ?? 0.7
        )

        // Submit to scheduler
        let (requestId, stream) = await scheduler.submit(
            inferenceRequest,
            priority: .normal
        )

        // Check if streaming is requested
        if request.stream == true {
            // SSE Streaming mode
            return Response.sse(req) { writer in
                let created = Int(Date().timeIntervalSince1970)

                do {
                    // Stream tokens as they arrive
                    for try await chunk in stream {
                        let streamChunk = ChatCompletionStreamChunk(
                            id: requestId.uuidString,
                            created: created,
                            model: request.model,
                            choices: [
                                ChatStreamChoice(
                                    delta: ChatMessage(role: "assistant", content: chunk.token),
                                    index: 0,
                                    finishReason: nil
                                )
                            ]
                        )

                        let json = try JSONEncoder().encode(streamChunk)
                        let jsonString = String(data: json, encoding: .utf8)!
                        try await writer.send(data: jsonString)
                    }

                    // Send final chunk with finish_reason
                    let finalChunk = ChatCompletionStreamChunk(
                        id: requestId.uuidString,
                        created: created,
                        model: request.model,
                        choices: [
                            ChatStreamChoice(
                                delta: ChatMessage(role: "assistant", content: ""),
                                index: 0,
                                finishReason: "stop"
                            )
                        ]
                    )
                    let finalJson = try JSONEncoder().encode(finalChunk)
                    let finalJsonString = String(data: finalJson, encoding: .utf8)!
                    try await writer.send(data: finalJsonString)

                    // Send [DONE] message
                    try await writer.send(data: "[DONE]")
                } catch {
                    req.logger.error("Chat streaming error: \(error)")
                    throw error
                }
            }
        } else {
            // Non-streaming mode
            var fullText = ""
            do {
                for try await chunk in stream {
                    fullText += chunk.token
                }
            } catch {
                throw Abort(.internalServerError, reason: "Generation failed: \(error.localizedDescription)")
            }

            // Return OpenAI-compatible response
            let response = ChatCompletionResponse(
                id: requestId.uuidString,
                created: Int(Date().timeIntervalSince1970),
                model: request.model,
                choices: [
                    ChatCompletionChoice(
                        message: ChatMessage(role: "assistant", content: fullText),
                        index: 0,
                        finishReason: "stop"
                    )
                ]
            )

            return try await response.encodeResponse(for: req)
        }
    }
}

// MARK: - Request Models

struct CompletionRequest: Content {
    let model: String
    let prompt: String
    let maxTokens: Int?
    let temperature: Double?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case maxTokens = "max_tokens"
        case temperature
        case stream
    }
}

struct ChatCompletionRequest: Content {
    let model: String
    let messages: [ChatMessage]
    let maxTokens: Int?
    let temperature: Double?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case stream
    }
}

struct ChatMessage: Content {
    let role: String
    let content: String
}

// MARK: - Response Models

struct CompletionResponse: Content {
    let id: String
    let object: String = "text_completion"
    let created: Int
    let model: String
    let choices: [CompletionChoice]
}

struct CompletionChoice: Content {
    let text: String
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text
        case index
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionResponse: Content {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [ChatCompletionChoice]
}

struct ChatCompletionChoice: Content {
    let message: ChatMessage
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case message
        case index
        case finishReason = "finish_reason"
    }
}

// MARK: - Observability Models (Phase 5.4)

struct HealthResponse: Content {
    let status: String
    let model: String
    let batcher: String
    let timestamp: Date
}

struct MetricsResponse: Content {
    let requests: RequestMetrics
    let batcher: BatcherMetrics
    let gpu: GPUMetrics
}

struct RequestMetrics: Content {
    let pending: Int
    let active: Int
    let completed: Int
    let failed: Int
    let cancelled: Int
}

struct BatcherMetrics: Content {
    let activeSlots: Int
    let totalSlots: Int
    let utilization: Double
    let stepCount: Int

    enum CodingKeys: String, CodingKey {
        case activeSlots = "active_slots"
        case totalSlots = "total_slots"
        case utilization
        case stepCount = "step_count"
    }
}

struct GPUMetrics: Content {
    let averageUtilization: Double
    let currentUtilization: Double
    let sampleCount: Int

    enum CodingKeys: String, CodingKey {
        case averageUtilization = "average_utilization"
        case currentUtilization = "current_utilization"
        case sampleCount = "sample_count"
    }
}

// MARK: - Streaming Response Models

struct CompletionStreamChunk: Content {
    let id: String
    let object: String = "text_completion.chunk"
    let created: Int
    let model: String
    let choices: [CompletionStreamChoice]
}

struct CompletionStreamChoice: Content {
    let text: String
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text
        case index
        case finishReason = "finish_reason"
    }
}

struct ChatCompletionStreamChunk: Content {
    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [ChatStreamChoice]
}

struct ChatStreamChoice: Content {
    let delta: ChatMessage
    let index: Int
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case delta
        case index
        case finishReason = "finish_reason"
    }
}

// MARK: - SSE Helper

extension Response {
    /// Create a Server-Sent Events (SSE) response for streaming
    static func sse(
        _ req: Request,
        onStream: @escaping @Sendable (SSEWriter) async throws -> Void
    ) -> Response {
        Response(
            status: .ok,
            headers: HTTPHeaders([
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "keep-alive"),
                ("X-Accel-Buffering", "no")
            ]),
            body: .init(managedAsyncStream: { writer in
                let sseWriter = SSEWriter(writer: writer)
                do {
                    try await onStream(sseWriter)
                } catch {
                    req.logger.error("SSE stream error: \(error)")
                }
            })
        )
    }
}

struct SSEWriter: Sendable {
    let writer: AsyncBodyStreamWriter

    func send(event: String? = nil, data: String) async throws {
        var message = ""
        if let event = event {
            message += "event: \(event)\n"
        }
        message += "data: \(data)\n\n"
        try await writer.write(.buffer(.init(string: message)))
    }
}

// MARK: - Request ID Middleware (Phase 5.4)

struct RequestIDMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        // Get or generate request ID
        let requestID = request.headers.first(name: "X-Request-ID") ?? UUID().uuidString

        // Add to request headers if not present
        if request.headers.first(name: "X-Request-ID") == nil {
            request.headers.add(name: "X-Request-ID", value: requestID)
        }

        // Call next responder
        var response = try await next.respond(to: request)

        // Add to response headers
        response.headers.add(name: "X-Request-ID", value: requestID)

        return response
    }
}

// MARK: - Chat Template Formatter

struct ChatTemplateFormatter {
    /// Convert chat messages to prompt using template format
    /// Format: <|system|>...<|user|>...<|assistant|>
    func format(messages: [ChatMessage]) -> String {
        var prompt = ""

        for message in messages {
            switch message.role {
            case "system":
                prompt += "<|system|>\n\(message.content)\n"
            case "user":
                prompt += "<|user|>\n\(message.content)\n"
            case "assistant":
                prompt += "<|assistant|>\n\(message.content)\n"
            default:
                // Ignore unknown roles
                break
            }
        }

        // Add assistant prefix for generation
        prompt += "<|assistant|>\n"

        return prompt
    }
}

import Vapor
import Core
import Scheduler

/// Configure API routes
public func routes(_ app: Application, scheduler: RequestScheduler? = nil) throws {
    // Health check endpoint
    app.get("health") { req async -> Response in
        return Response(status: .ok, body: .init(string: "OK"))
    }

    // Readiness check endpoint
    app.get("ready") { req async -> Response in
        // TODO: Check if model is loaded and ready
        return Response(status: .ok, body: .init(string: "Ready"))
    }

    // Cancel request endpoint (Phase 3.2)
    app.delete("v1", "requests", ":id") { req async throws -> HTTPStatus in
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

    // OpenAI-compatible completions endpoint
    app.post("v1", "completions") { req async throws -> CompletionResponse in
        let request = try req.content.decode(CompletionRequest.self)

        // TODO: Phase 5 - Implement completion endpoint
        // 1. Validate request
        // 2. Submit to scheduler
        // 3. Wait for completion
        // 4. Return response

        throw Abort(.notImplemented, reason: "Completions endpoint not yet implemented")
    }

    // OpenAI-compatible chat completions endpoint
    app.post("v1", "chat", "completions") { req async throws -> ChatCompletionResponse in
        let request = try req.content.decode(ChatCompletionRequest.self)

        // TODO: Phase 5 - Implement chat completion endpoint
        // 1. Convert chat messages to prompt
        // 2. Submit to scheduler
        // 3. Handle streaming if requested
        // 4. Return response

        throw Abort(.notImplemented, reason: "Chat completions endpoint not yet implemented")
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

import Foundation

// MARK: - Token Chunk

/// A single token chunk in the generation stream
public struct TokenChunk: Sendable {
    public let requestId: UUID
    public let token: String
    public let index: Int
    public let timestamp: Date

    public init(requestId: UUID, token: String, index: Int, timestamp: Date = Date()) {
        self.requestId = requestId
        self.token = token
        self.index = index
        self.timestamp = timestamp
    }
}

// MARK: - Generation Info

/// Final generation statistics
public struct GenerationInfo: Sendable {
    public let requestId: UUID
    public let totalTokens: Int
    public let duration: TimeInterval
    public let tokensPerSecond: Double
    public let finishReason: FinishReason

    public init(
        requestId: UUID,
        totalTokens: Int,
        duration: TimeInterval,
        finishReason: FinishReason
    ) {
        self.requestId = requestId
        self.totalTokens = totalTokens
        self.duration = duration
        self.tokensPerSecond = duration > 0 ? Double(totalTokens) / duration : 0
        self.finishReason = finishReason
    }
}

// MARK: - Token Stream

/// AsyncSequence wrapper for token streaming with backpressure
public struct TokenStream: AsyncSequence, Sendable {
    public typealias Element = TokenChunk

    private let stream: AsyncThrowingStream<TokenChunk, Error>
    private let continuation: AsyncThrowingStream<TokenChunk, Error>.Continuation

    public init() {
        var continuation: AsyncThrowingStream<TokenChunk, Error>.Continuation!
        stream = AsyncThrowingStream { cont in
            continuation = cont
        }
        self.continuation = continuation
    }

    // AsyncSequence conformance
    public func makeAsyncIterator() -> AsyncThrowingStream<TokenChunk, Error>.Iterator {
        stream.makeAsyncIterator()
    }

    // Stream control methods
    internal func yield(_ chunk: TokenChunk) {
        continuation.yield(chunk)
    }

    internal func finish() {
        continuation.finish()
    }

    internal func finish(throwing error: Error) {
        continuation.finish(throwing: error)
    }
}

// MARK: - Stream Errors

/// Errors that can occur during streaming
public enum StreamError: Error, LocalizedError {
    case streamClosed
    case timeout
    case cancelled
    case engineError(String)

    public var errorDescription: String? {
        switch self {
        case .streamClosed:
            return "Token stream was closed"
        case .timeout:
            return "Request timed out"
        case .cancelled:
            return "Request was cancelled"
        case .engineError(let message):
            return "Inference engine error: \(message)"
        }
    }
}

// MARK: - Token Stream Registry

/// Actor that manages active token streams by request ID
public actor TokenStreamRegistry {
    private var streams: [UUID: StreamController] = [:]

    /// Internal controller for a token stream
    private struct StreamController {
        let continuation: AsyncThrowingStream<TokenChunk, Error>.Continuation
        var isClosed: Bool = false
    }

    /// Register a new token stream
    public func register(requestId: UUID) -> TokenStream {
        var continuation: AsyncThrowingStream<TokenChunk, Error>.Continuation!
        let stream = AsyncThrowingStream<TokenChunk, Error> { cont in
            continuation = cont
        }

        streams[requestId] = StreamController(continuation: continuation)

        return TokenStream(stream: stream, continuation: continuation)
    }

    /// Yield a token chunk to the stream
    public func yield(requestId: UUID, chunk: TokenChunk) {
        guard let controller = streams[requestId], !controller.isClosed else {
            return
        }
        controller.continuation.yield(chunk)
    }

    /// Finish the stream successfully
    public func finish(requestId: UUID) {
        guard var controller = streams[requestId], !controller.isClosed else {
            return
        }
        controller.continuation.finish()
        controller.isClosed = true
        streams.removeValue(forKey: requestId)
    }

    /// Finish the stream with an error
    public func finish(requestId: UUID, error: Error) {
        guard var controller = streams[requestId], !controller.isClosed else {
            return
        }
        controller.continuation.finish(throwing: error)
        controller.isClosed = true
        streams.removeValue(forKey: requestId)
    }

    /// Check if a stream is registered
    public func isRegistered(requestId: UUID) -> Bool {
        streams[requestId] != nil
    }

    /// Get count of active streams
    public var activeStreamCount: Int {
        streams.count
    }

    /// Cancel all streams
    public func cancelAll() {
        for (requestId, _) in streams {
            finish(requestId: requestId, error: StreamError.cancelled)
        }
    }
}

// MARK: - TokenStream Extension for Internal Use

extension TokenStream {
    /// Internal initializer with explicit continuation
    fileprivate init(
        stream: AsyncThrowingStream<TokenChunk, Error>,
        continuation: AsyncThrowingStream<TokenChunk, Error>.Continuation
    ) {
        self.stream = stream
        self.continuation = continuation
    }
}

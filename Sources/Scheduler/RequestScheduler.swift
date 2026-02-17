import Foundation
import Core
import Logging

// MARK: - Scheduler Configuration

/// Configuration for the request scheduler
public struct SchedulerConfig: Sendable {
    public let maxConcurrentRequests: Int
    public let requestTimeout: TimeInterval
    public let timeoutCheckInterval: TimeInterval
    public let maxQueueSize: Int

    public init(
        maxConcurrentRequests: Int = 100,
        requestTimeout: TimeInterval = 300.0,  // 5 minutes
        timeoutCheckInterval: TimeInterval = 10.0,  // 10 seconds
        maxQueueSize: Int = 1000
    ) {
        self.maxConcurrentRequests = maxConcurrentRequests
        self.requestTimeout = requestTimeout
        self.timeoutCheckInterval = timeoutCheckInterval
        self.maxQueueSize = maxQueueSize
    }
}

// MARK: - Scheduler Statistics

/// Statistics about scheduler performance
public struct SchedulerStats: Sendable {
    public let totalSubmitted: Int
    public let totalCompleted: Int
    public let totalFailed: Int
    public let totalTimeout: Int
    public let totalCancelled: Int
    public let currentPending: Int
    public let currentActive: Int
    public let averageQueueTime: TimeInterval
    public let averageProcessingTime: TimeInterval

    public var successRate: Double {
        let total = totalCompleted + totalFailed + totalTimeout + totalCancelled
        return total > 0 ? Double(totalCompleted) / Double(total) : 0
    }
}

// MARK: - Request Scheduler

/// Actor-isolated scheduler for managing concurrent inference requests with priority and timeout handling
public actor RequestScheduler {
    private let logger = Logger(label: "request-scheduler")
    private let config: SchedulerConfig

    // Queue management
    private var pendingQueue: PriorityRequestQueue
    private var activeRequests: [UUID: StatefulRequest] = [:]
    private var completedRequests: [UUID: StatefulRequest] = [:]

    // Stream management
    private let streamRegistry: TokenStreamRegistry

    // Statistics
    private var totalSubmitted: Int = 0
    private var totalCompleted: Int = 0
    private var totalFailed: Int = 0
    private var totalTimeout: Int = 0
    private var totalCancelled: Int = 0
    private var queueTimes: [TimeInterval] = []
    private var processingTimes: [TimeInterval] = []

    // Timeout handling
    private var timeoutTask: Task<Void, Never>?

    public init(config: SchedulerConfig = SchedulerConfig()) {
        self.config = config
        self.pendingQueue = PriorityRequestQueue()
        self.streamRegistry = TokenStreamRegistry()
    }

    // MARK: - Request Submission

    /// Submit a new inference request with priority
    /// - Parameters:
    ///   - request: The inference request
    ///   - priority: Request priority level
    /// - Returns: Tuple of (request ID, token stream)
    public func submit(
        _ request: InferenceRequest,
        priority: RequestPriority = .normal
    ) async -> (UUID, TokenStream) {
        totalSubmitted += 1

        let statefulRequest = StatefulRequest(
            request: request,
            priority: priority
        )

        pendingQueue.enqueue(statefulRequest)

        let stream = await streamRegistry.register(requestId: statefulRequest.id)

        logger.debug("Request submitted", metadata: [
            "request_id": "\(statefulRequest.id)",
            "priority": "\(priority)",
            "queue_length": "\(pendingQueue.count)"
        ])

        // Start timeout checker if not already running
        if timeoutTask == nil {
            startTimeoutChecker()
        }

        return (statefulRequest.id, stream)
    }

    // MARK: - Batch Management

    /// Dequeue next batch of requests for processing
    /// - Parameter size: Maximum batch size
    /// - Returns: Array of requests to process
    public func dequeueNextBatch(size: Int) -> [StatefulRequest] {
        guard size > 0 else { return [] }

        let dequeuedBatch = pendingQueue.dequeue(count: size)

        // Mark as active and track
        var activeBatch: [StatefulRequest] = []
        for var request in dequeuedBatch {
            request.markActive()
            activeRequests[request.id] = request
            activeBatch.append(request)

            // Record queue time
            let queueTime = request.age
            queueTimes.append(queueTime)
            if queueTimes.count > 100 {
                queueTimes.removeFirst()
            }
        }

        if !activeBatch.isEmpty {
            logger.debug("Dequeued batch", metadata: [
                "batch_size": "\(activeBatch.count)",
                "active_count": "\(activeRequests.count)",
                "pending_count": "\(pendingQueue.count)"
            ])
        }

        return activeBatch
    }

    // MARK: - Request State Management

    /// Mark request as streaming
    public func markStreaming(requestId: UUID) {
        guard var request = activeRequests[requestId] else {
            logger.warning("Attempted to mark unknown request as streaming", metadata: [
                "request_id": "\(requestId)"
            ])
            return
        }

        request.markStreaming()
        activeRequests[requestId] = request

        logger.trace("Request streaming", metadata: ["request_id": "\(requestId)"])
    }

    /// Emit a token to the request's stream
    public func emitToken(requestId: UUID, token: String, index: Int) async {
        let chunk = TokenChunk(
            requestId: requestId,
            token: token,
            index: index
        )

        await streamRegistry.yield(requestId: requestId, chunk: chunk)

        logger.trace("Token emitted", metadata: [
            "request_id": "\(requestId)",
            "index": "\(index)"
        ])
    }

    /// Complete a request successfully
    public func complete(requestId: UUID, info: GenerationInfo) async {
        guard var request = activeRequests.removeValue(forKey: requestId) else {
            logger.warning("Attempted to complete unknown request", metadata: [
                "request_id": "\(requestId)"
            ])
            return
        }

        request.markCompleted()
        completedRequests[requestId] = request

        // Track statistics
        totalCompleted += 1
        if let processingTime = request.processingTime {
            processingTimes.append(processingTime)
            if processingTimes.count > 100 {
                processingTimes.removeFirst()
            }
        }

        // Close stream
        await streamRegistry.finish(requestId: requestId)

        logger.info("Request completed", metadata: [
            "request_id": "\(requestId)",
            "tokens": "\(info.totalTokens)",
            "duration": "\(String(format: "%.2f", info.duration))s",
            "tokens_per_second": "\(String(format: "%.2f", info.tokensPerSecond))",
            "finish_reason": "\(info.finishReason)"
        ])
    }

    /// Fail a request with an error
    public func fail(requestId: UUID, error: Error) async {
        guard var request = activeRequests.removeValue(forKey: requestId) else {
            logger.warning("Attempted to fail unknown request", metadata: [
                "request_id": "\(requestId)"
            ])
            return
        }

        request.markFailed()
        completedRequests[requestId] = request

        totalFailed += 1

        // Close stream with error
        await streamRegistry.finish(requestId: requestId, error: error)

        logger.error("Request failed", metadata: [
            "request_id": "\(requestId)",
            "error": "\(error.localizedDescription)"
        ])
    }

    /// Cancel a request
    public func cancel(requestId: UUID) async {
        // Try to remove from pending queue first
        if let removed = pendingQueue.remove(id: requestId) {
            var request = removed
            request.markCancelled()
            completedRequests[requestId] = request
            totalCancelled += 1

            await streamRegistry.finish(requestId: requestId, error: StreamError.cancelled)

            logger.debug("Request cancelled (was pending)", metadata: [
                "request_id": "\(requestId)"
            ])
            return
        }

        // Otherwise remove from active requests
        if var request = activeRequests.removeValue(forKey: requestId) {
            request.markCancelled()
            completedRequests[requestId] = request
            totalCancelled += 1

            await streamRegistry.finish(requestId: requestId, error: StreamError.cancelled)

            logger.debug("Request cancelled (was active)", metadata: [
                "request_id": "\(requestId)"
            ])
        }
    }

    /// Cancel all pending and active requests
    public func cancelAll() async {
        // Get all request IDs
        let activeIds = Array(activeRequests.keys)
        let pendingIds = pendingQueue.allRequests.map { $0.id }
        let allRequestIds = activeIds + pendingIds

        logger.info("Cancelling all requests", metadata: [
            "total_count": "\(allRequestIds.count)"
        ])

        // Cancel each request
        for requestId in allRequestIds {
            await cancel(requestId: requestId)
        }
    }

    /// Get the status of a request
    /// - Parameter requestId: The request ID to check
    /// - Returns: The request status, or nil if not found
    public func getStatus(requestId: UUID) -> RequestStatus? {
        // Check active requests
        if let request = activeRequests[requestId] {
            return request.status
        }

        // Check completed requests
        if let request = completedRequests[requestId] {
            return request.status
        }

        // Check pending queue
        if let request = pendingQueue.find(id: requestId) {
            return request.status
        }

        return nil
    }

    // MARK: - Timeout Handling

    /// Start the timeout checker task
    private func startTimeoutChecker() {
        timeoutTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self = self else { break }
                try? await Task.sleep(for: .seconds(self.config.timeoutCheckInterval))

                await self.handleTimeouts()
            }
        }
    }

    /// Check for and handle timed-out requests
    public func handleTimeouts() async {
        var timedOutIds: [UUID] = []

        // Check active requests for timeouts
        for (id, request) in activeRequests {
            if request.age > config.requestTimeout {
                timedOutIds.append(id)
            }
        }

        // Handle timed-out requests
        for id in timedOutIds {
            guard var request = activeRequests.removeValue(forKey: id) else {
                continue
            }

            request.markTimeout()
            completedRequests[id] = request
            totalTimeout += 1

            await streamRegistry.finish(requestId: id, error: StreamError.timeout)

            logger.warning("Request timed out", metadata: [
                "request_id": "\(id)",
                "age": "\(String(format: "%.2f", request.age))s",
                "timeout_threshold": "\(config.requestTimeout)s"
            ])
        }

        if !timedOutIds.isEmpty {
            logger.info("Timeout check completed", metadata: [
                "timed_out_count": "\(timedOutIds.count)"
            ])
        }
    }

    // MARK: - Observability

    /// Get current scheduler statistics
    public var stats: SchedulerStats {
        let avgQueueTime = queueTimes.isEmpty ? 0 : queueTimes.reduce(0, +) / Double(queueTimes.count)
        let avgProcessingTime = processingTimes.isEmpty ? 0 : processingTimes.reduce(0, +) / Double(processingTimes.count)

        return SchedulerStats(
            totalSubmitted: totalSubmitted,
            totalCompleted: totalCompleted,
            totalFailed: totalFailed,
            totalTimeout: totalTimeout,
            totalCancelled: totalCancelled,
            currentPending: pendingQueue.count,
            currentActive: activeRequests.count,
            averageQueueTime: avgQueueTime,
            averageProcessingTime: avgProcessingTime
        )
    }

    /// Get request by ID
    public func getRequest(id: UUID) -> StatefulRequest? {
        if let request = activeRequests[id] {
            return request
        }
        if let request = completedRequests[id] {
            return request
        }
        return pendingQueue.find(id: id)
    }

    /// Get the current queue length
    public var queueLength: Int {
        pendingQueue.count
    }

    /// Get number of active requests
    public var activeCount: Int {
        activeRequests.count
    }

    /// Clean up old completed requests (keep last 1000)
    public func cleanupCompleted() {
        if completedRequests.count > 1000 {
            let sortedIds = completedRequests
                .sorted { $0.value.completedAt ?? Date.distantPast < $1.value.completedAt ?? Date.distantPast }
                .map { $0.key }

            let toRemove = sortedIds.prefix(completedRequests.count - 1000)
            for id in toRemove {
                completedRequests.removeValue(forKey: id)
            }

            logger.debug("Cleaned up old completed requests", metadata: [
                "removed_count": "\(toRemove.count)"
            ])
        }
    }

    /// Stop the scheduler (cleanup)
    public func stop() {
        timeoutTask?.cancel()
        timeoutTask = nil
        logger.info("Scheduler stopped")
    }
}

// MARK: - Request Model

/// Represents an inference request
public struct InferenceRequest: Sendable {
    public var id: UUID?
    public let prompt: String
    public let maxTokens: Int
    public let temperature: Double
    public let topP: Double
    public let stream: Bool
    public let createdAt: Date

    public init(
        prompt: String,
        maxTokens: Int = 100,
        temperature: Double = 0.7,
        topP: Double = 1.0,
        stream: Bool = true
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.stream = stream
        self.createdAt = Date()
    }
}

/// Represents the response for an inference request
public struct InferenceResponse {
    public let id: UUID
    public let text: String
    public let tokensGenerated: Int
    public let completionTime: TimeInterval

    public init(id: UUID, text: String, tokensGenerated: Int, completionTime: TimeInterval) {
        self.id = id
        self.text = text
        self.tokensGenerated = tokensGenerated
        self.completionTime = completionTime
    }
}

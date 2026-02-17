import Foundation

// MARK: - Request Status

/// Lifecycle states for inference requests
public enum RequestStatus: String, Sendable, Codable {
    case pending      // Waiting in queue
    case active       // Being processed
    case streaming    // Actively generating tokens
    case completed    // Successfully finished
    case failed       // Error occurred
    case timeout      // Exceeded time limit
    case cancelled    // Cancelled by user or system

    /// Whether the request is in a terminal state
    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .timeout, .cancelled:
            return true
        case .pending, .active, .streaming:
            return false
        }
    }

    /// Whether the request is actively processing
    public var isActive: Bool {
        switch self {
        case .active, .streaming:
            return true
        case .pending, .completed, .failed, .timeout, .cancelled:
            return false
        }
    }
}

// MARK: - Request Priority

/// Priority levels for request scheduling
public enum RequestPriority: Int, Comparable, Sendable, Codable {
    case low = 0
    case normal = 1
    case high = 2
    case critical = 3

    public static func < (lhs: RequestPriority, rhs: RequestPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Finish Reason

/// Reason for generation completion
public enum FinishReason: String, Sendable, Codable {
    case stop         // Natural stop (EOS token)
    case length       // Reached max tokens
    case timeout      // Timeout exceeded
    case error        // Error occurred
    case cancelled    // User cancelled
}

// MARK: - Stateful Request

/// Request with full lifecycle state tracking
public struct StatefulRequest: Sendable {
    // Identity
    public let id: UUID
    public let request: InferenceRequest

    // State
    public var status: RequestStatus
    public let priority: RequestPriority

    // Timestamps
    public let submittedAt: Date
    public var startedAt: Date?
    public var completedAt: Date?

    // Computed properties
    public var age: TimeInterval {
        Date().timeIntervalSince(submittedAt)
    }

    public var processingTime: TimeInterval? {
        guard let started = startedAt else { return nil }
        if let completed = completedAt {
            return completed.timeIntervalSince(started)
        }
        return Date().timeIntervalSince(started)
    }

    public var isActive: Bool {
        status.isActive
    }

    public var isTerminal: Bool {
        status.isTerminal
    }

    // Initializer
    public init(
        id: UUID = UUID(),
        request: InferenceRequest,
        priority: RequestPriority = .normal
    ) {
        self.id = id
        self.request = request
        self.status = .pending
        self.priority = priority
        self.submittedAt = Date()
        self.startedAt = nil
        self.completedAt = nil
    }

    // State transitions
    public mutating func markActive() {
        status = .active
        if startedAt == nil {
            startedAt = Date()
        }
    }

    public mutating func markStreaming() {
        status = .streaming
        if startedAt == nil {
            startedAt = Date()
        }
    }

    public mutating func markCompleted() {
        status = .completed
        completedAt = Date()
    }

    public mutating func markFailed() {
        status = .failed
        completedAt = Date()
    }

    public mutating func markTimeout() {
        status = .timeout
        completedAt = Date()
    }

    public mutating func markCancelled() {
        status = .cancelled
        completedAt = Date()
    }
}

// MARK: - Batch Slot

/// Represents a slot in the continuous batch
public struct BatchSlot: Sendable {
    // Identity
    public let slotId: Int
    public let request: StatefulRequest

    // Generation state
    public var generatedTokens: [Int]
    public var kvCacheBlockIds: [Int]  // PagedKVCache integration (Phase 4)
    public var isFinished: Bool
    public var finishReason: FinishReason?

    // Computed properties
    public var totalTokens: Int {
        generatedTokens.count
    }

    public var nextPosition: Int {
        generatedTokens.count
    }

    // Initializer
    public init(slotId: Int, request: StatefulRequest) {
        self.slotId = slotId
        self.request = request
        self.generatedTokens = []
        self.kvCacheBlockIds = []
        self.isFinished = false
        self.finishReason = nil
    }

    // State mutations
    public mutating func appendToken(_ token: Int) {
        generatedTokens.append(token)
    }

    public mutating func finish(reason: FinishReason) {
        isFinished = true
        finishReason = reason
    }

    // Helper to check if max tokens reached
    public func hasReachedMaxTokens() -> Bool {
        return generatedTokens.count >= request.request.maxTokens
    }
}

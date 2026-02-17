import Foundation

// MARK: - Priority Request Queue

/// Efficient priority-based queue with FIFO ordering within priority levels
public struct PriorityRequestQueue {
    // Separate queue for each priority level
    private var queues: [RequestPriority: [StatefulRequest]]

    public init() {
        self.queues = [
            .low: [],
            .normal: [],
            .high: [],
            .critical: []
        ]
    }

    // MARK: - Core Operations

    /// Enqueue a request at its priority level
    /// - Parameter request: The request to enqueue
    public mutating func enqueue(_ request: StatefulRequest) {
        queues[request.priority, default: []].append(request)
    }

    /// Dequeue the highest-priority request (FIFO within priority)
    /// - Returns: The next request, or nil if queue is empty
    public mutating func dequeue() -> StatefulRequest? {
        // Check priorities from highest to lowest
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            if var queue = queues[priority], !queue.isEmpty {
                let request = queue.removeFirst()
                queues[priority] = queue
                return request
            }
        }
        return nil
    }

    /// Dequeue multiple requests up to the specified count
    /// - Parameter count: Maximum number of requests to dequeue
    /// - Returns: Array of requests (may be fewer than count)
    public mutating func dequeue(count: Int) -> [StatefulRequest] {
        guard count > 0 else { return [] }

        var result: [StatefulRequest] = []
        result.reserveCapacity(min(count, self.count))

        while result.count < count, let request = dequeue() {
            result.append(request)
        }

        return result
    }

    // MARK: - Query Operations

    /// Total number of requests in the queue
    public var count: Int {
        queues.values.reduce(0) { $0 + $1.count }
    }

    /// Whether the queue is empty
    public var isEmpty: Bool {
        count == 0
    }

    /// Count of requests at each priority level
    public var countsByPriority: [RequestPriority: Int] {
        queues.mapValues { $0.count }
    }

    /// Count at a specific priority level
    public func count(for priority: RequestPriority) -> Int {
        queues[priority]?.count ?? 0
    }

    /// Peek at the next request without removing it
    public func peek() -> StatefulRequest? {
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            if let queue = queues[priority], let first = queue.first {
                return first
            }
        }
        return nil
    }

    // MARK: - Advanced Operations

    /// Remove a specific request by ID
    /// - Parameter id: Request ID to remove
    /// - Returns: The removed request, or nil if not found
    public mutating func remove(id: UUID) -> StatefulRequest? {
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            if var queue = queues[priority],
               let index = queue.firstIndex(where: { $0.id == id }) {
                let request = queue.remove(at: index)
                queues[priority] = queue
                return request
            }
        }
        return nil
    }

    /// Find a request by ID without removing it
    /// - Parameter id: Request ID to find
    /// - Returns: The request, or nil if not found
    public func find(id: UUID) -> StatefulRequest? {
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            if let request = queues[priority]?.first(where: { $0.id == id }) {
                return request
            }
        }
        return nil
    }

    /// Remove all requests
    public mutating func removeAll() {
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            queues[priority] = []
        }
    }

    /// Get all requests at a specific priority level
    /// - Parameter priority: Priority level to query
    /// - Returns: Array of requests at that priority
    public func requests(at priority: RequestPriority) -> [StatefulRequest] {
        queues[priority] ?? []
    }

    /// Get all requests across all priorities
    public var allRequests: [StatefulRequest] {
        var all: [StatefulRequest] = []
        // Return in priority order
        for priority in [RequestPriority.critical, .high, .normal, .low] {
            all.append(contentsOf: queues[priority] ?? [])
        }
        return all
    }

    // MARK: - Statistics

    /// Average age of requests in the queue
    public var averageAge: TimeInterval {
        let requests = allRequests
        guard !requests.isEmpty else { return 0 }
        let totalAge = requests.reduce(0.0) { $0 + $1.age }
        return totalAge / Double(requests.count)
    }

    /// Oldest request in the queue
    public var oldestRequest: StatefulRequest? {
        allRequests.max { $0.age < $1.age }
    }

    /// Maximum age of any request in the queue
    public var maxAge: TimeInterval {
        oldestRequest?.age ?? 0
    }
}

// MARK: - CustomStringConvertible

extension PriorityRequestQueue: CustomStringConvertible {
    public var description: String {
        let counts = countsByPriority
        return "PriorityQueue(total: \(count), critical: \(counts[.critical] ?? 0), high: \(counts[.high] ?? 0), normal: \(counts[.normal] ?? 0), low: \(counts[.low] ?? 0))"
    }
}

import Foundation
import Vapor

/// Thread-safe rate limiter with sliding window
public actor RateLimiter {
    private struct UsageWindow: Sendable {
        var count: Int
        var resetTime: Date
    }

    private var usage: [String: UsageWindow] = [:]
    private let windowDuration: TimeInterval = 3600  // 1 hour

    public init() {}

    /// Check if request is within rate limit
    public func checkLimit(for userId: String, quota: Int) async throws {
        let now = Date()

        if let window = usage[userId] {
            if now >= window.resetTime {
                // Reset window
                usage[userId] = UsageWindow(count: 1, resetTime: now.addingTimeInterval(windowDuration))
            } else if window.count >= quota {
                // Limit exceeded
                let retryAfter = Int(window.resetTime.timeIntervalSince(now))
                throw Abort(.tooManyRequests, headers: [
                    "X-RateLimit-Limit": "\(quota)",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "\(Int(window.resetTime.timeIntervalSince1970))",
                    "Retry-After": "\(retryAfter)"
                ], reason: "Rate limit exceeded. Try again in \(retryAfter) seconds.")
            } else {
                // Increment count
                usage[userId] = UsageWindow(count: window.count + 1, resetTime: window.resetTime)
            }
        } else {
            // First request
            usage[userId] = UsageWindow(count: 1, resetTime: now.addingTimeInterval(windowDuration))
        }
    }

    /// Get current usage for a user
    public func getUsage(for userId: String) async -> (count: Int, remaining: Int, resetTime: Date)? {
        guard let window = usage[userId] else {
            return nil
        }

        let now = Date()
        if now >= window.resetTime {
            return (count: 0, remaining: 0, resetTime: now.addingTimeInterval(windowDuration))
        }

        return (count: window.count, remaining: 0, resetTime: window.resetTime)
    }

    /// Reset usage for a user (admin function)
    public func reset(for userId: String) async {
        usage.removeValue(forKey: userId)
    }

    /// Clear old usage windows (cleanup task)
    public func cleanup() async {
        let now = Date()
        usage = usage.filter { $0.value.resetTime > now }
    }
}

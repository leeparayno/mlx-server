import Foundation
import Vapor

/// Represents an authenticated user with quota information
public struct User: Sendable, Codable {
    public let id: String
    public var quota: Int  // requests per hour
    public var used: Int
    public var resetTime: Date

    public init(id: String, quota: Int = 1000, used: Int = 0, resetTime: Date = Date().addingTimeInterval(3600)) {
        self.id = id
        self.quota = quota
        self.used = used
        self.resetTime = resetTime
    }

    public var hasQuotaRemaining: Bool {
        let now = Date()
        if now >= resetTime {
            return true  // Window has reset
        }
        return used < quota
    }

    public var quotaRemaining: Int {
        let now = Date()
        if now >= resetTime {
            return quota  // Window has reset
        }
        return max(0, quota - used)
    }
}

/// Storage key for user in request storage
public struct UserKey: StorageKey {
    public typealias Value = User
}

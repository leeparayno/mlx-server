import Foundation

/// Thread-safe storage for API keys and user information
public actor APIKeyStore {
    private var keys: [String: User] = [:]

    public init() {
        // Initialize with some default keys for testing
        let defaultKey = "sk-test-12345"
        keys[defaultKey] = User(id: "test-user", quota: 1000, used: 0)
    }

    /// Validate an API key and return the associated user
    public func validate(_ apiKey: String) async -> User? {
        return keys[apiKey]
    }

    /// Create a new API key for a user
    public func create(for userId: String, quota: Int = 1000) async -> String {
        let apiKey = "sk-\(UUID().uuidString)"
        keys[apiKey] = User(id: userId, quota: quota, used: 0)
        return apiKey
    }

    /// Revoke an API key
    public func revoke(_ apiKey: String) async {
        keys.removeValue(forKey: apiKey)
    }

    /// Get all API keys (for admin purposes)
    public func listKeys() async -> [String: String] {
        return keys.mapValues { $0.id }
    }

    /// Update user quota
    public func updateQuota(for apiKey: String, newQuota: Int) async {
        if var user = keys[apiKey] {
            user.quota = newQuota
            keys[apiKey] = user
        }
    }

    /// Get user by API key
    public func getUser(for apiKey: String) async -> User? {
        return keys[apiKey]
    }

    /// Update user usage count
    public func incrementUsage(for apiKey: String) async {
        if var user = keys[apiKey] {
            let now = Date()

            // Reset if window expired
            if now >= user.resetTime {
                user.used = 1
                user.resetTime = now.addingTimeInterval(3600)
            } else {
                user.used += 1
            }

            keys[apiKey] = user
        }
    }
}

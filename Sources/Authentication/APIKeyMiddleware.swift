import Vapor

/// Middleware for API key authentication
public struct APIKeyMiddleware: AsyncMiddleware {
    private let keyStore: APIKeyStore
    private let rateLimiter: RateLimiter

    public init(keyStore: APIKeyStore, rateLimiter: RateLimiter) {
        self.keyStore = keyStore
        self.rateLimiter = rateLimiter
    }

    public func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        // Extract API key from Authorization header
        guard let authHeader = request.headers.first(name: .authorization) else {
            throw Abort(.unauthorized, headers: [
                "WWW-Authenticate": "Bearer"
            ], reason: "Missing Authorization header")
        }

        // Check format: "Bearer <api-key>"
        guard authHeader.hasPrefix("Bearer ") else {
            throw Abort(.unauthorized, headers: [
                "WWW-Authenticate": "Bearer"
            ], reason: "Invalid Authorization header format. Expected: Bearer <api-key>")
        }

        let apiKey = String(authHeader.dropFirst(7))  // Remove "Bearer "

        // Validate API key
        guard let user = await keyStore.validate(apiKey) else {
            throw Abort(.unauthorized, reason: "Invalid API key")
        }

        // Check rate limit
        try await rateLimiter.checkLimit(for: user.id, quota: user.quota)

        // Increment usage
        await keyStore.incrementUsage(for: apiKey)

        // Store user in request storage
        request.storage[UserKey.self] = user

        // Add rate limit headers to response
        var response = try await next.respond(to: request)

        if let usage = await rateLimiter.getUsage(for: user.id) {
            response.headers.add(name: "X-RateLimit-Limit", value: "\(user.quota)")
            response.headers.add(name: "X-RateLimit-Remaining", value: "\(max(0, user.quota - usage.count))")
            response.headers.add(name: "X-RateLimit-Reset", value: "\(Int(usage.resetTime.timeIntervalSince1970))")
        }

        return response
    }
}

/// Extension to get authenticated user from request
extension Request {
    public var user: User? {
        return self.storage[UserKey.self]
    }
}

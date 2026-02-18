// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-server",

    // Minimum platform versions
    platforms: [
        .macOS(.v15)  // Requires macOS 15+ (Tahoe 26.3+ recommended for optimal JACCL performance)
    ],

    // Products define the executables and libraries produced by this package
    products: [
        // Main server executable
        .executable(
            name: "mlx-server",
            targets: ["MLXServer"]
        ),

        // Core inference library (reusable)
        .library(
            name: "MLXCore",
            targets: ["Core"]
        ),

        // Benchmarking executable
        .executable(
            name: "mlx-benchmark",
            targets: ["Benchmark"]
        )
    ],

    // Dependencies on external packages
    // IMPORTANT: Keep all dependencies at latest versions for maximum performance
    // Updated: February 2026 - All versions verified as latest stable releases
    dependencies: [
        // MLX Swift - Core ML framework (v0.30.6 - Feb 10, 2025)
        // - Wired Memory Management System
        // - Improved performance on macOS 26.3+
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.30.6"
        ),

        // MLX Swift LM - High-level LLM utilities (v2.30.3 - Jan 22, 2025)
        // - Latest model support (GLM 4.7, LFM2 VL, SwissAI Apertus)
        // - Performance optimizations and bug fixes
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            exact: "2.30.3"
        ),

        // Vapor - HTTP server framework (v4.121.2 - Feb 10, 2025)
        // - Sendable conformance for Swift 6 concurrency
        // - Latest performance optimizations
        .package(
            url: "https://github.com/vapor/vapor.git",
            from: "4.121.2"
        ),

        // Swift Argument Parser - CLI arguments (v1.7.0)
        // - @ParentCommand property wrapper
        // - Improved shell completion
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            from: "1.7.0"
        ),

        // Swift Log - Structured logging (v1.9.1)
        // - Enhanced lock implementation
        // - Swift 6.0 language mode support
        .package(
            url: "https://github.com/apple/swift-log.git",
            from: "1.9.1"
        ),

        // Swift Metrics - Observability (v2.7.1)
        // - Release mode build optimizations
        // - TestMetrics improvements
        .package(
            url: "https://github.com/apple/swift-metrics.git",
            from: "2.7.1"
        ),

        // Swift Async Algorithms - Concurrency utilities (v1.1.2 - Feb 12, 2025)
        // - Critical deadlock fixes
        // - Swift 6 compatibility
        // - mapError algorithm support
        .package(
            url: "https://github.com/apple/swift-async-algorithms",
            from: "1.1.2"
        ),

        // Swift Transformers - Hugging Face Hub API (v1.1.6+)
        // - Required for ModelDownloader Hub API access
        // - Provides Hub module for model downloading
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.1.6"
        )
    ],

    // Targets define the modules and executables in this package
    targets: [
        // Main executable target
        .executableTarget(
            name: "MLXServer",
            dependencies: [
                "Core",
                "Scheduler",
                "Memory",
                "API",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Metrics", package: "swift-metrics")
            ],
            path: "Sources/MLXServer"
        ),

        // Core inference engine
        .target(
            name: "Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Core"
        ),

        // Continuous batching scheduler
        .target(
            name: "Scheduler",
            dependencies: [
                "Core",
                "Memory",
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Scheduler"
        ),

        // PagedAttention memory management
        .target(
            name: "Memory",
            dependencies: [
                "Core",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Memory"
        ),

        // Vapor API layer
        .target(
            name: "API",
            dependencies: [
                "Core",
                "Scheduler",
                "Memory",
                .product(name: "Vapor", package: "vapor"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/API"
        ),

        // Benchmark executable
        .executableTarget(
            name: "Benchmark",
            dependencies: [
                "Core",
                "Scheduler",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "benchmarks"
        ),

        // Test targets
        .testTarget(
            name: "CoreTests",
            dependencies: ["Core"],
            path: "Tests/CoreTests"
        ),

        .testTarget(
            name: "SchedulerTests",
            dependencies: ["Scheduler", "Core", "Memory"],
            path: "Tests/SchedulerTests"
        ),

        .testTarget(
            name: "APITests",
            dependencies: [
                "API",
                "Core",
                "Scheduler",
                "Memory",
                .product(name: "XCTVapor", package: "vapor")
            ],
            path: "Tests/APITests"
        )
    ]
)

# MLX Server: Swift Project Setup Guide

## Overview

This document outlines the recommended project structure and development workflow for the MLX inference server. We use **Swift Package Manager (SPM)** as the primary build system, with optional Xcode integration for debugging and profiling.

## Why Swift Package Manager?

### Advantages for Server Projects

1. **Server-First Design**
   - Standard for Swift server development (Vapor, Kitura, all major frameworks)
   - Cleaner, more portable structure than Xcode projects
   - No UI-specific cruft (storyboards, asset catalogs, etc.)

2. **MLX Swift Native**
   - Official MLX Swift library distributed as Swift Package
   - All mlx-swift-examples use SPM
   - Direct dependency integration via `Package.swift`

3. **CI/CD Friendly**
   - Command-line builds: `swift build`
   - Easy Docker containerization
   - No Xcode installation required on build servers
   - Reproducible builds across environments

4. **Editor Flexibility**
   - Works with any editor: VSCode, Vim, Cursor, Xcode
   - Modern Xcode (14+) opens SPM packages natively
   - Generate Xcode project on-demand: `swift package generate-xcodeproj`

5. **Version Control**
   - Minimal `.gitignore` footprint
   - No `.xcodeproj` bloat
   - Clean diffs (only `Package.swift` changes)

### When NOT to Use SPM

SPM is the wrong choice if you're building:
- ❌ macOS/iOS apps with SwiftUI/UIKit
- ❌ Projects requiring Xcode-specific features (asset catalogs, Interface Builder)
- ❌ Legacy projects already using Xcode

**For this server project:** ✅ SPM is the correct choice

## Project Structure

```
mlx-server/
├── Package.swift                 # SPM manifest (dependencies, targets)
├── Package.resolved              # Locked dependency versions
├── README.md
├── LICENSE
├── .gitignore
├── .swiftpm/                     # SPM configuration (auto-generated)
│   └── xcode/
├── Sources/                      # Source code
│   ├── MLXServer/               # Executable target (main entry point)
│   │   ├── main.swift
│   │   └── Server.swift
│   ├── Core/                    # Core inference logic (library target)
│   │   ├── ModelLoader.swift
│   │   ├── Tokenizer.swift
│   │   ├── InferenceEngine.swift
│   │   └── Config.swift
│   ├── Scheduler/               # Continuous batching (library target)
│   │   ├── RequestScheduler.swift
│   │   ├── ContinuousBatcher.swift
│   │   └── BatchState.swift
│   ├── Memory/                  # PagedAttention KV cache (library target)
│   │   ├── PagedKVCache.swift
│   │   └── BlockAllocator.swift
│   └── API/                     # Vapor API layer (library target)
│       ├── Routes.swift
│       ├── Middleware.swift
│       └── Controllers/
├── Tests/                       # Test suites
│   ├── CoreTests/
│   │   ├── InferenceTests.swift
│   │   └── TokenizerTests.swift
│   ├── SchedulerTests/
│   │   └── BatchingTests.swift
│   └── APITests/
│       └── RouteTests.swift
├── benchmarks/                  # Performance benchmarks
│   ├── benchmark.swift
│   └── scenarios.yaml
├── scripts/                     # Utility scripts
│   ├── setup.sh
│   ├── build-release.sh
│   └── profile.sh
├── config/                      # Configuration files
│   ├── development.yaml
│   ├── production.yaml
│   └── models.yaml
└── docs/                        # Documentation
    ├── ARCHITECTURE.md
    ├── API_REFERENCE.md
    └── PERFORMANCE_TUNING.md
```

## Package.swift Configuration

The `Package.swift` file is the heart of your SPM project. It defines:
- Swift tools version
- Package metadata
- Dependencies (external packages)
- Targets (executables, libraries, tests)

### Complete Package.swift Example

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-server",

    // Minimum platform versions
    platforms: [
        .macOS(.v14)  // Requires macOS 14+ for latest Swift concurrency
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
    dependencies: [
        // MLX Swift - Core ML framework
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.30.6"
        ),

        // Vapor - HTTP server framework
        .package(
            url: "https://github.com/vapor/vapor.git",
            from: "4.99.0"
        ),

        // Swift Argument Parser - CLI arguments
        .package(
            url: "https://github.com/apple/swift-argument-parser",
            from: "1.3.0"
        ),

        // Swift Log - Structured logging
        .package(
            url: "https://github.com/apple/swift-log.git",
            from: "1.5.0"
        ),

        // Swift Metrics - Observability
        .package(
            url: "https://github.com/apple/swift-metrics.git",
            from: "2.4.0"
        ),

        // Swift Prometheus - Metrics exporter
        .package(
            url: "https://github.com/swift-server/swift-prometheus.git",
            from: "1.0.0"
        ),

        // Swift Async Algorithms - Useful concurrency utilities
        .package(
            url: "https://github.com/apple/swift-async-algorithms",
            from: "1.0.0"
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
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/Core"
        ),

        // Continuous batching scheduler
        .target(
            name: "Scheduler",
            dependencies: [
                "Core",
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
                .product(name: "Vapor", package: "vapor"),
                .product(name: "Prometheus", package: "swift-prometheus"),
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
            dependencies: ["Scheduler", "Core"],
            path: "Tests/SchedulerTests"
        ),

        .testTarget(
            name: "APITests",
            dependencies: ["API", "Core"],
            path: "Tests/APITests"
        )
    ]
)
```

### Key Package.swift Sections Explained

**1. Swift Tools Version:**
```swift
// swift-tools-version: 5.10
```
- Minimum Swift version required
- Use 5.10+ for complete concurrency support
- Determines available Package.swift DSL features

**2. Platform Requirements:**
```swift
platforms: [.macOS(.v14)]
```
- macOS 14+ required for latest Swift concurrency
- Ensures access to unified memory APIs
- Metal compute shader support

**3. Products:**
```swift
.executable(name: "mlx-server", targets: ["MLXServer"])
.library(name: "MLXCore", targets: ["Core"])
```
- **Executables:** Runnable binaries
- **Libraries:** Reusable modules (can be imported by other packages)

**4. Dependencies:**
```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6")
```
- External package URL
- Version constraint: `from: "0.30.6"` = "0.30.6 or higher, but < 1.0.0"
- Other options: `.exact("1.0.0")`, `.upToNextMinor(from: "1.2.0")`

**5. Targets:**
```swift
.target(name: "Core", dependencies: [
    .product(name: "MLX", package: "mlx-swift")
])
```
- **Target:** A module (library or executable)
- **Dependencies:** Other targets or external products
- **Path:** Source directory (defaults to `Sources/[TargetName]`)

## Development Workflow

### Initial Project Setup

```bash
# 1. Create project directory
mkdir mlx-server
cd mlx-server

# 2. Initialize Swift package
swift package init --type executable

# 3. Edit Package.swift (use template above)
vim Package.swift

# 4. Resolve dependencies (download and build)
swift package resolve

# 5. Verify setup
swift build
```

### Daily Development Commands

```bash
# Build (debug mode, fast compile)
swift build

# Build (release mode, optimized)
swift build -c release

# Run executable
swift run mlx-server

# Run with arguments
swift run mlx-server --model llama-7b --port 8080

# Run tests
swift test

# Run specific test
swift test --filter CoreTests

# Clean build artifacts
swift package clean

# Update dependencies to latest compatible versions
swift package update

# Show dependency tree
swift package show-dependencies
```

### Build Configurations

**Debug Mode (default):**
```bash
swift build
# Output: .build/debug/mlx-server
```
- Fast compilation
- Debug symbols included
- No optimizations
- Assertions enabled
- **Use for:** Development, debugging

**Release Mode:**
```bash
swift build -c release
# Output: .build/release/mlx-server
```
- Slow compilation
- Full optimizations
- No debug symbols (unless `-g` flag)
- Assertions disabled
- **Use for:** Benchmarking, production deployment

### Running the Server

```bash
# Development (auto-rebuild on code changes with external tool)
swift run mlx-server

# Production (optimized binary)
swift build -c release
.build/release/mlx-server --config config/production.yaml

# With environment variables
MLX_MODEL_PATH=/models swift run mlx-server

# Background daemon
nohup .build/release/mlx-server &> logs/server.log &
```

## Using Xcode

### Opening the Package in Xcode

**Option 1: Direct Open (Recommended)**
```bash
open Package.swift
```
- Xcode 14+ opens SPM packages natively
- No `.xcodeproj` generated
- Changes to Package.swift automatically reflected

**Option 2: Generate Xcode Project**
```bash
swift package generate-xcodeproj
open mlx-server.xcodeproj
```
- Creates traditional `.xcodeproj`
- ⚠️ Re-run after Package.swift changes
- ⚠️ Don't commit `.xcodeproj` to git

### Xcode Scheme Configuration

After opening in Xcode:

1. **Select Scheme:** Product → Scheme → `mlx-server`
2. **Edit Scheme:** Product → Scheme → Edit Scheme (⌘<)
3. **Run Configuration:**
   - Build Configuration: Debug (for development) or Release (for profiling)
   - Arguments: Add CLI arguments under "Arguments Passed On Launch"
   - Environment Variables: Set `MLX_MODEL_PATH`, etc.
4. **Working Directory:** Set to project root under "Options" tab

### When to Use Xcode

**✅ Use Xcode for:**

1. **Debugging**
   - Breakpoints, step-through debugging
   - Variable inspection, memory graph
   - Thread sanitizer (detect data races)
   - Address sanitizer (detect memory issues)

2. **Profiling with Instruments**
   - Time Profiler (CPU hotspots)
   - Allocations (memory usage)
   - Leaks (memory leaks)
   - Metal System Trace (GPU activity)

3. **Auto-Completion & Navigation**
   - Jump to definition (⌘-click)
   - Find references
   - Refactoring tools

4. **Building & Running**
   - Quick iteration with ⌘R (build + run)
   - Visual test runner (⌘U)

**❌ Don't Need Xcode for:**
- Writing code (any editor works)
- Building release binaries (use CLI)
- CI/CD pipelines
- Docker containers

### Instruments Profiling Workflow

**Method 1: Profile from Xcode**
```bash
open Package.swift
# In Xcode: Product → Profile (⌘I)
# Select instrument template (Time Profiler, etc.)
```

**Method 2: Profile CLI Binary**
```bash
# Build release binary
swift build -c release

# Run Instruments from command line
instruments -t "Time Profiler" .build/release/mlx-server

# Or attach to running process
instruments -t "Metal System Trace" -p $(pgrep mlx-server)
```

**Common Instruments Templates:**
- **Time Profiler:** CPU usage, hot functions
- **Allocations:** Memory allocations over time
- **Leaks:** Memory leak detection
- **Metal System Trace:** GPU shader execution
- **System Trace:** CPU + GPU + I/O holistic view

### Debugging with LLDB

```bash
# Build with debug symbols
swift build

# Run with LLDB
lldb .build/debug/mlx-server

# LLDB commands
(lldb) breakpoint set --name main
(lldb) run
(lldb) thread backtrace
(lldb) frame variable
(lldb) continue
(lldb) quit
```

## VSCode Alternative

Many Swift developers prefer VSCode for server development:

### Setup VSCode for Swift

```bash
# 1. Install Swift extension
code --install-extension sswg.swift-lang

# 2. Install SourceKit-LSP (language server)
# Comes with Xcode Command Line Tools

# 3. Configure VSCode settings
cat > .vscode/settings.json << 'EOF'
{
  "swift.path": "/usr/bin/swift",
  "swift.buildPath": ".build",
  "swift.diagnosticsStyle": "llvm",
  "editor.formatOnSave": true,
  "[swift]": {
    "editor.defaultFormatter": "sswg.swift-lang"
  }
}
EOF

# 4. Add build tasks
cat > .vscode/tasks.json << 'EOF'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "swift build",
      "type": "shell",
      "command": "swift build",
      "group": {
        "kind": "build",
        "isDefault": true
      }
    },
    {
      "label": "swift test",
      "type": "shell",
      "command": "swift test",
      "group": "test"
    }
  ]
}
EOF
```

### VSCode Workflow

```bash
# Open project
code .

# Use integrated terminal for builds
swift build && swift run

# Set breakpoints in UI
# Run debugger with F5
```

## Dependency Management

### Adding New Dependencies

```swift
// Edit Package.swift
dependencies: [
    .package(url: "https://github.com/author/package", from: "1.0.0")
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "PackageName", package: "package")
        ]
    )
]
```

Then run:
```bash
swift package resolve  # Download and resolve new dependency
swift build           # Build with new dependency
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
swift package update

# Update specific dependency
swift package update mlx-swift

# Pin to specific version (edit Package.swift)
.package(url: "...", exact: "1.2.3")
```

### Viewing Dependency Graph

```bash
# Show dependency tree
swift package show-dependencies

# Example output:
# mlx-server
# ├── mlx-swift<https://github.com/ml-explore/mlx-swift@0.30.6>
# ├── vapor<https://github.com/vapor/vapor@4.99.0>
# │   ├── swift-nio<...>
# │   └── ...
# └── swift-log<...>
```

## Docker Containerization

### Dockerfile for MLX Server

```dockerfile
# Use official Swift image
FROM swift:5.10-jammy AS build

WORKDIR /app

# Copy package manifests
COPY Package.swift Package.resolved ./

# Resolve dependencies (cached layer)
RUN swift package resolve

# Copy source code
COPY Sources ./Sources
COPY Tests ./Tests

# Build release binary
RUN swift build -c release

# Runtime image (smaller)
FROM swift:5.10-slim-jammy

# Install MLX dependencies
RUN apt-get update && apt-get install -y \
    libmlx0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from build stage
COPY --from=build /app/.build/release/mlx-server /usr/local/bin/

# Copy config files
COPY config ./config

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/mlx-server"]
CMD ["--config", "config/production.yaml"]
```

### Building Docker Image

```bash
# Build image
docker build -t mlx-server:latest .

# Run container
docker run -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  -e MLX_MODEL_PATH=/models \
  mlx-server:latest
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-14  # macOS required for MLX

    steps:
      - uses: actions/checkout@v3

      - name: Select Xcode version
        run: sudo xcode-select -s /Applications/Xcode_15.0.app

      - name: Cache Swift packages
        uses: actions/cache@v3
        with:
          path: .build
          key: ${{ runner.os }}-spm-${{ hashFiles('Package.resolved') }}
          restore-keys: ${{ runner.os }}-spm-

      - name: Build
        run: swift build -v

      - name: Run tests
        run: swift test -v

      - name: Check formatting
        run: swift format lint --recursive .
```

## Best Practices

### 1. Dependency Versioning

```swift
// ✅ Good: Semantic versioning
.package(url: "...", from: "1.2.0")  // 1.2.0 <= version < 2.0.0

// ✅ Good: Exact version for stability
.package(url: "...", exact: "1.2.3")

// ⚠️ Caution: Branch tracking (unpredictable)
.package(url: "...", branch: "main")

// ❌ Bad: No version (causes issues)
.package(url: "...")
```

### 2. Target Organization

**Separate concerns into targets:**
- ✅ Small, focused targets (easier to test)
- ✅ Library targets for reusable code
- ✅ Executable targets only for entry points
- ❌ One giant target with everything

### 3. Test Organization

```swift
// Match source structure
Sources/Core/ModelLoader.swift
Tests/CoreTests/ModelLoaderTests.swift

Sources/Scheduler/Batcher.swift
Tests/SchedulerTests/BatcherTests.swift
```

### 4. Git Ignore

```.gitignore
# Swift Package Manager
.build/
.swiftpm/
Package.resolved  # Commit for executables, ignore for libraries

# Xcode (if generated)
*.xcodeproj
*.xcworkspace
xcuserdata/

# Build artifacts
*.o
*.swiftmodule
*.swiftdoc

# macOS
.DS_Store
```

### 5. Performance Optimization

**For benchmarking and production:**
```bash
# Release build with optimizations
swift build -c release

# Enable link-time optimization (LTO)
swift build -c release -Xswiftc -cross-module-optimization

# Profile-guided optimization (advanced)
swift build -c release -Xswiftc -profile-generate
# Run workload to generate profile data
swift build -c release -Xswiftc -profile-use
```

## Troubleshooting

### Common Issues

**1. "Package.resolved is out of sync"**
```bash
swift package resolve
```

**2. "Module 'X' not found"**
```bash
# Clean and rebuild
swift package clean
swift build
```

**3. Xcode shows stale code**
```bash
# Close Xcode, clean, reopen
swift package clean
open Package.swift
```

**4. Build fails with "Illegal instruction"**
```bash
# macOS version too old
# Requires macOS 14+ for this project
sw_vers
```

**5. MLX not found at runtime**
```bash
# Ensure MLX is installed system-wide
# Or use dynamic linking with RPATH
swift build -Xlinker -rpath -Xlinker /usr/local/lib
```

## Summary

**For MLX Server Development:**

1. ✅ **Use SPM** (Package.swift) as primary build system
2. ✅ **Open in Xcode** when needed (debugging, profiling)
3. ✅ **Build from CLI** for release and CI/CD
4. ❌ **Don't commit** `.xcodeproj` to version control
5. ✅ **Profile with Instruments** for performance optimization

**Quick Commands Reference:**
```bash
swift build                  # Build (debug)
swift build -c release       # Build (optimized)
swift run mlx-server         # Run executable
swift test                   # Run tests
swift package resolve        # Update dependencies
open Package.swift           # Open in Xcode
```

This setup provides the best balance of:
- **Simplicity:** Clean project structure
- **Flexibility:** Works with any editor
- **Performance:** Full optimization capabilities
- **Tooling:** Access to Xcode when needed

---

**Next Steps:**
1. Initialize project with `swift package init`
2. Copy Package.swift template from this document
3. Run `swift package resolve` to fetch dependencies
4. Start implementing Phase 1 of the single-node plan

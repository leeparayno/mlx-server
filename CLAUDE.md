# MLX Server

## Project Purpose

This project explores the highest performance possible for MLX-LM on Apple's most powerful personal computing hardware: Mac Studio with M3 Ultra and 512GB of RAM.

The goal is to achieve the highest aggregate performance from a total system that can scale from a single Mac Studio to multiple units clustered with Thunderbolt 5 cables using RDMA (Remote Direct Memory Access).

## Hardware Specifications

- **Platform**: Mac Studio with M3 Ultra
- **Memory**: 512GB RAM
- **Operating System**: macOS Tahoe 26.3+ (required for optimal JACCL bandwidth)
- **Swift**: 6.2.3 (latest stable release)
- **Metal**: Latest version with macOS Tahoe
- **Interconnect**: Thunderbolt 5 with RDMA support
- **Scalability**: Single node to multi-node clusters

## Technology Stack (Latest Releases - February 2026)

### Core ML Framework
- **MLX**: v0.30.6 (Feb 6, 2025)
  - Much faster bandwidth with JACCL on macOS >= 26.3
  - Improved quantized matrix-vector multiplication for small dimensions
  - Enhanced scaled dot-product attention on older hardware
- **mlx-swift**: v0.30.6 (Feb 10, 2025)
  - Wired Memory Management System for improved performance
  - Support for building on Linux (CPU only)
  - Fixed NAX hardware detection on iPhone 16 Pro
- **mlx-swift-lm**: v2.30.3 (Jan 22, 2025)
  - Support for GLM 4.7, LFM2 VL, SwissAI Apertus models
  - Chat re-hydration support
  - Performance optimizations and bug fixes

### Server Infrastructure
- **Vapor**: v4.121.2 (Feb 10, 2025)
  - Sendable conformance for improved Swift 6 concurrency
  - Latest performance optimizations
- **swift-argument-parser**: v1.7.0
  - @ParentCommand property wrapper for command composition
  - Improved shell completion scripts
- **swift-log**: v1.9.1
  - Enhanced lock implementation aligned with swift-nio
  - Swift 6.0 language mode support
- **swift-metrics**: v2.7.1
  - Release mode build optimizations
  - TestMetrics improvements
- **swift-async-algorithms**: v1.1.2 (Feb 12, 2025)
  - Deadlock fixes for improved reliability
  - Swift 6 compatibility
  - mapError algorithm support

## Performance Goals

Maximize MLX-LM inference throughput and efficiency by leveraging:
- **Unified memory architecture**: CPU and GPU share the same memory pool, eliminating data transfer overhead (2x speedup demonstrated on mixed workloads)
- **Hardware-accelerated ML operations**: Native Metal GPU acceleration with MLX's optimized kernels
- **JACCL high-bandwidth communication**: MLX 0.30.6 significantly improves bandwidth on macOS 26.3+
- **High-bandwidth inter-node communication**: Thunderbolt 5 with RDMA support for cluster scaling
- **Wired Memory Management**: mlx-swift 0.30.6's new system for improved memory performance
- **Swift 6 concurrency**: Actor-based architecture with modern async/await patterns
- **Distributed workload orchestration**: Multi-node clusters via Thunderbolt 5 RDMA

## Performance Considerations

**CRITICAL**: This project requires the absolute latest versions of all dependencies to avoid performance regressions:

1. **macOS 26.3+ is mandatory** for JACCL performance improvements in MLX 0.30.6
2. **mlx-swift 0.30.6** includes Wired Memory Management System - significant performance impact
3. **swift-async-algorithms 1.1.2** fixes critical deadlocks that could impact throughput
4. **Vapor 4.121.2** includes Swift 6 Sendable conformance for actor-based concurrency
5. **Swift 6.2.3** provides improved concurrency and performance optimizations

**Never use older versions** - each release contains performance improvements critical for high-throughput inference serving.

## Code Verification Requirements

**MANDATORY**: Every code change MUST be verified before completion. This is non-negotiable.

### Build Verification

**CRITICAL**: This project uses MLX Swift which requires Metal shader compilation. You MUST use `xcodebuild`, NOT `swift build`.

After making ANY code changes, you MUST run:
```bash
make build
```

Or manually:
```bash
xcodebuild build -workspace .swiftpm/xcode/package.xcworkspace -scheme mlx-server -destination 'platform=OS X'
```

The build must complete successfully with no errors or warnings.

### Test Verification
After making code changes that affect functionality, you MUST run:
```bash
make test
```

Or manually run the Xcode-built executable:
```bash
~/Library/Developer/Xcode/DerivedData/mlx-server-*/Build/Products/Debug/mlx-server --test
```

All tests must pass before the change is considered complete.

### Running the Server

Use the Makefile:
```bash
make run
```

Or run the Xcode-built executable directly:
```bash
~/Library/Developer/Xcode/DerivedData/mlx-server-*/Build/Products/Debug/mlx-server
```

### Common Swift Package Manager Issues

1. **MLX Metal Shaders**: MLX Swift requires Metal shader compilation which `swift build` CANNOT do
   - **Solution**: Always use `xcodebuild` (via `make build`) instead of `swift build`
   - **Why**: SwiftPM command line tools don't compile Metal shaders, but xcodebuild does
   - **Error if ignored**: "MLX error: Failed to load the default metallib"

2. **@main attribute with main.swift**: Files named `main.swift` in executable targets CANNOT use `@main` attribute
   - Solution: Rename the file (e.g., `MLXServer.swift`, `Benchmark.swift`)
   - Or: Remove `@main` and use top-level code instead

3. **Clean builds**: If encountering mysterious errors, run `make clean` or `swift package clean` first

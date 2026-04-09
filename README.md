# MLX Server

High-performance inference server for large language models on Apple Silicon, built with Swift and MLX.

## Overview

MLX Server is designed to maximize the performance of Mac Studio M3 Ultra hardware (512GB RAM) through:

- **Continuous Batching**: Dynamic request scheduling for 6-7x throughput improvement
- **PagedAttention**: Memory-efficient KV cache management supporting 100+ concurrent users
- **Swift Actor Concurrency**: Thread-safe request handling without locks
- **MLX Swift**: Native Metal GPU acceleration with unified memory

**Target Performance:** 200+ tokens/sec for 70B models (multi-user scenarios)

## Project Status

✅ **Phase 6.1 Complete** - Production API + Auth/Ratelimits implemented and verified

**Completed Phases:**
- Phase 1: Foundation
- Phase 2: Actor-Based Request Handling
- Phase 3: Continuous Batching
- Phase 4: PagedAttention + MLX Integration (Phase 4.4 verification complete)
- Phase 5: Production HTTP/SSE API
- Phase 6.1: Authentication & Authorization

See [docs/Development-Progress.md](docs/Development-Progress.md) for detailed progress and [docs/plans/2026-02-15-feat-single-node-swift-mlx-server-plan.md](docs/plans/2026-02-15-feat-single-node-swift-mlx-server-plan.md) for the full roadmap.

## Requirements

- **macOS:** 14.0+ (Sonoma)
- **Swift:** 5.10+
- **Hardware:** Mac Studio M3 Ultra with 512GB RAM (recommended)
- **Xcode:** 15.0+ (optional, for debugging/profiling)

## Quick Start

### Build

```bash
# Clone the repository
git clone https://github.com/leeparayno/mlx-server.git
cd mlx-server

# Resolve dependencies
swift package resolve

# Build (debug mode)
swift build

# Build (release mode, optimized)
swift build -c release
```

### Run

```bash
# Run with default settings
swift run mlx-server

# Specify model and port
swift run mlx-server --model mlx-community/Llama-3.2-1B-Instruct-4bit --port 8080

# Run release binary
.build/release/mlx-server --verbose
```

### Test

```bash
# Run all tests
swift test

# Run specific test
swift test --filter CoreTests
```

### Open in Xcode

```bash
# Open package directly (recommended)
open Package.swift

# Or generate Xcode project
swift package generate-xcodeproj
open mlx-server.xcodeproj
```

## Project Structure

```
mlx-server/
├── Package.swift              # Swift Package Manager manifest
├── Sources/
│   ├── MLXServer/            # Main executable
│   ├── Core/                 # Inference engine
│   ├── Scheduler/            # Continuous batching
│   ├── Memory/               # PagedAttention KV cache
│   └── API/                  # Vapor HTTP API
├── Tests/                    # Unit tests
├── benchmarks/               # Performance benchmarks
├── config/                   # Configuration files
└── docs/                     # Documentation
```

## Documentation

- [Swift Project Setup Guide](docs/SWIFT_PROJECT_SETUP.md) - Complete setup and workflow documentation
- [Single-Node Implementation Plan](docs/plans/2026-02-15-feat-single-node-swift-mlx-server-plan.md) - 12-week roadmap
- [Multi-Node RDMA Cluster Plan](docs/plans/2026-02-15-feat-multi-node-rdma-cluster-plan.md) - Distributed inference
- [Production Service Plan](docs/plans/2026-02-15-feat-production-mlx-inference-service-plan.md) - Enterprise deployment
- [MLX Swift Documentation](docs/COMPREHENSIVE_MLX_SWIFT_DOCUMENTATION.md) - MLX Swift API reference

## Architecture

### Phase 1: Foundation ✅ Complete

- [x] Project structure initialized
- [x] Package.swift configured with dependencies
- [x] Basic CLI with argument parsing
- [x] Model loading from Hugging Face Hub
- [x] Basic single-token inference

### Phase 2: Actor-Based Handling ✅ Complete

- [x] InferenceEngine actor
- [x] RequestScheduler actor
- [x] Concurrent request processing
- [x] Request prioritization

### Phase 3: Continuous Batching ✅ Complete

- [x] Dynamic batch formation
- [x] Slot management
- [x] GPU utilization optimization

### Phase 4: PagedAttention + MLX Integration ✅ Complete

- [x] Block-based KV cache
- [x] Memory allocation/deallocation
- [x] Cache eviction policies
- [x] Batch forward pass with MLX
- [x] Phase 4.4 verification (144/144 tests)

### Phase 5: API Layer ✅ Complete

- [x] OpenAI-compatible endpoints
- [x] SSE streaming
- [x] Observability (/health, /metrics)

### Phase 6.1: Auth & Rate Limiting ✅ Complete

- [x] API key authentication middleware
- [x] Per-user quotas + rate limiting
- [x] Security test suite

## Development

### Commands

```bash
# Build
swift build                    # Debug
swift build -c release         # Release (optimized)

# Run
swift run mlx-server          # Run executable
swift run mlx-server --help   # Show help

# Test
swift test                    # All tests
swift test --filter CoreTests # Specific tests

# Clean
swift package clean           # Remove build artifacts

# Update dependencies
swift package update          # Update to latest versions
```

### Debugging with Xcode

1. Open package: `open Package.swift`
2. Set breakpoints in source files
3. Run with ⌘R or Product → Run
4. Profile with ⌘I or Product → Profile

### Profiling with Instruments

```bash
# Build release binary
swift build -c release

# Profile with Time Profiler
instruments -t "Time Profiler" .build/release/mlx-server

# Profile GPU usage
instruments -t "Metal System Trace" .build/release/mlx-server
```

## API Reference

### Endpoints (Coming in Phase 5)

```bash
# Health check
GET /health

# Readiness check
GET /ready

# Completions (OpenAI-compatible)
POST /v1/completions
Content-Type: application/json

{
  "model": "llama-70b-4bit",
  "prompt": "Explain quantum computing",
  "max_tokens": 500,
  "temperature": 0.7,
  "stream": false
}

# Chat completions
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama-70b-4bit",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 500,
  "stream": true
}
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Throughput (70B, multi-user) | 200+ tok/s | TBD |
| Time to First Token | < 100ms | TBD |
| Inter-Token Latency | < 5ms | TBD |
| GPU Utilization | > 90% | TBD |
| Memory Efficiency | > 95% | TBD |

## Contributing

This is currently a research project. Contributions will be welcome once the foundation is stable (Phase 2+).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Swift bindings for MLX
- [Vapor](https://vapor.codes) - Swift web framework
- [vLLM](https://github.com/vllm-project/vllm) - Inspiration for continuous batching

## Related Projects

- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) - Python MLX language models
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - Official MLX Swift examples
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference engine

---

**Status:** Phase 6.1 (Authentication & Authorization) - Complete

**Next Milestone:** Phase 6.2+ Advanced optimizations (speculative decoding, KV quantization, prefix caching)

import Foundation
import ArgumentParser
import Core
import Scheduler
import Memory

@main
struct BenchmarkCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-benchmark",
        abstract: "Comprehensive benchmarking tool for MLX Server",
        version: "1.0.0"
    )

    @Option(name: .shortAndLong, help: "Model path or Hugging Face model ID")
    var model: String = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    @Option(name: .long, help: "Number of warmup iterations")
    var warmup: Int = 10

    @Option(name: .long, help: "Number of benchmark iterations")
    var iterations: Int = 100

    @Option(name: .long, help: "Concurrent requests")
    var concurrency: Int = 16

    @Option(name: .long, help: "Max tokens per request")
    var maxTokens: Int = 50

    @Option(name: .long, help: "Output format (json|csv|markdown)")
    var format: String = "markdown"

    @Flag(name: .long, help: "Run all benchmark scenarios")
    var all: Bool = false

    func run() async throws {
        print("🔬 MLX Server Benchmark")
        print("Model: \(model)")
        print("Iterations: \(iterations) (warmup: \(warmup))")
        print("Concurrency: \(concurrency)")
        print()

        // Load model
        print("Loading model...")
        let loader = ModelLoader()
        let modelContainer = try await loader.load(modelPath: model)
        print("✅ Model loaded")

        // Create inference engine
        let engine = InferenceEngine()
        await engine.initialize(modelContainer: modelContainer)

        // Create scheduler and batcher
        let scheduler = RequestScheduler()
        let batcher = ContinuousBatcher(
            scheduler: scheduler,
            engine: engine,
            config: ContinuousBatcher.Config(
                maxBatchSize: concurrency,
                eosTokenId: 2
            )
        )

        // Start batcher
        let batcherTask = Task {
            await batcher.start()
        }

        // Wait for batcher to start
        try await Task.sleep(for: .milliseconds(500))

        var results = BenchmarkResults()

        // Scenario 1: Single request latency
        print("\n📊 Scenario 1: Single Request Latency")
        let singleLatency = try await measureSingleRequestLatency(
            scheduler: scheduler,
            iterations: iterations,
            maxTokens: maxTokens
        )
        results.singleRequestLatency = singleLatency
        print("  Mean: \(String(format: "%.2f", singleLatency.mean))ms")
        print("  p50: \(String(format: "%.2f", singleLatency.p50))ms")
        print("  p95: \(String(format: "%.2f", singleLatency.p95))ms")

        // Scenario 2: Concurrent throughput
        print("\n📊 Scenario 2: Concurrent Throughput (\(concurrency) concurrent)")
        let throughput = try await measureConcurrentThroughput(
            scheduler: scheduler,
            concurrency: concurrency,
            iterations: iterations,
            maxTokens: maxTokens
        )
        results.concurrentThroughput = throughput
        print("  Throughput: \(String(format: "%.2f", throughput.tokensPerSecond)) tok/s")
        print("  Requests/s: \(String(format: "%.2f", throughput.requestsPerSecond))")
        print("  Avg Latency: \(String(format: "%.2f", throughput.averageLatency))ms")

        // Scenario 3: Memory efficiency
        print("\n📊 Scenario 3: Memory Efficiency")
        let memory = await measureMemoryEfficiency(batcher: batcher)
        results.memoryEfficiency = memory
        print("  KV Cache: \(String(format: "%.1f", memory.kvCacheMemoryMB))MB")
        print("  Total: \(String(format: "%.1f", memory.totalMemoryMB))MB")
        print("  Utilization: \(String(format: "%.1f", memory.cacheUtilization * 100))%")

        // Scenario 4: Batch size scaling (optional)
        if all {
            print("\n📊 Scenario 4: Batch Size Scaling")
            let scaling = try await measureBatchSizeScaling(
                engine: engine,
                sizes: [1, 4, 8, 16, 32],
                maxTokens: maxTokens
            )
            results.batchSizeScaling = scaling
            for (size, perf) in scaling.sorted(by: { $0.key < $1.key }) {
                print("  Batch \(size): \(String(format: "%.2f", perf)) tok/s")
            }
        }

        // Stop batcher
        await batcher.stop()
        batcherTask.cancel()

        // Output results
        print("\n" + String(repeating: "=", count: 60))
        print("Benchmark Complete!")
        print(String(repeating: "=", count: 60))

        switch format.lowercased() {
        case "json":
            try outputJSON(results)
        case "csv":
            try outputCSV(results)
        default:
            outputMarkdown(results)
        }
    }

    // MARK: - Measurement Functions

    private func measureSingleRequestLatency(
        scheduler: RequestScheduler,
        iterations: Int,
        maxTokens: Int
    ) async throws -> LatencyStats {
        var latencies: [Double] = []

        for i in 0..<iterations {
            let start = Date()
            let request = InferenceRequest(
                prompt: "Benchmark test \(i)",
                maxTokens: maxTokens
            )
            let (_, stream) = await scheduler.submit(request, priority: .normal)

            // Consume stream
            for try await _ in stream {}

            let latency = Date().timeIntervalSince(start) * 1000  // ms
            latencies.append(latency)
        }

        return calculateStats(latencies)
    }

    private func measureConcurrentThroughput(
        scheduler: RequestScheduler,
        concurrency: Int,
        iterations: Int,
        maxTokens: Int
    ) async throws -> ThroughputStats {
        let start = Date()
        var totalTokens = 0
        var latencies: [Double] = []

        await withTaskGroup(of: (Int, Double).self) { group in
            for i in 0..<iterations {
                group.addTask {
                    let reqStart = Date()
                    let request = InferenceRequest(
                        prompt: "Concurrent test \(i)",
                        maxTokens: maxTokens
                    )
                    let (_, stream) = await scheduler.submit(request, priority: .normal)

                    var tokenCount = 0
                    do {
                        for try await _ in stream {
                            tokenCount += 1
                        }
                    } catch {
                        // Handle errors gracefully
                    }

                    let latency = Date().timeIntervalSince(reqStart) * 1000
                    return (tokenCount, latency)
                }
            }

            for await (tokens, latency) in group {
                totalTokens += tokens
                latencies.append(latency)
            }
        }

        let duration = Date().timeIntervalSince(start)
        let avgLatency = latencies.reduce(0, +) / Double(latencies.count)

        return ThroughputStats(
            tokensPerSecond: Double(totalTokens) / duration,
            requestsPerSecond: Double(iterations) / duration,
            averageLatency: avgLatency
        )
    }

    private func measureMemoryEfficiency(batcher: ContinuousBatcher) async -> MemoryStats {
        // Get memory usage
        let processInfo = ProcessInfo.processInfo
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        let totalMemoryMB: Double
        if kerr == KERN_SUCCESS {
            totalMemoryMB = Double(info.resident_size) / 1024.0 / 1024.0
        } else {
            totalMemoryMB = 0
        }

        // Get KV cache stats
        let kvCacheMemoryMB: Double = 0  // TODO: Get from batcher/cache
        let cacheUtilization: Double = 0  // TODO: Calculate from batcher stats

        return MemoryStats(
            kvCacheMemoryMB: kvCacheMemoryMB,
            totalMemoryMB: totalMemoryMB,
            cacheUtilization: cacheUtilization
        )
    }

    private func measureBatchSizeScaling(
        engine: InferenceEngine,
        sizes: [Int],
        maxTokens: Int
    ) async throws -> [Int: Double] {
        var results: [Int: Double] = [:]

        for size in sizes {
            let start = Date()
            var totalTokens = 0

            // Simulate batch generation
            for _ in 0..<size {
                let tokens = try await engine.generate(
                    prompt: "Scaling test",
                    maxTokens: maxTokens,
                    temperature: 0.7
                )
                totalTokens += maxTokens
            }

            let duration = Date().timeIntervalSince(start)
            results[size] = Double(totalTokens) / duration
        }

        return results
    }

    // MARK: - Statistics

    private func calculateStats(_ values: [Double]) -> LatencyStats {
        let sorted = values.sorted()
        let count = sorted.count
        let mean = sorted.reduce(0, +) / Double(count)

        let variance = sorted.map { pow($0 - mean, 2) }.reduce(0, +) / Double(count)
        let stddev = sqrt(variance)

        let p50 = percentile(sorted, 50)
        let p95 = percentile(sorted, 95)
        let p99 = percentile(sorted, 99)

        return LatencyStats(
            mean: mean,
            stddev: stddev,
            p50: p50,
            p95: p95,
            p99: p99
        )
    }

    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        let index = (p / 100.0) * Double(sorted.count - 1)
        let lower = Int(floor(index))
        let upper = Int(ceil(index))

        if lower == upper {
            return sorted[lower]
        }

        let weight = index - Double(lower)
        return sorted[lower] * (1 - weight) + sorted[upper] * weight
    }

    // MARK: - Output Formats

    private func outputMarkdown(_ results: BenchmarkResults) {
        print("\n## Single Request Latency")
        if let lat = results.singleRequestLatency {
            print("- Mean: \(String(format: "%.2f", lat.mean))ms (±\(String(format: "%.2f", lat.stddev))ms)")
            print("- p50: \(String(format: "%.2f", lat.p50))ms")
            print("- p95: \(String(format: "%.2f", lat.p95))ms")
            print("- p99: \(String(format: "%.2f", lat.p99))ms")
        }

        print("\n## Concurrent Throughput")
        if let thr = results.concurrentThroughput {
            print("- Tokens/s: \(String(format: "%.2f", thr.tokensPerSecond))")
            print("- Requests/s: \(String(format: "%.2f", thr.requestsPerSecond))")
            print("- Avg Latency: \(String(format: "%.2f", thr.averageLatency))ms")
        }

        print("\n## Memory Efficiency")
        if let mem = results.memoryEfficiency {
            print("- KV Cache: \(String(format: "%.1f", mem.kvCacheMemoryMB))MB")
            print("- Total: \(String(format: "%.1f", mem.totalMemoryMB))MB")
            print("- Utilization: \(String(format: "%.1f", mem.cacheUtilization * 100))%")
        }

        if let scaling = results.batchSizeScaling {
            print("\n## Batch Size Scaling")
            for (size, perf) in scaling.sorted(by: { $0.key < $1.key }) {
                print("- Batch \(size): \(String(format: "%.2f", perf)) tok/s")
            }
        }
    }

    private func outputJSON(_ results: BenchmarkResults) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(results)
        if let json = String(data: data, encoding: .utf8) {
            print(json)
        }
    }

    private func outputCSV(_ results: BenchmarkResults) throws {
        print("metric,value")

        if let lat = results.singleRequestLatency {
            print("latency_mean,\(lat.mean)")
            print("latency_stddev,\(lat.stddev)")
            print("latency_p50,\(lat.p50)")
            print("latency_p95,\(lat.p95)")
            print("latency_p99,\(lat.p99)")
        }

        if let thr = results.concurrentThroughput {
            print("tokens_per_second,\(thr.tokensPerSecond)")
            print("requests_per_second,\(thr.requestsPerSecond)")
            print("avg_latency,\(thr.averageLatency)")
        }

        if let mem = results.memoryEfficiency {
            print("kv_cache_mb,\(mem.kvCacheMemoryMB)")
            print("total_memory_mb,\(mem.totalMemoryMB)")
            print("cache_utilization,\(mem.cacheUtilization)")
        }

        if let scaling = results.batchSizeScaling {
            for (size, perf) in scaling.sorted(by: { $0.key < $1.key }) {
                print("batch_\(size)_tokens_per_sec,\(perf)")
            }
        }
    }
}

// MARK: - Data Structures

struct BenchmarkResults: Codable {
    var singleRequestLatency: LatencyStats?
    var concurrentThroughput: ThroughputStats?
    var memoryEfficiency: MemoryStats?
    var batchSizeScaling: [Int: Double]?
}

struct LatencyStats: Codable {
    let mean: Double
    let stddev: Double
    let p50: Double
    let p95: Double
    let p99: Double
}

struct ThroughputStats: Codable {
    let tokensPerSecond: Double
    let requestsPerSecond: Double
    let averageLatency: Double
}

struct MemoryStats: Codable {
    let kvCacheMemoryMB: Double
    let totalMemoryMB: Double
    let cacheUtilization: Double
}

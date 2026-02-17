import Foundation
import Logging
import MLX
import Memory

// MARK: - Memory Pressure

/// Memory pressure levels for adaptive batching
public enum MemoryPressure: Sendable {
    case normal    // <80% memory used
    case high      // 80-90% memory used
    case critical  // >90% memory used
}

// MARK: - GPU Monitor

/// Actor for monitoring GPU utilization and memory pressure
public actor GPUMonitor {
    private let logger = Logger(label: "gpu-monitor")

    // Utilization tracking
    private var recentUtilization: [Double] = []
    private let windowSize: Int

    // Configuration
    public struct Config: Sendable {
        public let windowSize: Int
        public let targetUtilization: Double
        public let utilizationHysteresis: Double
        public let minBatchSize: Int
        public let maxBatchSize: Int
        public let batchSizeStep: Int
        public let totalMemoryGB: Int

        public init(
            windowSize: Int = 100,
            targetUtilization: Double = 0.90,
            utilizationHysteresis: Double = 0.05,
            minBatchSize: Int = 8,
            maxBatchSize: Int = 64,
            batchSizeStep: Int = 4,
            totalMemoryGB: Int = 512
        ) {
            self.windowSize = windowSize
            self.targetUtilization = targetUtilization
            self.utilizationHysteresis = utilizationHysteresis
            self.minBatchSize = minBatchSize
            self.maxBatchSize = maxBatchSize
            self.batchSizeStep = batchSizeStep
            self.totalMemoryGB = totalMemoryGB
        }
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
        self.windowSize = config.windowSize
    }

    // MARK: - Utilization Tracking

    /// Record a utilization measurement
    /// - Parameter utilization: Utilization value (0.0 to 1.0)
    public func recordUtilization(_ utilization: Double) {
        recentUtilization.append(utilization)
        if recentUtilization.count > windowSize {
            recentUtilization.removeFirst()
        }
    }

    /// Get average utilization over the window
    /// - Returns: Average utilization (0.0 to 1.0)
    public func averageUtilization() -> Double {
        guard !recentUtilization.isEmpty else { return 0.0 }
        return recentUtilization.reduce(0, +) / Double(recentUtilization.count)
    }

    /// Get current utilization (most recent measurement)
    /// - Returns: Current utilization (0.0 to 1.0), or 0.0 if no measurements
    public func currentUtilization() -> Double {
        return recentUtilization.last ?? 0.0
    }

    // MARK: - Batch Size Adjustment

    /// Recommend batch size adjustment based on utilization
    /// - Parameters:
    ///   - current: Current batch size
    ///   - target: Target utilization (defaults to config value)
    /// - Returns: Recommended batch size
    public func recommendBatchSizeAdjustment(
        current: Int,
        target: Double? = nil
    ) -> Int {
        let targetUtil = target ?? config.targetUtilization
        let avg = averageUtilization()

        let lowerBound = targetUtil - config.utilizationHysteresis
        let upperBound = targetUtil + config.utilizationHysteresis

        if avg < lowerBound {
            // Under-utilized, increase batch size
            let newSize = current + config.batchSizeStep
            return min(newSize, config.maxBatchSize)
        } else if avg > upperBound {
            // Over-utilized, decrease batch size
            let newSize = current - config.batchSizeStep
            return max(newSize, config.minBatchSize)
        } else {
            // Within target range, keep current size
            return current
        }
    }

    // MARK: - Memory Pressure

    /// Check current memory pressure with KV cache stats
    /// Phase 4.3: Integrates real MLX memory and KV cache block utilization
    /// - Parameter kvCacheStats: KV cache memory statistics
    /// - Returns: Memory pressure level
    public func checkMemoryPressure(kvCacheStats: MemoryStats) -> MemoryPressure {
        let allocatedGB = estimateAllocatedMemoryGB(kvCacheStats: kvCacheStats)
        let memoryUsageRatio = Double(allocatedGB) / Double(config.totalMemoryGB)
        let blockUsageRatio = kvCacheStats.utilizationPercent / 100.0

        // Use worst of memory or block utilization
        let usageRatio = max(memoryUsageRatio, blockUsageRatio)

        if usageRatio > 0.90 {
            logger.warning("Critical memory pressure", metadata: [
                "allocated_gb": "\(allocatedGB)",
                "total_gb": "\(config.totalMemoryGB)",
                "memory_usage": "\(String(format: "%.1f", memoryUsageRatio * 100))%",
                "block_usage": "\(String(format: "%.1f", kvCacheStats.utilizationPercent))%",
                "used_blocks": "\(kvCacheStats.usedBlocks)",
                "total_blocks": "\(kvCacheStats.totalBlocks)"
            ])
            return .critical
        } else if usageRatio > 0.80 {
            logger.info("High memory pressure", metadata: [
                "allocated_gb": "\(allocatedGB)",
                "memory_usage": "\(String(format: "%.1f", memoryUsageRatio * 100))%",
                "block_usage": "\(String(format: "%.1f", kvCacheStats.utilizationPercent))%"
            ])
            return .high
        } else {
            return .normal
        }
    }

    /// Estimate allocated memory in GB using real MLX tracking
    /// Phase 4.3: Uses MLX.memoryAllocated() and KV cache stats
    /// - Parameter kvCacheStats: KV cache memory statistics
    /// - Returns: Estimated memory usage in GB
    private func estimateAllocatedMemoryGB(kvCacheStats: MemoryStats) -> Int {
        // Get real MLX memory allocation (Phase 4.3)
        let mlxAllocatedBytes = MLX.Memory.activeMemory
        let mlxAllocatedGB = mlxAllocatedBytes / (1024 * 1024 * 1024)

        // Estimate KV cache memory (rough approximation)
        // Assume each token uses ~2KB for K/V (fp16, typical model dimensions)
        let kvCacheBytes = kvCacheStats.usedTokenCapacity * 2048
        let kvCacheGB = kvCacheBytes / (1024 * 1024 * 1024)

        let totalGB = mlxAllocatedGB + kvCacheGB

        return Int(totalGB)
    }

    // MARK: - Statistics

    /// Get monitoring statistics
    /// - Returns: Tuple of (avgUtil, currentUtil, samples)
    public func stats() -> (averageUtilization: Double, currentUtilization: Double, sampleCount: Int) {
        return (
            averageUtilization: averageUtilization(),
            currentUtilization: currentUtilization(),
            sampleCount: recentUtilization.count
        )
    }

    /// Reset utilization history
    public func reset() {
        recentUtilization.removeAll()
        logger.debug("GPU monitor reset")
    }
}

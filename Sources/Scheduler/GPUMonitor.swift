import Foundation
import Logging

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

    /// Check current memory pressure
    /// - Returns: Memory pressure level
    public func checkMemoryPressure() -> MemoryPressure {
        // TODO: Integrate with actual MLX memory tracking
        // For now, use placeholder logic based on available memory
        let allocatedGB = estimateAllocatedMemoryGB()
        let usageRatio = Double(allocatedGB) / Double(config.totalMemoryGB)

        if usageRatio > 0.90 {
            logger.warning("Critical memory pressure", metadata: [
                "allocated_gb": "\(allocatedGB)",
                "total_gb": "\(config.totalMemoryGB)",
                "usage": "\(String(format: "%.1f", usageRatio * 100))%"
            ])
            return .critical
        } else if usageRatio > 0.80 {
            logger.info("High memory pressure", metadata: [
                "allocated_gb": "\(allocatedGB)",
                "usage": "\(String(format: "%.1f", usageRatio * 100))%"
            ])
            return .high
        } else {
            return .normal
        }
    }

    /// Estimate allocated memory in GB
    /// - Returns: Estimated memory usage in GB
    private func estimateAllocatedMemoryGB() -> Int {
        // TODO: Phase 4 - Integrate with MLX.memoryAllocated()
        // For now, return a conservative estimate
        return 0  // Placeholder - will be replaced with actual MLX memory tracking
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

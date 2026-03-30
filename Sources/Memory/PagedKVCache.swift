import Foundation
import MLX
import Core
import Logging

/// Implements PagedAttention-style KV cache management
/// Divides memory into fixed-size pages (blocks) for efficient allocation
public actor PagedKVCache {
    private let logger = Logger(label: "paged-kv-cache")
    private let blockSize: Int  // Tokens per block (e.g., 16)
    private let numBlocks: Int   // Total blocks available
    private let quantization: QuantizationConfig
    private var blockPool: [CacheBlock]
    private var requestBlocks: [UUID: [Int]]  // Request ID -> Block IDs

    public init(blockSize: Int = 16, numBlocks: Int = 1024, quantization: QuantizationConfig = QuantizationConfig()) {
        self.blockSize = blockSize
        self.numBlocks = numBlocks
        self.quantization = quantization
        self.blockPool = (0..<numBlocks).map { CacheBlock(id: $0, blockSize: blockSize) }
        self.requestBlocks = [:]
    }

    /// Allocate blocks for a new request
    /// - Parameters:
    ///   - requestId: Request identifier
    ///   - numTokens: Number of tokens needed
    /// - Returns: Allocated block IDs
    public func allocate(for requestId: UUID, numTokens: Int) throws -> [Int] {
        let blocksNeeded = (numTokens + blockSize - 1) / blockSize  // Ceiling division

        // Find free blocks
        let freeBlocks = blockPool.enumerated()
            .filter { !$0.element.inUse }
            .prefix(blocksNeeded)
            .map { $0.offset }

        guard freeBlocks.count == blocksNeeded else {
            throw MemoryError.insufficientBlocks(needed: blocksNeeded, available: freeBlocks.count)
        }

        // Mark blocks as in use
        for blockId in freeBlocks {
            blockPool[blockId].inUse = true
        }

        requestBlocks[requestId] = Array(freeBlocks)

        logger.debug("Allocated blocks", metadata: [
            "request_id": "\(requestId)",
            "blocks": "\(freeBlocks.count)",
            "tokens": "\(numTokens)"
        ])

        return Array(freeBlocks)
    }

    /// Release blocks for a completed request
    /// - Parameter requestId: Request identifier
    public func release(for requestId: UUID) {
        guard let blockIds = requestBlocks.removeValue(forKey: requestId) else {
            return
        }

        // Mark blocks as free
        for blockId in blockIds {
            blockPool[blockId].inUse = false
            blockPool[blockId].clear()
        }

        logger.debug("Released blocks", metadata: [
            "request_id": "\(requestId)",
            "blocks": "\(blockIds.count)"
        ])
    }

    /// Get memory utilization statistics
    public var stats: MemoryStats {
        let usedBlocks = blockPool.filter { $0.inUse }.count
        let totalCapacity = numBlocks * blockSize
        let usedCapacity = usedBlocks * blockSize

        return MemoryStats(
            totalBlocks: numBlocks,
            usedBlocks: usedBlocks,
            freeBlocks: numBlocks - usedBlocks,
            utilizationPercent: Double(usedBlocks) / Double(numBlocks) * 100,
            totalTokenCapacity: totalCapacity,
            usedTokenCapacity: usedCapacity
        )
    }

    // MARK: - Phase 4.2: KV Cache Operations

    /// Retrieve K/V cache tensors for given block IDs
    /// - Parameter ids: Block IDs to retrieve
    /// - Returns: Tuple of (keys, values) MLXArrays
    public func getBlocks(ids: [Int]) throws -> (keys: MLXArray?, values: MLXArray?) {
        guard !ids.isEmpty else {
            return (nil, nil)
        }

        // Validate block IDs
        for id in ids {
            guard id >= 0 && id < blockPool.count else {
                throw MemoryError.blockNotFound(id)
            }
        }

        // For now, return nil since we haven't populated the blocks yet
        // In a full implementation, we'd concatenate K/V from all blocks
        if ids.count == 1 {
            let block = blockPool[ids[0]]
            if let kq = block.quantizedKeys, let vq = block.quantizedValues {
                let keys = TurboQuantMLX.dequantize(kq)
                let values = TurboQuantMLX.dequantize(vq)
                return (keys, values)
            }
            if let kq = block.quantizedKeysProd, let vq = block.quantizedValuesProd {
                let keys = TurboQuantMLX.dequantizeProd(kq)
                let values = TurboQuantMLX.dequantizeProd(vq)
                return (keys, values)
            }
            return (block.keys, block.values)
        }

        return (nil, nil)
    }

    /// Append new K/V values to blocks
    /// - Parameters:
    ///   - ids: Block IDs to append to
    ///   - keys: New key tensor to append
    ///   - values: New value tensor to append
    public func appendToBlocks(ids: [Int], keys: MLXArray, values: MLXArray) throws {
        guard !ids.isEmpty else { return }

        // Validate shapes match
        guard keys.shape == values.shape else {
            throw MemoryError.shapeMismatch
        }

        // Validate block IDs
        for id in ids {
            guard id >= 0 && id < blockPool.count else {
                throw MemoryError.blockNotFound(id)
            }
            guard blockPool[id].inUse else {
                throw MemoryError.blockNotInUse(id)
            }
        }

        // Append K/V to blocks
        // In a full implementation, we'd:
        // 1. Determine which block to write to based on current fill
        // 2. Concatenate new K/V with existing cache
        // 3. Update block currentLength

        // For now, store the latest K/V in the first block for scaffolding
        if let first = ids.first {
            if quantization.enabled {
                switch quantization.mode {
                case .mse:
                    blockPool[first].quantizedKeys = TurboQuantMLX.quantize(keys, config: quantization)
                    blockPool[first].quantizedValues = TurboQuantMLX.quantize(values, config: quantization)
                    blockPool[first].quantizedKeysProd = nil
                    blockPool[first].quantizedValuesProd = nil
                case .prod:
                    blockPool[first].quantizedKeysProd = TurboQuantMLX.quantizeProd(keys, config: quantization)
                    blockPool[first].quantizedValuesProd = TurboQuantMLX.quantizeProd(values, config: quantization)
                    blockPool[first].quantizedKeys = nil
                    blockPool[first].quantizedValues = nil
                }
                blockPool[first].keys = nil
                blockPool[first].values = nil
            } else {
                blockPool[first].keys = keys
                blockPool[first].values = values
                blockPool[first].quantizedKeys = nil
                blockPool[first].quantizedValues = nil
                blockPool[first].quantizedKeysProd = nil
                blockPool[first].quantizedValuesProd = nil
            }
        }

        logger.debug("Appended K/V to blocks", metadata: [
            "block_ids": "\(ids)",
            "shape": "\(keys.shape)",
            "quantized": "\(quantization.enabled)"
        ])
    }
}

// MARK: - Cache Block

/// Represents a single block in the paged cache
struct CacheBlock {
    let id: Int
    let blockSize: Int
    var inUse: Bool = false
    var keys: MLXArray?
    var values: MLXArray?
    var quantizedKeys: QuantizedVector?
    var quantizedValues: QuantizedVector?
    var quantizedKeysProd: QuantizedVectorProd?
    var quantizedValuesProd: QuantizedVectorProd?

    mutating func clear() {
        keys = nil
        values = nil
        quantizedKeys = nil
        quantizedValues = nil
        quantizedKeysProd = nil
        quantizedValuesProd = nil
    }
}

// MARK: - Memory Stats

/// Memory utilization statistics
public struct MemoryStats: Sendable {
    public let totalBlocks: Int
    public let usedBlocks: Int
    public let freeBlocks: Int
    public let utilizationPercent: Double
    public let totalTokenCapacity: Int
    public let usedTokenCapacity: Int
}

// MARK: - Errors

public enum MemoryError: Error, LocalizedError {
    case insufficientBlocks(needed: Int, available: Int)
    case blockNotFound(Int)
    case shapeMismatch
    case blockNotInUse(Int)

    public var errorDescription: String? {
        switch self {
        case .insufficientBlocks(let needed, let available):
            return "Insufficient blocks: needed \(needed), available \(available)"
        case .blockNotFound(let id):
            return "Block not found: \(id)"
        case .shapeMismatch:
            return "K/V tensor shapes do not match"
        case .blockNotInUse(let id):
            return "Block \(id) is not in use"
        }
    }
}

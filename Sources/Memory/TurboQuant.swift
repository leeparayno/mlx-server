import Foundation
import MLX

/// TurboQuant scaffolding for MLX
/// NOTE: This is an initial, MSE-only implementation with scalar quantization.
/// TODO: Add random rotation (structured orthogonal transform) and QJL residual stage.

public struct QuantizationConfig: Sendable {
    public var enabled: Bool
    public var bitWidth: Int

    public init(enabled: Bool = false, bitWidth: Int = 2) {
        self.enabled = enabled
        self.bitWidth = bitWidth
    }
}

public struct QuantizedVector: Sendable {
    public let indices: [UInt8]
    public let norm: Float
    public let bitWidth: Int
    public let dimension: Int

    public init(indices: [UInt8], norm: Float, bitWidth: Int, dimension: Int) {
        self.indices = indices
        self.norm = norm
        self.bitWidth = bitWidth
        self.dimension = dimension
    }
}

/// Approximate Lloyd-Max centroids for N(0,1) (sampled k-means on 200k samples)
/// Scaled at runtime by 1/sqrt(d) to match TurboQuant assumptions.
public enum TurboQuantCodebooks {
    public static let standardNormal: [Int: [Float]] = [
        1: [-0.8005832, 0.7982868],
        2: [-1.5159471, -0.45459235, 0.4502761, 1.5121450],
        3: [-2.1138350, -1.3111417, -0.7318221, -0.23609458, 0.2354913, 0.72909325, 1.3139455, 2.1254150],
        4: [-2.5076079, -1.8050863, -1.3400542, -0.99455965, -0.7178350, -0.48526016, -0.28037643, -0.0923133,
            0.09097275, 0.2781811, 0.48298168, 0.7161325, 0.99247175, 1.3362553, 1.7866939, 2.4786823]
    ]
}

public struct TurboQuantMSE {
    public let bitWidth: Int
    public let dimension: Int
    public let centroids: [Float]

    public init(bitWidth: Int, dimension: Int) {
        self.bitWidth = bitWidth
        self.dimension = dimension
        let base = TurboQuantCodebooks.standardNormal[bitWidth] ?? []
        let scale = 1.0 / sqrt(Float(dimension))
        self.centroids = base.map { $0 * scale }
    }

    /// Quantize a vector (MSE-only, no rotation yet)
    public func quantize(_ x: [Float]) -> QuantizedVector {
        let norm = sqrt(x.reduce(0) { $0 + $1 * $1 })
        let invNorm: Float = norm > 0 ? 1.0 / norm : 1.0
        var indices: [UInt8] = []
        indices.reserveCapacity(x.count)

        for v in x {
            let vNorm = v * invNorm
            // find nearest centroid (linear scan; k<=16)
            var bestIdx = 0
            var bestDist = Float.greatestFiniteMagnitude
            for (i, c) in centroids.enumerated() {
                let d = abs(vNorm - c)
                if d < bestDist {
                    bestDist = d
                    bestIdx = i
                }
            }
            indices.append(UInt8(bestIdx))
        }

        return QuantizedVector(indices: indices, norm: norm, bitWidth: bitWidth, dimension: dimension)
    }

    /// Dequantize a vector
    public func dequantize(_ q: QuantizedVector) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(q.indices.count)
        for idx in q.indices {
            let i = Int(idx)
            let c = (i < centroids.count) ? centroids[i] : 0.0
            out.append(c * q.norm)
        }
        return out
    }
}

// MARK: - MLX helpers

public enum TurboQuantMLX {
    /// Quantize an MLXArray (expects 1D)
    public static func quantize(_ x: MLXArray, bitWidth: Int) -> QuantizedVector {
        let arr = x.asArray(Float.self)
        let q = TurboQuantMSE(bitWidth: bitWidth, dimension: arr.count)
        return q.quantize(arr)
    }

    /// Dequantize to MLXArray (1D)
    public static func dequantize(_ q: QuantizedVector) -> MLXArray {
        let deq = TurboQuantMSE(bitWidth: q.bitWidth, dimension: q.dimension).dequantize(q)
        return MLXArray(deq)
    }
}

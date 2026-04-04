import Foundation
import MLX

/// TurboQuant scaffolding for MLX
/// NOTE: This is an initial, MSE-only implementation with scalar quantization.
/// TODO: Add random rotation (structured orthogonal transform) and QJL residual stage.

public enum QuantizationMode: String, Sendable {
    case mse
    case prod
}

public enum QuantizationImplementation: String, Sendable {
    case mlx
    case turbo
}

public struct QuantizationConfig: Sendable {
    public var enabled: Bool
    public var bitWidth: Int
    public var rotationEnabled: Bool
    public var rotationSeed: UInt64
    public var mode: QuantizationMode
    public var implementation: QuantizationImplementation
    public var qjlSeed: UInt64
    public var groupSize: Int
    public var quantizedKVStart: Int

    public init(
        enabled: Bool = false,
        bitWidth: Int = 2,
        rotationEnabled: Bool = true,
        rotationSeed: UInt64 = 1337,
        mode: QuantizationMode = .mse,
        implementation: QuantizationImplementation = .mlx,
        qjlSeed: UInt64 = 4242,
        groupSize: Int = 64,
        quantizedKVStart: Int = 0
    ) {
        self.enabled = enabled
        self.bitWidth = bitWidth
        self.rotationEnabled = rotationEnabled
        self.rotationSeed = rotationSeed
        self.mode = mode
        self.implementation = implementation
        self.qjlSeed = qjlSeed
        self.groupSize = groupSize
        self.quantizedKVStart = quantizedKVStart
    }
}

public struct QuantizedVector: Sendable {
    public let indices: [UInt8]
    public let norm: Float
    public let bitWidth: Int
    public let dimension: Int
    public let rotationEnabled: Bool
    public let rotationSeed: UInt64

    public init(indices: [UInt8], norm: Float, bitWidth: Int, dimension: Int, rotationEnabled: Bool, rotationSeed: UInt64) {
        self.indices = indices
        self.norm = norm
        self.bitWidth = bitWidth
        self.dimension = dimension
        self.rotationEnabled = rotationEnabled
        self.rotationSeed = rotationSeed
    }
}

public struct QuantizedVectorProd: Sendable {
    public let indices: [UInt8]
    public let norm: Float
    public let bitWidth: Int
    public let dimension: Int
    public let rotationEnabled: Bool
    public let rotationSeed: UInt64
    public let qjlSigns: [UInt8]
    public let residualNorm: Float
    public let qjlSeed: UInt64

    public init(
        indices: [UInt8],
        norm: Float,
        bitWidth: Int,
        dimension: Int,
        rotationEnabled: Bool,
        rotationSeed: UInt64,
        qjlSigns: [UInt8],
        residualNorm: Float,
        qjlSeed: UInt64
    ) {
        self.indices = indices
        self.norm = norm
        self.bitWidth = bitWidth
        self.dimension = dimension
        self.rotationEnabled = rotationEnabled
        self.rotationSeed = rotationSeed
        self.qjlSigns = qjlSigns
        self.residualNorm = residualNorm
        self.qjlSeed = qjlSeed
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
    public let rotationEnabled: Bool
    public let rotationSeed: UInt64

    public init(bitWidth: Int, dimension: Int, rotationEnabled: Bool = true, rotationSeed: UInt64 = 1337) {
        self.bitWidth = bitWidth
        self.dimension = dimension
        self.rotationEnabled = rotationEnabled
        self.rotationSeed = rotationSeed
        let base = TurboQuantCodebooks.standardNormal[bitWidth] ?? []
        let scale = 1.0 / sqrt(Float(dimension))
        self.centroids = base.map { $0 * scale }
    }

    /// Quantize a vector (MSE-only, structured rotation)
    public func quantize(_ x: [Float]) -> QuantizedVector {
        let norm = sqrt(x.reduce(0) { $0 + $1 * $1 })
        let invNorm: Float = norm > 0 ? 1.0 / norm : 1.0
        var working = x.map { $0 * invNorm }

        if rotationEnabled {
            working = TurboQuantRotation.apply(working, seed: rotationSeed)
        }

        var indices: [UInt8] = []
        indices.reserveCapacity(working.count)

        for vNorm in working {
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

        return QuantizedVector(
            indices: indices,
            norm: norm,
            bitWidth: bitWidth,
            dimension: dimension,
            rotationEnabled: rotationEnabled,
            rotationSeed: rotationSeed
        )
    }

    /// Dequantize a vector
    public func dequantize(_ q: QuantizedVector) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(q.indices.count)
        for idx in q.indices {
            let i = Int(idx)
            let c = (i < centroids.count) ? centroids[i] : 0.0
            out.append(c)
        }

        if q.rotationEnabled {
            out = TurboQuantRotation.applyInverse(out, seed: q.rotationSeed)
        }

        return out.map { $0 * q.norm }
    }
}

public struct TurboQuantProd {
    public let bitWidth: Int
    public let dimension: Int
    public let mseQuant: TurboQuantMSE
    public let qjlSeed: UInt64

    public init(bitWidth: Int, dimension: Int, rotationEnabled: Bool, rotationSeed: UInt64, qjlSeed: UInt64) {
        self.bitWidth = bitWidth
        self.dimension = dimension
        self.mseQuant = TurboQuantMSE(bitWidth: max(1, bitWidth - 1), dimension: dimension, rotationEnabled: rotationEnabled, rotationSeed: rotationSeed)
        self.qjlSeed = qjlSeed
    }

    public func quantize(_ x: [Float]) -> QuantizedVectorProd {
        let mse = mseQuant.quantize(x)
        let xMSE = mseQuant.dequantize(mse)

        var residual: [Float] = []
        residual.reserveCapacity(x.count)
        for i in 0..<x.count {
            residual.append(x[i] - xMSE[i])
        }

        let residualNorm = sqrt(residual.reduce(0) { $0 + $1 * $1 })
        let invResNorm: Float = residualNorm > 0 ? 1.0 / residualNorm : 1.0
        var rNormed = residual.map { $0 * invResNorm }

        // Structured QJL: use same rotation primitive, then sign
        rNormed = TurboQuantRotation.apply(rNormed, seed: qjlSeed)
        var qjlSigns: [UInt8] = []
        qjlSigns.reserveCapacity(rNormed.count)
        for v in rNormed {
            qjlSigns.append(v >= 0 ? 1 : 0)
        }

        return QuantizedVectorProd(
            indices: mse.indices,
            norm: mse.norm,
            bitWidth: bitWidth,
            dimension: dimension,
            rotationEnabled: mse.rotationEnabled,
            rotationSeed: mse.rotationSeed,
            qjlSigns: qjlSigns,
            residualNorm: residualNorm,
            qjlSeed: qjlSeed
        )
    }

    public func dequantize(_ q: QuantizedVectorProd) -> [Float] {
        let base = mseQuant.dequantize(
            QuantizedVector(
                indices: q.indices,
                norm: q.norm,
                bitWidth: max(1, q.bitWidth - 1),
                dimension: q.dimension,
                rotationEnabled: q.rotationEnabled,
                rotationSeed: q.rotationSeed
            )
        )

        // Reconstruct residual via structured QJL
        var signs: [Float] = []
        signs.reserveCapacity(q.qjlSigns.count)
        for s in q.qjlSigns {
            signs.append(s == 1 ? 1.0 : -1.0)
        }
        var r = TurboQuantRotation.applyInverse(signs, seed: q.qjlSeed)
        let scale = (sqrt(Float.pi / 2.0) / Float(q.dimension)) * q.residualNorm
        r = r.map { $0 * scale }

        var out: [Float] = []
        out.reserveCapacity(base.count)
        for i in 0..<base.count {
            out.append(base[i] + r[i])
        }
        return out
    }
}

// MARK: - MLX helpers

// MARK: - Structured rotation (Hadamard + sign + permutation)

public enum TurboQuantRotation {
    public static func apply(_ x: [Float], seed: UInt64) -> [Float] {
        guard isPowerOfTwo(x.count) else { return x }
        let perm = permutation(count: x.count, seed: seed)
        let sign = signVector(count: x.count, seed: seed ^ 0xA5A5A5A5)

        var y = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            y[i] = x[perm[i]] * sign[i]
        }
        hadamardInPlace(&y)
        return y
    }

    public static func applyInverse(_ x: [Float], seed: UInt64) -> [Float] {
        guard isPowerOfTwo(x.count) else { return x }
        var y = x
        hadamardInPlace(&y)

        let perm = permutation(count: x.count, seed: seed)
        let sign = signVector(count: x.count, seed: seed ^ 0xA5A5A5A5)

        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let v = y[i] * sign[i]
            out[perm[i]] = v
        }
        return out
    }

    private static func hadamardInPlace(_ x: inout [Float]) {
        let n = x.count
        var h = 1
        while h < n {
            var i = 0
            while i < n {
                for j in i..<(i + h) {
                    let a = x[j]
                    let b = x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
                }
                i += h * 2
            }
            h *= 2
        }
        let scale = 1.0 / sqrt(Float(n))
        for i in 0..<n { x[i] *= scale }
    }

    private static func isPowerOfTwo(_ n: Int) -> Bool {
        return n > 0 && (n & (n - 1)) == 0
    }

    private static func permutation(count: Int, seed: UInt64) -> [Int] {
        var rng = SplitMix64(seed: seed)
        var arr = Array(0..<count)
        for i in stride(from: count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            arr.swapAt(i, j)
        }
        return arr
    }

    private static func signVector(count: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        var s = [Float](repeating: 1.0, count: count)
        for i in 0..<count {
            s[i] = (rng.next() & 1 == 0) ? 1.0 : -1.0
        }
        return s
    }
}

// MARK: - PRNG

public struct SplitMix64 {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

public enum TurboQuantMLX {
    /// Quantize an MLXArray (expects 1D)
    public static func quantize(_ x: MLXArray, config: QuantizationConfig) -> QuantizedVector {
        let arr = x.asArray(Float.self)
        let q = TurboQuantMSE(
            bitWidth: config.bitWidth,
            dimension: arr.count,
            rotationEnabled: config.rotationEnabled,
            rotationSeed: config.rotationSeed
        )
        return q.quantize(arr)
    }

    public static func quantizeProd(_ x: MLXArray, config: QuantizationConfig) -> QuantizedVectorProd {
        let arr = x.asArray(Float.self)
        let q = TurboQuantProd(
            bitWidth: config.bitWidth,
            dimension: arr.count,
            rotationEnabled: config.rotationEnabled,
            rotationSeed: config.rotationSeed,
            qjlSeed: config.qjlSeed
        )
        return q.quantize(arr)
    }

    /// Dequantize to MLXArray (1D)
    public static func dequantize(_ q: QuantizedVector) -> MLXArray {
        let deq = TurboQuantMSE(
            bitWidth: q.bitWidth,
            dimension: q.dimension,
            rotationEnabled: q.rotationEnabled,
            rotationSeed: q.rotationSeed
        ).dequantize(q)
        return MLXArray(deq)
    }

    public static func dequantizeProd(_ q: QuantizedVectorProd) -> MLXArray {
        let deq = TurboQuantProd(
            bitWidth: q.bitWidth,
            dimension: q.dimension,
            rotationEnabled: q.rotationEnabled,
            rotationSeed: q.rotationSeed,
            qjlSeed: q.qjlSeed
        ).dequantize(q)
        return MLXArray(deq)
    }
}

import Foundation
import MLX
import MLXLMCommon
import Memory

/// TurboQuant-backed KV cache (scaffolding)
/// NOTE: This stores quantized per-token vectors and dequantizes on read.
/// This is correct but not performance-optimized.
public final class TurboQuantKVCache: BaseKVCache {
    public struct Config: Sendable {
        public let quantization: QuantizationConfig
        public init(quantization: QuantizationConfig) {
            self.quantization = quantization
        }
    }

    private enum Entry {
        case mse(QuantizedVector)
        case prod(QuantizedVectorProd)
    }

    private var keys: [Entry] = []
    private var values: [Entry] = []
    private var B: Int = 0
    private var kvHeads: Int = 0
    private var headDim: Int = 0

    private let config: Config

    public init(config: Config) {
        self.config = config
        super.init()
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let b = keys.dim(0)
        let h = keys.dim(1)
        let l = keys.dim(2)
        let d = keys.dim(3)

        if self.offset == 0 {
            self.B = b
            self.kvHeads = h
            self.headDim = d
        }

        let kArr = keys.asArray(Float.self)
        let vArr = values.asArray(Float.self)

        // Quantize per (b, h, t) vector of length headDim
        for bb in 0..<b {
            for hh in 0..<h {
                for tt in 0..<l {
                    let base = ((bb * h + hh) * l + tt) * d
                    let sliceK = Array(kArr[base..<(base + d)])
                    let sliceV = Array(vArr[base..<(base + d)])

                    switch config.quantization.mode {
                    case .mse:
                        let qk = TurboQuantMSE(
                            bitWidth: config.quantization.bitWidth,
                            dimension: d,
                            rotationEnabled: config.quantization.rotationEnabled,
                            rotationSeed: config.quantization.rotationSeed
                        ).quantize(sliceK)
                        let qv = TurboQuantMSE(
                            bitWidth: config.quantization.bitWidth,
                            dimension: d,
                            rotationEnabled: config.quantization.rotationEnabled,
                            rotationSeed: config.quantization.rotationSeed
                        ).quantize(sliceV)
                        self.keys.append(.mse(qk))
                        self.values.append(.mse(qv))
                    case .prod:
                        let qk = TurboQuantProd(
                            bitWidth: config.quantization.bitWidth,
                            dimension: d,
                            rotationEnabled: config.quantization.rotationEnabled,
                            rotationSeed: config.quantization.rotationSeed,
                            qjlSeed: config.quantization.qjlSeed
                        ).quantize(sliceK)
                        let qv = TurboQuantProd(
                            bitWidth: config.quantization.bitWidth,
                            dimension: d,
                            rotationEnabled: config.quantization.rotationEnabled,
                            rotationSeed: config.quantization.rotationSeed,
                            qjlSeed: config.quantization.qjlSeed
                        ).quantize(sliceV)
                        self.keys.append(.prod(qk))
                        self.values.append(.prod(qv))
                    }
                }
            }
        }

        self.offset += l

        // Dequantize full cache to return
        let totalTokens = self.offset
        let total = B * kvHeads * totalTokens * headDim
        var outKeys = [Float](repeating: 0, count: total)
        var outValues = [Float](repeating: 0, count: total)

        for bb in 0..<B {
            for hh in 0..<kvHeads {
                for tt in 0..<totalTokens {
                    let idx = ((bb * kvHeads + hh) * totalTokens + tt)
                    let base = idx * headDim

                    let kVec: [Float]
                    let vVec: [Float]

                    switch keys[idx] {
                    case .mse(let q):
                        kVec = TurboQuantMSE(
                            bitWidth: q.bitWidth,
                            dimension: q.dimension,
                            rotationEnabled: q.rotationEnabled,
                            rotationSeed: q.rotationSeed
                        ).dequantize(q)
                    case .prod(let q):
                        kVec = TurboQuantProd(
                            bitWidth: q.bitWidth,
                            dimension: q.dimension,
                            rotationEnabled: q.rotationEnabled,
                            rotationSeed: q.rotationSeed,
                            qjlSeed: q.qjlSeed
                        ).dequantize(q)
                    }

                    switch values[idx] {
                    case .mse(let q):
                        vVec = TurboQuantMSE(
                            bitWidth: q.bitWidth,
                            dimension: q.dimension,
                            rotationEnabled: q.rotationEnabled,
                            rotationSeed: q.rotationSeed
                        ).dequantize(q)
                    case .prod(let q):
                        vVec = TurboQuantProd(
                            bitWidth: q.bitWidth,
                            dimension: q.dimension,
                            rotationEnabled: q.rotationEnabled,
                            rotationSeed: q.rotationSeed,
                            qjlSeed: q.qjlSeed
                        ).dequantize(q)
                    }

                    for i in 0..<headDim {
                        outKeys[base + i] = kVec[i]
                        outValues[base + i] = vVec[i]
                    }
                }
            }
        }

        let outK = MLXArray(outKeys).reshaped([B, kvHeads, totalTokens, headDim])
        let outV = MLXArray(outValues).reshaped([B, kvHeads, totalTokens, headDim])

        return (outK, outV)
    }

    public override var state: [MLXArray] {
        get { [] }
        set { }
    }

    public override var metaState: [String] {
        get { [String(offset), String(B), String(kvHeads), String(headDim)] }
        set { }
    }

    public override var isTrimmable: Bool { false }

    @discardableResult
    public override func trim(_ n: Int) -> Int { 0 }

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(config: config)
        new.offset = self.offset
        new.B = self.B
        new.kvHeads = self.kvHeads
        new.headDim = self.headDim
        new.keys = self.keys
        new.values = self.values
        return new
    }
}

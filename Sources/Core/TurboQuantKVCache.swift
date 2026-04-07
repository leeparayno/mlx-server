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
    private var dequantKeys: MLXArray?
    private var dequantValues: MLXArray?
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

        // Prebuild quantizers for this head dimension
        let mseQ = TurboQuantMSE(
            bitWidth: config.quantization.bitWidth,
            dimension: d,
            rotationEnabled: config.quantization.rotationEnabled,
            rotationSeed: config.quantization.rotationSeed
        )
        let prodQ = TurboQuantProd(
            bitWidth: config.quantization.bitWidth,
            dimension: d,
            rotationEnabled: config.quantization.rotationEnabled,
            rotationSeed: config.quantization.rotationSeed,
            qjlSeed: config.quantization.qjlSeed
        )

        // Quantize per (b, h, t) vector of length headDim
        for bb in 0..<b {
            for hh in 0..<h {
                for tt in 0..<l {
                    let base = ((bb * h + hh) * l + tt) * d
                    let sliceK = Array(kArr[base..<(base + d)])
                    let sliceV = Array(vArr[base..<(base + d)])

                    switch config.quantization.mode {
                    case .mse:
                        let qk = mseQ.quantize(sliceK)
                        let qv = mseQ.quantize(sliceV)
                        self.keys.append(.mse(qk))
                        self.values.append(.mse(qv))
                    case .prod:
                        let qk = prodQ.quantize(sliceK)
                        let qv = prodQ.quantize(sliceV)
                        self.keys.append(.prod(qk))
                        self.values.append(.prod(qv))
                    }
                }
            }
        }

        self.offset += l

        // Dequantize only the newly added tokens and append
        let newTotal = b * h * l * d
        var outKeysNew = [Float](repeating: 0, count: newTotal)
        var outValuesNew = [Float](repeating: 0, count: newTotal)

        // entries are appended in order (bb,hh,tt)
        var writeIndex = 0

        for bb in 0..<B {
            for hh in 0..<kvHeads {
                for tt in (self.offset - l)..<self.offset {
                    let idx = ((bb * kvHeads + hh) * self.offset + tt)

                    let kVec: [Float]
                    let vVec: [Float]

                    switch keys[idx] {
                    case .mse(let q):
                        kVec = mseQ.dequantize(q)
                    case .prod(let q):
                        kVec = prodQ.dequantize(q)
                    }

                    switch values[idx] {
                    case .mse(let q):
                        vVec = mseQ.dequantize(q)
                    case .prod(let q):
                        vVec = prodQ.dequantize(q)
                    }

                    for i in 0..<headDim {
                        outKeysNew[writeIndex + i] = kVec[i]
                        outValuesNew[writeIndex + i] = vVec[i]
                    }
                    writeIndex += headDim
                }
            }
        }

        let newK = MLXArray(outKeysNew).reshaped([B, kvHeads, l, headDim])
        let newV = MLXArray(outValuesNew).reshaped([B, kvHeads, l, headDim])

        if let currentK = dequantKeys, let currentV = dequantValues {
            dequantKeys = concatenated([currentK, newK], axis: 2)
            dequantValues = concatenated([currentV, newV], axis: 2)
        } else {
            dequantKeys = newK
            dequantValues = newV
        }

        return (dequantKeys!, dequantValues!)
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
        new.dequantKeys = self.dequantKeys
        new.dequantValues = self.dequantValues
        return new
    }
}

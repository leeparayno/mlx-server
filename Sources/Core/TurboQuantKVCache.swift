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
    private let step: Int = 256

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
        // Also dequantize immediately for MSE mode to avoid a second pass
        let newTotal = b * h * l * d
        var outKeysNew = [Float](repeating: 0, count: newTotal)
        var outValuesNew = [Float](repeating: 0, count: newTotal)
        var writeIndex = 0

        // Fast batch MSE path (rotation disabled) using midpoints
        if config.quantization.mode == .mse && !config.quantization.rotationEnabled {
            let centroids = mseQ.centroids
            var midpoints: [Float] = []
            midpoints.reserveCapacity(centroids.count - 1)
            for i in 0..<(centroids.count - 1) {
                midpoints.append((centroids[i] + centroids[i + 1]) * 0.5)
            }

            func quantizeVector(_ slice: ArraySlice<Float>) -> (indices: [UInt8], deq: [Float], norm: Float) {
                var norm: Float = 0
                for v in slice { norm += v * v }
                norm = sqrt(norm)
                let inv = norm > 0 ? 1.0 / norm : 1.0

                var indices: [UInt8] = []
                indices.reserveCapacity(slice.count)
                var deq: [Float] = []
                deq.reserveCapacity(slice.count)

                for v in slice {
                    let x = v * inv
                        while idx < midpoints.count && x > midpoints[idx] { idx += 1 }
                    indices.append(UInt8(idx))
                    deq.append(centroids[idx] * norm)
                }
                return (indices, deq, norm)
            }

            for bb in 0..<b {
                for hh in 0..<h {
                    for tt in 0..<l {
                        let base = ((bb * h + hh) * l + tt) * d
                        let sliceK = kArr[base..<(base + d)]
                        let sliceV = vArr[base..<(base + d)]

                        let qk = quantizeVector(sliceK)
                        let qv = quantizeVector(sliceV)

                        self.keys.append(.mse(QuantizedVector(indices: qk.indices, norm: qk.norm, bitWidth: config.quantization.bitWidth, dimension: d, rotationEnabled: false, rotationSeed: config.quantization.rotationSeed)))
                        self.values.append(.mse(QuantizedVector(indices: qv.indices, norm: qv.norm, bitWidth: config.quantization.bitWidth, dimension: d, rotationEnabled: false, rotationSeed: config.quantization.rotationSeed)))

                        for i in 0..<headDim {
                            outKeysNew[writeIndex + i] = qk.deq[i]
                            outValuesNew[writeIndex + i] = qv.deq[i]
                        }
                        writeIndex += headDim
                    }
                }
            }
        } else {
            // Vectorized Hadamard paths
            if config.quantization.mode == .mse && config.quantization.rotationEnabled {
                let n = b * h * l
                let kMatrix = MLXArray(kArr).reshaped([n, d])
                let vMatrix = MLXArray(vArr).reshaped([n, d])

                // Norms and normalization
                let kNorm = sqrt(sum(kMatrix * kMatrix, axis: -1))
                let vNorm = sqrt(sum(vMatrix * vMatrix, axis: -1))
                let kNormed = kMatrix / kNorm.reshaped([n, 1])
                let vNormed = vMatrix / vNorm.reshaped([n, 1])

                // Rotate
                let kRot = TurboQuantRotation.applyBatch(kNormed, seed: config.quantization.rotationSeed)
                let vRot = TurboQuantRotation.applyBatch(vNormed, seed: config.quantization.rotationSeed)

                // Quantize rotated using midpoints
                let centroids = mseQ.centroids
                var midpoints: [Float] = []
                midpoints.reserveCapacity(centroids.count - 1)
                for i in 0..<(centroids.count - 1) {
                    midpoints.append((centroids[i] + centroids[i + 1]) * 0.5)
                }

                let kRotArr = kRot.asArray(Float.self)
                let vRotArr = vRot.asArray(Float.self)
                let kNormArr = kNorm.asArray(Float.self)
                let vNormArr = vNorm.asArray(Float.self)

                // Build indices and dequantized rotated
                var deqKRot = [Float](repeating: 0, count: newTotal)
                var deqVRot = [Float](repeating: 0, count: newTotal)

                for i in 0..<n {
                    let start = i * d
                    var idxsK = [UInt8]()
                    var idxsV = [UInt8]()
                    idxsK.reserveCapacity(d)
                    idxsV.reserveCapacity(d)

                    for j in 0..<d {
                        let xk = kRotArr[start + j]
                        let xv = vRotArr[start + j]

                        var ik = 0
                        while ik < midpoints.count && xk > midpoints[ik] { ik += 1 }
                        var iv = 0
                        while iv < midpoints.count && xv > midpoints[iv] { iv += 1 }

                        idxsK.append(UInt8(ik))
                        idxsV.append(UInt8(iv))

                        deqKRot[start + j] = centroids[ik]
                        deqVRot[start + j] = centroids[iv]
                    }

                    let qk = QuantizedVector(
                        indices: idxsK,
                        norm: kNormArr[i],
                        bitWidth: config.quantization.bitWidth,
                        dimension: d,
                        rotationEnabled: true,
                        rotationSeed: config.quantization.rotationSeed
                    )
                    let qv = QuantizedVector(
                        indices: idxsV,
                        norm: vNormArr[i],
                        bitWidth: config.quantization.bitWidth,
                        dimension: d,
                        rotationEnabled: true,
                        rotationSeed: config.quantization.rotationSeed
                    )
                    self.keys.append(.mse(qk))
                    self.values.append(.mse(qv))
                }

                // Dequantize: inverse rotate and rescale by norms
                let kDeq = TurboQuantRotation.applyBatchInverse(MLXArray(deqKRot).reshaped([n, d]), seed: config.quantization.rotationSeed)
                let vDeq = TurboQuantRotation.applyBatchInverse(MLXArray(deqVRot).reshaped([n, d]), seed: config.quantization.rotationSeed)

                let outK = kDeq * kNorm.reshaped([n, 1])
                let outV = vDeq * vNorm.reshaped([n, 1])

                let outKArr = outK.asArray(Float.self)
                let outVArr = outV.asArray(Float.self)
                for i in 0..<newTotal {
                    outKeysNew[i] = outKArr[i]
                    outValuesNew[i] = outVArr[i]
                }

                writeIndex = newTotal
            } else if config.quantization.mode == .prod && config.quantization.rotationEnabled {
                let n = b * h * l
                let kMatrix = MLXArray(kArr).reshaped([n, d])
                let vMatrix = MLXArray(vArr).reshaped([n, d])

                // MSE quantization per vector (still scalar)
                var mseDeqK = [Float](repeating: 0, count: newTotal)
                var mseDeqV = [Float](repeating: 0, count: newTotal)
                var qkList: [QuantizedVector] = []
                var qvList: [QuantizedVector] = []
                qkList.reserveCapacity(n)
                qvList.reserveCapacity(n)

                for bb in 0..<b {
                    for hh in 0..<h {
                        for tt in 0..<l {
                            let base = ((bb * h + hh) * l + tt) * d
                            let sliceK = Array(kArr[base..<(base + d)])
                            let sliceV = Array(vArr[base..<(base + d)])

                            let qk = mseQ.quantize(sliceK)
                            let qv = mseQ.quantize(sliceV)
                            qkList.append(qk)
                            qvList.append(qv)

                            let dk = mseQ.dequantize(qk)
                            let dv = mseQ.dequantize(qv)
                            for i in 0..<headDim {
                                mseDeqK[base + i] = dk[i]
                                mseDeqV[base + i] = dv[i]
                            }
                        }
                    }
                }

                // Residuals
                let resK = (kMatrix - MLXArray(mseDeqK).reshaped([n, d]))
                let resV = (vMatrix - MLXArray(mseDeqV).reshaped([n, d]))

                // Norms
                let resKNorm = sqrt(sum(resK * resK, axis: -1))
                let resVNorm = sqrt(sum(resV * resV, axis: -1))

                let resKNormed = resK / resKNorm.reshaped([n, 1])
                let resVNormed = resV / resVNorm.reshaped([n, 1])

                // QJL: Hadamard + sign
                let qjlK = TurboQuantRotation.applyBatch(resKNormed, seed: config.quantization.qjlSeed)
                let qjlV = TurboQuantRotation.applyBatch(resVNormed, seed: config.quantization.qjlSeed)
                let signsK = qjlK .>= 0
                let signsV = qjlV .>= 0

                // Dequantize residual via inverse Hadamard
                let signKFloat = (signsK * 2) - 1
                let signVFloat = (signsV * 2) - 1
                let resKRec = TurboQuantRotation.applyBatchInverse(signKFloat, seed: config.quantization.qjlSeed)
                let resVRec = TurboQuantRotation.applyBatchInverse(signVFloat, seed: config.quantization.qjlSeed)

                let scale = sqrt(Float.pi / 2.0) / Float(d)
                let resKScaled = resKRec * resKNorm.reshaped([n, 1]) * scale
                let resVScaled = resVRec * resVNorm.reshaped([n, 1]) * scale

                let outK = MLXArray(mseDeqK).reshaped([n, d]) + resKScaled
                let outV = MLXArray(mseDeqV).reshaped([n, d]) + resVScaled

                let outKArr = outK.asArray(Float.self)
                let outVArr = outV.asArray(Float.self)
                let signsKArr = signsK.asArray(Bool.self)
                let signsVArr = signsV.asArray(Bool.self)
                let resKNormArr = resKNorm.asArray(Float.self)
                let resVNormArr = resVNorm.asArray(Float.self)

                // store prod entries
                for i in 0..<n {
                    let start = i * d
                    let qk = qkList[i]
                    let qv = qvList[i]

                    let signBase = i * d
                    let qjlSignsK = signsKArr[signBase..<(signBase + d)].map { $0 ? UInt8(1) : UInt8(0) }
                    let qjlSignsV = signsVArr[signBase..<(signBase + d)].map { $0 ? UInt8(1) : UInt8(0) }

                    let kProd = QuantizedVectorProd(
                        indices: qk.indices,
                        norm: qk.norm,
                        bitWidth: config.quantization.bitWidth,
                        dimension: d,
                        rotationEnabled: true,
                        rotationSeed: config.quantization.rotationSeed,
                        qjlSigns: Array(qjlSignsK),
                        residualNorm: resKNormArr[i],
                        qjlSeed: config.quantization.qjlSeed
                    )
                    let vProd = QuantizedVectorProd(
                        indices: qv.indices,
                        norm: qv.norm,
                        bitWidth: config.quantization.bitWidth,
                        dimension: d,
                        rotationEnabled: true,
                        rotationSeed: config.quantization.rotationSeed,
                        qjlSigns: Array(qjlSignsV),
                        residualNorm: resVNormArr[i],
                        qjlSeed: config.quantization.qjlSeed
                    )

                    self.keys.append(.prod(kProd))
                    self.values.append(.prod(vProd))

                    for j in 0..<d {
                        outKeysNew[start + j] = outKArr[start + j]
                        outValuesNew[start + j] = outVArr[start + j]
                    }
                }

                writeIndex = newTotal
            } else {
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

                                // inline dequantize
                                let kVec = mseQ.dequantize(qk)
                                let vVec = mseQ.dequantize(qv)
                                for i in 0..<headDim {
                                    outKeysNew[writeIndex + i] = kVec[i]
                                    outValuesNew[writeIndex + i] = vVec[i]
                                }
                            case .prod:
                                let qk = prodQ.quantize(sliceK)
                                let qv = prodQ.quantize(sliceV)
                                self.keys.append(.prod(qk))
                                self.values.append(.prod(qv))

                                // prod dequant (still required for attention)
                                let kVec = prodQ.dequantize(qk)
                                let vVec = prodQ.dequantize(qv)
                                for i in 0..<headDim {
                                    outKeysNew[writeIndex + i] = kVec[i]
                                    outValuesNew[writeIndex + i] = vVec[i]
                                }
                            }

                            writeIndex += headDim
                        }
                    }
                }
            }
        }

        self.offset += l

        let newK = MLXArray(outKeysNew).reshaped([B, kvHeads, l, headDim])
        let newV = MLXArray(outValuesNew).reshaped([B, kvHeads, l, headDim])

        // Expand dequant buffers if needed (like KVCacheSimple)
        let prev = self.offset - l
        if dequantKeys == nil || (prev + l) > dequantKeys!.dim(2) {
            let newSteps = ((step + l - 1) / step) * step
            let shape = [B, kvHeads, newSteps, headDim]
            let zerosK = MLXArray.zeros(shape, dtype: newK.dtype)
            let zerosV = MLXArray.zeros(shape, dtype: newV.dtype)

            if let currentK = dequantKeys, let currentV = dequantValues {
                let trimmedK = currentK[.ellipsis, ..<prev, 0...]
                let trimmedV = currentV[.ellipsis, ..<prev, 0...]
                dequantKeys = concatenated([trimmedK, zerosK], axis: 2)
                dequantValues = concatenated([trimmedV, zerosV], axis: 2)
            } else {
                dequantKeys = zerosK
                dequantValues = zerosV
            }
        }

        dequantKeys?[.ellipsis, prev ..< self.offset, 0...] = newK
        dequantValues?[.ellipsis, prev ..< self.offset, 0...] = newV

        let outK = dequantKeys![.ellipsis, ..<self.offset, 0...]
        let outV = dequantValues![.ellipsis, ..<self.offset, 0...]

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
        new.dequantKeys = self.dequantKeys
        new.dequantValues = self.dequantValues
        return new
    }
}

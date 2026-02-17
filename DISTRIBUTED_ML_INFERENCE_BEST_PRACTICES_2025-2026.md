# Distributed ML Inference Best Practices (2025-2026)

**Research Date:** February 15, 2026
**Focus Areas:** Continuous batching, PagedAttention, RDMA networking, Swift HPC, MLX distributed computing

---

## Executive Summary

This document synthesizes current best practices for high-performance distributed ML inference based on 2025-2026 research from production systems (vLLM, TensorRT-LLM), academic papers, and official documentation. Key findings focus on memory-efficient serving through PagedAttention, low-latency RDMA networking for Apple Silicon clusters, and distributed coordination patterns.

---

## 1. Distributed ML Inference Architecture

### 1.1 Continuous Batching

**Definition:** Continuous (or "dynamic") batching allows new requests to join existing batches mid-execution, maximizing GPU utilization compared to static batching.

**Key Implementations:**

#### vLLM Continuous Batching (Source: Official docs)
- **Core principle:** Requests are batched dynamically as they arrive, rather than waiting for fixed batch boundaries
- **Benefit:** Achieves 2-4× higher throughput vs. static batching systems (FasterTransformer, Orca)
- **Integration:** Works seamlessly with PagedAttention for memory management

**Architecture Pattern:**
```python
from vllm import LLM

# Continuous batching is enabled by default
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    max_num_batched_tokens=8192,  # Controls batch size
    max_num_seqs=256  # Max concurrent sequences
)

# System automatically batches incoming requests
outputs = llm.generate(prompts, sampling_params)
```

**Recent Research (Feb 2026):**
- **OServe** (arXiv:2602.12151): Introduces spatial-temporal workload orchestration
  - Handles heterogeneous requests (different compute/memory needs)
  - Adaptive switching for changing workload patterns
  - **Results:** 2× average improvement, up to 1.5× on real-world traces

### 1.2 PagedAttention for KV Cache Management

**Innovation:** Treats KV cache like OS virtual memory with fixed-size pages, eliminating fragmentation.

**Source:** "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)

#### Core Architecture

**Memory Organization:**
```python
# KV cache structure in vLLM
def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata: AttentionMetadata,
            kv_cache: torch.Tensor,
            output: torch.Tensor):

    # Split into key and value caches
    key_cache, value_cache = PagedAttention.split_kv_cache(
        kv_cache,
        self.num_kv_heads,
        self.head_size
    )

    # Handle FP8 quantized cache
    if self.kv_cache_dtype.startswith("fp8"):
        key_cache = key_cache.view(self.fp8_dtype)
        value_cache = value_cache.view(self.fp8_dtype)

    # Compute attention with paged cache
    chunked_prefill_paged_decode(
        query=query,
        key=key,
        value=value,
        output=output,
        kv_cache_dtype=self.kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=attn_metadata.block_table,
        query_start_loc=attn_metadata.query_start_loc,
        seq_lens=attn_metadata.seq_lens,
        max_seq_len=attn_metadata.max_seq_len,
        max_query_len=attn_metadata.max_query_len,
        sliding_window=self.sliding_window[0],
        sm_scale=self.scale
    )

    return output
```

**Key Benefits:**
- **Near-zero memory waste** through paging
- **Flexible sharing** within and across requests
- **2-4× throughput improvement** vs. traditional approaches

**KV Cache Configuration:**
```python
# Automatic memory planning
global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

# Project to workers
projected_groups_per_worker = [
    _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
    for worker_spec in kv_cache_specs
]

# Auto-fit max model length if not specified
if vllm_config.model_config.original_max_model_len == -1:
    _auto_fit_max_model_len(
        vllm_config,
        projected_groups_per_worker,
        available_memory
    )

# Shrink to minimum blocks across workers
min_num_blocks = min(
    kv_cache_config.num_blocks
    for kv_cache_config in kv_cache_configs
)
```

### 1.3 Advanced KV Cache Optimization (2026)

#### PrefillShare Architecture (arXiv:2602.12029)

**Problem:** Multi-agent systems repeatedly process identical prompts across different models, wasting compute and memory.

**Solution:** Disaggregated architecture with shared prefill module
```
┌─────────────────┐
│ Shared Prefill  │  ← Frozen weights, generates KV cache once
│     Module      │
└────────┬────────┘
         │ KV Cache
    ┌────┼────┬────────┐
    ▼    ▼    ▼        ▼
┌─────┐ ┌─────┐ ┌─────┐
│Decode│ │Decode│ │Decode│ ← Fine-tuned per task
│ LLM 1│ │ LLM 2│ │ LLM 3│
└─────┘ └─────┘ └─────┘
```

**Implementation Pattern:**
- Factorize model into prefill + decode modules
- Freeze prefill module (shared)
- Fine-tune only decode modules (task-specific)
- Separate GPUs for prefill/decode to minimize interference

**Results:**
- **4.5× lower p95 latency**
- **3.9× higher throughput**
- Full fine-tuning accuracy maintained

### 1.4 Multi-Node Tensor Parallelism

**Tensor Parallelism (TP):** Shards model weights across GPUs within layers

**Pipeline Parallelism (PP):** Distributes different layers across GPUs

**Combined Pattern (for 70B+ models):**
```python
from vllm import LLM

# 8 GPUs total: 4-way TP × 2-way PP
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,      # Split weights 4 ways
    pipeline_parallel_size=2,    # Split layers 2 ways
    distributed_executor_backend="ray"
)
```

**Multi-Node Configuration:**
```bash
# Launch across 2 nodes with 4 GPUs each
vllm serve meta-llama/Llama-3.3-70B-Instruct \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2 \
     --distributed-executor-backend ray \
     --master-addr 192.168.1.100 \
     --master-port 29501 \
     --nnodes 2 \
     --node-rank 0  # Run on each node with appropriate rank
```

**Configuration Parameters (vLLM):**
```python
{
    "nnodes": 2,                          # Number of nodes
    "node_rank": 0,                        # Rank of this node
    "tensor_parallel_size": 4,             # TP degree
    "pipeline_parallel_size": 2,           # PP degree
    "prefill_context_parallel_size": 1,    # Context parallelism
    "data_parallel_size": 1,               # Data parallelism
    "distributed_executor_backend": "ray", # or "mp" for multiprocessing
    "master_addr": "192.168.1.100",       # Primary node address
    "master_port": 29501                   # Coordination port
}
```

**Ray Integration for Multi-Node:**
```python
# Ray placement groups for worker coordination
placement_group = None  # Auto-managed by vLLM
ray_runtime_env = None  # Optional environment config

# World size calculation
world_size = tensor_parallel_size × pipeline_parallel_size
world_size_across_dp = world_size × data_parallel_size
```

**Best Practices:**
1. **Start with TP only** for models that fit
2. **Add PP** when TP alone is insufficient
3. **Use Ray** for multi-node (more flexible than multiprocessing)
4. **Monitor memory** with `--max-model-len` to prevent OOM
5. **Test locally first** with `--nnodes 1` before scaling

---

## 2. RDMA Networking for Low-Latency Inference

### 2.1 RDMA Fundamentals

**RDMA (Remote Direct Memory Access):** Network protocol enabling zero-copy data transfer directly between memory of remote systems, bypassing CPU and OS kernel.

**Key Advantages:**
- **Zero-copy:** Data transfers without CPU involvement
- **Kernel bypass:** Direct hardware access reduces latency
- **Ultra-low latency:** Sub-microsecond network latency possible
- **High bandwidth:** Saturates link speeds efficiently

**vs. Traditional TCP/IP:**
```
Traditional:           RDMA:
App → Socket          App → RDMA Verbs
  ↓                     ↓
Kernel                NIC (Direct)
  ↓                     ↓
NIC                   Remote Memory
  ↓
Network
  ↓
Remote NIC
  ↓
Kernel
  ↓
Remote App
```

### 2.2 RDMA over Thunderbolt for Mac Studio Clusters

**MLX + JACCL Integration**

**JACCL (Jack and Angelos' Collective Communication Library):** Purpose-built for low-latency RDMA communication between Apple Silicon Macs via Thunderbolt.

**Requirements:**
- macOS 26.2 or later
- Thunderbolt 4/5 cables
- Fully connected mesh topology (each Mac connected to all others)
- RDMA enabled in recovery mode

**Setup Process:**

1. **Enable RDMA (per Mac):**
```bash
# Boot into recovery mode
# Open Terminal from Utilities menu
rdma_ctl enable
# Reboot
```

2. **Verify RDMA devices:**
```bash
ibv_devices
# Should show Thunderbolt RDMA devices
```

3. **Configure MLX with JACCL:**
```python
import mlx.core as mx

# Initialize with JACCL backend
world = mx.distributed.init(backend="jaccl")
print(f"Rank: {world.rank()}, Size: {world.size()}")

# Collective operations
local_data = mx.ones(10)
global_sum = mx.distributed.all_sum(local_data)
gathered = mx.distributed.all_gather(mx.array([world.rank()]))
```

4. **Launch distributed job:**
```bash
# With JACCL backend and hostfile
mlx.launch --verbose \
           --backend jaccl \
           --hostfile m3-ultra-jaccl.json \
           --env MLX_METAL_FAST_SYNCH=1 \
           -- python distributed_inference.py

# Hostfile format (m3-ultra-jaccl.json):
{
  "hosts": [
    {"hostname": "mac-studio-1.local", "slots": 1},
    {"hostname": "mac-studio-2.local", "slots": 1},
    {"hostname": "mac-studio-3.local", "slots": 1}
  ]
}
```

**Performance Optimization:**
```bash
# Environment variables for optimal performance
export MLX_METAL_FAST_SYNCH=1  # Fast CPU-GPU sync
export MLX_IBV_DEVICES=mlx5_0  # Specific RDMA device
```

### 2.3 Zero-Copy Network Architecture Patterns

**Traditional (Copy-Heavy):**
```python
# Data copies: GPU → CPU → Network → CPU → GPU
def send_traditional(tensor):
    cpu_tensor = tensor.cpu()           # Copy 1: GPU → CPU
    network_buffer = serialize(cpu_tensor)  # Copy 2: Tensor → Buffer
    socket.send(network_buffer)          # Copy 3: Buffer → Network
```

**RDMA Zero-Copy:**
```python
# Direct GPU memory access over network
def send_rdma(tensor):
    # Pin GPU memory for RDMA
    rdma_buffer = register_memory(tensor.data_ptr(), tensor.nbytes)

    # Direct network transfer (no copies)
    rdma_post_send(rdma_buffer, remote_key, remote_addr)

    # GPU memory directly visible to remote GPU
```

**MLX with JACCL (Zero-Copy Pattern):**
```python
import mlx.core as mx

# Metal-backed tensor (GPU memory)
local_tensor = mx.random.normal(shape=(1000, 1000))

# Zero-copy collective communication
# Data stays in GPU memory across network
global_tensor = mx.distributed.all_reduce(
    local_tensor,
    op="sum"  # or "max", "min", "prod"
)

# All-gather without copies
gathered = mx.distributed.all_gather(local_tensor)
```

**Communication Topology:**
```
Mac Studio 1 (M2 Ultra)     Mac Studio 2 (M2 Ultra)
┌─────────────────┐         ┌─────────────────┐
│ GPU Memory      │◄────────┤ GPU Memory      │
│  RDMA-mapped    │ Thunder-│  RDMA-mapped    │
└────────┬────────┘  bolt   └────────┬────────┘
         │            Link            │
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │ Mac Studio 3│
              │ (M2 Ultra)  │
              └─────────────┘
```

### 2.4 Low-Latency Communication Patterns

**Batched Collective Operations:**
```python
# Bad: Many small operations (high latency overhead)
for i in range(100):
    result = mx.distributed.all_reduce(small_tensor)

# Good: Batch communications
large_tensor = mx.concatenate([tensors...])
result = mx.distributed.all_reduce(large_tensor)
results = mx.split(result, num_splits=100)
```

**Overlapping Computation and Communication:**
```python
import mlx.core as mx

# Pipeline: compute on rank i while communicating rank i-1
async def distributed_inference(inputs):
    # Start communication early
    comm_future = mx.distributed.all_reduce_async(previous_result)

    # Compute while communication in flight
    current_result = model.forward(inputs)

    # Wait for communication to complete
    gathered = await comm_future

    return current_result, gathered
```

**Ring-based Reductions (for large data):**
```python
# RING backend: efficient for Thunderbolt/Ethernet
world = mx.distributed.init(backend="ring")

# Data flows in a ring: 0 → 1 → 2 → 3 → 0
# More efficient than all-to-all for large tensors
result = mx.distributed.all_reduce(large_tensor)
```

---

## 3. Swift High-Performance Computing

### 3.1 Swift Concurrency for ML Workloads

**Swift Actor Model:** Provides isolated, thread-safe state management critical for concurrent ML inference.

**Basic Pattern:**
```swift
actor InferenceQueue {
    private var currentBatch: [Tensor] = []
    private let batchSize: Int
    private let model: MLModel

    init(model: MLModel, batchSize: Int) {
        self.model = model
        self.batchSize = batchSize
    }

    // Actor-isolated: safe concurrent access
    func enqueue(_ input: Tensor) async -> Tensor {
        currentBatch.append(input)

        if currentBatch.count >= batchSize {
            let batch = currentBatch
            currentBatch = []
            return await processBatch(batch)
        }

        return await waitForBatch()
    }

    private func processBatch(_ batch: [Tensor]) async -> Tensor {
        // Safe: Actor ensures only one thread executes at a time
        return await model.predict(batch)
    }
}
```

**Parallel Inference with TaskGroup:**
```swift
func processRequests(_ requests: [Request]) async throws -> [Result] {
    try await withThrowingTaskGroup(of: Result.self) { group in
        for request in requests {
            group.addTask {
                // Each request processed concurrently
                return await inferenceQueue.enqueue(request.tensor)
            }
        }

        var results: [Result] = []
        for try await result in group {
            results.append(result)
        }
        return results
    }
}
```

### 3.2 Swift/Metal Integration Patterns

**Metal Compute Pipeline for ML:**
```swift
import Metal
import MetalPerformanceShaders

class MetalInferenceEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let computePipeline: MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw InferenceError.noMetalDevice
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Load compute shader
        let library = device.makeDefaultLibrary()!
        let function = library.makeFunction(name: "matmul_kernel")!
        self.computePipeline = try device.makeComputePipelineState(function: function)
    }

    // Zero-copy buffer allocation
    func allocateBuffer<T>(count: Int) -> MTLBuffer? {
        let size = count * MemoryLayout<T>.stride
        return device.makeBuffer(
            length: size,
            options: [.storageModeShared]  // Shared CPU/GPU memory
        )
    }

    // Async compute dispatch
    func compute(input: MTLBuffer, output: MTLBuffer) async throws {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(computePipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        let gridSize = MTLSize(width: 1024, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: 32, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

        encoder.endEncoding()
        commandBuffer.commit()

        await commandBuffer.waitUntilCompleted()
    }
}
```

### 3.3 Swift Networking with UnsafePointers for Zero-Copy

**Traditional Approach (Copies):**
```swift
// Multiple copies: Tensor → Data → Network
func send(tensor: Tensor) async throws {
    let data = Data(tensor.bytes)  // Copy 1
    try await connection.send(data) // Copy 2 (internal)
}
```

**Zero-Copy Pattern:**
```swift
import Network

class ZeroCopyNetworking {
    let connection: NWConnection

    // Send tensor data without intermediate copies
    func sendZeroCopy(tensor: UnsafeMutableRawPointer, byteCount: Int) async throws {
        // Create dispatch data referencing existing memory
        let dispatchData = DispatchData(
            bytesNoCopy: UnsafeRawBufferPointer(
                start: tensor,
                count: byteCount
            ),
            deallocator: .custom(nil, { /* tensor cleanup */ })
        )

        // Send without copying
        try await connection.sendData(dispatchData)
    }

    // Receive directly into preallocated buffer
    func receiveZeroCopy(into buffer: UnsafeMutableRawPointer,
                        byteCount: Int) async throws {
        let data = try await connection.receive(minimumIncompleteLength: byteCount,
                                                maximumLength: byteCount)

        // Copy directly into destination (single copy)
        data.copyBytes(to: buffer.assumingMemoryBound(to: UInt8.self),
                      count: byteCount)
    }
}
```

**MLX Swift Integration:**
```swift
import MLX
import MLXLinalg

actor DistributedMLXInference {
    let model: MLXModel
    let networking: ZeroCopyNetworking

    func distributedInference(input: MLXArray) async throws -> MLXArray {
        // Get raw pointer to MLX array data (zero-copy access)
        input.withUnsafeBytes { pointer in
            // Send over network without copying
            try await networking.sendZeroCopy(
                tensor: pointer.baseAddress!,
                byteCount: pointer.count
            )
        }

        // Receive result into preallocated MLX array
        let outputShape = [1024, 768]
        var output = MLXArray.zeros(outputShape)

        output.withUnsafeMutableBytes { pointer in
            try await networking.receiveZeroCopy(
                into: pointer.baseAddress!,
                byteCount: pointer.count
            )
        }

        return output
    }
}
```

### 3.4 Swift Performance Patterns for ML

**Actor Isolation for Thread Safety:**
```swift
actor KVCacheManager {
    private var cache: [String: MLXArray] = [:]
    private var lruQueue: [String] = []
    private let maxSize: Int

    func get(key: String) -> MLXArray? {
        return cache[key]
    }

    func set(key: String, value: MLXArray) {
        if cache.count >= maxSize {
            evictLRU()
        }
        cache[key] = value
        lruQueue.append(key)
    }

    private func evictLRU() {
        guard let oldest = lruQueue.first else { return }
        cache.removeValue(forKey: oldest)
        lruQueue.removeFirst()
    }
}

// Usage: Thread-safe by design
let cacheManager = KVCacheManager(maxSize: 1000)
await cacheManager.set(key: "prompt_123", value: kvCache)
let cached = await cacheManager.get(key: "prompt_123")
```

**Structured Concurrency for Batch Processing:**
```swift
func processBatches(_ batches: [[Tensor]]) async throws -> [Result] {
    try await withThrowingTaskGroup(of: Result.self) { group in
        for batch in batches {
            // Limit concurrency to avoid resource contention
            if group.isEmpty || group.count < maxConcurrentBatches {
                group.addTask {
                    return await inferenceEngine.process(batch)
                }
            }
        }

        return try await group.reduce(into: []) { $0.append($1) }
    }
}
```

---

## 4. MLX Distributed Computing

### 4.1 MLX Distributed API Overview

**Communication Backends:**

| Backend | Transport | Use Case | Latency | Setup |
|---------|-----------|----------|---------|-------|
| **RING** | TCP sockets | General-purpose, Thunderbolt/Ethernet | Medium | Always available |
| **JACCL** | RDMA over Thunderbolt | Low-latency Mac clusters | Ultra-low | macOS 26.2+, recovery mode |
| **MPI** | Message Passing Interface | HPC environments | Low | MPI installation required |
| **NCCL** | NVIDIA proprietary | CUDA multi-GPU | Ultra-low | NVIDIA GPUs only |

**Backend Selection:**
```python
import mlx.core as mx

# Auto-select (defaults to RING)
world = mx.distributed.init()

# Explicit backend selection
world_ring = mx.distributed.init(backend="ring")
world_jaccl = mx.distributed.init(backend="jaccl")  # Best for Mac clusters
world_mpi = mx.distributed.init(backend="mpi")
world_nccl = mx.distributed.init(backend="nccl")    # CUDA only
```

### 4.2 JACCL Backend Usage

**Prerequisites:**
1. macOS 26.2+ on all Macs
2. RDMA enabled (recovery mode): `rdma_ctl enable`
3. Fully connected Thunderbolt topology
4. Hostfile configuration

**Hostfile Format (JSON):**
```json
{
  "hosts": [
    {
      "hostname": "mac-studio-1.local",
      "slots": 1,
      "rdma_device": "mlx5_0"
    },
    {
      "hostname": "mac-studio-2.local",
      "slots": 1,
      "rdma_device": "mlx5_0"
    },
    {
      "hostname": "mac-studio-3.local",
      "slots": 1,
      "rdma_device": "mlx5_0"
    }
  ],
  "coordinator": "mac-studio-1.local:50051"
}
```

**Launch Distributed Job:**
```bash
# Launch with JACCL backend
mlx.launch --verbose \
           --backend jaccl \
           --hostfile cluster-config.json \
           --env MLX_METAL_FAST_SYNCH=1 \
           --env MLX_IBV_DEVICES=mlx5_0 \
           -- python distributed_training.py

# For local testing (2 processes)
mlx.launch --backend jaccl -n 2 test_script.py
```

**Environment Variables (Manual Launch):**
```bash
# Required for JACCL
export MLX_RANK=0                           # Process rank (0, 1, 2, ...)
export MLX_JACCL_COORDINATOR=192.168.1.1:50051  # Coordinator address
export MLX_IBV_DEVICES=mlx5_0               # RDMA device
export MLX_METAL_FAST_SYNCH=1               # Performance optimization
```

### 4.3 Multi-GPU/Multi-Node Coordination Patterns

**Collective Operations:**

```python
import mlx.core as mx
import mlx.nn as nn

class DistributedModel:
    def __init__(self):
        self.world = mx.distributed.init(backend="jaccl")
        self.rank = self.world.rank()
        self.size = self.world.size()

        # Shard model across GPUs
        self.model = self._create_sharded_model()

    def _create_sharded_model(self):
        # Each rank gets different layers or tensor shards
        if self.rank == 0:
            return nn.Linear(1024, 2048)
        else:
            return nn.Linear(2048, 1024)

    def forward(self, x):
        # Compute local forward pass
        local_output = self.model(x)

        # All-reduce to synchronize gradients
        global_output = mx.distributed.all_reduce(
            local_output,
            op="sum"
        )

        return global_output / self.size  # Average across ranks

    def gather_results(self, local_result):
        # Collect results from all ranks
        all_results = mx.distributed.all_gather(local_result)
        return all_results

# Usage
model = DistributedModel()
output = model.forward(input_tensor)
```

**Tensor Parallelism Pattern:**
```python
def tensor_parallel_linear(x, weight, bias, world):
    """
    Distribute linear layer computation across GPUs.
    Each GPU handles a shard of the output dimension.
    """
    rank = world.rank()
    size = world.size()

    # Shard weights across GPUs
    out_features = weight.shape[0]
    shard_size = out_features // size
    start_idx = rank * shard_size
    end_idx = (rank + 1) * shard_size

    # Local computation
    weight_shard = weight[start_idx:end_idx]
    bias_shard = bias[start_idx:end_idx] if bias is not None else None

    local_output = x @ weight_shard.T
    if bias_shard is not None:
        local_output += bias_shard

    # All-gather to reconstruct full output
    full_output = mx.distributed.all_gather(local_output)
    return mx.concatenate(full_output, axis=-1)
```

**Pipeline Parallelism Pattern:**
```python
class PipelineParallelModel:
    def __init__(self, layers, world):
        self.world = world
        self.rank = world.rank()
        self.size = world.size()

        # Distribute layers across ranks
        layers_per_rank = len(layers) // self.size
        start = self.rank * layers_per_rank
        end = (self.rank + 1) * layers_per_rank
        self.local_layers = layers[start:end]

    def forward(self, x):
        # Process through local layers
        for layer in self.local_layers:
            x = layer(x)

        # Send to next rank in pipeline
        if self.rank < self.size - 1:
            # Send to next stage
            mx.distributed.send(x, dest=self.rank + 1)
            return None
        else:
            # Last rank returns result
            return x

    def backward(self, grad):
        # Backward pass through pipeline (reversed)
        for layer in reversed(self.local_layers):
            grad = layer.backward(grad)

        # Send gradients to previous rank
        if self.rank > 0:
            mx.distributed.send(grad, dest=self.rank - 1)
```

### 4.4 MLX-Distributed Performance Optimization

**Batched Communication:**
```python
# Bad: Many small communications
for tensor in tensors:
    result = mx.distributed.all_reduce(tensor)

# Good: Batch into single communication
stacked = mx.stack(tensors)
result = mx.distributed.all_reduce(stacked)
results = mx.split(result, len(tensors))
```

**Topology-Aware Communication:**
```bash
# Visualize communication topology
mlx.distributed_config --dot > topology.dot
dot -Tpng topology.dot -o topology.png

# Optimize for ring topology
mlx.launch --backend ring --hostfile hosts.json script.py
```

**Memory-Efficient Gradient Accumulation:**
```python
def distributed_training_step(model, data, world, accumulation_steps=4):
    """
    Accumulate gradients over multiple steps before synchronizing.
    Reduces communication frequency.
    """
    accumulated_loss = 0

    for step in range(accumulation_steps):
        # Local forward + backward
        loss = model.forward(data[step])
        gradients = mx.grad(loss)
        accumulated_loss += loss.item()

        # Accumulate gradients locally
        if step < accumulation_steps - 1:
            # Don't synchronize yet
            continue

    # Synchronize once after accumulation
    for param_name, grad in gradients.items():
        synced_grad = mx.distributed.all_reduce(grad, op="sum")
        model.parameters[param_name] -= learning_rate * synced_grad / world.size()

    return accumulated_loss / accumulation_steps
```

**Overlapping Computation and Communication:**
```python
def overlap_compute_comm(model, inputs, world):
    """
    Start communication early while continuing computation.
    """
    # Compute layer 1
    hidden1 = model.layer1(inputs)

    # Start communication for layer 1 output (async)
    comm_handle = mx.distributed.all_reduce_async(hidden1)

    # Continue computing layer 2 while layer 1 communicates
    hidden2 = model.layer2(hidden1)

    # Wait for layer 1 communication to complete
    hidden1_synced = comm_handle.wait()

    # Use synchronized result
    output = model.layer3(hidden1_synced + hidden2)
    return output
```

---

## 5. Production System Architectures

### 5.1 vLLM Architecture (2025-2026)

**System Components:**
```
┌─────────────────────────────────────────────────────┐
│                   vLLM Engine                       │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Scheduler   │  │  KV Cache   │  │  Worker    │ │
│  │ (Continuous  │→ │  Manager    │→ │   Pool     │ │
│  │  Batching)   │  │ (PagedAttn) │  │ (TP/PP/DP) │ │
│  └──────────────┘  └─────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
           ↓                    ↓
    ┌─────────────┐      ┌──────────────┐
    │  Ray/MP     │      │   CUDA/HIP   │
    │  Cluster    │      │   Kernels    │
    └─────────────┘      └──────────────┘
```

**Key Features:**
- Continuous batching (dynamic request addition)
- PagedAttention (near-zero memory waste)
- Multiple parallelism strategies (TP, PP, DP)
- FlashAttention/FlashInfer integration
- Quantization support (FP8, INT4, AWQ, SmoothQuant)
- Speculative decoding

**Multi-Node Setup:**
```python
# Start vLLM server with distributed inference
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    pipeline_parallel_size=2,
    distributed_executor_backend="ray",
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.95
)

# Continuous batching handles incoming requests automatically
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

**Recent Updates (v0.15.1, Feb 2026):**
- Enhanced multi-device support (AMD, Intel, TPU, Huawei Ascend)
- Improved CUDA graph execution
- Better KV cache sharing across requests
- Optimized speculative decoding

### 5.2 TensorRT-LLM Architecture

**Design Philosophy:**
- PyTorch-based for modularity
- Custom attention kernels
- Advanced quantization (FP8, FP4, INT4 AWQ, INT8 SmoothQuant)
- Triton Inference Server integration

**Performance Highlights (2026):**
- **40,000+ tokens/sec** on NVIDIA B200 GPUs
- **1,000 TPS/user** with Llama 4 Maverick on Blackwell

**Architecture:**
```python
import tensorrt_llm as trtllm

# Build engine with optimizations
builder = trtllm.Builder()
network = builder.create_network()

# Enable PagedAttention for KV cache
network.plugin_config.set_paged_kv_cache(True)
network.plugin_config.set_tokens_per_block(64)

# Multi-GPU tensor parallelism
network.set_tensor_parallel(world_size=4)

# Build optimized engine
engine = builder.build_engine(network)

# Run inference
session = trtllm.InferenceSession(engine)
outputs = session.infer(input_ids, max_length=512)
```

---

## 6. Best Practices Summary

### 6.1 Memory Management

1. **Always use PagedAttention** for production LLM serving
2. **Set block size** to balance memory granularity and waste (typically 16-64 tokens)
3. **Enable KV cache sharing** when serving multiple similar requests
4. **Monitor memory usage** and adjust `max_model_len` dynamically
5. **Use quantization** (FP8/INT4) for larger batch sizes

### 6.2 Distributed Inference

1. **Start with tensor parallelism** (single-node multi-GPU)
2. **Add pipeline parallelism** only when TP insufficient
3. **Use Ray for multi-node** (more flexible than multiprocessing)
4. **Test locally first** before scaling to clusters
5. **Monitor network bandwidth** and latency between nodes

### 6.3 RDMA Networking (Mac Clusters)

1. **Use JACCL backend** for Mac Studio clusters with Thunderbolt
2. **Enable RDMA in recovery mode** on all nodes
3. **Fully connected topology** required (mesh, not star)
4. **Set `MLX_METAL_FAST_SYNCH=1`** for optimal performance
5. **Batch communications** to amortize latency
6. **Test with `mlx.launch -n 2`** locally before multi-node

### 6.4 Swift Integration

1. **Use Actors** for thread-safe inference queues
2. **TaskGroups** for parallel request processing
3. **Zero-copy with UnsafePointers** for network I/O
4. **Metal shared buffers** for CPU-GPU data sharing
5. **Async/await** for non-blocking inference pipelines

### 6.5 Performance Optimization

1. **Continuous batching** over static batching
2. **Overlap computation and communication** when possible
3. **Gradient accumulation** to reduce communication frequency
4. **Profile with system trace** to identify bottlenecks
5. **Monitor GPU utilization** (aim for >80%)

---

## 7. Recent Research Papers (2026)

### 7.1 LLM Serving Optimization

**OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration**
- **arXiv:** 2602.12151 (Feb 13, 2026)
- **Key Innovation:** Workload-aware scheduling for heterogeneous requests
- **Results:** 2× average improvement, 1.5× on real-world traces
- **Use Case:** Production systems with variable request patterns

**PrefillShare: A Shared Prefill Module for KV Reuse**
- **arXiv:** 2602.12029 (Feb 13, 2026)
- **Key Innovation:** Disaggregated prefill/decode with shared KV cache
- **Results:** 4.5× lower p95 latency, 3.9× higher throughput
- **Use Case:** Multi-agent systems with repeated prompts

**GORGO: Maximizing KV-Cache Reuse**
- **arXiv:** 2602.11688 (Feb 13, 2026)
- **Focus:** Cross-region load balancing with KV cache reuse
- **Use Case:** Geographically distributed serving

**PAM: Processing Across Memory Hierarchy**
- **arXiv:** 2602.11521 (Feb 13, 2026)
- **Focus:** KV cache management across memory hierarchy
- **Use Case:** Memory-constrained environments

**PARD: Enhancing Goodput via Proactive Request Dropping**
- **arXiv:** 2602.08747 (Feb 10, 2026)
- **Focus:** Pipeline optimization through intelligent request dropping
- **Use Case:** High-load scenarios with SLA constraints

**BOute: Cost-Efficient LLM Serving**
- **arXiv:** 2602.10729 (Feb 12, 2026)
- **Focus:** Multi-objective Bayesian optimization for heterogeneous GPUs
- **Use Case:** Cost optimization in cloud deployments

---

## 8. Implementation Checklist

### For Mac Studio Clusters (MLX + JACCL):

- [ ] Update all Macs to macOS 26.2+
- [ ] Enable RDMA via recovery mode (`rdma_ctl enable`)
- [ ] Verify RDMA devices (`ibv_devices`)
- [ ] Set up fully connected Thunderbolt topology
- [ ] Create hostfile configuration (JSON)
- [ ] Test locally with `mlx.launch -n 2`
- [ ] Set environment variables (`MLX_METAL_FAST_SYNCH=1`)
- [ ] Implement collective communication (all_reduce, all_gather)
- [ ] Profile with `mlx.distributed_config --dot`
- [ ] Monitor network bandwidth and latency

### For vLLM Deployment:

- [ ] Install vLLM (`pip install vllm`)
- [ ] Choose parallelism strategy (TP/PP/DP)
- [ ] Configure Ray cluster (if multi-node)
- [ ] Set `max_model_len` based on available memory
- [ ] Enable PagedAttention (default)
- [ ] Test with single GPU first
- [ ] Scale to multi-GPU with tensor parallelism
- [ ] Add pipeline parallelism if needed
- [ ] Monitor memory usage and adjust block size
- [ ] Enable quantization for larger batches (FP8/INT4)

### For Swift + Metal Integration:

- [ ] Define Actor-based inference queue
- [ ] Implement TaskGroup for parallel processing
- [ ] Create Metal compute pipeline
- [ ] Allocate shared CPU/GPU buffers (`.storageModeShared`)
- [ ] Implement zero-copy networking with UnsafePointers
- [ ] Test Metal kernel performance
- [ ] Profile with Instruments (Metal System Trace)
- [ ] Optimize thread group sizes
- [ ] Monitor GPU utilization

---

## 9. Tools and Resources

### Documentation
- **vLLM:** https://docs.vllm.ai/en/latest/
- **MLX:** https://ml-explore.github.io/mlx/build/html/
- **TensorRT-LLM:** https://github.com/NVIDIA/TensorRT-LLM
- **Metal:** https://developer.apple.com/metal/
- **Swift Concurrency:** https://docs.swift.org/swift-book/

### Code Repositories
- **vLLM:** https://github.com/vllm-project/vllm
- **MLX:** https://github.com/ml-explore/mlx
- **MLX Swift:** https://github.com/ml-explore/mlx-swift
- **TensorRT-LLM:** https://github.com/NVIDIA/TensorRT-LLM

### Papers
- **PagedAttention (SOSP 2023):** https://arxiv.org/abs/2309.06180
- **OServe (2026):** https://arxiv.org/abs/2602.12151
- **PrefillShare (2026):** https://arxiv.org/abs/2602.12029

### Tools
- **mlx.launch:** Distributed job launcher for MLX
- **Ray:** Distributed computing framework
- **Instruments:** Apple performance profiling (Metal System Trace)
- **nsight:** NVIDIA profiling for TensorRT-LLM

---

## 10. Conclusion

The landscape of distributed ML inference in 2025-2026 is dominated by three key innovations:

1. **PagedAttention** for memory-efficient KV cache management (2-4× throughput improvement)
2. **Continuous batching** for dynamic request handling and GPU utilization
3. **RDMA networking** (JACCL) for ultra-low latency communication on Apple Silicon clusters

Production systems like vLLM and TensorRT-LLM demonstrate that combining these techniques with multi-GPU tensor/pipeline parallelism achieves state-of-the-art inference performance. Recent 2026 research (OServe, PrefillShare) extends these concepts to handle heterogeneous workloads and multi-agent systems.

For Mac Studio clusters, the MLX framework with JACCL backend provides a unique opportunity to leverage Thunderbolt RDMA for low-latency distributed computing, rivaling traditional CUDA-based systems.

Swift's modern concurrency model (Actors, async/await) combined with Metal's zero-copy buffers creates an efficient stack for ML inference on Apple platforms, though production examples remain sparse compared to Python-based systems.

**Key Takeaway:** Start with single-GPU continuous batching + PagedAttention, scale with tensor parallelism, then add RDMA networking (JACCL for Macs) for multi-node deployments requiring ultra-low latency.

---

**Document Version:** 1.0
**Last Updated:** February 15, 2026
**Sources:** vLLM docs, MLX docs, arXiv papers (2026), official API documentation

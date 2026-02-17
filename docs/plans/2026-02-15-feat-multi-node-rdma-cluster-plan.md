---
title: Multi-Node RDMA Cluster for Distributed MLX Inference
type: feat
date: 2026-02-15
status: draft
priority: high
estimated_duration: 10-14 weeks
prerequisites: Single-node Swift server implementation (2026-02-15-feat-single-node-swift-mlx-server-plan.md)
---

# Multi-Node RDMA Cluster for Distributed MLX Inference

## Overview

Build a distributed MLX inference cluster using multiple Mac Studio M3 Ultra units connected via Thunderbolt 5 RDMA networking. Implement tensor parallelism to shard large models (70B+) across nodes and achieve 2-4x throughput improvement over single-node setups through horizontal scaling and optimized inter-node communication.

## Problem Statement

Single Mac Studio M3 Ultra with 512GB RAM can serve 70B models effectively, but faces limitations:
1. **Model Size Ceiling:** Cannot load 200B+ models entirely in memory
2. **Throughput Limit:** Single GPU bounded at ~200 tok/s even with continuous batching
3. **Redundancy:** Single point of failure for production deployments
4. **Scaling:** Cannot dynamically add capacity during high-demand periods

## Proposed Solution

Implement a distributed inference cluster using:
- **JACCL Backend:** RDMA over Thunderbolt 5 for ultra-low latency communication (< 1μs)
- **Tensor Parallelism (TP):** Shard model weights across multiple nodes
- **Pipeline Parallelism (PP):** Distribute layers across nodes for very large models
- **Global Scheduler:** Coordinate request routing and load balancing
- **Zero-Copy Networking:** Direct GPU-to-GPU memory transfers without CPU intermediation

**Target Performance:** 500+ tok/s for 70B models (3 nodes), ability to serve 200B+ models.

## Technical Approach

### Cluster Topology

```
                    ┌─────────────────────┐
                    │  Gateway Node       │
                    │  (Mac Studio #1)    │
                    │  • API Layer        │
                    │  • Global Scheduler │
                    │  • Load Balancer    │
                    └──────────┬──────────┘
                               │ Thunderbolt 5 RDMA
                               │ (Fully Connected Mesh)
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐ ┌────▼────────┐ ┌────▼────────┐
    │ Compute Node #2  │ │ Compute #3  │ │ Compute #4  │
    │ Model Shard 1/3  │ │ Shard 2/3   │ │ Shard 3/3   │
    │ TP Rank 1        │ │ TP Rank 2   │ │ TP Rank 3   │
    └──────────────────┘ └─────────────┘ └─────────────┘
           ▲                   ▲                 ▲
           └───────────────────┴─────────────────┘
              All-to-All Communication (RDMA)
```

### Implementation Phases

#### Phase 1: RDMA Infrastructure Setup (Weeks 1-2)

**Goal:** Configure Thunderbolt 5 RDMA networking across Mac Studios

Tasks:
- [ ] Enable RDMA on each Mac Studio (recovery mode)
  ```bash
  # Boot into recovery mode, then:
  rdma_ctl enable
  ```
- [ ] Configure fully-connected mesh topology (each Mac connects to all others)
- [ ] Test basic RDMA connectivity with `rdma_test` utilities
- [ ] Verify bandwidth: expect 120 Gbps per Thunderbolt 5 link
- [ ] Set up static IP addressing for cluster network
- [ ] Create `cluster_config.json` with node mappings

**Deliverables:**
- Network topology diagram (physical and logical)
- RDMA validation test results
- Configuration files: `cluster_config.json`, `rdma_setup.sh`

**Success Criteria:**
- All nodes can ping each other over RDMA network
- Bandwidth test shows > 100 Gbps per link
- Latency test shows < 1μs node-to-node

#### Phase 2: MLX Distributed Integration (Weeks 3-4)

**Goal:** Integrate mlx-distributed with JACCL backend

Tasks:
- [ ] Install mlx-distributed on all nodes
- [ ] Configure JACCL backend with Thunderbolt topology
- [ ] Implement distributed context initialization
- [ ] Test collective operations: all_reduce, all_gather, broadcast
- [ ] Profile communication overhead for various tensor sizes
- [ ] Set up environment variables for optimal performance

**Code Example:**
```swift
// Initialize distributed world
import MLXDistributed

let config = DistributedConfig(
    backend: .jaccl,
    hostfile: "cluster_config.json",
    environment: [
        "MLX_METAL_FAST_SYNCH": "1",
        "RDMA_DEVICE": "tbolt0"
    ]
)

let world = try await DistributedWorld.initialize(config: config)
print("Rank \(world.rank) of \(world.size) initialized")
```

**Deliverables:**
- `Sources/Distributed/ClusterInit.swift`
- Collective operation benchmarks
- Configuration templates

**Success Criteria:**
- Successfully initialize 4-node distributed world
- all_reduce operations complete in < 5ms for 1GB tensors
- Zero communication failures in 1-hour stress test

#### Phase 3: Tensor Parallelism Implementation (Weeks 5-7)

**Goal:** Implement model sharding across nodes

Tasks:
- [ ] Design sharding strategy for transformer layers
  - Shard attention weights across TP dimension
  - Shard FFN weights across TP dimension
  - Keep embeddings replicated or sharded based on vocab size
- [ ] Implement ColumnParallelLinear and RowParallelLinear layers
- [ ] Add all_reduce synchronization after each layer
- [ ] Load sharded checkpoints from Hugging Face
- [ ] Verify numerical correctness vs single-node inference

**Architecture:**
```swift
struct TensorParallelLinear {
    let weight: MLXArray  // Sharded: [hidden_dim, hidden_dim/world_size]
    let rank: Int
    let worldSize: Int

    func forward(_ input: MLXArray) async -> MLXArray {
        // Local matmul on shard
        let localOutput = matmul(input, weight)

        // All-reduce to combine results
        let globalOutput = await world.allReduce(localOutput)

        return globalOutput
    }
}
```

**Deliverables:**
- `Sources/Distributed/TensorParallel.swift`
- `Sources/Model/DistributedTransformer.swift`
- Numerical validation tests

**Success Criteria:**
- Distributed 70B model output matches single-node output (< 0.01% difference)
- Successfully load and shard model in < 30 seconds
- Memory usage evenly distributed across nodes

#### Phase 4: Global Request Scheduler (Weeks 8-9)

**Goal:** Implement intelligent request routing and load balancing

Tasks:
- [ ] Create `GlobalScheduler` Actor on gateway node
- [ ] Implement load-aware request assignment
- [ ] Add queue management for overload scenarios
- [ ] Implement request migration for load balancing
- [ ] Add node health monitoring and failover

**Scheduler Logic:**
```swift
actor GlobalScheduler {
    private var nodeLoads: [NodeID: LoadMetrics] = [:]
    private var pendingRequests: [Request] = []

    func assignRequest(_ request: Request) async -> NodeID {
        // Find least-loaded node with available capacity
        let targetNode = nodeLoads
            .filter { $0.value.availableSlots > 0 }
            .min { $0.value.currentLoad < $1.value.currentLoad }
            ?.key

        guard let node = targetNode else {
            // All nodes at capacity, queue request
            pendingRequests.append(request)
            throw SchedulerError.noCapacity
        }

        return node
    }
}
```

**Deliverables:**
- `Sources/Distributed/GlobalScheduler.swift`
- Load balancing algorithm documentation
- Node health check implementation

**Success Criteria:**
- Evenly distribute 1000 requests across 4 nodes (± 10% variance)
- Automatic failover when node becomes unavailable
- < 10ms scheduling latency per request

#### Phase 5: Pipeline Parallelism (Optional, Weeks 10-11)

**Goal:** Support models > 200B parameters via pipeline parallelism

Tasks:
- [ ] Divide model into pipeline stages (e.g., 4 stages for 200B model)
- [ ] Implement micro-batching for pipeline efficiency
- [ ] Add stage-to-stage communication (point-to-point transfers)
- [ ] Implement gradient accumulation (if training)
- [ ] Balance load across pipeline stages

**Pipeline Architecture:**
```
Stage 1 (Node 1): Layers 0-20  ─→ Stage 2 (Node 2): Layers 21-40
                                     ↓
Stage 4 (Node 4): Layers 61-80 ←─ Stage 3 (Node 3): Layers 41-60
```

**Deliverables:**
- `Sources/Distributed/PipelineParallel.swift`
- Pipeline schedule optimizer
- Micro-batching implementation

**Success Criteria:**
- Successfully run 200B model across 4 nodes
- Pipeline bubble time < 10% (> 90% compute utilization)
- Throughput improvement of 1.5x over TP-only

#### Phase 6: Zero-Copy Optimizations (Week 12)

**Goal:** Eliminate CPU overhead in distributed communication

Tasks:
- [ ] Implement GPU-to-GPU direct transfers via RDMA
- [ ] Use `UnsafeRawPointer` for zero-copy buffer passing
- [ ] Profile memory bandwidth utilization
- [ ] Optimize tensor layout for RDMA transfers (contiguous memory)
- [ ] Implement persistent communication buffers

**Code Example:**
```swift
func sendTensor(_ tensor: MLXArray, to rank: Int) async {
    // Get raw pointer to GPU memory
    let buffer = tensor.unsafeRawPointer()

    // Direct RDMA send without CPU copy
    try await rdmaConnection.send(
        buffer: buffer,
        size: tensor.byteSize,
        destinationRank: rank,
        zeroCopy: true
    )
}
```

**Deliverables:**
- Zero-copy communication primitives
- Memory bandwidth profiling report
- Optimization guide

**Success Criteria:**
- Zero CPU copies during tensor transfers (verified with profiling)
- Communication bandwidth > 110 Gbps per link
- CPU utilization < 10% during inference

#### Phase 7: Fault Tolerance & Monitoring (Week 13)

**Goal:** Production-grade reliability and observability

Tasks:
- [ ] Implement checkpointing for in-flight requests
- [ ] Add automatic node recovery after failure
- [ ] Create cluster-wide metrics aggregation
- [ ] Build monitoring dashboard (Grafana + Prometheus)
- [ ] Implement distributed tracing (spans across nodes)
- [ ] Add alerting for node failures and performance degradation

**Deliverables:**
- `Sources/Monitoring/ClusterMetrics.swift`
- Grafana dashboard JSON
- Alerting rules configuration

**Success Criteria:**
- Survive single-node failure with < 5s recovery time
- Complete request draining before node shutdown
- Real-time visibility into cluster health

#### Phase 8: Benchmarking & Optimization (Week 14)

**Goal:** Validate performance targets and optimize bottlenecks

Tasks:
- [ ] Run comprehensive benchmarks (1-4 nodes, various models)
- [ ] Compare vs single-node baseline
- [ ] Profile communication overhead vs computation time
- [ ] Optimize critical paths identified in profiling
- [ ] Create performance tuning guide

**Benchmark Suite:**
| Configuration | Model | Expected Throughput | Measured |
|---------------|-------|---------------------|----------|
| 1 node | 70B | 200 tok/s | TBD |
| 2 nodes (TP=2) | 70B | 350 tok/s | TBD |
| 3 nodes (TP=3) | 70B | 500 tok/s | TBD |
| 4 nodes (PP=4) | 200B | 150 tok/s | TBD |

**Deliverables:**
- Benchmark results report
- Performance comparison charts
- Optimization recommendations

**Success Criteria:**
- Achieve 500+ tok/s for 70B model (3 nodes, TP=3)
- Successfully serve 200B model (4 nodes, PP=4)
- Scaling efficiency > 80% (throughput scales linearly with nodes)

## Acceptance Criteria

### Functional Requirements
- [ ] Support 2-8 Mac Studio nodes in cluster
- [ ] Shard models up to 200B parameters
- [ ] Automatic failover for single-node failures
- [ ] Load balancing across compute nodes
- [ ] Backward compatible with single-node API

### Performance Requirements
- [ ] 500+ tok/s for 70B model (3 nodes, TP=3)
- [ ] 150+ tok/s for 200B model (4 nodes, PP=4)
- [ ] Node-to-node latency < 1μs (RDMA)
- [ ] Communication overhead < 20% of total time
- [ ] Scaling efficiency > 80%

### Reliability Requirements
- [ ] 99.99% uptime (4 nines)
- [ ] Automatic recovery from single-node failures
- [ ] Graceful degradation under partial failures
- [ ] Zero data loss for in-flight requests

## Technical Considerations

### RDMA Networking
- **Topology:** Fully-connected mesh required for all-to-all communication
- **Cable Quality:** Use certified Thunderbolt 5 cables (40Gbps or 120Gbps)
- **Network Isolation:** Dedicate Thunderbolt ports for cluster traffic only
- **Bandwidth Planning:** 70B model layer transfer = ~280MB, target < 2ms

### Synchronization Overhead
- **TP Communication:** all_reduce after each layer (80 layers = 80 syncs)
- **Frequency:** Every token generation triggers full model pass
- **Optimization:** Overlap communication with computation where possible

### Load Balancing
- **Sticky Sessions:** Route user to same node for context continuity
- **KV Cache Sharing:** Consider shared KV cache layer for session migration
- **Request Migration:** Expensive due to KV cache transfer

### Security
- [ ] Encrypt RDMA traffic (TLS over RDMA)
- [ ] Authenticate nodes before cluster join
- [ ] Isolate cluster network from public internet
- [ ] Audit logs for all cluster events

## Success Metrics

### Performance KPIs
- **Throughput:** 500+ tok/s (70B, 3 nodes)
- **Latency:** < 150ms TTFT, < 10ms inter-token
- **Scaling Efficiency:** > 80% (linear scaling)
- **Availability:** 99.99% uptime

### Cost Efficiency
- **Hardware Utilization:** > 85% GPU utilization across cluster
- **Tokens per Dollar:** Measure cost-effectiveness of horizontal scaling
- **Power Efficiency:** Watts per token generated

## Dependencies & Risks

### Dependencies
- **MLX Distributed:** Requires version with JACCL backend support
- **macOS:** 26.2+ for RDMA support
- **Thunderbolt 5:** Certified cables and Mac Studio with TB5 ports
- **Single-Node Server:** Must complete single-node implementation first

### Risks & Mitigation
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| RDMA instability | High | Medium | Extensive testing, fallback to TCP |
| Communication bottleneck | High | Medium | Profile early, optimize tensor sizes |
| Node synchronization drift | Medium | Low | Implement distributed clock sync |
| Hardware failure | Medium | Low | N+1 redundancy, automatic failover |
| Network congestion | Medium | Medium | QoS prioritization, traffic shaping |

## Future Considerations

### Advanced Clustering Features
- **Dynamic Node Addition:** Hot-add nodes without cluster restart
- **Heterogeneous Hardware:** Mix M3 and M4 Ultra nodes
- **Multi-Tenant Isolation:** Run multiple isolated clusters on shared hardware
- **Speculative Execution:** Use idle nodes for speculative decoding

### Integration with Cloud Services
- **Hybrid Cloud:** Burst to cloud GPUs during peak demand
- **Edge Deployment:** Deploy cluster at edge locations for low latency
- **Federated Learning:** Coordinate across multiple clusters

## References & Research

### Internal References
- Single-node plan: `2026-02-15-feat-single-node-swift-mlx-server-plan.md`
- Distributed research: `/Users/lee.parayno/code4/business/mlx-server/DISTRIBUTED_ML_INFERENCE_BEST_PRACTICES_2025-2026.md`
- RDMA guide: `/Users/lee.parayno/code4/business/mlx-server/docs/COMPREHENSIVE_MLX_SWIFT_DOCUMENTATION.md`

### External References
- JACCL Documentation: https://github.com/ml-explore/mlx-distributed
- Thunderbolt 5 Spec: https://www.intel.com/content/www/us/en/io/thunderbolt/thunderbolt-5-technology.html
- Megatron-LM (TP/PP patterns): https://github.com/NVIDIA/Megatron-LM
- vLLM Distributed: https://docs.vllm.ai/en/latest/serving/distributed_serving.html

### Research Papers
- PagedAttention: https://arxiv.org/abs/2309.06180
- OServe (2026): Spatial-temporal orchestration
- PrefillShare (2026): Shared KV cache for multi-LLM systems

---

**Next Steps:** Complete single-node implementation first, then proceed with RDMA setup.

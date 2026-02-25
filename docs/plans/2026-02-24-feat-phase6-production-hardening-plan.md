# Phase 6: Production Hardening - Implementation Plan

**Date:** February 24, 2026
**Status:** Planning
**Prerequisites:** Phase 5 Complete (84 tests passing, OpenAI-compatible API operational)

## Overview

Phase 6 hardens the MLX Server for production deployment by adding authentication, real model integration, deployment automation, and comprehensive documentation. This phase transforms the server from a functional prototype to a production-ready service.

**Key Outcomes:**
- API key authentication and rate limiting
- Real MLX model loading and inference
- Docker/container deployment
- Production-grade documentation
- Performance validation with real models

## Current State

### What Works (Phase 5)
- ✅ OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- ✅ SSE streaming for real-time tokens
- ✅ Request scheduler with priority queuing
- ✅ Continuous batching with slot management
- ✅ PagedKVCache for memory efficiency
- ✅ Health and metrics endpoints
- ✅ 84 tests passing (100% pass rate)

### What's Missing
1. **Authentication:** No API key validation, open to all
2. **Real Models:** Using mock inference, not actual MLX models
3. **Deployment:** No containerization or deployment automation
4. **Documentation:** No user guides or deployment docs
5. **Rate Limiting:** No per-user quotas or throttling

## Phase 6 Implementation

### Phase 6.1: Authentication & Authorization ⏳

**Goal:** Implement API key authentication and rate limiting

**Tasks:**

1. **Create API Key Middleware**
   ```swift
   // Sources/Security/APIKeyMiddleware.swift
   struct APIKeyMiddleware: AsyncMiddleware {
       let keyStore: APIKeyStore

       func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
           // Extract API key from Authorization header
           guard let authHeader = request.headers.first(name: .authorization),
                 authHeader.hasPrefix("Bearer ") else {
               throw Abort(.unauthorized, reason: "Missing API key")
           }

           let apiKey = String(authHeader.dropFirst(7))

           // Validate API key
           guard let user = await keyStore.validate(apiKey) else {
               throw Abort(.unauthorized, reason: "Invalid API key")
           }

           // Store user in request storage
           request.storage[UserKey.self] = user

           return try await next.respond(to: request)
       }
   }
   ```

2. **Implement API Key Store**
   ```swift
   // Sources/Security/APIKeyStore.swift
   actor APIKeyStore {
       private var keys: [String: User] = [:]

       func validate(_ apiKey: String) async -> User? {
           return keys[apiKey]
       }

       func create(for userId: String) async -> String {
           let apiKey = UUID().uuidString
           keys[apiKey] = User(id: userId, quota: 1000, used: 0)
           return apiKey
       }

       func revoke(_ apiKey: String) async {
           keys.removeValue(forKey: apiKey)
       }
   }

   struct User: Sendable {
       let id: String
       var quota: Int  // requests per hour
       var used: Int
       var resetTime: Date
   }
   ```

3. **Add Rate Limiting**
   ```swift
   // Sources/Security/RateLimiter.swift
   actor RateLimiter {
       private var usage: [String: (count: Int, resetTime: Date)] = [:]

       func checkLimit(for userId: String, quota: Int) async throws {
           let now = Date()

           if let entry = usage[userId] {
               if now >= entry.resetTime {
                   // Reset window
                   usage[userId] = (count: 1, resetTime: now.addingTimeInterval(3600))
               } else if entry.count >= quota {
                   throw Abort(.tooManyRequests, reason: "Rate limit exceeded")
               } else {
                   usage[userId] = (count: entry.count + 1, resetTime: entry.resetTime)
               }
           } else {
               usage[userId] = (count: 1, resetTime: now.addingTimeInterval(3600))
           }
       }
   }
   ```

4. **Write Authentication Tests**
   - Test: Missing API key returns 401
   - Test: Invalid API key returns 401
   - Test: Valid API key allows request
   - Test: Rate limit exceeded returns 429
   - Test: Rate limit resets after window
   - Test: Different users have separate limits

**Success Criteria:**
- API requires valid Authorization header
- Invalid keys rejected with 401
- Rate limiting enforces per-user quotas
- All authentication tests pass

**Deliverables:**
- `Sources/Security/APIKeyMiddleware.swift` (~40 lines)
- `Sources/Security/APIKeyStore.swift` (~60 lines)
- `Sources/Security/RateLimiter.swift` (~50 lines)
- `Tests/SecurityTests/AuthenticationTests.swift` (~200 lines)

### Phase 6.2: Real MLX Model Integration ⏳

**Goal:** Replace mock inference with actual MLX model loading and inference

**Tasks:**

1. **Update InferenceEngine for Real Models**
   ```swift
   // Sources/Core/InferenceEngine.swift
   actor InferenceEngine {
       private var modelContainer: ModelContainer?
       private var tokenizer: Tokenizer?

       func loadModel(path: String) async throws {
           logger.info("Loading model from \(path)")

           // Load model using mlx-swift-lm
           let config = ModelConfiguration(
               id: path,
               quantization: .q4_0  // 4-bit quantization
           )

           modelContainer = try await ModelContainer(config: config)
           tokenizer = try await Tokenizer(modelName: path)

           logger.info("Model loaded successfully")
       }

       func forwardBatch(_ input: BatchInput) async throws -> [Int] {
           guard let container = modelContainer,
                 let tokenizer = tokenizer else {
               throw InferenceError.modelNotLoaded
           }

           // Stack token IDs: [batch_size, 1]
           let inputIds = MLXArray(input.tokens.map { [$0] })

           // Get KV cache blocks
           let kvCache = getKVCache(for: input.kvCacheBlockIds)

           // Single batched forward pass
           let logits = try await container.model.forward(
               inputIds,
               cache: kvCache
           )

           // Sample next tokens
           let nextTokens = sample(logits, temperature: input.temperature)

           // Update KV cache
           try await updateKVCache(kvCache, for: input.kvCacheBlockIds)

           return nextTokens
       }

       private func sample(_ logits: MLXArray, temperature: Float) -> [Int] {
           // Temperature sampling with top-p
           let probs = softmax(logits / temperature)
           return topPSample(probs, p: 0.9)
       }
   }
   ```

2. **Add Model Download CLI**
   ```swift
   // Sources/MLXServer/Commands/DownloadCommand.swift
   struct DownloadCommand: AsyncParsableCommand {
       static let configuration = CommandConfiguration(
           commandName: "download",
           abstract: "Download a model from Hugging Face"
       )

       @Option(name: .long, help: "Model ID (e.g., mlx-community/Qwen2.5-0.5B-Instruct-4bit)")
       var model: String

       @Option(name: .long, help: "Local cache directory")
       var cacheDir: String = "~/.cache/mlx-server/models"

       mutating func run() async throws {
           print("Downloading model: \(model)")

           // Use mlx-swift-lm's Hub integration
           let downloader = ModelDownloader(cacheDir: cacheDir)
           try await downloader.download(modelId: model)

           print("Model downloaded to: \(cacheDir)/\(model)")
       }
   }
   ```

3. **Update Server Initialization**
   ```swift
   // Sources/MLXServer/MLXServer.swift
   // Update main() to load real model

   if let modelPath = parsed.model {
       logger.info("Loading model: \(modelPath)")
       try await engine.loadModel(path: modelPath)
   } else if !parsed.test {
       throw ValidationError("Model path required (use --model or --test flag)")
   }
   ```

4. **Write Model Integration Tests**
   - Test: Model loading succeeds
   - Test: Inference produces valid tokens
   - Test: Batched inference works
   - Test: Temperature affects sampling
   - Test: Token generation completes correctly

**Success Criteria:**
- Real MLX model loads successfully
- Inference produces coherent text
- Batching works with real model
- Performance meets Phase 4.4 targets (>10 tok/s)

**Deliverables:**
- Updated `Sources/Core/InferenceEngine.swift` (~200 lines added)
- `Sources/MLXServer/Commands/DownloadCommand.swift` (~80 lines)
- Updated `Sources/MLXServer/MLXServer.swift` (~20 lines)
- `Tests/CoreTests/RealModelTests.swift` (~150 lines)

### Phase 6.3: Production Deployment ⏳

**Goal:** Create deployment artifacts for production use

**Tasks:**

1. **Create Dockerfile**
   ```dockerfile
   # Dockerfile
   FROM swift:6.2.3

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       libxml2-dev \
       && rm -rf /var/lib/apt/lists/*

   # Copy project files
   COPY Package.swift Package.resolved ./
   COPY Sources ./Sources
   COPY Tests ./Tests

   # Build release binary
   RUN swift build -c release --arch arm64

   # Copy binary to final location
   RUN cp .build/release/mlx-server /usr/local/bin/

   # Create cache directory
   RUN mkdir -p /root/.cache/mlx-server/models

   # Expose port
   EXPOSE 8080

   # Health check
   HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
       CMD curl -f http://localhost:8080/health || exit 1

   # Run server
   CMD ["mlx-server", "--model", "/models/default", "--port", "8080"]
   ```

2. **Create Docker Compose**
   ```yaml
   # docker-compose.yml
   version: '3.8'

   services:
     mlx-server:
       build: .
       ports:
         - "8080:8080"
       volumes:
         - ./models:/models:ro
         - mlx-cache:/root/.cache/mlx-server
       environment:
         - LOG_LEVEL=info
         - MAX_BATCH_SIZE=32
       restart: unless-stopped
       deploy:
         resources:
           limits:
             memory: 64G

     # Optional: Load balancer
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
       depends_on:
         - mlx-server

   volumes:
     mlx-cache:
   ```

3. **Create Kubernetes Manifests**
   ```yaml
   # k8s/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: mlx-server
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: mlx-server
     template:
       metadata:
         labels:
           app: mlx-server
       spec:
         containers:
         - name: mlx-server
           image: mlx-server:latest
           ports:
           - containerPort: 8080
           resources:
             requests:
               memory: "32Gi"
             limits:
               memory: "64Gi"
           livenessProbe:
             httpGet:
               path: /health
               port: 8080
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8080
             initialDelaySeconds: 10
             periodSeconds: 5

   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: mlx-server
   spec:
     selector:
       app: mlx-server
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8080
     type: LoadBalancer
   ```

4. **Create Deployment Scripts**
   ```bash
   #!/bin/bash
   # scripts/deploy.sh

   set -e

   # Build Docker image
   echo "Building Docker image..."
   docker build -t mlx-server:latest .

   # Tag for registry
   docker tag mlx-server:latest your-registry/mlx-server:latest

   # Push to registry
   echo "Pushing to registry..."
   docker push your-registry/mlx-server:latest

   # Deploy to Kubernetes
   echo "Deploying to Kubernetes..."
   kubectl apply -f k8s/

   # Wait for rollout
   kubectl rollout status deployment/mlx-server

   echo "Deployment complete!"
   ```

**Success Criteria:**
- Docker image builds successfully
- Container runs and serves requests
- Kubernetes deployment succeeds
- Health checks pass
- Load balancer routes traffic correctly

**Deliverables:**
- `Dockerfile` (~40 lines)
- `docker-compose.yml` (~30 lines)
- `k8s/deployment.yaml` (~60 lines)
- `k8s/service.yaml` (~15 lines)
- `scripts/deploy.sh` (~30 lines)
- `nginx.conf` (~40 lines)

### Phase 6.4: Documentation ⏳

**Goal:** Create comprehensive user and deployment documentation

**Tasks:**

1. **User Guide** (`docs/USER_GUIDE.md`)
   - Installation instructions
   - API authentication
   - Making requests (curl, SDK examples)
   - Streaming vs non-streaming
   - Error handling
   - Rate limits and quotas

2. **Deployment Guide** (`docs/DEPLOYMENT_GUIDE.md`)
   - System requirements
   - Model download and preparation
   - Docker deployment
   - Kubernetes deployment
   - Configuration options
   - Monitoring setup
   - Troubleshooting

3. **API Reference** (Update `docs/API.md`)
   - Authentication endpoints
   - Rate limit headers
   - Error codes and responses
   - OpenAPI/Swagger spec

4. **Operations Guide** (`docs/OPERATIONS_GUIDE.md`)
   - Health monitoring
   - Log aggregation
   - Metrics collection
   - Backup and recovery
   - Scaling considerations
   - Performance tuning

**Success Criteria:**
- Documentation is clear and comprehensive
- All examples work correctly
- Deployment guide enables successful setup
- API reference matches implementation

**Deliverables:**
- `docs/USER_GUIDE.md` (~400 lines)
- `docs/DEPLOYMENT_GUIDE.md` (~500 lines)
- Updated `docs/API.md` (+200 lines)
- `docs/OPERATIONS_GUIDE.md` (~300 lines)

## Success Criteria

### Functional Requirements
- [ ] API key authentication required for all endpoints
- [ ] Rate limiting enforces per-user quotas
- [ ] Real MLX model loads and generates coherent text
- [ ] Docker container runs successfully
- [ ] Kubernetes deployment succeeds
- [ ] All documentation complete

### Performance Requirements
- [ ] Real model achieves >10 tok/s (test model baseline)
- [ ] Authentication overhead <1ms per request
- [ ] Container startup <60 seconds
- [ ] Health checks respond <10ms

### Quality Requirements
- [ ] All existing tests still pass (84 tests)
- [ ] New tests for authentication (6 tests)
- [ ] New tests for real models (5 tests)
- [ ] Zero regressions from Phase 5

## Timeline

**Week 1:**
- Days 1-2: Authentication & Authorization (Phase 6.1)
- Days 3-4: Real Model Integration (Phase 6.2)

**Week 2:**
- Days 1-2: Production Deployment (Phase 6.3)
- Days 3-4: Documentation (Phase 6.4)
- Day 5: Testing and validation

**Total: 2 weeks**

## Dependencies

**Existing:**
- ✅ Phase 5 complete (OpenAI API, streaming, observability)
- ✅ Vapor 4.121.2
- ✅ MLX Swift 0.30.6

**New:**
- `mlx-swift-lm` v2.30.3 (model loading, tokenizer)
- Docker Engine
- Kubernetes (optional, for k8s deployment)

## Risks and Mitigations

### Risk 1: Model Loading Performance
**Risk:** Real model loading takes >60 seconds
**Mitigation:**
- Pre-download models in Docker image
- Use 4-bit quantization
- Document expected startup time

### Risk 2: Container Size
**Risk:** Docker image >20GB
**Mitigation:**
- Use multi-stage builds
- Exclude test dependencies from release
- Consider model volume mounts

### Risk 3: Authentication Overhead
**Risk:** API key validation adds >10ms latency
**Mitigation:**
- In-memory key store (no DB lookup)
- Cache validation results
- Profile and optimize hot path

## Next Steps After Phase 6

### Phase 7: Advanced Optimizations
1. Speculative decoding (2-3x speedup)
2. Prefix caching (shared prompts)
3. Quantized KV cache (4x memory savings)
4. Model quantization options (q4, q6, q8)

## References

**Internal:**
- Phase 5 Summary: `docs/Phase-5-Implementation-Summary.md`
- Development Progress: `docs/Development-Progress.md`
- CLAUDE.md: Project charter and requirements

**External:**
- mlx-swift-lm: https://github.com/ml-explore/mlx-swift-examples
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Kubernetes Deployment: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

---

**Created:** 2026-02-24
**Author:** Claude Code + Lee Parayno
**Status:** Planning - Ready for Implementation

# MLX Server Deployment Guide

**Version:** 1.0.0
**Last Updated:** February 24, 2026
**Status:** Production Ready

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

**Minimum (Development):**
- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB RAM
- 50GB free disk space

**Recommended (Production):**
- Mac Studio M3 Ultra or later
- 512GB RAM
- 500GB SSD
- macOS Tahoe 26.3+ (for optimal JACCL performance)

### Software Requirements

- **macOS:** 15.0+ (26.3+ recommended)
- **Swift:** 6.2.3+
- **Xcode:** 16.0+ (for Metal shader compilation)
- **Docker:** 20.10+ (for containerized deployment)
- **Kubernetes:** 1.28+ (for K8s deployment)

## Local Development

### Build from Source

```bash
# Clone repository
git clone https://github.com/your-org/mlx-server.git
cd mlx-server

# Build with make (uses xcodebuild for Metal shaders)
make build

# Run tests
make test

# Run server
./mlx-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8080
```

### Configuration

Create a configuration file (optional):

```bash
# config.json
{
  "port": 8080,
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "log_level": "info",
  "max_batch_size": 32
}
```

Run with config:

```bash
./mlx-server --config config.json
```

### Environment Variables

```bash
export MLX_PORT=8080
export MLX_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
export MLX_LOG_LEVEL="info"

./mlx-server
```

## Docker Deployment

### Build Docker Image

**On macOS (with Metal support):**

```bash
# Build macOS-specific image
./scripts/docker-build.sh

# Or manually:
docker build -t mlx-server:latest -f Dockerfile.macOS .
```

**On Linux (no Metal, for development only):**

```bash
docker build -t mlx-server:latest -f Dockerfile .
```

### Run Container

```bash
# Using provided script
./scripts/docker-run.sh

# Or manually:
docker run -d \
  --name mlx-server \
  -p 8080:8080 \
  -e MLX_MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit" \
  -v mlx-cache:/root/.cache/mlx-server \
  mlx-server:latest
```

### Docker Compose

```bash
# Start all services (server + nginx)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Health Check

```bash
# Check if container is healthy
docker ps
curl http://localhost:8080/health
```

## Kubernetes Deployment

### Quick Start

```bash
# Deploy to Kubernetes
./scripts/k8s-deploy.sh

# Or manually:
kubectl apply -f k8s/
```

### Step-by-Step Deployment

**1. Create Namespace (Optional)**

```bash
kubectl create namespace mlx-server
```

**2. Apply Configuration**

```bash
kubectl apply -f k8s/configmap.yaml -n mlx-server
kubectl apply -f k8s/secret.yaml -n mlx-server
```

**3. Deploy Application**

```bash
kubectl apply -f k8s/deployment.yaml -n mlx-server
kubectl apply -f k8s/service.yaml -n mlx-server
```

**4. Configure Ingress (Optional)**

```bash
kubectl apply -f k8s/ingress.yaml -n mlx-server
```

**5. Enable Autoscaling (Optional)**

```bash
kubectl apply -f k8s/hpa.yaml -n mlx-server
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n mlx-server

# Check services
kubectl get svc -n mlx-server

# View logs
kubectl logs -f deployment/mlx-server -n mlx-server

# Wait for rollout
kubectl rollout status deployment/mlx-server -n mlx-server
```

### Port Forward (Testing)

```bash
kubectl port-forward svc/mlx-server 8080:80 -n mlx-server
curl http://localhost:8080/health
```

### Scale Deployment

```bash
# Manual scaling
kubectl scale deployment mlx-server --replicas=3 -n mlx-server

# Auto-scaling (using HPA)
kubectl apply -f k8s/hpa.yaml -n mlx-server
```

## Configuration

### Server Options

| Option | Environment | Default | Description |
|--------|------------|---------|-------------|
| `--port` | `MLX_PORT` | 8080 | HTTP server port |
| `--model` | `MLX_MODEL` | Qwen2.5-0.5B-Instruct-4bit | Model ID or path |
| `--log-level` | `MLX_LOG_LEVEL` | info | Log level (trace/debug/info/warning/error) |
| `--config` | - | - | Config file path |
| `--test` | `MLX_TEST` | false | Test mode: run single inference and exit |

### Model Selection

**Small Models (Development):**
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (500MB, fast)
- `mlx-community/Qwen2-1.5B-Instruct-4bit` (1.5B params)

**Medium Models (Production):**
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (7B params, 4GB)
- `mlx-community/Mistral-7B-Instruct-v0.2-4bit` (7B params)

**Large Models (High Quality):**
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` (47B params, 26GB)
- `mlx-community/Llama-3-70B-Instruct-4bit` (70B params, 40GB)

### Authentication

**Default API Key (Development):**
```
sk-test-12345
```

**Create New API Keys:**

```swift
// Via APIKeyStore (programmatically)
let key = await apiKeyStore.create(for: "user-id", quota: 5000)
```

**Environment Variables:**

```bash
export DEFAULT_API_KEY="sk-your-production-key"
```

### Resource Limits

**Docker:**

```yaml
# docker-compose.yml
services:
  mlx-server:
    deploy:
      resources:
        reservations:
          memory: 16G
        limits:
          memory: 64G
```

**Kubernetes:**

```yaml
# k8s/deployment.yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
  limits:
    memory: "64Gi"
    cpu: "16"
```

## Monitoring

### Health Endpoints

**Health Check:**
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model": "loaded",
  "batcher": "running",
  "timestamp": "2026-02-24T12:00:00Z"
}
```

**Readiness Check:**
```bash
curl http://localhost:8080/ready
```

**Metrics:**
```bash
curl http://localhost:8080/metrics
```

Response:
```json
{
  "requests": {
    "pending": 5,
    "active": 10,
    "completed": 1234,
    "failed": 12,
    "cancelled": 3
  },
  "batcher": {
    "activeSlots": 10,
    "totalSlots": 32,
    "utilization": 0.3125,
    "stepCount": 5678
  },
  "gpu": {
    "averageUtilization": 0.85,
    "currentUtilization": 0.92,
    "sampleCount": 100
  }
}
```

### Logging

**View Logs:**

```bash
# Docker
docker logs -f mlx-server

# Kubernetes
kubectl logs -f deployment/mlx-server -n mlx-server

# Follow specific pod
kubectl logs -f pod/<pod-name> -n mlx-server
```

**Log Levels:**
- `trace`: Very detailed debugging
- `debug`: Detailed operational information
- `info`: General operational information (default)
- `warning`: Warning messages
- `error`: Error messages

### Prometheus Integration

The `/metrics` endpoint is compatible with Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'mlx-server'
    static_configs:
      - targets: ['mlx-server:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Troubleshooting

### Common Issues

**1. Metal Shader Compilation Fails**

**Error:**
```
MLX error: Failed to load the default metallib
```

**Solution:**
- Use `xcodebuild` instead of `swift build`
- Run `make build` (uses xcodebuild automatically)

**2. Model Download Fails**

**Error:**
```
Failed to load model: Network error
```

**Solution:**
- Check internet connection
- Verify model ID is correct
- Check Hugging Face Hub status
- Pre-download model:
  ```bash
  mlx-server download --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
  ```

**3. Out of Memory**

**Error:**
```
MLX error: Out of memory
```

**Solution:**
- Use smaller model
- Reduce `max_batch_size`
- Use 4-bit quantized models
- Increase system memory (Docker/K8s limits)

**4. Authentication Fails**

**Error:**
```
401 Unauthorized: Missing Authorization header
```

**Solution:**
- Add `Authorization: Bearer <api-key>` header
- Use default key: `sk-test-12345`
- Create new API key if needed

**5. Rate Limit Exceeded**

**Error:**
```
429 Too Many Requests: Rate limit exceeded
```

**Solution:**
- Wait for rate limit window to reset (see `Retry-After` header)
- Increase user quota (if admin)
- Use different API key

### Debug Mode

Enable detailed logging:

```bash
./mlx-server --log-level debug --verbose
```

Or with Docker:

```bash
docker run -e MLX_LOG_LEVEL=debug -e MLX_VERBOSE=true mlx-server:latest
```

### Performance Issues

**Low Throughput:**

1. Check GPU utilization: `curl http://localhost:8080/metrics`
2. Increase batch size if utilization < 80%
3. Verify macOS version (26.3+ for JACCL performance)
4. Check if model is loaded in memory

**High Latency:**

1. Use smaller model for development
2. Enable speculative decoding (Phase 7)
3. Enable prefix caching (Phase 7)
4. Check network latency to client

### Container Issues

**Container Won't Start:**

```bash
# Check container logs
docker logs mlx-server

# Check container status
docker ps -a

# Restart container
docker restart mlx-server
```

**Kubernetes Pod Crashing:**

```bash
# Check pod events
kubectl describe pod <pod-name> -n mlx-server

# Check pod logs
kubectl logs <pod-name> -n mlx-server --previous

# Check resource usage
kubectl top pod <pod-name> -n mlx-server
```

## Support

### Documentation

- **API Reference:** `docs/API.md`
- **User Guide:** `docs/USER_GUIDE.md`
- **Operations Guide:** `docs/OPERATIONS_GUIDE.md`

### Community

- **GitHub Issues:** https://github.com/your-org/mlx-server/issues
- **Discussions:** https://github.com/your-org/mlx-server/discussions

### Reporting Bugs

When reporting issues, include:

1. MLX Server version
2. macOS version
3. Hardware specs (Mac model, RAM)
4. Error messages and logs
5. Steps to reproduce

---

**Last Updated:** February 24, 2026
**Version:** 1.0.0
**Status:** Production Ready

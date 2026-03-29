#!/bin/bash
# Run MLX Server in Docker

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-mlx-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-mlx-server}"
PORT="${PORT:-8080}"
MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "================================"
echo "Starting MLX Server Container"
echo "================================"
echo "Container: ${CONTAINER_NAME}"
echo "Port: ${PORT}"
echo "Model: ${MODEL}"
echo "Log Level: ${LOG_LEVEL}"
echo "================================"

# Stop existing container if running
if docker ps -a | grep -q "${CONTAINER_NAME}"; then
  echo "Stopping existing container..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Run container
echo "Starting container..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8080" \
  -e MLX_PORT=8080 \
  -e MLX_MODEL="${MODEL}" \
  -e MLX_LOG_LEVEL="${LOG_LEVEL}" \
  -v mlx-cache:/root/.cache/mlx-server \
  --restart unless-stopped \
  "${IMAGE_NAME}:${IMAGE_TAG}"

echo "✅ Container started: ${CONTAINER_NAME}"
echo ""
echo "View logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "Check health:"
echo "  curl http://localhost:${PORT}/health"
echo ""
echo "Stop container:"
echo "  docker stop ${CONTAINER_NAME}"

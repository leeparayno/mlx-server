#!/bin/bash
# Docker build script for MLX Server

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-mlx-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.macOS}"  # Use Dockerfile for non-macOS
REGISTRY="${REGISTRY:-}"  # e.g., docker.io/username or gcr.io/project

echo "================================"
echo "MLX Server Docker Build"
echo "================================"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo "================================"

# Build Docker image
echo "Building Docker image..."
docker build \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  .

echo "✅ Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"

# Tag for registry if specified
if [ -n "${REGISTRY}" ]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
  echo "Tagging for registry: ${FULL_IMAGE}"
  docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
  echo "✅ Tagged: ${FULL_IMAGE}"
fi

# Show image info
echo ""
echo "Image information:"
docker images | grep "${IMAGE_NAME}" | head -3

echo ""
echo "To run locally:"
echo "  docker run -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to registry:"
echo "  docker push ${FULL_IMAGE}"

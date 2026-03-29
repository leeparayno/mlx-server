#!/bin/bash
# Kubernetes deployment script for MLX Server

set -e

# Configuration
NAMESPACE="${NAMESPACE:-default}"
KUBECTL="${KUBECTL:-kubectl}"

echo "================================"
echo "MLX Server Kubernetes Deployment"
echo "================================"
echo "Namespace: ${NAMESPACE}"
echo "================================"

# Check if kubectl is available
if ! command -v ${KUBECTL} &> /dev/null; then
  echo "❌ kubectl not found. Please install kubectl first."
  exit 1
fi

# Check if cluster is accessible
if ! ${KUBECTL} cluster-info &> /dev/null; then
  echo "❌ Cannot connect to Kubernetes cluster."
  echo "Please configure kubectl to connect to your cluster."
  exit 1
fi

# Create namespace if it doesn't exist
if ! ${KUBECTL} get namespace ${NAMESPACE} &> /dev/null; then
  echo "Creating namespace: ${NAMESPACE}"
  ${KUBECTL} create namespace ${NAMESPACE}
fi

# Apply Kubernetes manifests
echo ""
echo "Applying Kubernetes manifests..."

# ConfigMap and Secret first
echo "→ ConfigMap..."
${KUBECTL} apply -f k8s/configmap.yaml -n ${NAMESPACE}

echo "→ Secret..."
${KUBECTL} apply -f k8s/secret.yaml -n ${NAMESPACE}

# Deployment and PVC
echo "→ Deployment..."
${KUBECTL} apply -f k8s/deployment.yaml -n ${NAMESPACE}

# Services
echo "→ Services..."
${KUBECTL} apply -f k8s/service.yaml -n ${NAMESPACE}

# Ingress (optional)
if [ -f "k8s/ingress.yaml" ]; then
  echo "→ Ingress..."
  ${KUBECTL} apply -f k8s/ingress.yaml -n ${NAMESPACE}
fi

# HPA (optional)
if [ -f "k8s/hpa.yaml" ]; then
  echo "→ HorizontalPodAutoscaler..."
  ${KUBECTL} apply -f k8s/hpa.yaml -n ${NAMESPACE}
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Check deployment status:"
echo "  ${KUBECTL} get pods -n ${NAMESPACE}"
echo "  ${KUBECTL} get svc -n ${NAMESPACE}"
echo ""
echo "View logs:"
echo "  ${KUBECTL} logs -f deployment/mlx-server -n ${NAMESPACE}"
echo ""
echo "Wait for rollout:"
echo "  ${KUBECTL} rollout status deployment/mlx-server -n ${NAMESPACE}"
echo ""
echo "Port forward (for testing):"
echo "  ${KUBECTL} port-forward svc/mlx-server 8080:80 -n ${NAMESPACE}"

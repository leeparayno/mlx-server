#!/bin/bash
# Delete Kubernetes deployment

set -e

NAMESPACE="${NAMESPACE:-default}"
KUBECTL="${KUBECTL:-kubectl}"

echo "================================"
echo "MLX Server Kubernetes Cleanup"
echo "================================"
echo "Namespace: ${NAMESPACE}"
echo "================================"

read -p "Are you sure you want to delete the deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled."
  exit 0
fi

echo "Deleting Kubernetes resources..."

# Delete in reverse order
if [ -f "k8s/hpa.yaml" ]; then
  ${KUBECTL} delete -f k8s/hpa.yaml -n ${NAMESPACE} --ignore-not-found
fi

if [ -f "k8s/ingress.yaml" ]; then
  ${KUBECTL} delete -f k8s/ingress.yaml -n ${NAMESPACE} --ignore-not-found
fi

${KUBECTL} delete -f k8s/service.yaml -n ${NAMESPACE} --ignore-not-found
${KUBECTL} delete -f k8s/deployment.yaml -n ${NAMESPACE} --ignore-not-found
${KUBECTL} delete -f k8s/secret.yaml -n ${NAMESPACE} --ignore-not-found
${KUBECTL} delete -f k8s/configmap.yaml -n ${NAMESPACE} --ignore-not-found

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Note: PVC (mlx-server-cache) is NOT deleted to preserve data."
echo "To delete PVC:"
echo "  ${KUBECTL} delete pvc mlx-server-cache -n ${NAMESPACE}"

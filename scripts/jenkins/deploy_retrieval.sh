#!/usr/bin/env bash
set -euo pipefail

IMAGE_REF="${1:?usage: deploy_retrieval.sh <image-ref> [namespace]}"
NAMESPACE="${2:-asterscope}"

if [[ -z "${KUBECONFIG:-}" ]]; then
  echo "KUBECONFIG is required for deployment" >&2
  exit 1
fi

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "${NAMESPACE}" apply -f deploy/k8s/retrieval-deployment.yaml
kubectl -n "${NAMESPACE}" set image deployment/asterscope-retrieval retrieval="${IMAGE_REF}"
kubectl -n "${NAMESPACE}" rollout status deployment/asterscope-retrieval --timeout=300s

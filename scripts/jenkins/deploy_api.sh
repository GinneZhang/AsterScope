#!/usr/bin/env bash
set -euo pipefail

IMAGE_REF="${1:?usage: deploy_api.sh <image-ref> [namespace]}"
NAMESPACE="${2:-asterscope}"

if [[ -z "${KUBECONFIG:-}" ]]; then
  echo "KUBECONFIG is required for deployment" >&2
  exit 1
fi

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if [[ -n "${OPENAI_API_KEY:-}" || -n "${API_KEY:-}" || -n "${POSTGRES_PASSWORD:-}" || -n "${NEO4J_PASSWORD:-}" ]]; then
  secret_args=()
  [[ -n "${OPENAI_API_KEY:-}" ]] && secret_args+=(--from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}")
  [[ -n "${API_KEY:-}" ]] && secret_args+=(--from-literal=API_KEY="${API_KEY}")
  [[ -n "${POSTGRES_HOST:-}" ]] && secret_args+=(--from-literal=POSTGRES_HOST="${POSTGRES_HOST}")
  [[ -n "${POSTGRES_PORT:-}" ]] && secret_args+=(--from-literal=POSTGRES_PORT="${POSTGRES_PORT}")
  [[ -n "${POSTGRES_DB:-}" ]] && secret_args+=(--from-literal=POSTGRES_DB="${POSTGRES_DB}")
  [[ -n "${POSTGRES_USER:-}" ]] && secret_args+=(--from-literal=POSTGRES_USER="${POSTGRES_USER}")
  [[ -n "${POSTGRES_PASSWORD:-}" ]] && secret_args+=(--from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}")
  [[ -n "${REDIS_HOST:-}" ]] && secret_args+=(--from-literal=REDIS_HOST="${REDIS_HOST}")
  [[ -n "${REDIS_PORT:-}" ]] && secret_args+=(--from-literal=REDIS_PORT="${REDIS_PORT}")
  [[ -n "${REDIS_PASSWORD:-}" ]] && secret_args+=(--from-literal=REDIS_PASSWORD="${REDIS_PASSWORD}")
  [[ -n "${NEO4J_URI:-}" ]] && secret_args+=(--from-literal=NEO4J_URI="${NEO4J_URI}")
  [[ -n "${NEO4J_USER:-}" ]] && secret_args+=(--from-literal=NEO4J_USER="${NEO4J_USER}")
  [[ -n "${NEO4J_PASSWORD:-}" ]] && secret_args+=(--from-literal=NEO4J_PASSWORD="${NEO4J_PASSWORD}")

  kubectl -n "${NAMESPACE}" create secret generic asterscope-secrets \
    "${secret_args[@]}" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

kubectl -n "${NAMESPACE}" apply -f deploy/k8s/api-deployment.yaml
kubectl -n "${NAMESPACE}" set image deployment/asterscope-api api="${IMAGE_REF}"
kubectl -n "${NAMESPACE}" rollout status deployment/asterscope-api --timeout=300s

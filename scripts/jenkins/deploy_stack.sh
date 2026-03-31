#!/usr/bin/env bash
set -euo pipefail

API_IMAGE_REF="${1:?usage: deploy_stack.sh <api-image-ref> <retrieval-image-ref> [namespace] [release-name]}"
RETRIEVAL_IMAGE_REF="${2:?usage: deploy_stack.sh <api-image-ref> <retrieval-image-ref> [namespace] [release-name]}"
NAMESPACE="${3:-asterscope}"
RELEASE_NAME="${4:-asterscope}"

if [[ -z "${KUBECONFIG:-}" ]]; then
  echo "KUBECONFIG is required for deployment" >&2
  exit 1
fi

if ! command -v helm >/dev/null 2>&1; then
  echo "helm is required for deploy_stack.sh" >&2
  exit 1
fi

split_image_ref() {
  local image_ref="$1"
  local repo="${image_ref%:*}"
  local tag="${image_ref##*:}"
  if [[ "${repo}" == "${tag}" ]]; then
    echo "unable to parse image ref: ${image_ref}" >&2
    exit 1
  fi
  printf '%s\n%s\n' "${repo}" "${tag}"
}

mapfile -t api_parts < <(split_image_ref "${API_IMAGE_REF}")
mapfile -t retrieval_parts < <(split_image_ref "${RETRIEVAL_IMAGE_REF}")

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if [[ -n "${OPENAI_API_KEY:-}" || -n "${API_KEY:-}" || -n "${POSTGRES_PASSWORD:-}" || -n "${NEO4J_PASSWORD:-}" ]]; then
  secret_args=()
  secret_args+=(--set externalSecrets.enabled=false)
  [[ -n "${OPENAI_API_KEY:-}" ]] && secret_args+=(--set-string "env.OPENAI_API_KEY=${OPENAI_API_KEY}")
  [[ -n "${API_KEY:-}" ]] && secret_args+=(--set-string "env.API_KEY=${API_KEY}")
  [[ -n "${POSTGRES_HOST:-}" ]] && secret_args+=(--set-string "env.POSTGRES_HOST=${POSTGRES_HOST}")
  [[ -n "${POSTGRES_PORT:-}" ]] && secret_args+=(--set-string "env.POSTGRES_PORT=${POSTGRES_PORT}")
  [[ -n "${POSTGRES_DB:-}" ]] && secret_args+=(--set-string "env.POSTGRES_DB=${POSTGRES_DB}")
  [[ -n "${POSTGRES_USER:-}" ]] && secret_args+=(--set-string "env.POSTGRES_USER=${POSTGRES_USER}")
  [[ -n "${POSTGRES_PASSWORD:-}" ]] && secret_args+=(--set-string "env.POSTGRES_PASSWORD=${POSTGRES_PASSWORD}")
  [[ -n "${REDIS_HOST:-}" ]] && secret_args+=(--set-string "env.REDIS_HOST=${REDIS_HOST}")
  [[ -n "${REDIS_PORT:-}" ]] && secret_args+=(--set-string "env.REDIS_PORT=${REDIS_PORT}")
  [[ -n "${REDIS_PASSWORD:-}" ]] && secret_args+=(--set-string "env.REDIS_PASSWORD=${REDIS_PASSWORD}")
  [[ -n "${NEO4J_URI:-}" ]] && secret_args+=(--set-string "env.NEO4J_URI=${NEO4J_URI}")
  [[ -n "${NEO4J_USER:-}" ]] && secret_args+=(--set-string "env.NEO4J_USER=${NEO4J_USER}")
  [[ -n "${NEO4J_PASSWORD:-}" ]] && secret_args+=(--set-string "env.NEO4J_PASSWORD=${NEO4J_PASSWORD}")
else
  secret_args=(--set externalSecrets.enabled=true)
fi

helm upgrade --install "${RELEASE_NAME}" deploy/helm/asterscope \
  --namespace "${NAMESPACE}" \
  --set-string "fullnameOverride=${RELEASE_NAME}" \
  --set-string "image.api.repository=${api_parts[0]}" \
  --set-string "image.api.tag=${api_parts[1]}" \
  --set-string "image.retrieval.repository=${retrieval_parts[0]}" \
  --set-string "image.retrieval.tag=${retrieval_parts[1]}" \
  "${secret_args[@]}"

kubectl -n "${NAMESPACE}" rollout status deployment/"${RELEASE_NAME}"-api --timeout=300s
kubectl -n "${NAMESPACE}" rollout status deployment/"${RELEASE_NAME}"-retrieval --timeout=300s

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-unit}"
REPORT_DIR="${REPORT_DIR:-reports}"
mkdir -p "${REPORT_DIR}"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export API_KEY="${API_KEY:-jenkins-ci-key}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-test-openai-key}"
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_DB="${POSTGRES_DB:-asterscope}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

case "${MODE}" in
  unit)
    python -m pytest -q \
      tests/test_parser.py \
      tests/test_memory.py \
      tests/test_cypher.py \
      tests/test_resilience.py \
      tests/test_evidence_selection.py \
      tests/test_generation_packing.py \
      tests/test_benchmark_support.py \
      tests/test_benchmark_reporting.py \
      --junitxml="${REPORT_DIR}/unit.xml"
    ;;
  integration)
    python -m pytest -q \
      tests/test_e2e.py \
      --junitxml="${REPORT_DIR}/integration.xml"
    ;;
  all)
    "${0}" unit
    "${0}" integration
    ;;
  *)
    echo "Unknown test mode: ${MODE}" >&2
    exit 2
    ;;
esac

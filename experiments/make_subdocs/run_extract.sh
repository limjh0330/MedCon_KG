#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRACT_SCRIPT="${REPO_ROOT}/experiments/make_subdocs/extract_info.py"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
  echo "extract_info.py not found: ${EXTRACT_SCRIPT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${EXTRACT_SCRIPT}" "$@"

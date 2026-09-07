#!/usr/bin/env bash
# Fetch FastAPI documentation from the official repository.
# Usage: ./indexing/fetch_docs.sh
#
# To switch to another doc source (e.g. Kubernetes), modify REPO_URL and SPARSE_PATH.

set -euo pipefail

DATA_DIR="data/raw"
REPO_URL="https://github.com/fastapi/fastapi.git"
SPARSE_PATH="docs/en/docs"
TARGET_DIR="${DATA_DIR}/fastapi"

if [ -d "${TARGET_DIR}" ]; then
    echo "Documentation already exists at ${TARGET_DIR}, skipping clone."
    echo "Run 'rm -rf ${TARGET_DIR}' to re-fetch."
    exit 0
fi

mkdir -p "${DATA_DIR}"

echo "Cloning ${REPO_URL} (sparse: ${SPARSE_PATH})..."
git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TARGET_DIR}"
cd "${TARGET_DIR}"
git sparse-checkout set "${SPARSE_PATH}"

DOC_COUNT=$(find "${SPARSE_PATH}" -name "*.md" | wc -l | tr -d ' ')
echo "✓ Done. Fetched ${DOC_COUNT} markdown files into ${TARGET_DIR}/${SPARSE_PATH}"

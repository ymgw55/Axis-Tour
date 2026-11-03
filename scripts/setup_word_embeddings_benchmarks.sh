#!/usr/bin/env bash
# Fetch word-embeddings-benchmarks at a pinned commit and apply our patch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/kudkudak/word-embeddings-benchmarks.git"
UPSTREAM_COMMIT="c78272b8c1374e5e518915a240ab2b348b59f44e"
TARGET_DIR="${REPO_ROOT}/word-embeddings-benchmarks"

if [ -e "${TARGET_DIR}" ]; then
    echo "${TARGET_DIR} already exists. Remove it first to re-run setup." >&2
    exit 1
fi

git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout --quiet "${UPSTREAM_COMMIT}"
git -C "${TARGET_DIR}" apply "${REPO_ROOT}/patches/word-embeddings-benchmarks.patch"
echo "word-embeddings-benchmarks is ready at ${TARGET_DIR}"

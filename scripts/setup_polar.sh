#!/usr/bin/env bash
# Fetch POLAR at a pinned commit and create POLAR/polar_glove.ipynb
# from POLAR/main.ipynb (outputs cleared) by applying our patch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/Sandipan99/POLAR.git"
UPSTREAM_COMMIT="2b5740eea23b4b5769fb10ceff893f269b9da285"
TARGET_DIR="${REPO_ROOT}/POLAR"

if [ -e "${TARGET_DIR}" ]; then
    echo "${TARGET_DIR} already exists. Remove it first to re-run setup." >&2
    exit 1
fi

git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout --quiet "${UPSTREAM_COMMIT}"
python - "${TARGET_DIR}/main.ipynb" "${TARGET_DIR}/polar_glove.ipynb" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
with open(sys.argv[2], "w") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
PY
git -C "${TARGET_DIR}" apply "${REPO_ROOT}/patches/polar_glove.patch"
echo "POLAR/polar_glove.ipynb is ready"

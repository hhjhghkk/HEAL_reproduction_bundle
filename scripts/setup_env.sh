#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Create a Python 3.8 environment manually, then run pip commands below."
  exit 1
fi

conda create -n heal-repro python=3.8 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate heal-repro
python -m pip install -U pip
python -m pip install -r "$ROOT/requirements_reproduction.txt"

if [ ! -d "$ROOT/workspace/embodied-agent-interface" ]; then
  echo "EAI source is missing. Run: bash scripts/fetch_all.sh"
  exit 1
fi
python -m pip install -e "$ROOT/workspace/embodied-agent-interface"

echo "Environment ready."
echo "Activate later with: conda activate heal-repro"

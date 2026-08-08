#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$ROOT/workspace"
mkdir -p "$WS"

EAI_REPO="https://github.com/embodied-agent-interface/embodied-agent-interface.git"
HEAL_REPO="Trishna13/HEAL"
EAI_SDIST_URL="https://files.pythonhosted.org/packages/62/ae/be6df7f55b175dec2564662bd95c2d93fed7373b63cb8f567ee77b8c114b/eai_eval-1.0.5.tar.gz"
EAI_SDIST_SHA256="61a13f38b8e60414540b44cbcb51e3b8cddd2fba9150fc0965e089940715dabf"

printf '\n[1/2] Fetch EAI source code\n'
if [ -d "$WS/embodied-agent-interface/.git" ]; then
  git -C "$WS/embodied-agent-interface" pull --ff-only
elif command -v git >/dev/null 2>&1; then
  if ! git clone --depth 1 "$EAI_REPO" "$WS/embodied-agent-interface"; then
    echo "git clone failed; falling back to the pinned PyPI source distribution."
    python "$ROOT/scripts/fetch_eai_sdist.py"
  fi
else
  python "$ROOT/scripts/fetch_eai_sdist.py"
fi

printf '\n[2/2] Fetch HEAL dataset\n'
python "$ROOT/scripts/download_heal_dataset.py"

echo
echo "Done."
echo "EAI:  $WS/embodied-agent-interface"
echo "HEAL: $WS/HEAL_dataset"

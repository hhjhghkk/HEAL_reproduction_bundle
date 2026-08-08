#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-virtualhome}"
MODE="${2:-generate_prompts}"

eai-eval \
  --dataset "$DATASET" \
  --eval-type goal_interpretation \
  --mode "$MODE"

#!/usr/bin/env bash
set -euo pipefail

# Sequentially run refinement-only and refinement+tracking on one GPU, then
# compare both with the existing test_09 coarse diagnostic.
#
# Usage:
#   GPU=0 bash llada_server_fix/run_huskyqa_refinement_experiments.sh 5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$ROOT/yzx_test"
RUNNER="$TEST_ROOT/llada_server_fix/run_huskyqa_online_latent_variant.sh"
PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
GPU="${GPU:-0}"
LIMIT="${1:-5}"
COARSE_ROOT="${COARSE_ROOT:-$TEST_ROOT/test/test_09_online_latent_anchor}"
REFINE_ROOT="${REFINE_ROOT:-$TEST_ROOT/test/test_10_refine_only}"
TRACKING_ROOT="${TRACKING_ROOT:-$TEST_ROOT/test/test_11_refine_tracking}"
REFINE_PORT="${REFINE_PORT:-7394}"
TRACKING_PORT="${TRACKING_PORT:-7395}"
COMPARISON_DIR="${COMPARISON_DIR:-$TRACKING_ROOT/comparison}"

COARSE_TIMING="$COARSE_ROOT/timing/huskyqa_full_timings.jsonl"
if [[ ! -f "$COARSE_TIMING" ]]; then
  echo "Missing coarse baseline: $COARSE_TIMING" >&2
  exit 2
fi

GPU="$GPU" PORT="$REFINE_PORT" PYTHON="$PYTHON" \
  bash "$RUNNER" online_latent_refine "$REFINE_ROOT" "$LIMIT"

GPU="$GPU" PORT="$TRACKING_PORT" PYTHON="$PYTHON" \
  bash "$RUNNER" online_latent_refine_tracking "$TRACKING_ROOT" "$LIMIT"

mkdir -p "$COMPARISON_DIR"
"$PYTHON" "$TEST_ROOT/analyze_anchor_refinement_variants.py" \
  --coarse-timing "$COARSE_TIMING" \
  --refine-timing "$REFINE_ROOT/timing/huskyqa_full_timings.jsonl" \
  --tracking-timing "$TRACKING_ROOT/timing/huskyqa_full_timings.jsonl" \
  --output-dir "$COMPARISON_DIR" \
  |& tee "$TRACKING_ROOT/logs/comparison.log"

echo "Completed refinement experiments"
echo "Comparison: $COMPARISON_DIR/report.md"

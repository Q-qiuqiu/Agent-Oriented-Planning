#!/usr/bin/env bash
set -euo pipefail

# Run one read-only refined online latent diagnostic variant.
# Usage:
#   GPU=0 PORT=7394 bash run_huskyqa_online_latent_variant.sh \
#     online_latent_refine test/test_10_refine_only 5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$ROOT/yzx_test"
SERVER_DIR="$TEST_ROOT/llada_server_fix"

METHOD="${1:-}"
OUTPUT_ROOT="${2:-}"
LIMIT="${3:-5}"
if [[ "$METHOD" != "online_latent_refine" \
  && "$METHOD" != "online_latent_refine_tracking" ]]; then
  echo "METHOD must be online_latent_refine or online_latent_refine_tracking" >&2
  exit 2
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT is required" >&2
  exit 2
fi

PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
GPU="${GPU:-0}"
PORT="${PORT:-7394}"
PLANNER_SCRIPT="${PLANNER_SCRIPT:-$TEST_ROOT/huskyqa_test/build_subtask_full_benchmark_v2.py}"
HUSKYQA_INPUT="${HUSKYQA_INPUT:-$TEST_ROOT/benchmarks/huskyqa/huskyqa_raw.json}"
PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-1024}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-32}"
ANCHOR_MIN_LOGIT_MARGIN="${ANCHOR_MIN_LOGIT_MARGIN:--6.0}"
REFINEMENT_RADIUS="${REFINEMENT_RADIUS:-4}"
REFINEMENT_ANCHOR_WEIGHT="${REFINEMENT_ANCHOR_WEIGHT:-1.0}"
REFINEMENT_AGENT_WEIGHT="${REFINEMENT_AGENT_WEIGHT:-1.0}"
MAX_TRACK_MISSES="${MAX_TRACK_MISSES:-3}"
MAX_TRACK_MATCH_DISTANCE="${MAX_TRACK_MATCH_DISTANCE:-64}"
TRACK_STABLE_OBSERVATIONS="${TRACK_STABLE_OBSERVATIONS:-2}"
TRACK_SCORE_WEIGHT="${TRACK_SCORE_WEIGHT:-0.05}"
RESUME="${RESUME:-0}"

if [[ "$LIMIT" != "all" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer or all" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$PLANNER_SCRIPT" || ! -f "$HUSKYQA_INPUT" ]]; then
  echo "Missing Python, planner script, or HuskyQA input" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
RESULT_DIR="$OUTPUT_ROOT/results"
TIMING_DIR="$OUTPUT_ROOT/timing"
LOG_DIR="$OUTPUT_ROOT/logs"
ANALYSIS_DIR="$OUTPUT_ROOT/analysis"
TIMING_FILE="$TIMING_DIR/huskyqa_full_timings.jsonl"

if [[ "$RESUME" != "1" ]]; then
  for path in \
    "$RESULT_DIR/plans.json" "$RESULT_DIR/expanded.json" "$TIMING_FILE" \
    "$LOG_DIR/server.log" "$LOG_DIR/client.log"; do
    if [[ -e "$path" ]]; then
      echo "Refusing to overwrite existing experiment output: $path" >&2
      exit 2
    fi
  done
fi
mkdir -p "$RESULT_DIR" "$TIMING_DIR" "$LOG_DIR" "$ANALYSIS_DIR"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

if command -v ss >/dev/null 2>&1 \
  && ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
  echo "Port $PORT is already in use" >&2
  exit 2
fi

printf '%s\n' \
  "method=$METHOD" "diagnostic_only=true" "gpu=$GPU" "port=$PORT" \
  "limit=$LIMIT" "model=$MODEL" "cache_mode=dual" "block_size=32" \
  "steps_per_block=32" "threshold=0.9" \
  "anchor_min_logit_margin=$ANCHOR_MIN_LOGIT_MARGIN" \
  "refinement_radius=$REFINEMENT_RADIUS" \
  "refinement_anchor_weight=$REFINEMENT_ANCHOR_WEIGHT" \
  "refinement_agent_weight=$REFINEMENT_AGENT_WEIGHT" \
  "max_track_misses=$MAX_TRACK_MISSES" \
  "max_track_match_distance=$MAX_TRACK_MATCH_DISTANCE" \
  "track_stable_observations=$TRACK_STABLE_OBSERVATIONS" \
  "track_score_weight=$TRACK_SCORE_WEIGHT" "extra_model_forwards=0" \
  >"$OUTPUT_ROOT/run_config.txt"

echo "Starting $METHOD | GPU=$GPU port=$PORT"
(
  cd "$SERVER_DIR"
  exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON" llada_server.py \
    --method "$METHOD" \
    --model_path "$MODEL" --served_model_name "$MODEL" \
    --device cuda --host 127.0.0.1 --port "$PORT" \
    --cache_mode dual --block_size 32 --max_gen_length 1024 \
    --steps_per_block 32 --threshold 0.9 \
    --agent_anchor_margin "$ANCHOR_MIN_LOGIT_MARGIN" \
    --oracle-score-chunk-size "$SCORE_CHUNK_SIZE" \
    --online-refinement-radius "$REFINEMENT_RADIUS" \
    --online-refinement-anchor-weight "$REFINEMENT_ANCHOR_WEIGHT" \
    --online-refinement-agent-weight "$REFINEMENT_AGENT_WEIGHT" \
    --online-max-track-misses "$MAX_TRACK_MISSES" \
    --online-max-track-match-distance "$MAX_TRACK_MATCH_DISTANCE" \
    --online-track-stable-observations "$TRACK_STABLE_OBSERVATIONS" \
    --online-track-score-weight "$TRACK_SCORE_WEIGHT" \
    --plan_json_repair --agent-timing-log-dir "$TIMING_DIR" \
    --log_level info
) >"$LOG_DIR/server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server exited during startup; inspect $LOG_DIR/server.log" >&2
    exit 3
  fi
  if NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Server startup timed out; inspect $LOG_DIR/server.log" >&2
  exit 3
fi

limit_args=()
if [[ "$LIMIT" != "all" ]]; then
  limit_args=(--limit "$LIMIT")
fi
(
  cd "$TEST_ROOT"
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    "$PYTHON" "$PLANNER_SCRIPT" \
    --input "$HUSKYQA_INPUT" \
    --planner-api-url "http://127.0.0.1:$PORT/v1" \
    --planner-api-key empty --planner-model "$MODEL" \
    --planner-temperature 0 --planner-max-tokens "$PLANNER_MAX_TOKENS" \
    --timeout "$REQUEST_TIMEOUT" "${limit_args[@]}" \
    --plans-output "$RESULT_DIR/plans.json" \
    --benchmark-output "$RESULT_DIR/expanded.json"
) |& tee "$LOG_DIR/client.log"

stop_server
server_pid=""

"$PYTHON" "$TEST_ROOT/analyze_online_latent_anchor.py" \
  --timing "$TIMING_FILE" --output-dir "$ANALYSIS_DIR" \
  |& tee "$LOG_DIR/analysis.log"
echo "Completed: $OUTPUT_ROOT"

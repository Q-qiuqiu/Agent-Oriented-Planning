#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only online latent Agent-anchor experiment.
#
# Usage:
#   GPU=0 bash llada_server_fix/run_huskyqa_online_latent_anchor.sh \
#     test/test_09_online_latent_anchor 5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$ROOT/yzx_test"
SERVER_DIR="$TEST_ROOT/llada_server_fix"

OUTPUT_ROOT="${1:-$TEST_ROOT/test/test_09_online_latent_anchor}"
LIMIT="${2:-5}"
PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
GPU="${GPU:-0}"
PORT="${PORT:-7393}"
PLANNER_SCRIPT="${PLANNER_SCRIPT:-$TEST_ROOT/huskyqa_test/build_subtask_full_benchmark_v2.py}"
HUSKYQA_INPUT="${HUSKYQA_INPUT:-$TEST_ROOT/benchmarks/huskyqa/huskyqa_raw.json}"
PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-1024}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-32}"
ANCHOR_MIN_LOGIT_MARGIN="${ANCHOR_MIN_LOGIT_MARGIN:--6.0}"
ANCHOR_POSITION_TOLERANCE="${ANCHOR_POSITION_TOLERANCE:-4}"
ANCHOR_STABLE_OBSERVATIONS="${ANCHOR_STABLE_OBSERVATIONS:-2}"
RESUME="${RESUME:-0}"

if [[ "$LIMIT" != "all" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer or 'all': $LIMIT" >&2
  exit 2
fi
if [[ ! "$SCORE_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SCORE_CHUNK_SIZE must be positive" >&2
  exit 2
fi
if [[ ! "$ANCHOR_POSITION_TOLERANCE" =~ ^[0-9]+$ ]]; then
  echo "ANCHOR_POSITION_TOLERANCE must be non-negative" >&2
  exit 2
fi
if [[ ! "$ANCHOR_STABLE_OBSERVATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ANCHOR_STABLE_OBSERVATIONS must be positive" >&2
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
    "$RESULT_DIR/plans.json" \
    "$RESULT_DIR/expanded.json" \
    "$TIMING_FILE" \
    "$LOG_DIR/server.log" \
    "$LOG_DIR/client.log"; do
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
  "method=online_latent_diagnostic" \
  "diagnostic_only=true" \
  "gpu=$GPU" \
  "port=$PORT" \
  "limit=$LIMIT" \
  "model=$MODEL" \
  "cache_mode=dual" \
  "block_size=32" \
  "steps_per_block=32" \
  "threshold=0.9" \
  "score_chunk_size=$SCORE_CHUNK_SIZE" \
  "anchor_min_logit_margin=$ANCHOR_MIN_LOGIT_MARGIN" \
  "anchor_position_tolerance=$ANCHOR_POSITION_TOLERANCE" \
  "anchor_stable_observations=$ANCHOR_STABLE_OBSERVATIONS" \
  "extra_model_forwards=0" \
  >"$OUTPUT_ROOT/run_config.txt"

echo "Starting online-latent diagnostic | GPU=$GPU port=$PORT"
(
  cd "$SERVER_DIR"
  exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON" llada_server.py \
    --method online_latent_diagnostic \
    --model_path "$MODEL" \
    --served_model_name "$MODEL" \
    --device cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    --cache_mode dual \
    --block_size 32 \
    --max_gen_length 1024 \
    --steps_per_block 32 \
    --threshold 0.9 \
    --agent_anchor_margin "$ANCHOR_MIN_LOGIT_MARGIN" \
    --oracle-score-chunk-size "$SCORE_CHUNK_SIZE" \
    --online-anchor-position-tolerance "$ANCHOR_POSITION_TOLERANCE" \
    --online-anchor-stable-observations "$ANCHOR_STABLE_OBSERVATIONS" \
    --plan_json_repair \
    --agent-timing-log-dir "$TIMING_DIR" \
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

echo "Running HuskyQA limit=$LIMIT"
(
  cd "$TEST_ROOT"
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
    "$PYTHON" "$PLANNER_SCRIPT" \
    --input "$HUSKYQA_INPUT" \
    --planner-api-url "http://127.0.0.1:$PORT/v1" \
    --planner-api-key empty \
    --planner-model "$MODEL" \
    --planner-temperature 0 \
    --planner-max-tokens "$PLANNER_MAX_TOKENS" \
    --timeout "$REQUEST_TIMEOUT" \
    "${limit_args[@]}" \
    --plans-output "$RESULT_DIR/plans.json" \
    --benchmark-output "$RESULT_DIR/expanded.json"
) |& tee "$LOG_DIR/client.log"

stop_server
server_pid=""

"$PYTHON" "$TEST_ROOT/analyze_online_latent_anchor.py" \
  --timing "$TIMING_FILE" \
  --output-dir "$ANALYSIS_DIR" |& tee "$LOG_DIR/analysis.log"

echo "Completed: $OUTPUT_ROOT"

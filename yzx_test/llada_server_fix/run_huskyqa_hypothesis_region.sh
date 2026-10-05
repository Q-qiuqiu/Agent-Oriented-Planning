#!/usr/bin/env bash
set -euo pipefail

# Run the read-only merged/validated region hypothesis diagnostic.
# Usage: GPU=0 PORT=7397 bash run_huskyqa_hypothesis_region.sh \
#   test/test_13_hypothesis_region 5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$ROOT/yzx_test"
SERVER_DIR="$TEST_ROOT/llada_server_fix"
OUTPUT_ROOT="${1:-$TEST_ROOT/test/test_13_hypothesis_region}"
LIMIT="${2:-5}"
PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
GPU="${GPU:-0}"
PORT="${PORT:-7397}"
PLANNER_SCRIPT="${PLANNER_SCRIPT:-$TEST_ROOT/huskyqa_test/build_subtask_full_benchmark_v2.py}"
HUSKYQA_INPUT="${HUSKYQA_INPUT:-$TEST_ROOT/benchmarks/huskyqa/huskyqa_raw.json}"
PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-1024}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-32}"
ANCHOR_MIN_LOGIT_MARGIN="${ANCHOR_MIN_LOGIT_MARGIN:--6.0}"
REGION_RADIUS="${REGION_RADIUS:-4}"
MERGE_DISTANCE="${MERGE_DISTANCE:-6}"
MERGE_GAP="${MERGE_GAP:-2}"
MIN_SEEN="${MIN_SEEN:-2}"
MIN_SUPPORT="${MIN_SUPPORT:-0.5}"
MAX_CENTER_JUMP="${MAX_CENTER_JUMP:-12}"
DUPLICATE_OBSERVATIONS="${DUPLICATE_OBSERVATIONS:-2}"
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
  "method=online_latent_hypothesis" "diagnostic_only=true" \
  "gpu=$GPU" "port=$PORT" "limit=$LIMIT" "model=$MODEL" \
  "region_radius=$REGION_RADIUS" "region_scoring=agent_only_top2_mean" \
  "merge_distance=$MERGE_DISTANCE" "merge_gap=$MERGE_GAP" \
  "min_seen=$MIN_SEEN" "min_support=$MIN_SUPPORT" \
  "max_center_jump=$MAX_CENTER_JUMP" \
  "duplicate_observations=$DUPLICATE_OBSERVATIONS" \
  "extra_model_forwards=0" >"$OUTPUT_ROOT/run_config.txt"

echo "Starting online_latent_hypothesis | GPU=$GPU port=$PORT"
(
  cd "$SERVER_DIR"
  exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON" llada_server.py \
    --method online_latent_hypothesis \
    --model_path "$MODEL" --served_model_name "$MODEL" \
    --device cuda --host 127.0.0.1 --port "$PORT" \
    --cache_mode dual --block_size 32 --max_gen_length 1024 \
    --steps_per_block 32 --threshold 0.9 \
    --agent_anchor_margin "$ANCHOR_MIN_LOGIT_MARGIN" \
    --oracle-score-chunk-size "$SCORE_CHUNK_SIZE" \
    --online-region-radius "$REGION_RADIUS" \
    --online-region-top-k 2 --online-region-temperature 1.0 \
    --online-region-main-aggregation top2_mean \
    --online-max-track-misses "$MAX_TRACK_MISSES" \
    --online-max-track-match-distance "$MAX_TRACK_MATCH_DISTANCE" \
    --online-track-stable-observations "$TRACK_STABLE_OBSERVATIONS" \
    --online-track-score-weight "$TRACK_SCORE_WEIGHT" \
    --hypothesis-merge-distance "$MERGE_DISTANCE" \
    --hypothesis-merge-gap "$MERGE_GAP" \
    --hypothesis-min-seen "$MIN_SEEN" \
    --hypothesis-min-support "$MIN_SUPPORT" \
    --hypothesis-max-center-jump "$MAX_CENTER_JUMP" \
    --hypothesis-duplicate-observations "$DUPLICATE_OBSERVATIONS" \
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
"$PYTHON" "$TEST_ROOT/analyze_hypothesis_region.py" \
  --timing "$TIMING_FILE" --output-dir "$ANALYSIS_DIR" \
  |& tee "$LOG_DIR/analysis.log"
echo "Completed: $OUTPUT_ROOT"
echo "Report: $ANALYSIS_DIR/report.md"

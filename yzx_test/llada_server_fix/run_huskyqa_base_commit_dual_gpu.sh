#!/usr/bin/env bash
set -euo pipefail

# Run the same HuskyQA planner workload against base and commit concurrently.
# Each method owns one GPU, one port, and an isolated result/log directory.
#
# Usage:
#   GPU_BASE=0 GPU_COMMIT=1 \
#     bash run_huskyqa_base_commit_dual_gpu.sh [OUTPUT_ROOT] [LIMIT]
#
# Examples:
#   bash run_huskyqa_base_commit_dual_gpu.sh \
#     benchmarks/huskyqa_smoke/run_01 5
#   GPU_BASE=2 GPU_COMMIT=3 \
#     bash run_huskyqa_base_commit_dual_gpu.sh \
#     /data/home/yzx/huskyqa_runs/full_01 all
#
# Optional environment variables:
#   PYTHON, MODEL, PLANNER_SCRIPT, HUSKYQA_INPUT
#   GPU_BASE, GPU_COMMIT, PORT_BASE, PORT_COMMIT
#   PLANNER_MAX_TOKENS, REQUEST_TIMEOUT, STARTUP_TIMEOUT
#   RESUME=1  Allow the planner/timing writers to resume an existing run.

usage() {
  sed -n '3,22p' "${BASH_SOURCE[0]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$ROOT/yzx_test"
SERVER_DIR="$TEST_ROOT/llada_server_fix"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${1:-$TEST_ROOT/benchmarks/huskyqa_base_commit/$RUN_STAMP}"
LIMIT="${2:-5}"

PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
PLANNER_SCRIPT="${PLANNER_SCRIPT:-huskyqa_test/build_subtask_full_benchmark_v2.py}"
HUSKYQA_INPUT="${HUSKYQA_INPUT:-$TEST_ROOT/benchmarks/huskyqa/huskyqa_raw.json}"

GPU_BASE="${GPU_BASE:-0}"
GPU_COMMIT="${GPU_COMMIT:-1}"
PORT_BASE="${PORT_BASE:-7390}"
PORT_COMMIT="${PORT_COMMIT:-7391}"

PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-1024}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
RESUME="${RESUME:-0}"

if [[ "$PLANNER_SCRIPT" != /* ]]; then
  PLANNER_SCRIPT="$TEST_ROOT/$PLANNER_SCRIPT"
fi

if [[ "$GPU_BASE" == "$GPU_COMMIT" ]]; then
  echo "GPU_BASE and GPU_COMMIT must be different: $GPU_BASE" >&2
  exit 2
fi
if [[ "$PORT_BASE" == "$PORT_COMMIT" ]]; then
  echo "PORT_BASE and PORT_COMMIT must be different: $PORT_BASE" >&2
  exit 2
fi
if [[ "$LIMIT" != "all" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer or 'all': $LIMIT" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1: $RESUME" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$PLANNER_SCRIPT" ]]; then
  echo "Planner script not found: $PLANNER_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$HUSKYQA_INPUT" ]]; then
  echo "HuskyQA input not found: $HUSKYQA_INPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

stop_process() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

wait_for_server() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

prepare_output() {
  local method="$1"
  local method_root="$OUTPUT_ROOT/$method"
  local occupied=(
    "$method_root/results/plans.json"
    "$method_root/results/expanded.json"
    "$method_root/timing/huskyqa_full_timings.jsonl"
    "$method_root/logs/server.log"
    "$method_root/logs/client.log"
  )
  local path
  if [[ "$RESUME" != "1" ]]; then
    for path in "${occupied[@]}"; do
      if [[ -e "$path" ]]; then
        echo "Refusing existing $method output: $path" >&2
        echo "Choose a new OUTPUT_ROOT or set RESUME=1." >&2
        return 1
      fi
    done
  fi
  mkdir -p \
    "$method_root/results" \
    "$method_root/timing" \
    "$method_root/logs"
}

run_planner_client() {
  local method="$1"
  local port="$2"
  local method_root="$OUTPUT_ROOT/$method"
  local limit_args=()
  if [[ "$LIMIT" != "all" ]]; then
    limit_args=(--limit "$LIMIT")
  fi

  (
    cd "$TEST_ROOT"
    NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      "$PYTHON" "$PLANNER_SCRIPT" \
      --input "$HUSKYQA_INPUT" \
      --planner-api-url "http://127.0.0.1:$port/v1" \
      --planner-api-key empty \
      --planner-model "$MODEL" \
      --planner-temperature 0 \
      --planner-max-tokens "$PLANNER_MAX_TOKENS" \
      --timeout "$REQUEST_TIMEOUT" \
      "${limit_args[@]}" \
      --plans-output "$method_root/results/plans.json" \
      --benchmark-output "$method_root/results/expanded.json"
  ) |& tee "$method_root/logs/client.log"
}

run_method() {
  local method="$1"
  local gpu="$2"
  local port="$3"
  local method_root="$OUTPUT_ROOT/$method"
  local server_pid=""

  trap 'stop_process "$server_pid"' EXIT INT TERM
  prepare_output "$method"

  if command -v ss >/dev/null 2>&1 \
    && ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
    echo "[$method] port $port is already in use" >&2
    return 2
  fi

  echo "[$method] starting server | gpu=$gpu port=$port"
  (
    cd "$SERVER_DIR"
    exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      "$PYTHON" llada_server.py \
      --method "$method" \
      --model_path "$MODEL" \
      --served_model_name "$MODEL" \
      --device cuda \
      --host 127.0.0.1 \
      --port "$port" \
      --cache_mode dual \
      --block_size 32 \
      --max_gen_length 1024 \
      --steps_per_block 32 \
      --threshold 0.9 \
      --plan_json_repair \
      --agent-timing-log-dir "$method_root/timing" \
      --log_level info
  ) >"$method_root/logs/server.log" 2>&1 &
  server_pid=$!

  if ! wait_for_server "$port" "$server_pid"; then
    echo "[$method] server failed; inspect $method_root/logs/server.log" >&2
    return 3
  fi

  echo "[$method] server ready; running HuskyQA limit=$LIMIT"
  run_planner_client "$method" "$port"
  echo "[$method] completed"

  stop_process "$server_pid"
  server_pid=""
  trap - EXIT INT TERM
}

BASE_LANE_PID=""
COMMIT_LANE_PID=""

stop_lanes() {
  stop_process "$BASE_LANE_PID"
  stop_process "$COMMIT_LANE_PID"
}
trap stop_lanes INT TERM

echo "Output root: $OUTPUT_ROOT"
echo "Base:   GPU $GPU_BASE, port $PORT_BASE"
echo "Commit: GPU $GPU_COMMIT, port $PORT_COMMIT"

printf '%s\n' \
  "output_root=$OUTPUT_ROOT" \
  "limit=$LIMIT" \
  "python=$PYTHON" \
  "model=$MODEL" \
  "planner_script=$PLANNER_SCRIPT" \
  "huskyqa_input=$HUSKYQA_INPUT" \
  "gpu_base=$GPU_BASE" \
  "gpu_commit=$GPU_COMMIT" \
  "port_base=$PORT_BASE" \
  "port_commit=$PORT_COMMIT" \
  "commit_policy=latent_region_stable2" \
  "hypothesis_min_seen=2" \
  "hypothesis_min_support=0.5" \
  "region_radius=4" \
  "region_scorer=agent_only_top2_mean" \
  "latent_stable_observations=2" \
  "extra_model_forwards=0" \
  >"$OUTPUT_ROOT/run_config.txt"

run_method base "$GPU_BASE" "$PORT_BASE" &
BASE_LANE_PID=$!
run_method commit "$GPU_COMMIT" "$PORT_COMMIT" &
COMMIT_LANE_PID=$!

status=0
wait "$BASE_LANE_PID" || status=$?
wait "$COMMIT_LANE_PID" || status=$?
trap - INT TERM

if [[ "$status" -ne 0 ]]; then
  echo "Run failed with status $status. Inspect per-method logs under $OUTPUT_ROOT." >&2
  exit "$status"
fi

echo "Completed base and commit HuskyQA runs."
echo "Base results:   $OUTPUT_ROOT/base/results"
echo "Base timing:    $OUTPUT_ROOT/base/timing/huskyqa_full_timings.jsonl"
echo "Commit results: $OUTPUT_ROOT/commit/results"
echo "Commit timing:  $OUTPUT_ROOT/commit/timing/huskyqa_full_timings.jsonl"

echo "Verifying paired generation invariants and final commit metrics"
VERIFY_ARGS=()
if [[ "$LIMIT" != "all" ]]; then
  VERIFY_ARGS=(--expected-requests "$LIMIT")
fi
"$PYTHON" "$TEST_ROOT/verify_final_commit_run.py" \
  "$OUTPUT_ROOT" "${VERIFY_ARGS[@]}"
RUN_ROOT="$OUTPUT_ROOT" "$PYTHON" "$TEST_ROOT/commit_time_caculate.py"

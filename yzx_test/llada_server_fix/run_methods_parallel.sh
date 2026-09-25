#!/usr/bin/env bash
set -euo pipefail

# Final four-method runner. Two GPU lanes run independently:
#   GPU_A: HuskyQA -> MMLU
#   GPU_B: IIRC -> ChronoQA
# Usage:
#   GPU_A=2 GPU_B=3 METHODS="base commit plan all" \
#     bash run_methods_parallel.sh TAG 5

TAG="${1:-final_methods_sanity_01}"
LIMIT="${2:-5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_DIR="$ROOT/yzx_test/llada_server_fix"
TEST_ROOT="$ROOT/yzx_test"
OUT_ROOT="$TEST_ROOT/benchmarks/final_methods/$TAG"
PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
METHODS="${METHODS:-base commit plan all}"

stop_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

wait_server() {
  local port="$1" pid="$2"
  for _ in $(seq 1 600); do
    if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
    if NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_client() {
  local benchmark="$1" port="$2" out="$3"
  local script="${benchmark}_test/build_subtask_full_benchmark_v2.py"
  local key_args=()
  if [[ "$benchmark" == "huskyqa" || "$benchmark" == "chronoqa" ]]; then
    key_args=(--planner-api-key empty)
  fi
  (
    cd "$TEST_ROOT"
    NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      "$PYTHON" "$script" \
      --planner-api-url "http://127.0.0.1:$port/v1" \
      "${key_args[@]}" \
      --planner-model "$MODEL" \
      --planner-max-tokens 1024 \
      --limit "$LIMIT" \
      --plans-output "$out/plans.json" \
      --benchmark-output "$out/expanded.json"
  ) |& tee "$out/client.log"
}

run_lane() {
  local gpu="$1" port="$2"
  shift 2
  local benchmark method pid="" out
  trap 'stop_server "$pid"' EXIT
  for benchmark in "$@"; do
    for method in $METHODS; do
      out="$OUT_ROOT/$benchmark/$method"
      mkdir -p "$out"
      if [[ -e "$out/timings.jsonl" || -e "$out/plans.json" ]]; then
        echo "Refusing existing output: $out (choose a new TAG)" >&2
        return 2
      fi
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
          --policy reasonplan \
          --agent_timing_slots 16 \
          --plan_json_repair \
          --agent_timing_log_path "$out/timings.jsonl" \
          --log_level info
      ) >"$out/server.log" 2>&1 &
      pid=$!
      if ! wait_server "$port" "$pid"; then
        echo "Server failed: $benchmark/$method; see $out/server.log" >&2
        return 3
      fi
      run_client "$benchmark" "$port" "$out"
      stop_server "$pid"
      pid=""
    done
  done
  trap - EXIT
}

mkdir -p "$OUT_ROOT"
status=0
run_lane "$GPU_A" 7390 huskyqa mmlu & lane_a=$!
run_lane "$GPU_B" 7391 iirc chronoqa & lane_b=$!
wait "$lane_a" || status=$?
wait "$lane_b" || status=$?
[[ "$status" -eq 0 ]] || exit "$status"
echo "Results: $OUT_ROOT"

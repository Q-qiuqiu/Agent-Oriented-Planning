#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_fixed_canvas_parallel.sh TAG sanity   # 5 + 5
#   bash run_fixed_canvas_parallel.sh TAG complete # resume missing baseline, then fixed 50 + 50
#   bash run_fixed_canvas_parallel.sh TAG full     # requires a 50 + 50 baseline
#   bash run_fixed_canvas_parallel.sh TAG all      # sanity, validate, then full

TAG="${1:-fixed_canvas_dynamic_end_01}"
STAGE="${2:-sanity}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_DIR="$ROOT/yzx_test/llada_server_fix"
TEST_ROOT="$ROOT/yzx_test"
OUT_ROOT="$TEST_ROOT/benchmarks/fixed_canvas/$TAG"
ALGORITHM_VERSION="fixed_canvas_dynamic_end_v3"
PYTHON="${PYTHON:-/home/yzx/miniconda3/envs/llada/bin/python}"
MODEL="${MODEL:-/data/labshare/Param/llada}"
HUSKY_GPU="${HUSKY_GPU:-6}"
MMLU_GPU="${MMLU_GPU:-7}"
METHODS=(fixed_canvas_vanilla fixed_canvas_plan_first)
BASELINE_ROOT="${BASELINE_ROOT:-$TEST_ROOT/benchmarks/fixed_canvas/fixed_canvas_earlystop_01}"

if [[ -d "$OUT_ROOT/huskyqa" && ! -f "$OUT_ROOT/algorithm_version.txt" ]]; then
  echo "Refusing to reuse legacy output directory: $OUT_ROOT" >&2
  echo "Use a new TAG (for example fixed_canvas_dynamic_end_01)." >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"
if [[ -f "$OUT_ROOT/algorithm_version.txt" ]]; then
  existing_version="$(<"$OUT_ROOT/algorithm_version.txt")"
  if [[ "$existing_version" != "$ALGORITHM_VERSION" ]]; then
    echo "Output TAG belongs to $existing_version, expected $ALGORITHM_VERSION" >&2
    exit 2
  fi
else
  printf '%s\n' "$ALGORITHM_VERSION" >"$OUT_ROOT/algorithm_version.txt"
fi

# Preserve the prior baseline files.  The complete stage copies its first five
# records into the new TAG and the benchmark builder resumes at query six.
# Other stages use a read-only link to the prior baseline.
for bench in huskyqa mmlu; do
  mkdir -p "$OUT_ROOT/$bench"
  if [[ ! -e "$OUT_ROOT/$bench/dual_vanilla" ]]; then
    if [[ ! -d "$BASELINE_ROOT/$bench/dual_vanilla" ]]; then
      echo "Missing prior baseline: $BASELINE_ROOT/$bench/dual_vanilla" >&2
      exit 2
    fi
    if [[ "$STAGE" == "complete" ]]; then
      cp -a "$BASELINE_ROOT/$bench/dual_vanilla" "$OUT_ROOT/$bench/dual_vanilla"
    else
      ln -s "$BASELINE_ROOT/$bench/dual_vanilla" "$OUT_ROOT/$bench/dual_vanilla"
    fi
  fi
done

check_full_baseline() {
  for bench in huskyqa mmlu; do
    baseline_count="$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$OUT_ROOT/$bench/dual_vanilla/plans.json")"
    if [[ "$baseline_count" -lt 50 ]]; then
      echo "Cannot run 50-query comparison: $bench has only $baseline_count baseline plans." >&2
      exit 2
    fi
  done
}

if [[ "$STAGE" == "full" || "$STAGE" == "all" ]]; then
  check_full_baseline
fi

stop_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

wait_server() {
  local port="$1"
  local pid="$2"
  for _ in $(seq 1 600); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Server process $pid exited before becoming healthy" >&2
      return 1
    fi
    if NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for port $port" >&2
  return 1
}

run_benchmark_worker() {
  local bench="$1"
  local gpu="$2"
  local port="$3"
  local limit="$4"
  shift 4
  local method out repair_option pid=""
  trap 'stop_server "$pid"' EXIT
  for method in "$@"; do
    if [[ "$method" == "dual_vanilla" ]]; then
      repair_option="--no-plan_json_repair"
    else
      repair_option="--plan_json_repair"
    fi
    out="$OUT_ROOT/$bench/$method"
    mkdir -p "$out"
    echo "[$bench][$method] loading model on GPU $gpu (limit=$limit)"
    (
      cd "$SERVER_DIR"
      exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$PYTHON" llada_server.py \
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
          --agent_probe_period 0 \
          --agent_timing_slots 16 \
          --structure_mode "$method" \
          --reasoning_budget 499 \
          --plan_budget 499 \
          "$repair_option" \
          --agent_timing_log_path "$out/timings.jsonl" \
          --log_level info
    ) >"$out/server_${limit}.log" 2>&1 &
    pid=$!
    wait_server "$port" "$pid"

    if [[ "$bench" == "huskyqa" ]]; then
      (
        cd "$TEST_ROOT"
        NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
          "$PYTHON" huskyqa_test/build_subtask_full_benchmark_v2.py \
            --planner-api-url "http://127.0.0.1:$port/v1" \
            --planner-api-key empty \
            --planner-model "$MODEL" \
            --planner-max-tokens 1024 \
            --limit "$limit" \
            --plans-output "$out/plans.json" \
            --benchmark-output "$out/expanded.json"
      ) |& tee "$out/client_${limit}.log"
    else
      (
        cd "$TEST_ROOT"
        NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
          "$PYTHON" mmlu_test/build_subtask_full_benchmark_v2.py \
            --planner-api-url "http://127.0.0.1:$port/v1" \
            --planner-model "$MODEL" \
            --planner-max-tokens 1024 \
            --limit "$limit" \
            --plans-output "$out/plans.json" \
            --benchmark-output "$out/expanded.json"
      ) |& tee "$out/client_${limit}.log"
    fi
    stop_server "$pid"
    pid=""
    echo "[$bench][$method] completed"
  done
  trap - EXIT
}

run_phase() {
  local limit="$1"
  shift
  run_benchmark_worker huskyqa "$HUSKY_GPU" 7310 "$limit" "$@" &
  local husky_pid=$!
  run_benchmark_worker mmlu "$MMLU_GPU" 7311 "$limit" "$@" &
  local mmlu_pid=$!
  local status=0
  wait "$husky_pid" || status=$?
  wait "$mmlu_pid" || status=$?
  return "$status"
}

case "$STAGE" in
  sanity)
    run_phase 5 "${METHODS[@]}"
    "$PYTHON" "$SERVER_DIR/analyze_canvas_capacity.py" "$OUT_ROOT" --model "$MODEL" --reasoning-budget 499 --plan-budget 499
    "$PYTHON" "$SERVER_DIR/analyze_fixed_canvas_experiment.py" "$OUT_ROOT"
    "$PYTHON" "$SERVER_DIR/check_fixed_canvas_sanity.py" "$OUT_ROOT" --expected-per-method 10
    ;;
  full)
    run_phase 50 "${METHODS[@]}"
    "$PYTHON" "$SERVER_DIR/analyze_canvas_capacity.py" "$OUT_ROOT" --model "$MODEL" --reasoning-budget 499 --plan-budget 499
    "$PYTHON" "$SERVER_DIR/analyze_fixed_canvas_experiment.py" "$OUT_ROOT"
    ;;
  all)
    run_phase 5 "${METHODS[@]}"
    "$PYTHON" "$SERVER_DIR/analyze_canvas_capacity.py" "$OUT_ROOT" --model "$MODEL" --reasoning-budget 499 --plan-budget 499
    "$PYTHON" "$SERVER_DIR/analyze_fixed_canvas_experiment.py" "$OUT_ROOT"
    "$PYTHON" "$SERVER_DIR/check_fixed_canvas_sanity.py" "$OUT_ROOT" --expected-per-method 10
    run_phase 50 "${METHODS[@]}"
    "$PYTHON" "$SERVER_DIR/analyze_canvas_capacity.py" "$OUT_ROOT" --model "$MODEL" --reasoning-budget 499 --plan-budget 499
    "$PYTHON" "$SERVER_DIR/analyze_fixed_canvas_experiment.py" "$OUT_ROOT"
    ;;
  complete)
    if [[ -L "$OUT_ROOT/huskyqa/dual_vanilla" || -L "$OUT_ROOT/mmlu/dual_vanilla" ]]; then
      echo "Use a new TAG for complete: its baseline must be an independent copy." >&2
      exit 2
    fi
    run_phase 50 dual_vanilla
    check_full_baseline
    run_phase 50 "${METHODS[@]}"
    "$PYTHON" "$SERVER_DIR/analyze_canvas_capacity.py" "$OUT_ROOT" --model "$MODEL" --reasoning-budget 499 --plan-budget 499
    "$PYTHON" "$SERVER_DIR/analyze_fixed_canvas_experiment.py" "$OUT_ROOT"
    ;;
  *)
    echo "stage must be sanity, full, all, or complete" >&2
    exit 2
    ;;
esac

echo "Results: $OUT_ROOT"

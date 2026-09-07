#!/usr/bin/env bash
set -uo pipefail

MODEL_SIZE="1b"
PLAN_VARIANT="full_llama3"

# Summary API configuration. These values override summary_evaluate.py.
SUMMARY_API_URL="http://10.137.144.97:7002/v1"
SUMMARY_API_KEY="empty"
#SUMMARY_MODEL="/data/labshare/Param/llada"
#SUMMARY_MODEL="/mnt/home/yzx/models/LLADA/"

SUMMARY_MODEL="/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct"
SUMMARY_TEMPERATURE="0.0"
SUMMARY_TIMEOUT="120"

# Run assignments sequentially; edit this list for each experiment batch.
ASSIGNMENTS=(
  "g_g_g"
  "q_q_q"
  "l_l_l"
  "m_m_m"
  "d_d_d"
  "qm_qm_qm"
  "qc_qc_qc"
  "i_i_i"
  "s_s_s"
)
SUMMARY_ARGS=(
  # "--force"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${TEST_ROOT}"

echo "Batch configuration"
echo "  benchmark=chronoqa"
echo "  stage=summary_evaluate"
echo "  model_size=${MODEL_SIZE}"
echo "  plan_variant=${PLAN_VARIANT}"
echo "  results_dir=chronoqa_test/results_${MODEL_SIZE}_${PLAN_VARIANT}"
echo "  assignments=${ASSIGNMENTS[*]}"
echo "  summary_model=${SUMMARY_MODEL}"
echo "  summary_api_url=${SUMMARY_API_URL}"
echo "====================="

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"
  output_path="chronoqa_test/results_${MODEL_SIZE}_${PLAN_VARIANT}/summary_result_${assignment}.json"

  if "${PYTHON_BIN}" -B "${SCRIPT_DIR}/summary_evaluate.py" \
      --assignment "${assignment}" \
      --model-size "${MODEL_SIZE}" \
      --plan-variant "${PLAN_VARIANT}" \
      --summary-api-url "${SUMMARY_API_URL}" \
      --summary-api-key "${SUMMARY_API_KEY}" \
      --summary-model "${SUMMARY_MODEL}" \
      --summary-temperature "${SUMMARY_TEMPERATURE}" \
      --summary-timeout "${SUMMARY_TIMEOUT}" \
      "${SUMMARY_ARGS[@]}" >/dev/null 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi

  "${PYTHON_BIN}" -B - "${output_path}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
success = sum(
    not row.get("summary_error")
    and isinstance(row.get("final_answer"), str)
    and bool(row["final_answer"].strip())
    for row in rows
)
report = {
    "count": len(rows),
    "success_count": success,
    "failure_count": len(rows) - success,
}
if int(sys.argv[2]):
    report["process_exit_code"] = int(sys.argv[2])
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

  if (( index + 1 < ${#ASSIGNMENTS[@]} )); then
    echo "====================="
  fi
done

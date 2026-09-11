#!/usr/bin/env bash
set -uo pipefail

MODEL_SIZE="1b"
PLAN_VARIANT="base_llama3"

# Judge API configuration. These values override subtask_hetro.py.
JUDGE_API_URL="http://10.137.144.97:7001/v1"
#JUDGE_API_URL="http://222.30.44.50:45104/v1/chat/completions"
JUDGE_API_KEY="empty"
JUDGE_MODEL="/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507"
#JUDGE_MODEL="/public/models/Qwen3-30B-A3B/"
JUDGE_TEMPERATURE="0.0"
JUDGE_TIMEOUT="120"

# Run assignments sequentially. Edit this list for each experiment batch.
ASSIGNMENTS=(
  "g_g_g"
  "l_l_l"
  "m_m_m"
  "q_q_q"
  "d_d_d"
  "qm_qm_qm"
  "qc_qc_qc"
  "i_i_i"
  "s_s_s"
)

# Optional arguments passed to every subtask_hetro.py judge invocation.
JUDGE_ARGS=(
  # "--force"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${TEST_ROOT}/batch_output.sh"
cd "${TEST_ROOT}"

echo "Batch configuration"
echo "  benchmark=huskyqa"
echo "  stage=subtask_judge"
echo "  model_size=${MODEL_SIZE}"
echo "  plan_variant=${PLAN_VARIANT}"
echo "  results_dir=huskyqa_test/results_${MODEL_SIZE}_${PLAN_VARIANT}"
echo "  assignments=${ASSIGNMENTS[*]}"
echo "  judge_model=${JUDGE_MODEL}"
echo "  judge_api_url=${JUDGE_API_URL}"
echo "====================="

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"

  output_path="huskyqa_test/results_${MODEL_SIZE}_${PLAN_VARIANT}/subtask_hetro_scores_${assignment}.json"

  if run_with_error_output "${PYTHON_BIN}" -B "${SCRIPT_DIR}/subtask_hetro.py" \
      --mode judge \
      --model-size "${MODEL_SIZE}" \
      --plan-variant "${PLAN_VARIANT}" \
      --assignment "${assignment}" \
      --judge-api-url "${JUDGE_API_URL}" \
      --judge-api-key "${JUDGE_API_KEY}" \
      --judge-model "${JUDGE_MODEL}" \
      --judge-temperature "${JUDGE_TEMPERATURE}" \
      --judge-timeout "${JUDGE_TIMEOUT}" \
      "${JUDGE_ARGS[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  "${PYTHON_BIN}" -B - "${output_path}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data.get("summary", {}) if isinstance(data, dict) else {}
else:
    report = {"error": f"Score file was not created: {path}"}
if exit_code:
    report = {**report, "process_exit_code": exit_code}
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

  if (( index + 1 < ${#ASSIGNMENTS[@]} )); then
    echo "====================="
  fi
done

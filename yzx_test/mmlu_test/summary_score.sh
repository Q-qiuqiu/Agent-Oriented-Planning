#!/usr/bin/env bash
set -uo pipefail

MODEL_SIZE="1b"
PLAN_VARIANT="full_llada"

# MMLU-Pro uses option exact match, so no external Judge configuration is needed.
ASSIGNMENTS=(
  "q_q_q"
  "g_g_g"
  "l_l_l"
  "m_m_m"
  "d_d_d"
  "qc_qc_qc"
  "qm_qm_qm"
  "i_i_i"
  "s_s_s"
)

SCORE_ARGS=()

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${TEST_ROOT}/batch_output.sh"
cd "${TEST_ROOT}"

echo "Batch configuration"
echo "  benchmark=mmlu"
echo "  stage=summary_score"
echo "  model_size=${MODEL_SIZE}"
echo "  plan_variant=${PLAN_VARIANT}"
echo "  results_dir=mmlu_test/results_${MODEL_SIZE}_${PLAN_VARIANT}"
echo "  assignments=${ASSIGNMENTS[*]}"
echo "  scoring=option_exact_match"
echo "====================="

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"

  output_path="mmlu_test/results_${MODEL_SIZE}_${PLAN_VARIANT}/summary_score_${assignment}.json"

  if run_with_error_output "${PYTHON_BIN}" -B "${SCRIPT_DIR}/summary_score.py" \
      --assignment "${assignment}" \
      --model-size "${MODEL_SIZE}" \
      --plan-variant "${PLAN_VARIANT}" \
      "${SCORE_ARGS[@]}"; then
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

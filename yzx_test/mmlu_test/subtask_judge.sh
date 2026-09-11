#!/usr/bin/env bash
set -uo pipefail

MODEL_SIZE="1b"
PLAN_VARIANT="base_llada"

# MMLU-Pro uses option exact match, so no external Judge API is required.
ASSIGNMENTS=(
  "g_g_g"
  "q_q_q"
  "l_l_l"
  "m_m_m"
  "s_s_s"
  "i_i_i"
  "qc_qc_qc"
  "qm_qm_qm"
  "d_d_d"
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
echo "  benchmark=mmlu"
echo "  stage=subtask_judge"
echo "  model_size=${MODEL_SIZE}"
echo "  plan_variant=${PLAN_VARIANT}"
echo "  results_dir=mmlu_test/results_${MODEL_SIZE}_${PLAN_VARIANT}"
echo "  assignments=${ASSIGNMENTS[*]}"
echo "  scoring=option_exact_match"
echo "====================="

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"

  output_path="mmlu_test/results_${MODEL_SIZE}_${PLAN_VARIANT}/subtask_hetro_scores_${assignment}.json"

  if run_with_error_output "${PYTHON_BIN}" -B "${SCRIPT_DIR}/subtask_hetro.py" \
      --mode judge \
      --model-size "${MODEL_SIZE}" \
      --plan-variant "${PLAN_VARIANT}" \
      --assignment "${assignment}" \
      "${JUDGE_ARGS[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  "${PYTHON_BIN}" -B -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    report = {
        "by_agent": summary.get("by_agent", {}),
        **{
            key: summary.get(key)
            for key in ("count", "correct", "accuracy", "parse_failure_count")
        },
    }
else:
    report = {"error": f"Score file was not created: {path}"}
if exit_code:
    report = {**report, "process_exit_code": exit_code}
print(json.dumps(report, ensure_ascii=False, indent=2))
' "${output_path}" "${exit_code}"

  if (( index + 1 < ${#ASSIGNMENTS[@]} )); then
    echo "====================="
  fi
done

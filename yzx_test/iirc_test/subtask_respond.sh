#!/usr/bin/env bash
set -uo pipefail

handle_interrupt() {
  trap - INT TERM
  printf '\nBatch interrupted; stopping remaining assignments.\n' >&2
  exit 130
}

trap handnle_interrupt INT TERM

MODEL_SIZE="1b"
PLAN_VARIANT="full_llama3"

# Run assignments sequentially. Keep this list aligned with later pipeline stages.
ASSIGNMENTS=(
  "q_q_q"
  "g_g_g"
  "l_l_l"
  "m_m_m"
  # "d_d_d"
  # "qm_qm_qm"
  # "qc_qc_qc"
  # "i_i_i"
  # "s_s_s"
)

# Optional arguments passed to every subtask_hetro.py respond invocation.
RESPOND_ARGS=(
  # "--force"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${TEST_ROOT}/batch_output.sh"
cd "${TEST_ROOT}"

echo "Batch configuration"
echo "  benchmark=iirc"
echo "  stage=subtask_respond"
echo "  model_size=${MODEL_SIZE}"
echo "  plan_variant=${PLAN_VARIANT}"
echo "  results_dir=iirc_test/results_${MODEL_SIZE}_${PLAN_VARIANT}"
echo "  assignments=${ASSIGNMENTS[*]}"
echo "====================="

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"

  output_path="iirc_test/results_${MODEL_SIZE}_${PLAN_VARIANT}/subtask_hetro_responses_${assignment}.json"

  if run_with_error_output "${PYTHON_BIN}" -B "${SCRIPT_DIR}/subtask_hetro.py" \
      --mode respond \
      --model-size "${MODEL_SIZE}" \
      --plan-variant "${PLAN_VARIANT}" \
      --assignment "${assignment}" \
      "${RESPOND_ARGS[@]}"; then
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
    rows = json.loads(path.read_text(encoding="utf-8"))
    success = sum(
        row.get("error") is None
        and bool(row.get("steps"))
        and all(
            step.get("error") is None and bool(step.get("response"))
            for step in row["steps"]
        )
        for row in rows
    )
    steps = [step for row in rows for step in row.get("steps", [])]
    report = {
        "count": len(rows),
        "success_count": success,
        "failure_count": len(rows) - success,
        "step_count": len(steps),
        "step_failure_count": sum(
            step.get("error") is not None or not step.get("response") for step in steps
        ),
    }
else:
    report = {"error": f"Response file was not created: {path}"}
if exit_code:
    report["process_exit_code"] = exit_code
print(json.dumps(report, ensure_ascii=False, indent=2))
' "${output_path}" "${exit_code}"

  if (( index + 1 < ${#ASSIGNMENTS[@]} )); then
    echo "====================="
  fi
done

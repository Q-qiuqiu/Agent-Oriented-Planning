#!/usr/bin/env bash
set -uo pipefail

# MMLU-Pro uses option exact match, so no external Judge configuration is needed.
ASSIGNMENTS=(
  "g_g_g"
  "q_q_q"
  "l_l_l"
  "m_m_m"
  "g_q_l"
)

SCORE_ARGS=()

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${TEST_ROOT}"

for index in "${!ASSIGNMENTS[@]}"; do
  assignment="${ASSIGNMENTS[$index]}"
  echo "assignment=${assignment}"

  output_path="$("${PYTHON_BIN}" -B -c '
import sys
sys.path.insert(0, sys.argv[1])
import summary_score
print(f"{summary_score.RESULTS_DIR}/summary_evaluate_{sys.argv[2]}.json")
' "${SCRIPT_DIR}" "${assignment}")"

  if "${PYTHON_BIN}" -B "${SCRIPT_DIR}/summary_score.py" \
      --assignment "${assignment}" "${SCORE_ARGS[@]}" >/dev/null 2>&1; then
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

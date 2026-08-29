#!/usr/bin/env bash
set -uo pipefail

# Judge API configuration. These values override summary_score.py.
JUDGE_API_URL="http://10.137.144.97:7001/v1"
JUDGE_API_KEY="empty"
JUDGE_MODEL="/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507"
JUDGE_TEMPERATURE="0.0"
JUDGE_TIMEOUT="120"

ASSIGNMENTS=(
  "g_g_g"
  "h_h_h"
  "l_l_l"
  "g_q_l"
  "q_q_q"
)

SCORE_ARGS=(
  # "--force"
)

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
print(f"{summary_score.RESULTS_DIR}/summary_score_{sys.argv[2]}.json")
' "${SCRIPT_DIR}" "${assignment}")"

  if "${PYTHON_BIN}" -B "${SCRIPT_DIR}/summary_score.py" \
      --assignment "${assignment}" \
      --judge-api-url "${JUDGE_API_URL}" \
      --judge-api-key "${JUDGE_API_KEY}" \
      --judge-model "${JUDGE_MODEL}" \
      --judge-temperature "${JUDGE_TEMPERATURE}" \
      --judge-timeout "${JUDGE_TIMEOUT}" \
      "${SCORE_ARGS[@]}" >/dev/null 2>&1; then
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

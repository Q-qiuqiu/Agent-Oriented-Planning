#!/usr/bin/env bash
set -uo pipefail

# Run these assignments sequentially. Edit this list for each experiment batch.
ASSIGNMENTS=(
  "f_q_m"
  # "g_q_l"
)

# Optional arguments passed to every summary_evaluate.py invocation.
SUMMARY_ARGS=(
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
import summary_evaluate
print(f"{summary_evaluate.RESULTS_DIR}/summary_result_{sys.argv[2]}.json")
' "${SCRIPT_DIR}" "${assignment}")"

  if "${PYTHON_BIN}" -B "${SCRIPT_DIR}/summary_evaluate.py" \
      --assignment "${assignment}" "${SUMMARY_ARGS[@]}" >/dev/null 2>&1; then
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
rows = []
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", []) if isinstance(data, dict) else data

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
if exit_code:
    report["process_exit_code"] = exit_code
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

  if (( index + 1 < ${#ASSIGNMENTS[@]} )); then
    echo "====================="
  fi
done

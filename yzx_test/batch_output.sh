#!/usr/bin/env bash

# Run a pipeline stage while showing only failed per-record progress lines.
# Stderr remains visible so process-level failures and tracebacks are not hidden.
run_with_error_output() {
  "$@" | awk '
    /\| error=/ && !/\| error=None([[:space:]]*$|[[:space:]]*\|)/ {
      print
      fflush()
    }
  '
  local pipeline_status=("${PIPESTATUS[@]}")
  return "${pipeline_status[0]}"
}

# Parse the common execution-mode argument used by all assignment batch scripts.
# Serial execution remains the default. ``--batch-worker`` is an internal flag
# used when a parallel parent re-enters the same script for one assignment.
parse_assignment_batch_args() {
  ASSIGNMENT_BATCH_MODE="serial"
  ASSIGNMENT_BATCH_WORKER=0
  while (( $# )); do
    case "$1" in
      --batch)
        ASSIGNMENT_BATCH_MODE="parallel"
        ;;
      --batch-worker)
        ASSIGNMENT_BATCH_WORKER=1
        ;;
      -h|--help)
        echo "Usage: $(basename -- "${BASH_SOURCE[1]}") [--batch]"
        echo "  default   Run ASSIGNMENTS sequentially."
        echo "  --batch   Run all ASSIGNMENTS concurrently."
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Usage: $(basename -- "${BASH_SOURCE[1]}") [--batch]" >&2
        exit 2
        ;;
    esac
    shift
  done

  if (( ASSIGNMENT_BATCH_WORKER )); then
    if [[ -z "${BATCH_ASSIGNMENT:-}" ]]; then
      echo "Internal batch worker is missing BATCH_ASSIGNMENT." >&2
      exit 2
    fi
    ASSIGNMENTS=("${BATCH_ASSIGNMENT}")
  fi
}

assignment_batch_is_parallel_parent() {
  [[ "${ASSIGNMENT_BATCH_MODE:-serial}" == "parallel" ]] \
    && (( ! ${ASSIGNMENT_BATCH_WORKER:-0} ))
}

_batch_kill_tree() {
  local parent_pid="$1"
  local child_pid
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r child_pid; do
      [[ -n "${child_pid}" ]] || continue
      _batch_kill_tree "${child_pid}"
    done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
  fi
  kill -TERM "${parent_pid}" 2>/dev/null || true
}

_batch_stop_children() {
  local pid
  for pid in "${ASSIGNMENT_BATCH_PIDS[@]:-}"; do
    _batch_kill_tree "${pid}"
  done
  for pid in "${ASSIGNMENT_BATCH_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

_batch_handle_interrupt() {
  trap - INT TERM
  printf '\nParallel batch interrupted; stopping all assignments.\n' >&2
  _batch_stop_children
  exit 130
}

# Re-enter one batch script per assignment. This keeps every script's existing
# command, summary, and error filtering logic identical in serial and parallel
# modes. Worker configuration headers are removed and every remaining line is
# prefixed so concurrent output remains attributable.
run_assignment_scripts_parallel() {
  local script_path="$1"
  shift
  local -a assignments=("$@")
  local assignment pid
  local failed=0
  ASSIGNMENT_BATCH_PIDS=()

  trap _batch_handle_interrupt INT TERM
  for assignment in "${assignments[@]}"; do
    (
      BATCH_ASSIGNMENT="${assignment}" "${script_path}" --batch-worker
    ) > >(
      awk -v prefix="[${assignment}] " '
        BEGIN { skipping_header = 1 }
        skipping_header && /^=====================$/ {
          skipping_header = 0
          next
        }
        skipping_header { next }
        { print prefix $0; fflush() }
      '
    ) 2> >(
      awk -v prefix="[${assignment}] " '
        { print prefix $0 > "/dev/stderr"; fflush("/dev/stderr") }
      '
    ) &
    pid=$!
    ASSIGNMENT_BATCH_PIDS+=("${pid}")
  done

  for pid in "${ASSIGNMENT_BATCH_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  trap - INT TERM
  ASSIGNMENT_BATCH_PIDS=()
  return "${failed}"
}

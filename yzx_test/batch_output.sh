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

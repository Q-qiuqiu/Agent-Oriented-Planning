# Agent-Oriented Planning Experiments

This directory contains the rebuilt benchmark-specific evaluation workflow.
HuskyQA is currently configured for three agent roles and recommends no more
than five sub-agent calls per query. Longer plans are still accepted. A role may
be called repeatedly without implying a new model cold start. Run all commands
from this directory so relative paths resolve correctly.

See `huskyqa_test/README.md` for the complete HuskyQA pipeline and
`mmlu_test/README.md` for the three-perspective MMLU-Pro pipeline.
`iirc_test/README.md` contains the three-agent IIRC evidence and reasoning
workflow.

Timing simulation is provided by `time_report.py`. Planner-time prefetching
fills up to the configured device count, skips duplicate model loads for tasks
in different dependency waves, and keeps duplicate copies only when the same
model is needed concurrently in one wave. `time_report_crossbench.py` applies
the same policy to interleaved benchmark arrivals. Its benchmark assignments
must be updated when comparing the rebuilt IIRC and MMLU workflows.

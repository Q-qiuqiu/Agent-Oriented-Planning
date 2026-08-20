# Planner and Sub-Agent Evaluation

This directory contains two isolated evaluation suites with the same main
pipeline:

1. planner decomposition and offline plan storage;
2. plan-only evaluation;
3. role-fit or heterogeneous sub-agent execution;
4. offline response summarization;
5. final-answer evaluation.

Run all commands from this directory. Model names, API endpoints, temperatures,
timeouts, and default paths are grouped in `CONFIG` or `AGENT_CONFIG` near the
top of each executable script.

## Layout

- `benchmarks/huskyqa/`: HuskyQA input, plans, and expanded subtasks.
- `benchmarks/iirc/`: IIRC source data, flattened dev questions, and SQLite FTS5
  article index.
- `huskyqa_test/`: the existing HuskyQA pipeline, caches, and historical results.
- `iirc_test/`: the IIRC-adapted pipeline and its result directory.

See `huskyqa_test/README.md` and `iirc_test/README.md` for the commands.

## Version-Control Scope

Git contains the legacy source code and documentation only. Benchmark JSON,
SQLite indexes, search caches, model responses, scores, timing logs, and Python
bytecode are generated or local artifacts and are intentionally excluded. Run
the benchmark preparation scripts to rebuild required data in a fresh clone.

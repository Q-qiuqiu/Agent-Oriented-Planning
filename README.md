# Planner and Sub-Agent Experiments

The active experiment code is under [`yzx_test`](yzx_test/README.md).

Current benchmark workflows:

- `yzx_test/huskyqa_test`: HuskyQA planner decomposition, heterogeneous
  sub-agent execution, summary evaluation, timing, and collision analysis.
- `yzx_test/mmlu_test`: sampled MMLU-Pro workflow with independent knowledge,
  reasoning, and option-elimination agents.

Generated datasets, plans, responses, scores, caches, model logs, SQLite files,
and Python bytecode are local artifacts and are excluded from Git.

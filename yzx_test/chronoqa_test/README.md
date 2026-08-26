# ChronoQA Three-Agent Workflow

This workflow uses a deterministic 360-question subset of the official
ChronoQA dataset: 120 absolute, 120 aggregate, and 120 relative temporal
questions. Run commands from `yzx_test` so relative paths resolve correctly.

## Agent Roles

- `evidence_agent`: independently extracts decisive facts, events, entities,
  and dates from the supplied golden evidence.
- `temporal_agent`: independently resolves time expressions against the
  question date and performs temporal comparison or aggregation.
- `verification_agent`: independently cross-checks evidence, chronology,
  numbers, and contradictions.

Every valid plan uses all three roles exactly once with `dep: []`; the subtask
executor sends all three requests concurrently. Responses end with
`最终答案：...`.

The plan therefore contains 3 executable calls per query. The expanded
`chronoqa_subtask_*.json` role-fit file contains 9 rows per query because each
of the 3 planned tasks is also evaluated under all 3 candidate roles; the
heterogeneous end-to-end executor reads the plan file and still runs only 3.

## Pipeline

```bash
python3 chronoqa_test/prepare_chronoqa.py
python3 chronoqa_test/build_subtask_benchmark.py
# Or generate planning reasoning followed by the same JSON plan:
python3 chronoqa_test/build_subtask_full_benchmark.py

python3 chronoqa_test/evaluate_agent_fit.py --mode all
python3 chronoqa_test/subtask_hetro.py --mode all --assignment g_q_l
python3 chronoqa_test/summary_evaluate.py --assignment g_q_l
python3 chronoqa_test/summary_score.py --assignment g_q_l
python3 chronoqa_test/plan_evaluate.py
```

`evaluate_agent_fit.py`, `subtask_hetro.py`, and `summary_score.py` use the
official ChronoQA binary LLM-judge rule: semantically correct answers score 1,
and incorrect answers score 0. Successful records are skipped on rerun; failed
records are retried unless the corresponding retry setting is disabled.

## Analysis

```bash
python3 chronoqa_test/benchmark_planner_latency.py
python3 chronoqa_test/analyze_device_collisions.py
python3 chronoqa_test/time_report.py
```

Edit `MODEL_SIZE`, `AGENT_ASSIGNMENT`, and `PLAN_VARIANT` at the top of the
execution, summary, scoring, and timing scripts so one result family stays
aligned.

# HuskyQA Three-Agent Evaluation

Run commands from the `yzx_test` directory. API endpoints and model names are
kept in the `CONFIG` dictionaries near the top of each script.

## Agent Policy

The benchmark defines exactly three roles:

- `search_agent`: retrieves all external facts in one consolidated call.
- `calculation_agent`: performs all numerical or programmatic work in one call.
- `reasoning_agent`: handles optional non-numerical reasoning.

A plan should normally contain no more than five steps, but this is a soft
recommendation: longer valid plans are accepted. Any of the three roles may
occur repeatedly; this means another request to an already loaded model, not
another model cold start. Independent searches should run without dependencies,
and related calculations should normally be merged into a later task that waits
for all required results. Repeated calculation or reasoning calls remain valid
when they represent distinct dependency stages. The final answer is produced
separately by the summary model.

## Data And Outputs

- `benchmarks/huskyqa/huskyqa_raw.json`: original 292 HuskyQA questions.
- `benchmarks/huskyqa/huskyqa_plans_llada.json`: planner records.
- `benchmarks/huskyqa/huskyqa_subtask_llada.json`: each planned task expanded
  across the three roles for model-role evaluation.
- `huskyqa_test/cache/`: web-search caches.
- `huskyqa_test/results/`: plan and role-fit evaluation results.
- `huskyqa_test/results_1b_llada/`: heterogeneous execution and summary results.

## Pipeline

Generate normalized plans and the expanded role-fit benchmark:

```bash
python3 huskyqa_test/build_subtask_benchmark.py
```

Evaluate plan completeness and non-redundancy:

```bash
python3 huskyqa_test/evaluate.py
```

Generate role-fit responses and judge the saved responses:

```bash
python3 huskyqa_test/evaluate_agent_fit.py --mode respond
python3 huskyqa_test/evaluate_agent_fit.py --mode judge
```

Execute planner-selected roles with heterogeneous models and judge each task:

```bash
python3 huskyqa_test/subtask_hetro.py --mode respond
python3 huskyqa_test/subtask_hetro.py --mode judge
```

Produce and score final answers:

```bash
python3 huskyqa_test/summary_evaluate.py
python3 huskyqa_test/summary_score.py
```

Use `build_subtask_full_benchmark.py` only when planner reasoning plus the JSON
plan is needed. `benchmark_planner_latency.py` measures one query with that
verbose response format. `plan_evaluate.py` is a compatibility entry point for
`evaluate.py`.

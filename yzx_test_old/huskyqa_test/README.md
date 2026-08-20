# HuskyQA Evaluation

Run commands from the `yzx_test` directory. Edit the `CONFIG` and
`AGENT_CONFIG` dictionaries near the top of each script to set model paths and
OpenAI-compatible API endpoints.

## Data

- `benchmarks/huskyqa/huskyqa_raw.json`: original questions.
- `benchmarks/huskyqa/huskyqa_plans_llama3.json`: saved planner output.
- `benchmarks/huskyqa/huskyqa_subtask_llama3.json`: each planned subtask expanded
  across the four agent roles for model-role evaluation.
- `huskyqa_test/cache/`: saved web-search results.
- `huskyqa_test/results/`: offline responses and judged outputs.

## Pipeline

Generate plans and the expanded role-fit benchmark:

```bash
python3 huskyqa_test/build_subtask_benchmark.py
```

Evaluate plan completeness and non-redundancy:

```bash
python3 huskyqa_test/evaluate.py
```

Generate role-fit responses, then judge the saved responses:

```bash
python3 huskyqa_test/evaluate_agent_fit.py --mode respond
python3 huskyqa_test/evaluate_agent_fit.py --mode judge
```

Execute each planner-selected role with its configured heterogeneous model, then
judge each subtask:

```bash
python3 huskyqa_test/subtask_hetro.py --mode respond
python3 huskyqa_test/subtask_hetro.py --mode judge
```

Produce and score final answers:

```bash
python3 huskyqa_test/summary_evaluate.py
python3 huskyqa_test/summary_score.py
```

`plan_evaluate.py` remains as a compatibility alias for `evaluate.py`.

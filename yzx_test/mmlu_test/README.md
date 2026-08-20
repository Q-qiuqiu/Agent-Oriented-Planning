# MMLU-Pro Three-Agent Workflow

This workflow uses a deterministic 350-question subset of the official
`TIGER-Lab/MMLU-Pro` test split: 25 questions from each of its 14 categories.
Run commands from `yzx_test` so all relative paths resolve correctly.

## Agent Roles

- `knowledge_agent`: solves independently from domain knowledge, definitions,
  laws, principles, and facts.
- `reasoning_agent`: solves independently through deduction, calculation, and
  condition analysis.
- `elimination_agent`: checks each option and eliminates incorrect choices.

Every valid plan contains all three roles exactly once with `dep: []`. The
subtask executor sends the three requests concurrently. Each response must end
with `Final answer: X`.

## Pipeline

1. Prepare or reproduce the sampled benchmark:

   ```bash
   python3 mmlu_test/prepare_mmlu_pro.py
   ```

2. Generate ordinary plans and the expanded role-fit benchmark:

   ```bash
   python3 mmlu_test/build_subtask_benchmark.py
   ```

   To generate visible planning reasoning before the same JSON plan:

   ```bash
   python3 mmlu_test/build_subtask_full_benchmark.py
   ```

3. Evaluate a local model separately in each role:

   ```bash
   python3 mmlu_test/evaluate_agent_fit.py --mode all
   ```

4. Execute heterogeneous role models in parallel:

   ```bash
   python3 mmlu_test/subtask_hetro.py --mode all
   ```

5. Produce one final answer from the three saved responses:

   ```bash
   python3 mmlu_test/summary_evaluate.py
   ```

6. Compute the standard option exact-match accuracy:

   ```bash
   python3 mmlu_test/summary_score.py
   ```

7. Evaluate plan quality with deterministic structure checks plus the external
   judge:

   ```bash
   python3 mmlu_test/evaluate.py
   ```

## Analysis

```bash
python3 mmlu_test/benchmark_planner_latency.py
python3 mmlu_test/analyze_device_collisions.py
python3 mmlu_test/time_report.py
```

`subtask_hetro.py`, `summary_evaluate.py`, and `summary_score.py` derive result
filenames from `MODEL_SIZE`, `AGENT_ASSIGNMENT`, and `PLAN_VARIANT` at the top
of each file. Keep those three values aligned. Failed records are retried on a
normal rerun; successful records with unchanged inputs are skipped.

MMLU-Pro's primary result is deterministic multiple-choice exact match. The
external judge is intentionally not used for final answer accuracy.

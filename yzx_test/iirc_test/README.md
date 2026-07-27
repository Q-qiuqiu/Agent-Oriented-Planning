# IIRC Evaluation

This suite keeps the HuskyQA pipeline structure but adapts the evidence source
and sample format to IIRC. Run every command from the `yzx_test` directory.
Edit `CONFIG` and `AGENT_CONFIG` near the top of each executable script before
running model calls.

## Agent Roles

- `search_agent`: generates a search query, retrieves from the local IIRC
  Wikipedia corpus, and rewrites the retrieved snippets into an answer.
- `commonsense_agent`: reads the initial passage and performs evidence synthesis.
- `math_agent`: performs arithmetic and step-by-step mathematical reasoning.
- `code_agent`: writes and executes Python for precise calculations.

The search agent does not use DDGS or an external search API. It performs one
model call to generate a query, retrieves up to five records from SQLite FTS5,
then performs a second model call to answer from those snippets.

## Data Preparation

`prepare_iirc.py` flattens the nested dev split into 1,301 question records and
builds a 56,550-document SQLite FTS5 index without loading the 1.1 GB article
object into memory.

```bash
python3 iirc_test/prepare_iirc.py
```

Outputs:

- `benchmarks/iirc/iirc_dev_flat.json`
- `benchmarks/iirc/context_articles.sqlite3`

Inference receives only the original question, initial article, and available
linked article titles. `gold_question_links` and `gold_context` are retained in
offline records for later analysis but are never added to planner, sub-agent, or
summary prompts.

## Planner

Generate and save plans, then expand every planned subtask across all four roles
for role-fit testing:

```bash
python3 iirc_test/build_subtask_benchmark.py
```

Outputs:

- `benchmarks/iirc/iirc_plans_llama3.json`
- `benchmarks/iirc/iirc_subtask_llama3.json`

The script resumes successful planner records already present in the plan file.

## Plan-Only Evaluation

Use the original completeness and non-redundancy detector:

```bash
python3 iirc_test/evaluate.py
```

Output: `iirc_test/results/plan_evaluate.json`.

## Model-Role Evaluation

Run a local model against the selected roles, save all responses, then use the
separate judge API:

```bash
python3 iirc_test/evaluate_agent_fit.py --mode respond
python3 iirc_test/evaluate_agent_fit.py --mode judge
```

The benchmark input is `benchmarks/iirc/iirc_subtask_llama3.json`. Change
`CONFIG["models"]`, `CONFIG["agents"]`, and the local/judge API blocks at the top
of the script for each model.

## Heterogeneous Execution

Assign one model endpoint to each role in `AGENT_CONFIG`, execute the saved
planner output, and then judge the saved subtask responses:

```bash
python3 iirc_test/subtask_hetro.py --mode respond
python3 iirc_test/subtask_hetro.py --mode judge
```

Outputs:

- `iirc_test/results/subtask_hetro_responses_g_q_g_l.json`
- `iirc_test/results/subtask_hetro_scores_g_q_g_l.json`

## Final Answer

Summarize all dependency-aware subtask responses:

```bash
python3 iirc_test/summary_evaluate.py
```

Then score the saved final answers using the original CompareGPT judge:

```bash
python3 iirc_test/summary_score.py
```

The final score summary contains the original judge accuracy plus deterministic
IIRC normalized exact match and token F1. `plan_evaluate.py` remains as a
compatibility alias for `evaluate.py`.

## Device Collision Analysis

Count each query's planner-selected agent calls and simulate dependency-aware
LRU model replacement:

```bash
python3 iirc_test/analyze_device_collisions.py
```

The default configurations compare two and three devices under both the shared
three-model mapping and four independent agent models. Detailed per-query call
counts are saved in
`iirc_test/results/device_collision_summary_lru.json`.

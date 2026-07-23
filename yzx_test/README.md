# yzx_test

This is a minimal copy of the Agent-Oriented-Planning flow for testing a planner.

Flow:

1. Input a query.
2. Generate or provide a JSON plan.
3. Normalize plan fields (`agent`, `name`, or `name_1` are accepted).
4. Dispatch subtasks to sub-agents.
5. Summarize sub-agent responses into a final answer.

## Run with the built-in LLM planner

```bash
python simple_run.py "If a plane carries 300 passengers, how many full flights are needed for 1000 people?"
```

## Run with an external planner output

```bash
python simple_run.py "query here" --plan-file plan.json
```

`plan.json` should be a list like:

```json
[
  {
    "id": 1,
    "task": "Calculate how many full 300-passenger flights are needed for 1000 people.",
    "agent": "math_agent",
    "dep": []
  }
]
```

API config is loaded from environment variables:

- `AOP_API_URL`
- `AOP_AUTHORIZATION`

or from `../keys/gptapi_key.json`.

## Evaluate Local Models For Agent Roles

This keeps the repository's original evaluation logic: each local model answers
tasks under an agent prompt, then an external judge scores correctness,
relevance, and completeness.

Build planner-derived subtasks first:

```bash
python build_subtask_benchmark.py \
  --input benchmarks/huskyqa_raw.json \
  --planner-api-url https://api.openai.com/v1 \
  --planner-api-key "$OPENAI_API_KEY" \
  --planner-model gpt-4o \
  --plans-output benchmarks/huskyqa_plans.json \
  --benchmark-output benchmarks/huskyqa_subtask_agent_fit.json
```

Generate and save local model responses offline:

```bash
python evaluate_agent_fit.py \
  --mode respond \
  --local-api-url http://localhost:8000/v1 \
  --models qwen2.5-7b llama3.2-3b \
  --benchmarks benchmarks/huskyqa_subtask_agent_fit.json \
  --responses agent_fit_responses.json
```

Judge saved responses with a stronger model:

```bash
python evaluate_agent_fit.py \
  --mode judge \
  --responses agent_fit_responses.json \
  --judge-api-url https://api.openai.com/v1 \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-model gpt-4o \
  --output agent_fit_results.json
```

Search Agent evaluation follows the original repository pipeline during
`respond`: the local model generates a search query, search reads or updates the
configured cache, and the same local model rewrites the snippets into the
saved `scorer_response`. The later `judge` stage is fully offline with respect
to the local model and search service; it only scores that saved response.

To regenerate every Search Agent row while preserving existing Math, Code, and
Commonsense rows, configure `agents` as `["search_agent"]` and run:

```bash
python evaluate_agent_fit.py --mode respond --agents search_agent --force
```

Then rejudge only those regenerated Search rows while retaining existing scores
for the other roles:

```bash
python evaluate_agent_fit.py --mode judge --agents search_agent
```

All planner, sub-agent, synthesis, and judge temperature defaults in this copy
are `0.0`. Existing response and score files are not regenerated automatically;
use `--force` when deterministic outputs must replace older runs.

The output file defaults to `agent_fit_results.json`. The summary ranks which
agent roles each local model is most suitable for.

The built-in benchmark is only a small smoke test with two tasks per agent
role. For a real evaluation, pass a larger JSON file with `--benchmarks`.
The original repository evaluated over dataset-derived subtasks rather than a
handful of examples.

Downloaded HuskyQA files:

- `benchmarks/huskyqa_raw.json`: original 292 HuskyQA questions.
- `benchmarks/huskyqa_agent_fit_expanded.json`: each HuskyQA question expanded
  across `math_agent`, `code_agent`, `search_agent`, and `commonsense_agent`
  for direct use with `evaluate_agent_fit.py`.

Run the downloaded benchmark:

```bash
python evaluate_agent_fit.py \
  --mode all \
  --local-api-url http://localhost:8000/v1 \
  --models your-local-model \
  --judge-api-url https://api.openai.com/v1 \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-model gpt-4o \
  --benchmarks benchmarks/huskyqa_agent_fit_expanded.json
```

Search defaults to the `ddgs` Python package, so no search API key or search
URL is required. Install the dependency before running:

```bash
pip install -U ddgs
```

Search uses the parallel multi-backend DDGS logic from `test_ddgs.py`. Each
backend has an independent client and retry loop, traffic uses `ddgs_proxy`, and
backend failures do not discard successful results from other backends. Results
are deduplicated by URL and ranked by backend agreement and average rank. The
final ten results are cached in `benchmarks/search_cache_ddgs_parallel_proxy.json`.
The default `cache_fallback` mode reads the cache first. Missing, empty, or
invalid entries are searched live and replaced in the cache. Set
`search_backend` to `"cache"` to disable live search and read only this cache.

## Run Planner-Selected Heterogeneous Agents

Use `benchmarks/huskyqa_plans_llama3.json` as input when each planner step should
be routed only to its selected agent. Edit `AGENT_CONFIG` at the top of
`subtask_hetro.py` to bind a separate OpenAI-compatible model/API to each role.

Generate and persist step responses first:

```bash
python subtask_hetro.py --mode respond
```

Judge the saved responses later with the independent judge configuration:

```bash
python subtask_hetro.py --mode judge
```

## Evaluate Plans And Final Answers

The evaluation is split into three independent scripts. Edit each script's
`CONFIG` block before running.

Evaluate a plan JSON with the original repository's plan detector prompt:

```bash
python evaluate.py
```

The default input is `benchmarks/huskyqa_plans_llama3.json`, and the output is
`plan_evaluate.json`. This checks completeness and non-redundancy. A standalone
JSON array of plan steps is also accepted when `query` is configured.

Read `subtask_hetro_responses.json`, summarize each query's sub-agent responses,
and save the final answers:

```bash
python summary_evaluate.py
```

The output is `summary_result.json`. Set `query` or `source_index` to process
only one original query.

Evaluate the saved final answers with the original CompareGPT prompt:

```bash
python summary_score.py
```

This reads `summary_result.json`, compares each final answer with its reference
answer, and saves `summary_evaluate.json`. The reported `accuracy` is the
proportion of CompareGPT `yes` decisions among records with a reference answer.

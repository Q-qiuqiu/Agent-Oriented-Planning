# IIRC Three-Agent Workflow

This directory adapts the original IIRC workflow to three Agent roles. Plans
should normally use at most five calls, but longer valid plans are retained and
executed when the question genuinely requires them. Run commands from
`yzx_test`.

## Roles And Execution Waves

- `context_agent`: extracts relevant evidence from the initial article.
- `retrieval_agent`: retrieves missing evidence from the local IIRC Wikipedia
  SQLite index.
- `reasoning_agent`: combines both evidence results, performs any required
  reasoning or calculation, checks answerability, and returns the answer.

The common three-call plan has this dependency structure:

```text
Wave 1: context_agent (dep=[]) || retrieval_agent (dep=[])
Wave 2: reasoning_agent (dep=[1, 2])
```

Plans may repeat a role or add later dependency waves. Dependencies must always
point to earlier steps, and every dependency-free step in a wave can execute in
parallel. The heterogeneous executor schedules those waves automatically. The
retrieval role uses
one planned Agent task but internally performs the established two-stage tool
workflow: generate a local search query, retrieve top-5 SQLite snippets, then
rewrite those snippets into evidence with the same model. This does not load a
different model.

## Data Preparation

Expected local inputs:

```text
benchmarks/iirc/dev.json
benchmarks/iirc/context_articles.json
```

Build the flattened 1301-question dev benchmark and FTS5 index:

```bash
python3 iirc_test/prepare_iirc.py
```

Use `--skip-sqlite` when an existing `context_articles.sqlite3` is available.
Dataset JSON and SQLite files are ignored by Git.

## Pipeline

Generate normal plans and the expanded role-fit benchmark:

```bash
python3 iirc_test/build_subtask_benchmark.py
```

Generate plans with an additional planning-reasoning section:

```bash
python3 iirc_test/build_subtask_full_benchmark.py
```

Evaluate local models separately by role:

```bash
python3 iirc_test/evaluate_agent_fit.py --mode all
```

Execute heterogeneous role models and judge the saved subtask responses:

```bash
python3 iirc_test/subtask_hetro.py --mode all
```

Generate and score final answers:

```bash
python3 iirc_test/summary_evaluate.py
python3 iirc_test/summary_score.py
```

Evaluate plans and analyze first-batch model collisions:

```bash
python3 iirc_test/evaluate.py
python3 iirc_test/analyze_device_collisions.py
python3 iirc_test/benchmark_planner_latency.py
python3 iirc_test/time_report.py
```

Model endpoints and output naming are configured at the top of each script.
`MODEL_SIZE`, `AGENT_ASSIGNMENT`, and `PLAN_VARIANT` must remain aligned across
the execution, summary, and scoring stages.

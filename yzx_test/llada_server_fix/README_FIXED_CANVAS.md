# Fixed Structural Canvas + PLAN-First Denoising

This directory is an independent copy of `yzx_test/llada_server/`.  The source
directory is not imported or modified by this experiment.

## Canvas budget and PLAN completion

The current two-sentence Dual Vanilla sanity outputs (HuskyQA 5 + MMLU 5)
have the following combined LLaDA-tokenizer lengths:

| Region | P50 | P95 | Max | Selected budget |
|---|---:|---:|---:|---:|
| Planning reasoning | 303 | 324.2 | 331 | 499 (default) |
| PLAN JSON | 335 | 384.9 | 389 | 499 maximum capacity |

With the current tokenizer, the prefix/middle/suffix structural sequences use
7/14/5 tokens. The 998 non-delimiter positions are split at runtime according
to `--reasoning_ratio 0.5 --plan_ratio 0.5`, producing 499/499. Explicit
`--reasoning-budget` or `--plan-budget` overrides are also supported; one
omitted budget receives the exact remainder. If both explicit budgets leave
unused capacity, the initial canvas is shortened rather than padded with MASK,
EOS, or PAD tokens.

Reasoning and PLAN budgets are maximum capacities, not required output lengths.
The beginning delimiters (`PLANNING_REASONING` and `PLAN_JSON`) are fixed;
`END_PLANNING_REASONING` and `END_PLAN_JSON` must materialize naturally. Once
an END marker appears, the decoder finishes any preceding masks, then removes
the unused capacity after the marker. PLAN completion additionally requires a
valid parsed JSON plan. No JSON key, brace, Agent identity, or PLAN content is
fixed.

## Modes

- `dual_vanilla`: unchanged copied Dual Cache path. With
  `--agent_probe_period 0`, Agent observation is read-only and adds no forward.
- `fixed_canvas_vanilla`: fixed canvas, reasoning region then PLAN region.
- `fixed_canvas_plan_first`: identical canvas, PLAN region then reasoning.

## Recommended run

From the repository root:

```bash
# Run HuskyQA and MMLU concurrently on GPUs 6 and 7.
# This copies the existing 5+5 Dual Vanilla baseline and resumes its missing
# 45+45 queries, then runs 50+50 for both fixed-canvas modes.
HUSKY_GPU=6 MMLU_GPU=7 bash \
  yzx_test/llada_server_fix/run_fixed_canvas_parallel.sh \
  fixed_canvas_dynamic_end_full_01 complete
```

Results are written under
`yzx_test/benchmarks/fixed_canvas/fixed_canvas_dynamic_end_full_01/`. The final
report is `fixed_canvas_report.md`. Its primary table reports parse success,
First-3 Agent tuple agreement, full ordered Agent sequence agreement, and
paired First-3 lead versus Dual Vanilla. Reasoning length remains a secondary
health diagnostic.

To re-run analysis without inference:

```bash
/home/yzx/miniconda3/envs/llada/bin/python \
  yzx_test/llada_server_fix/analyze_fixed_canvas_experiment.py \
  yzx_test/benchmarks/fixed_canvas/fixed_canvas_dynamic_end_full_01
```

The runner honors `PYTHON` and `MODEL` environment overrides.

## Manual server example

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test/llada_server_fix
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python llada_server.py \
  --model_path /data/labshare/Param/llada \
  --served_model_name /data/labshare/Param/llada \
  --device cuda --host 127.0.0.1 --port 7310 \
  --cache_mode dual --block_size 32 --max_gen_length 1024 \
  --steps_per_block 32 --threshold 0.9 --policy reasonplan \
  --structure_mode fixed_canvas_plan_first \
  --reasoning_ratio 0.5 --plan_ratio 0.5 \
  --agent_probe_period 0 --agent_timing_slots 16 \
  --plan_json_repair \
  --agent_timing_log_path /tmp/fixed_canvas_plan_first_timings.jsonl
```

Replace `fixed_canvas_plan_first` with `dual_vanilla` or
`fixed_canvas_vanilla` for the other two modes.

# Final LLaDA Agent-prefetch server

This directory exposes one server with four methods:

| Method | Decoding | Agent prediction |
|---|---|---|
| `base` | Original Dual Vanilla | None; natural Agent timing only |
| `commit` | Original Dual Vanilla | Read-only Global + Local + Natural fusion |
| `plan` | Fixed Canvas, PLAN-first, Dynamic END | None; natural timing only |
| `all` | Fixed Canvas, PLAN-first, Dynamic END | Local ordered prediction + Natural zero-lag fallback |

No observer changes `x`, decoder masks, transfer order, block order, or NFE.
`plan` and `all` intentionally use PLAN-first decoding; `base` and `commit`
share the original Dual Vanilla trajectory.

For `all`, PLAN-first creates the materialization window. The observer only
uses safe PLAN-local predictions inside that window. If no local prediction
wins, the corresponding slot falls back immediately to natural decoding, so
its fused prefetch time cannot be later than the natural PLAN timing.

## Start one server

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test/llada_server_fix
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python llada_server.py \
  --method commit \
  --model_path /data/labshare/Param/llada \
  --served_model_name /data/labshare/Param/llada \
  --device cuda --host 127.0.0.1 --port 7390 \
  --cache_mode dual --block_size 32 --max_gen_length 1024 \
  --steps_per_block 32 --threshold 0.9 \
  --agent_timing_slots 16 \
  --agent-timing-log-dir /tmp/commit_timings
```

Replace `commit` with `base`, `plan`, or `all`.

## Compact timing log

The server infers the benchmark from the Agent registry and writes one of
`huskyqa_full_timings.jsonl`, `iirc_full_timings.jsonl`,
`mmlu_full_timings.jsonl`, or `chronoqa_full_timings.jsonl` in the configured
timing directory. Each JSONL record contains only request identity/status plus:

- `predicted_agents`: predicted Agent name, source, and prediction time;
- `natural_agents`: naturally decoded Agent name and decode time;
- `generation_seconds` and `nfe`.

`commit` keeps both prediction and natural occurrence lists. `plan` keeps only
the natural occurrence list. First-3 and all-Agent times are derived offline
from those lists instead of being redundantly written by the server.

## Parallel benchmark runner

```bash
GPU_A=2 GPU_B=3 METHODS="base commit plan all" \
  bash yzx_test/llada_server_fix/run_methods_parallel.sh final_01 5
```

Results are written under `yzx_test/benchmarks/final_methods/final_01/`.

Run only the step-level Local `all` experiment:

```bash
GPU_A=2 GPU_B=3 \
  bash yzx_test/llada_server_fix/run_all_local_step_parallel.sh \
  all_local_step_01 5
```

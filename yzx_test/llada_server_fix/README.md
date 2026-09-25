# Final LLaDA Agent-prefetch server

This directory exposes one server with four methods:

| Method | Decoding | Agent prediction |
|---|---|---|
| `base` | Original Dual Vanilla | None; natural Agent timing only |
| `commit` | Original Dual Vanilla | Read-only Global + Local + Natural fusion |
| `plan` | Fixed Canvas, PLAN-first, Dynamic END | None; natural timing only |
| `all` | Fixed Canvas, PLAN-first, Dynamic END | Read-only PLAN-region prediction |

No observer changes `x`, decoder masks, transfer order, block order, or NFE.
`plan` and `all` intentionally use PLAN-first decoding; `base` and `commit`
share the original Dual Vanilla trajectory.

## Start one server

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test/llada_server_fix
CUDA_VISIBLE_DEVICES=0 /home/yzx/miniconda3/envs/llada/bin/python llada_server.py \
  --method commit \
  --model_path /data/labshare/Param/llada \
  --served_model_name /data/labshare/Param/llada \
  --device cuda --host 127.0.0.1 --port 7390 \
  --cache_mode dual --block_size 32 --max_gen_length 1024 \
  --steps_per_block 32 --threshold 0.9 --policy reasonplan \
  --agent_timing_slots 16 \
  --agent_timing_log_path /tmp/commit_timings.jsonl
```

Replace `commit` with `base`, `plan`, or `all`.

## Compact timing log

Each JSONL record contains only request identity/status plus:

- `predicted_agents`: predicted Agent name, source, and prediction time;
- `natural_agents`: naturally decoded Agent name and decode time;
- `first3_prediction_seconds`, `first3_natural_seconds`;
- `all_natural_seconds`;
- `generation_seconds` and `nfe`.

## Parallel benchmark runner

```bash
GPU_A=2 GPU_B=3 METHODS="base commit plan all" \
  bash yzx_test/llada_server_fix/run_methods_parallel.sh final_01 5
```

Results are written under `yzx_test/benchmarks/final_methods/final_01/`.

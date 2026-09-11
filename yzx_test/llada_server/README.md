# LLaDA Server

This directory is the AOP-owned copy of the Fast-dLLM v1 LLaDA runtime. Make
future server changes here so the benchmark clients and planner server remain
in the same repository.

## Entrypoints

- `fastdllm_server.py`: original baseline OpenAI-compatible LLaDA server.
- `base_llada_server.py`: separately instrumented baseline server. It passively
  monitors naturally materialized JSON `agent` fields for timing, but never
  changes logits, masks, tokens, decoding order, or the returned response.
- `llada_server.py`: planner server with Agent-name priority decoding, timing
  records, prompt policy handling, and conservative plan JSON repair.

The planner server extracts the active Agent registry from lines formatted as
`- name_agent: description` in the request system prompt. Its 13-name global
catalog is used only to repair close spelling errors; the repaired role must
still belong to the current request registry.

## Layout

- `generate.py`: masked-diffusion decoding and cache implementations.
- `model/`: local LLaDA Transformers model definition.
- `agent_priority.py`, `json_agent_priority.py`: early Agent recognition and
  priority decoding.
- `response_agent_timing.py`: read-only controller used by the baseline server
  to observe Agent-name materialization during diffusion.
- `agent_timing.py`: per-request Agent timing persistence.
- `planner_policy.py`: planner output-order policies.
- `planner_json_repair.py`: conservative syntax and Agent-name repair.
- `tests/`: focused tests for the copied runtime modules.

Runtime JSONL logs, model weights, caches, and generated outputs are not kept
in this directory.

## Launch

The original baseline servers configured by the benchmark launcher can be
started with:

```bash
bash yzx_test/start_vllm.sh llada4 0
```

Run the monitored baseline directly with the same environment variables used
by `fastdllm_server.py`:

```bash
FASTDLLM_MODEL_PATH=/data/labshare/Param/llada \
FASTDLLM_SERVED_MODEL_NAME=/data/labshare/Param/llada \
FASTDLLM_PORT=7004 \
python yzx_test/llada_server/base_llada_server.py \
  --gen-length 1024 \
  --block-size 32 \
  --cache-mode dual \
  --threshold 0.9 \
  --steps 1024
```

Agent timing is enabled by default in `base_llada_server.py`. It reads the
active Agent registry from the system prompt, identifies HuskyQA, IIRC, MMLU,
or ChronoQA, and writes the matching log under
`yzx_test/benchmarks/fastdllm_log/`:

```text
base_huskyqa_full_timings.jsonl
base_iirc_full_timings.jsonl
base_mmlu_full_timings.jsonl
base_chronoqa_full_timings.jsonl
```

Each record contains total `generation_seconds` and one entry per observed
Agent occurrence, including `first_observed_seconds`, `decision_seconds`, and
the diffusion step. These are real observation times from the generation loop,
not estimates based on the final token positions. Only JSON `agent` fields
between `PLAN_JSON` and `END_PLAN_JSON` are counted; Agent names mentioned in
planning reasoning are ignored. Monitoring overhead is part of
`generation_seconds` and is marked by
`monitor_overhead_included: true`. Disable it for an uninstrumented baseline:

```bash
python yzx_test/llada_server/base_llada_server.py --no-record-agent-timings
```

Run the Agent-aware planner server directly when early Agent recognition is
required:

```bash
python llada_server/llada_server.py \
  --model_path /data/labshare/Param/llada \
  --served_model_name /data/labshare/Param/llada \
  --port 7004 \
  --max_gen_length 1024 \
  --block_size 32 \
  --steps_per_block 32 \
  --cache_mode dual \
  --policy reasonplan
```

The code was copied from `/data/home/yzx/Fast-dLLM/v1/llada`. The upstream
license is retained as `LICENSE.fast-dllm`.

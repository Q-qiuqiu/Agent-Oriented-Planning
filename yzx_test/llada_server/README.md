# LLaDA Server

This directory is the AOP-owned copy of the Fast-dLLM v1 LLaDA runtime. Make
future server changes here so the benchmark clients and planner server remain
in the same repository.

## Entrypoints

- `fastdllm_server.py`: baseline OpenAI-compatible LLaDA server without Agent
  priority decoding.
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
- `agent_timing.py`: per-request Agent timing persistence.
- `planner_policy.py`: planner output-order policies.
- `planner_json_repair.py`: conservative syntax and Agent-name repair.
- `tests/`: focused tests for the copied runtime modules.

Runtime JSONL logs, model weights, caches, and generated outputs are not kept
in this directory.

## Launch

Baseline servers configured by the benchmark launcher can be started with:

```bash
bash yzx_test/start_vllm.sh llada4 0
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

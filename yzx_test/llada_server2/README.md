# LLaDA Server 2

This is an independent experimental copy of the Fast-dLLM v1 LLaDA runtime.
It keeps the caller's reasoning-first planner prompt and predicts the first
three plan Agent names earlier from intermediate diffusion states.

## Entrypoints

- `fastdllm_server.py`: original baseline OpenAI-compatible LLaDA server.
- `base_llada_server.py`: separately instrumented baseline server. It passively
  monitors naturally materialized JSON `agent` fields for timing, but never
  changes logits, masks, tokens, decoding order, or the returned response.
- `llada_server.py`: planner server. Under `reasonplan`, it marginalizes over
  moving JSON-field positions, accumulates Agent evidence across denoising
  observations, and emits ordered Agent-name prefetch decisions.

The planner server extracts the active Agent registry from lines formatted as
`- name_agent: description` in the request system prompt. Its 13-name global
catalog is used only to repair close spelling errors; the repaired role must
still belong to the current request registry.

## Layout

- `generate.py`: masked-diffusion decoding and cache implementations.
- `model/`: local LLaDA Transformers model definition.
- `marginal_agent_priority.py`: read-only reasonplan predictor. It never writes
  predicted names into the output canvas and has no device-count scheduler.
- `agent_priority.py`, `json_agent_priority.py`: original plan-first priority
  implementation retained for comparison.
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

For a local timing comparison, run the copied passive baseline with the same
model, decoding arguments, and client prompt:

```bash
FASTDLLM_MODEL_PATH=/data/labshare/Param/llada \
FASTDLLM_SERVED_MODEL_NAME=/data/labshare/Param/llada \
FASTDLLM_PORT=7004 \
python yzx_test/llada_server2/base_llada_server.py \
  --gen-length 1024 \
  --block-size 32 \
  --cache-mode dual \
  --threshold 0.9 \
  --steps 1024 \
  --agent-slots 3 \
  --agent-timing-slots 3
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
python yzx_test/llada_server2/base_llada_server.py --no-record-agent-timings
```

Run the Agent-aware planner server directly when early Agent recognition is
required:

```bash
python yzx_test/llada_server2/llada_server.py \
  --model_path /data/labshare/Param/llada \
  --served_model_name /data/labshare/Param/llada \
  --port 7004 \
  --max_gen_length 1024 \
  --block_size 32 \
  --steps_per_block 32 \
  --cache_mode dual \
  --agent_slots 3 \
  --agent_timing_slots 3 \
  --policy reasonplan \
  --agent_timing_log_path yzx_test/benchmarks/fastdllm_log/reasonplan_v2_timings.jsonl
```

An INFO log line named `agent_prefetch_decision` is emitted once for every
recognized slot. The controller always commits the first three slots and still
allows repeated Agent names. For every ordered field layout, it scores every
complete Agent sequence, aggregates identical sequences across layouts with
log-sum-exp, and selects the global MAP sequence. An arithmetic EMA combines
sequence distributions across observations. If confidence gates have not
fired, the third full-sequence observation forces a MAP decision so the final
Agent is still prefetched early.

The timing JSONL records `agent`, `final_agent`, `final_agent_seconds`,
`prediction_correct`, and `decision_source`. A wrong prediction emits an
immediate `agent_prefetch_switch` INFO event when the true JSON field becomes
visible; an external model loader can use that event to cancel the old load and
start the final Agent model. The Agent catalog is still taken from the current
request prompt; no model-to-device or expected-benefit policy is applied.

The code was copied from `/data/home/yzx/Fast-dLLM/v1/llada`. The upstream
license is retained as `LICENSE.fast-dllm`.

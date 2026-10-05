# Final LLaDA Agent-prefetch server

This directory exposes one server with four runtime methods plus isolated
diagnostic methods:

| Method | Decoding | Agent prediction |
|---|---|---|
| `base` | Original Dual Vanilla | None; natural Agent timing only |
| `commit` | Original Dual Vanilla | Latent-region stable-2 → Prefix → Natural first-valid commit |
| `plan` | Fixed Canvas, PLAN-first, Dynamic END | None; natural timing only |
| `all` | Fixed Canvas, PLAN-first, Dynamic END | Local ordered prediction + Natural zero-lag fallback |
| `oracle_latent` | Original Dual Vanilla | Diagnostic-only oracle-span replay of existing full-sequence logits |
| `online_latent_diagnostic` | Original Dual Vanilla | Diagnostic-only online logits-anchor localization with oracle control |
| `online_latent_refine` | Original Dual Vanilla | Coarse online anchors plus diagnostic ±4 joint template refinement |
| `online_latent_refine_tracking` | Original Dual Vanilla | Local refinement plus persistent monotonic slot tracking |
| `online_latent_region` | Original Dual Vanilla | Persistent coarse-anchor tracks plus diagnostic region-level Agent scoring |
| `online_latent_hypothesis` | Original Dual Vanilla | Merge/filter raw tracks before applying the unchanged region scorer |

No commit observer changes `x`, decoder masks, transfer order, block order, or NFE.
`plan` and `all` intentionally use PLAN-first decoding; `base` and `commit`
share the original Dual Vanilla trajectory.

The production `commit` policy is the causal stable-2 policy validated by
`test_08` through `test_13` and the offline commit simulator. Existing
full-sequence Dual Vanilla warmups feed latent anchor detection, persistent
hypothesis tracking, causal validation (`seen_count >= 2`, support at least
0.5), and an Agent-only radius-4 region scorer. The scorer uses normalized
sequence log probability with top-2-mean position aggregation. Two consecutive
valid observations with the same Agent produce a `latent_region` candidate;
unique natural prefix and complete natural value remain fallbacks. The observer
adds no model forward and never changes generated tokens or masks.

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
  --agent-timing-log-dir /tmp/commit_timings
```

Replace `commit` with `base`, `plan`, or `all`.

## Oracle latent diagnostic

`oracle_latent` scores every runtime-registry Agent name at every possible
generation position on existing full-sequence block warmups, then selects the
real positions only after the final JSON Agent spans are known. It does not add
model forwards, change decoder masks, write Agent tokens, commit, or prefetch.

Run the first five HuskyQA queries on GPU 0:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
GPU=0 bash llada_server_fix/run_huskyqa_oracle_latent.sh \
  test/test_08_oracle_latent 5
```

The run writes request trajectories under `timing/` and generates
`analysis/report.md`, `analysis/per_agent_summary.csv`,
`analysis/per_slot_timeline.csv`, and `analysis/prediction_trajectory.jsonl`.
Set `SEMANTIC_TIMING=/path/to/huskyqa_full_timings.jsonl` to compare against a
different visible-evidence semantic diagnostic log.

## Online latent anchor diagnostic

`online_latent_diagnostic` reuses the old JSON Agent logits-anchor detector on
each existing full-sequence warmup. It dynamically tracks every detected
anchor by output order, infers the value start from the matched pattern width,
and scores complete runtime-registry Agent names there. The same observation
also retains oracle-span scores, so anchor delay/error and lost stable lead can
be compared directly without a second generation run. It remains read-only
and adds no model forward, commit, token write, freeze, or prefetch.

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
GPU=0 bash llada_server_fix/run_huskyqa_online_latent_anchor.sh \
  test/test_09_online_latent_anchor 5
```

The analysis is written to `analysis/report.md`, with the full slot table in
`analysis/per_slot_timeline.csv` and observation trajectory in
`analysis/online_prediction_trajectory.jsonl`.

Run the local-refinement ablation and then the persistent-tracking variant on
the same GPU and same first five queries:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
GPU=0 bash llada_server_fix/run_huskyqa_refinement_experiments.sh 5
```

This preserves `test_09` and writes the two new runs to
`test/test_10_refine_only` and `test/test_11_refine_tracking`. Their unified
comparison is generated under `test/test_11_refine_tracking/comparison/`.

## Region latent Agent diagnostic

`online_latent_region` keeps persistent monotonic tracks but treats each track
as the center of a local Agent-containing region. It does not use the old
joint-score refinement to choose an exact value boundary. For every runtime
registry Agent it aggregates all alignments in the region using max,
top-2 mean, and soft position aggregation; agent-only and full-field-template
scores are both retained. The primary report uses agent-only top-2 mean.

Run the same first five HuskyQA queries with radius 4:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
GPU=0 PORT=7396 REGION_RADIUS=4 \
  bash llada_server_fix/run_huskyqa_region_latent.sh \
  test/test_12_region_latent 5
```

The diagnostic remains read-only and reports `extra_model_forwards=0`.
Results are written to `analysis/report.md`, `analysis/per_slot_timeline.csv`,
`analysis/region_prediction_trajectory.jsonl`, and the mapped plus unmapped
track stream in `analysis/all_track_region_trajectory.jsonl`.

Run the simplified track-merge and hypothesis-validation experiment:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
GPU=0 PORT=7397 \
  bash llada_server_fix/run_huskyqa_hypothesis_region.sh \
  test/test_13_hypothesis_region 5
```

Defaults are merge distance 6, merge gap 2 observations, minimum seen count
2, support ratio 0.5, and region radius 4. The scorer remains agent-only
top-2 mean and the experiment remains read-only.

Replay a completed hypothesis run through the causal offline commit simulator:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
python3 offline_latent_commit_simulator.py \
  --timing test/test_13_hypothesis_region/timing/huskyqa_full_timings.jsonl \
  --output-dir test/test_13_hypothesis_region/offline_commit_simulation
```

The simulator compares immediate, stable-2, and stable-3 latent candidates
against the recorded unique-prefix and natural times. It produces no model
calls and uses final Agent spans only after causal triggers have been replayed.

## Two-GPU HuskyQA base/commit smoke test

Run the same HuskyQA queries against base and commit concurrently, with one
model server per GPU:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test/llada_server_fix
GPU_BASE=0 GPU_COMMIT=1 \
  bash run_huskyqa_base_commit_dual_gpu.sh \
  /data/home/yzx/huskyqa_runs/smoke_01 5
```

Use `all` instead of `5` to process the complete input. Each method receives
separate `results/`, `timing/`, and `logs/` directories under the requested
output root. `PLANNER_SCRIPT`, `MODEL`, `PYTHON`, ports, input path, and timeout
settings can be overridden with environment variables; run the script with
`--help` for the complete list.

Analyze a completed schema-v7 run without editing either report script:

```bash
cd /data/home/yzx/Agent-Oriented-Planning/yzx_test
RUN_ROOT=/data/home/yzx/huskyqa_runs/smoke_01 python3 commit_time_caculate.py
RUN_ROOT=/data/home/yzx/huskyqa_runs/smoke_01 python3 report_agent_detection.py
```

## Compact timing log

The server infers the benchmark from the Agent registry and writes one of
`huskyqa_full_timings.jsonl`, `iirc_full_timings.jsonl`,
`mmlu_full_timings.jsonl`, or `chronoqa_full_timings.jsonl` in the configured
timing directory. For `commit`, each JSONL record contains a compact
`agent_commits` list with one row per final PLAN step. It records the winning
source (`latent_region`, `prefix`, or `natural`), committed/naturally decoded
Agent, monotonic wall times, hypothesis/region metadata, stability,
correctness, and lead time. Slot IDs are the only PLAN occurrence identifiers.
NFE is retained at request level to verify that the observer adds no forward.

Request-level fields contain first-valid accuracy, latent accuracy, correct
latent coverage, correct-only latent lead, source distributions, compact
false-trigger diagnostics, first/all commit time, NFE, extra model forwards,
and full generation time. Repeated Agent/model names remain distinct PLAN
slots while the logical prefetch marker reuses an Agent already marked for
loading. All event timestamps are captured in memory during decoding; JSON
encoding and the atomic JSONL write happen only after generation timing ends.

For `base`, decoding is unchanged and the passive observer records only fully
materialized registry Agent names (including repeats), their natural appearance
times, and basic request generation metadata. It performs no semantic/prefix
prediction, commit, prefetch, PLAN parsing, or fixed-first-three aggregation.

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

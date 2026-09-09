import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]

from base_llama3_builder import replace_json_array_example, run_builder
from build_subtask_benchmark import (
    AGENTS,
    expand_plans,
    load_queries,
    normalize_plan,
    print_agent_selection_summary,
)
from prompt import planner_prompt


PROMPT_VERSION = "huskyqa_base_llama3_prefetch_then_plan_v8"

OUTPUT_CONTRACT = """For the base_llama3 control strategy, output only one valid
JSON object with exactly two top-level fields in this order: `prefetch_agents`,
then `plan`.

Use this structure:
{
  "prefetch_agents": [
    {"id": 1, "agent": "search_agent"},
    {"id": 2, "agent": "search_agent"},
    {"id": 3, "agent": "calculation_agent"}
  ],
  "plan": [
    {
      "agent": "search_agent",
      "id": 1,
      "task": "Retrieve the first independent group of external facts",
      "reason": "The first fact group requires external evidence",
      "dep": []
    },
    {
      "agent": "search_agent",
      "id": 2,
      "task": "Retrieve the second independent group of external facts",
      "reason": "The second fact group requires a separate search",
      "dep": []
    },
    {
      "agent": "calculation_agent",
      "id": 3,
      "task": "Use all retrieved values to perform the requested calculation",
      "reason": "The final result requires numerical calculation",
      "dep": [1, 2]
    }
  ]
}

First determine the decomposition silently. Then write `prefetch_agents` before
any task, reason, or dependency text. List the Agent calls already identified in
execution order. This is a per-call sequence rather than a unique role set, so
repeated calls require repeated objects. Use consecutive ids starting at 1.

Next write a complete, self-contained `plan` under the original planning policy.
This `plan` is the authoritative output that will be executed and scored. Every
plan object must explicitly contain `agent`, `id`, `task`, `reason`, and `dep`;
repeated calls repeat the Agent name. Prefer no more than five calls, but retain
additional calls when genuinely necessary for a correct plan.

Do not shorten, merge, duplicate, or otherwise distort the final `plan` merely
to make it match `prefetch_agents`. If task detailing reveals an additional
necessary call, include it in `plan`. The two arrays are not required to have
the same length; correctness and completeness of `plan` take priority.
Do not output prose before or after the JSON object."""

BASE_LLAMA3_PROMPT = replace_json_array_example(
    planner_prompt,
    "Output only one valid JSON array in this exact schema. This example shows two\n"
    "independent retrievals followed by one consolidated calculation:",
    OUTPUT_CONTRACT,
).replace(
    "text outside the array.",
    "text outside the JSON object.",
)

CONFIG = {
    "input": "benchmarks/huskyqa/huskyqa_raw.json",
    "plans_output": "benchmarks/huskyqa/huskyqa_plans_base_llama3.json",
    "benchmark_output": "benchmarks/huskyqa/huskyqa_subtask_base_llama3.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "source": "agent-husky/HuskyQA",
    "prompt_version": PROMPT_VERSION,
    "planner_mode": "prefetch_then_plan",
    "planner_prompt": BASE_LLAMA3_PROMPT,
}


if __name__ == "__main__":
    run_builder(
        config=CONFIG,
        agents=AGENTS,
        load_queries=load_queries,
        normalize_plan=normalize_plan,
        expand_plans=expand_plans,
        print_summary=print_agent_selection_summary,
        description="Build HuskyQA base_llama3 agent-names-first plans.",
    )

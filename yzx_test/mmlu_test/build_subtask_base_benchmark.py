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
    print_summary,
)
from prompt import planner_prompt


PROMPT_VERSION = "mmlu_base_llama3_prefetch_then_plan_v4"

OUTPUT_CONTRACT = """For the base_llama3 control strategy, output only one valid
JSON object with exactly two top-level fields in this order: `prefetch_agents`,
then `plan`.

Use this exact structure:
{
  "prefetch_agents": [
    {"id": 1, "agent": "knowledge_agent"},
    {"id": 2, "agent": "reasoning_agent"},
    {"id": 3, "agent": "elimination_agent"}
  ],
  "plan": [
    {
      "agent": "knowledge_agent",
      "id": 1,
      "task": "Independently solve the complete question from domain knowledge and select one provided option",
      "reason": "Provides a complete fact- and principle-based solution",
      "dep": []
    },
    {
      "agent": "reasoning_agent",
      "id": 2,
      "task": "Independently solve the complete question through logic, calculation, and condition analysis and select one provided option",
      "reason": "Provides a complete derivation-based solution",
      "dep": []
    },
    {
      "agent": "elimination_agent",
      "id": 3,
      "task": "Independently inspect every provided option, eliminate incorrect choices, and select the strongest remaining option",
      "reason": "Provides a complete option-comparison solution",
      "dep": []
    }
  ]
}

First determine the decomposition silently. Then write `prefetch_agents` before
any task, reason, or dependency text. List the Agent calls already identified in
execution order, using consecutive ids starting at 1.

Next write a complete, self-contained `plan`. This `plan` is the authoritative
output that will be executed and scored. Every plan object must explicitly
contain `agent`, `id`, `task`, `reason`, and `dep`.

Do not shorten, duplicate, or otherwise distort the final `plan` merely to make
it match `prefetch_agents`. Correctness and completeness of `plan` take priority.
For this benchmark, the final plan must contain exactly the three independent
calls specified below.

Parallel independence requirements:
- Generate exactly three tasks and use each listed agent exactly once in the
  order shown above.
- Every agent receives the complete original question and every answer option.
  Each task must independently solve the entire question and select an answer.
- Every `dep` must be exactly `[]`. Never output `[1]`, `[2]`, `[1, 2]`, or any
  other dependency.
- Do not split successive reasoning or calculation stages across different
  agents. The reasoning_agent must perform the complete derivation itself.
- The elimination_agent must not verify or continue another agent's result. It
  must independently evaluate all options from the original input.
- The three independent responses are combined outside the planner, so the plan
  must not include a synthesis or finalization dependency.
- Do not output prose before or after the JSON object."""

BASE_LLAMA3_PROMPT = replace_json_array_example(
    planner_prompt,
    "Output only one valid JSON array containing exactly three tasks. Use each agent\n"
    "exactly once and set every dependency list to [] so all tasks can run in parallel:",
    OUTPUT_CONTRACT,
)

CONFIG = {
    "input": "benchmarks/mmlu/mmlu_pro_sampled.json",
    "plans_output": "benchmarks/mmlu/mmlu_plans_base_llama3.json",
    "benchmark_output": "benchmarks/mmlu/mmlu_subtask_base_llama3.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "source": "TIGER-Lab/MMLU-Pro",
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
        print_summary=print_summary,
        description="Build MMLU-Pro base_llama3 agent-names-first plans.",
    )

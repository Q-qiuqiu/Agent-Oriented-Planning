import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]

from base_llama3_builder import build_prefetch_reasoning_plan_prompt, run_builder
from build_subtask_benchmark import (
    AGENTS,
    expand_plans,
    load_queries,
    normalize_plan,
    print_summary,
)
# Use the v2 long-reasoning prompt (one ~2-sentence paragraph per agent plus a
# synthesis paragraph) so the reasoning length matches the base_lladav1 /
# base_llama3 v2 runs; only the PREFETCH_AGENTS prefix is added on top.
from build_subtask_full_benchmark_v2 import FULL_PLANNER_PROMPT


PROMPT_VERSION = "chronoqa_base_llama3_prefetch_reasoning_plan_v6_3sent"

BASE_LLAMA3_PROMPT = build_prefetch_reasoning_plan_prompt(
    FULL_PLANNER_PROMPT,
    language="zh",
)

CONFIG = {
    "input": "benchmarks/chronoqa/chronoqa_sampled.json",
    "plans_output": "benchmarks/chronoqa/chronoqa_plans_base_llama3_agentfirst.json",
    "benchmark_output": "benchmarks/chronoqa/chronoqa_subtask_base_llama3_agentfirst.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "source": "czy1999/ChronoQA",
    "prompt_version": PROMPT_VERSION,
    "planner_mode": "prefetch_reasoning_plan",
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
        description="Build ChronoQA base_llama3 agent-names-first plans.",
    )

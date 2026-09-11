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
from build_subtask_full_benchmark import FULL_PLANNER_PROMPT


PROMPT_VERSION = "mmlu_base_llama3_prefetch_reasoning_plan_v5"

BASE_LLAMA3_PROMPT = build_prefetch_reasoning_plan_prompt(FULL_PLANNER_PROMPT)

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
        description="Build MMLU-Pro base_llama3 agent-names-first plans.",
    )

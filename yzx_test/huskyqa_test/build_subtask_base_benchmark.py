"""Generate HuskyQA plans with the base Llama model as direct JSON arrays."""

import argparse

from build_subtask_benchmark import AGENTS, print_agent_selection_summary
from build_subtask_full_benchmark_v2 import (
    build_plans,
    expand_plans,
    load_queries,
    save_json,
)


CONFIG = {
    "input": "benchmarks/huskyqa/huskyqa_raw.json",
    "plans_output": "benchmarks/huskyqa/huskyqa_plans_base_llama3_agentfirst.json",
    "benchmark_output": "benchmarks/huskyqa/huskyqa_subtask_base_llama3_agentfirst.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "agents": AGENTS,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build base-Llama3 HuskyQA plans as direct JSON arrays."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument(
        "--planner-temperature", type=float, default=CONFIG["planner_temperature"]
    )
    parser.add_argument(
        "--planner-max-tokens", type=int, default=CONFIG["planner_max_tokens"]
    )
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--agents", nargs="+", default=CONFIG["agents"], choices=AGENTS)
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(vars(args))
    if not config["planner_api_url"]:
        raise ValueError("Missing planner API URL")
    if config["planner_max_tokens"] < 1:
        raise ValueError("planner_max_tokens must be at least 1")

    plans = build_plans(load_queries(config["input"]), config)
    save_json(config["plans_output"], plans)
    benchmark = expand_plans(plans, config["agents"])
    save_json(config["benchmark_output"], benchmark)
    print(f"Saved plans: {config['plans_output']} ({len(plans)} queries)")
    print(
        f"Saved benchmark: {config['benchmark_output']} "
        f"({len(benchmark)} rows)"
    )
    print_agent_selection_summary(plans)


if __name__ == "__main__":
    main()

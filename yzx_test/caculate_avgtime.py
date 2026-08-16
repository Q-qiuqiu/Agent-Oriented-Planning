import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Edit these paths directly when the generated files use different names.
CONFIG = {
    "inputs": [
        {
            "benchmark": "HuskyQA",
            "planner": "LLaDA",
            "path": ROOT
            / "benchmarks/huskyqa/huskyqa_plans_llada_now.json",
        },
        {
            "benchmark": "HuskyQA",
            "planner": "Llama",
            "path": ROOT
            / "benchmarks/huskyqa/huskyqa_plans_llama3.json",
        },
        {
            "benchmark": "IIRC",
            "planner": "LLaDA",
            "path": ROOT / "benchmarks/iirc/iirc_plans_llada.json",
        },
        {
            "benchmark": "IIRC",
            "planner": "Llama",
            "path": ROOT / "benchmarks/iirc/iirc_plans_llama3.json",
        },
    ],
}


def valid_time(record):
    value = record.get("time")
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def generated_response(record):
    raw_plan = record.get("raw_plan")
    return valid_time(record) and isinstance(raw_plan, str) and bool(raw_plan.strip())


def summarize_file(item):
    path = Path(item["path"])
    summary = {
        "benchmark": item["benchmark"],
        "planner": item["planner"],
        "path": str(path),
        "status": "ok",
        "record_count": 0,
        "generated_response_count": 0,
        "successful_count": 0,
        "failed_count": 0,
        "missing_time_count": 0,
        "avg_response_seconds": None,
        "median_response_seconds": None,
        "min_response_seconds": None,
        "max_response_seconds": None,
        "avg_success_seconds": None,
        "agent_counts": {},
        "total_agent_calls": 0,
    }
    if not path.exists():
        summary["status"] = "missing"
        return summary

    try:
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        summary["status"] = f"error: {exc}"
        return summary
    if not isinstance(records, list):
        summary["status"] = "error: top-level JSON value is not an array"
        return summary

    generated_times = [
        float(record["time"])
        for record in records
        if isinstance(record, dict) and generated_response(record)
    ]
    success_times = [
        float(record["time"])
        for record in records
        if isinstance(record, dict)
        and record.get("error") is None
        and valid_time(record)
    ]
    failed_count = sum(
        1
        for record in records
        if isinstance(record, dict) and record.get("error") is not None
    )
    timed_count = sum(
        1
        for record in records
        if isinstance(record, dict) and valid_time(record)
    )
    agent_counts = Counter(
        step.get("agent").strip()
        for record in records
        if isinstance(record, dict) and isinstance(record.get("plan"), list)
        for step in record["plan"]
        if isinstance(step, dict)
        and isinstance(step.get("agent"), str)
        and step.get("agent").strip()
    )

    summary.update(
        {
            "record_count": len(records),
            "generated_response_count": len(generated_times),
            "successful_count": len(success_times),
            "failed_count": failed_count,
            "missing_time_count": len(records) - timed_count,
            "avg_response_seconds": (
                statistics.fmean(generated_times) if generated_times else None
            ),
            "median_response_seconds": (
                statistics.median(generated_times) if generated_times else None
            ),
            "min_response_seconds": min(generated_times) if generated_times else None,
            "max_response_seconds": max(generated_times) if generated_times else None,
            "avg_success_seconds": (
                statistics.fmean(success_times) if success_times else None
            ),
            "agent_counts": dict(agent_counts),
            "total_agent_calls": sum(agent_counts.values()),
        }
    )
    return summary


def format_seconds(value):
    return f"{value:.4f}" if value is not None else "N/A"


def print_table(summaries):
    headers = [
        "Benchmark",
        "Planner",
        "Records",
        "Generated",
        "Success",
        "Failed",
        "No time",
        "Avg (s)",
        "Median (s)",
        "Min (s)",
        "Max (s)",
        "Success avg (s)",
        "Status",
    ]
    rows = []
    for item in summaries:
        rows.append(
            [
                item["benchmark"],
                item["planner"],
                str(item["record_count"]),
                str(item["generated_response_count"]),
                str(item["successful_count"]),
                str(item["failed_count"]),
                str(item["missing_time_count"]),
                format_seconds(item["avg_response_seconds"]),
                format_seconds(item["median_response_seconds"]),
                format_seconds(item["min_response_seconds"]),
                format_seconds(item["max_response_seconds"]),
                format_seconds(item["avg_success_seconds"]),
                item["status"],
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row):
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    print(render(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in rows:
        print(render(row))


def print_agent_table(summaries):
    preferred_order = [
        "code_agent",
        "math_agent",
        "search_agent",
        "commonsense_agent",
        "context_agent",
        "retrieval_agent",
        "reasoning_agent",
        "calculation_agent",
        "answerability_agent",
    ]
    benchmarks = list(dict.fromkeys(summary["benchmark"] for summary in summaries))
    for benchmark in benchmarks:
        benchmark_summaries = [
            summary
            for summary in summaries
            if summary["benchmark"] == benchmark
        ]
        present_agents = {
            agent
            for summary in benchmark_summaries
            for agent in summary["agent_counts"]
        }
        agents = [agent for agent in preferred_order if agent in present_agents]
        agents.extend(sorted(present_agents - set(agents)))

        headers = ["Planner", "Total calls", *agents]
        rows = []
        for summary in benchmark_summaries:
            counts = summary["agent_counts"]
            rows.append(
                [
                    summary["planner"],
                    str(summary["total_agent_calls"]),
                    *(str(counts.get(agent, 0)) for agent in agents),
                ]
            )

        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers))
        ]

        def render(row):
            return "| " + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            ) + " |"

        print(f"\n{benchmark} agent calls from normalized plans")
        print(render(headers))
        print("|-" + "-|-".join("-" * width for width in widths) + "-|")
        for row in rows:
            print(render(row))


def main():
    summaries = [summarize_file(item) for item in CONFIG["inputs"]]
    print_table(summaries)
    print_agent_table(summaries)


if __name__ == "__main__":
    main()

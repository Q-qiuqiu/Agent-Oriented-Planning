import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Edit these paths directly when the generated files use different names.
CONFIG = {
    "inputs": [
        {
            "benchmark": "HuskyQA",
            "planner": "LLaDA",
            "path": ROOT
            / "benchmarks/huskyqa/huskyqa_plans_full_llada.json",
        },
        {
            "benchmark": "HuskyQA",
            "planner": "Llama",
            "path": ROOT
            / "benchmarks/huskyqa/huskyqa_plans_full_llama3.json",
        },
        {
            "benchmark": "IIRC",
            "planner": "LLaDA",
            "path": ROOT / "benchmarks/iirc/iirc_plans_full_llada.json",
        },
        {
            "benchmark": "IIRC",
            "planner": "Llama",
            "path": ROOT / "benchmarks/iirc/iirc_plans_full_llama3.json",
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


def main():
    summaries = [summarize_file(item) for item in CONFIG["inputs"]]
    print_table(summaries)


if __name__ == "__main__":
    main()

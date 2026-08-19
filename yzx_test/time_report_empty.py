import math
from pathlib import Path

import time_report as timing


ROOT = Path(__file__).resolve().parent

# Every query starts with an empty cluster. Edit only the assignment and paths
# when switching result sets.
CONFIG = {
    "benchmarks": {
        "huskyqa": {
            "assignment": "qc_q_f_m",
            "planner_file": "benchmarks/huskyqa/huskyqa_plans_llada_now.json",
            "results_dir": "huskyqa_test/results_1b_llada_now",
            "timings_file": "benchmarks/fastdllm_log/huskyqa_timings.jsonl",
        },
        "iirc": {
            "assignment": "q_q_q_q_q",
            "planner_file": "benchmarks/iirc/iirc_plans_llada_now.json",
            "results_dir": "iirc_test/results_1b_llada_now",
            "timings_file": "benchmarks/fastdllm_log/iirc_timings.jsonl",
        },
    },
    "device_counts": [2, 3],
    "cold_start_file": "benchmarks/fastdllm_log/model_start_time.json",
    "prefetch_agent_limit": 4,
    "prefetch_time_field": "decision_seconds",
    "seconds_precision": 4,
}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_benchmark(name, config):
    assignment = timing.normalize_assignment(config["assignment"])
    planner_path = resolve_path(config["planner_file"])
    result_dir = resolve_path(config["results_dir"])
    timings_path = resolve_path(config["timings_file"])
    response_path = result_dir / f"{timing.RESPONSE_PREFIX}{assignment}.json"
    summary_path = result_dir / f"{timing.SUMMARY_PREFIX}{assignment}.json"

    paths = (planner_path, response_path, summary_path, timings_path)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(map(str, missing)))

    plans = timing.keyed_records(timing.load_json(planner_path), planner_path)
    responses = timing.keyed_records(timing.load_json(response_path), response_path)
    summaries = timing.keyed_records(timing.load_json(summary_path), summary_path)
    timing_records = timing.timing_records_for_plans(plans, timings_path)
    planner_times = timing.planner_times(plans)
    subtask_times = timing.subtask_times(responses)
    summary_times = timing.summary_times(summaries)
    successful_keys = set(planner_times) & set(subtask_times) & set(summary_times)

    return {
        "name": name,
        "assignment": assignment,
        "expected_count": len(plans),
        "ordered_keys": [key for key in plans if key in successful_keys],
        "responses": responses,
        "timings": timing_records,
        "planner_times": planner_times,
        "summary_times": summary_times,
        "agent_models": timing.agent_models_from_responses(responses),
    }


def simulate_empty_cluster(
    benchmark,
    device_count,
    cold_start_times,
    prefetch,
    prefetch_time_field,
    prefetch_agent_limit,
):
    values = []
    for key in benchmark["ordered_keys"]:
        devices = [
            {"model": None, "last_used": -1}
            for _ in range(device_count)
        ]
        clock = 0
        initial_ready_times = None
        if prefetch:
            initial_ready_times, clock = timing.prepare_planner_prefetch(
                benchmark["timings"].get(key),
                devices,
                benchmark["agent_models"],
                cold_start_times,
                clock,
                prefetch_time_field,
                prefetch_agent_limit,
            )
        subtask_time, _ = timing.simulate_query_time(
            benchmark["responses"][key],
            devices,
            cold_start_times,
            clock,
            initial_device_ready_times=initial_ready_times,
        )
        values.append(
            benchmark["planner_times"][key]
            + subtask_time
            + benchmark["summary_times"][key]
        )
    return timing.summarize(values, benchmark["expected_count"])


def format_number(value, precision):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def print_table(benchmark, rows, precision):
    headers = [
        "Mode",
        "Devices",
        "Successful",
        "Excluded",
        "Average (s)",
        "Min (s)",
        "Max (s)",
        "P95 (s)",
    ]
    display_rows = [
        [
            row["mode"],
            str(row["devices"]),
            str(row["count"]),
            str(row["missing"]),
            format_number(row["mean"], precision),
            format_number(row["min"], precision),
            format_number(row["max"], precision),
            format_number(row["p95"], precision),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in display_rows))
        for index in range(len(headers))
    ]

    def render(values):
        return "| " + " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ) + " |"

    print(
        f"\n{benchmark['name'].upper()} | assignment={benchmark['assignment']} "
        "| empty cluster before every query"
    )
    print(render(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in display_rows:
        print(render(row))


def main():
    cold_start_path = resolve_path(CONFIG["cold_start_file"])
    if not cold_start_path.exists():
        raise FileNotFoundError(cold_start_path)
    cold_start_times = timing.load_cold_start_times(cold_start_path)

    device_counts = list(dict.fromkeys(CONFIG["device_counts"]))
    if not device_counts or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in device_counts
    ):
        raise ValueError("CONFIG['device_counts'] must contain positive integers")

    prefetch_time_field = CONFIG["prefetch_time_field"]
    if prefetch_time_field not in {"decision_seconds", "confirmation_seconds"}:
        raise ValueError("Unsupported prefetch_time_field")
    prefetch_agent_limit = CONFIG["prefetch_agent_limit"]
    if (
        not isinstance(prefetch_agent_limit, int)
        or isinstance(prefetch_agent_limit, bool)
        or prefetch_agent_limit <= 0
    ):
        raise ValueError("prefetch_agent_limit must be a positive integer")

    for name, config in CONFIG["benchmarks"].items():
        benchmark = load_benchmark(name, config)
        rows = []
        for device_count in device_counts:
            baseline = simulate_empty_cluster(
                benchmark,
                device_count,
                cold_start_times,
                False,
                prefetch_time_field,
                prefetch_agent_limit,
            )
            prefetch = simulate_empty_cluster(
                benchmark,
                device_count,
                cold_start_times,
                True,
                prefetch_time_field,
                prefetch_agent_limit,
            )
            rows.append({"mode": "No prefetch", "devices": device_count, **baseline})
            rows.append({"mode": "Planner prefetch", "devices": device_count, **prefetch})
        print_table(benchmark, rows, CONFIG["seconds_precision"])


if __name__ == "__main__":
    main()

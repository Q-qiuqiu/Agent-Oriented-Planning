import math
from pathlib import Path

import time_report as timing


ROOT = Path(__file__).resolve().parent

# IIRC is split into contiguous partitions. Each run interleaves the complete
# HuskyQA set with one IIRC partition, matching analyze_multi_colliosions.py.
CONFIG = {
    "benchmarks": {
        "huskyqa": {
            "assignment": "f_q_m",
            "planner_file": "benchmarks/huskyqa/huskyqa_plans_llama3.json",
            "results_dir": "huskyqa_test/results_1b_llama",
            "timings_file": "benchmarks/fastdllm_log/huskyqa_timings.jsonl",
        },
        "iirc": {
            "assignment": "q_q_q_q_q",
            "planner_file": "benchmarks/iirc/iirc_plans_llada_now.json",
            "results_dir": "iirc_test/results_1b_llada_now",
            "timings_file": "benchmarks/fastdllm_log/iirc_timings.jsonl",
        },
    },
    # One group is H...H followed by I...I. Remaining queries continue in
    # source order after one benchmark is exhausted.
    "huskyqa_queries_per_group": 1,
    "iirc_queries_per_group": 1,
    "iirc_partitions": 5,
    "device_counts": [2, 3, 4],
    "cold_start_file": "benchmarks/fastdllm_log/model_start_time.json",
    # None means prefetch one useful model instance per available device.
    "prefetch_agent_limit": None,
    "prefetch_time_field": "decision_seconds",
    "seconds_precision": 4,
}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"CONFIG[{name!r}] must be a positive integer")


def partition_entries(entries, partition_count):
    validate_positive_integer("iirc_partitions", partition_count)
    if partition_count > len(entries):
        raise ValueError(
            "iirc_partitions cannot exceed IIRC query count: "
            f"{partition_count} > {len(entries)}"
        )

    base_size, larger_count = divmod(len(entries), partition_count)
    partitions = []
    start = 0
    for partition_index in range(partition_count):
        size = base_size + (partition_index < larger_count)
        end = start + size
        partitions.append(entries[start:end])
        start = end
    return partitions


def interleave_entries(huskyqa_entries, iirc_entries, huskyqa_count, iirc_count):
    validate_positive_integer("huskyqa_queries_per_group", huskyqa_count)
    validate_positive_integer("iirc_queries_per_group", iirc_count)

    arrivals = []
    huskyqa_index = 0
    iirc_index = 0
    while (
        huskyqa_index < len(huskyqa_entries)
        or iirc_index < len(iirc_entries)
    ):
        for _ in range(huskyqa_count):
            if huskyqa_index >= len(huskyqa_entries):
                break
            arrivals.append(huskyqa_entries[huskyqa_index])
            huskyqa_index += 1
        for _ in range(iirc_count):
            if iirc_index >= len(iirc_entries):
                break
            arrivals.append(iirc_entries[iirc_index])
            iirc_index += 1
    return arrivals


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

    entries = []
    for key in plans:
        successful = key in successful_keys
        entries.append(
            {
                "dataset": name,
                "key": key,
                "successful": successful,
                "planner_time": planner_times.get(key),
                "response": responses.get(key),
                "summary_time": summary_times.get(key),
                "timing": timing_records.get(key),
            }
        )

    return {
        "name": name,
        "assignment": assignment,
        "entries": entries,
        "agent_models": timing.agent_models_from_responses(responses),
        "successful_count": len(successful_keys),
        "planner_path": planner_path,
        "response_path": response_path,
        "summary_path": summary_path,
        "timings_path": timings_path,
    }


def simulate_arrivals(
    arrivals,
    datasets,
    device_count,
    cold_start_times,
    prefetch,
    prefetch_time_field,
    prefetch_agent_limit,
):
    devices = [
        {"model": None, "last_used": -1}
        for _ in range(device_count)
    ]
    clock = 0
    values = []

    for arrival in arrivals:
        if not arrival["successful"]:
            continue
        initial_ready_times = None
        if prefetch:
            initial_ready_times, clock = timing.prepare_planner_prefetch(
                arrival["timing"],
                devices,
                datasets[arrival["dataset"]]["agent_models"],
                cold_start_times,
                clock,
                prefetch_time_field,
                prefetch_agent_limit,
                plan_record=arrival["response"],
            )
        subtask_time, clock = timing.simulate_query_time(
            arrival["response"],
            devices,
            cold_start_times,
            clock,
            initial_device_ready_times=initial_ready_times,
        )
        values.append(
            float(arrival["planner_time"])
            + subtask_time
            + float(arrival["summary_time"])
        )
    return timing.summarize(values, len(arrivals))


def average_run_summaries(summaries):
    if not summaries:
        raise ValueError("No run summaries to average")
    fields = ("count", "missing", "mean", "min", "max", "p95")
    return {
        field: sum(summary[field] for summary in summaries) / len(summaries)
        for field in fields
    }


def format_number(value, precision):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def print_crossbench_table(title, rows, precision):
    headers = [
        "Devices",
        "Avg successful/run",
        "Avg excluded/run",
        "Average (s)",
        "Avg run min (s)",
        "Avg run max (s)",
        "Avg run P95 (s)",
    ]
    display_rows = [
        [
            str(row["devices"]),
            f"{row['count']:.2f}",
            f"{row['missing']:.2f}",
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

    print(f"\n{title}")
    print(render(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in display_rows:
        print(render(row))


def main():
    benchmark_configs = CONFIG["benchmarks"]
    if set(benchmark_configs) != {"huskyqa", "iirc"}:
        raise ValueError("CONFIG['benchmarks'] must contain huskyqa and iirc")

    datasets = {
        name: load_benchmark(name, config)
        for name, config in benchmark_configs.items()
    }
    cold_start_path = resolve_path(CONFIG["cold_start_file"])
    if not cold_start_path.exists():
        raise FileNotFoundError(cold_start_path)
    cold_start_times = timing.load_cold_start_times(cold_start_path)

    device_counts = list(dict.fromkeys(CONFIG["device_counts"]))
    if not device_counts:
        raise ValueError("CONFIG['device_counts'] must not be empty")
    for device_count in device_counts:
        validate_positive_integer("device_counts item", device_count)

    prefetch_time_field = CONFIG["prefetch_time_field"]
    if prefetch_time_field not in {"decision_seconds", "confirmation_seconds"}:
        raise ValueError("Unsupported prefetch_time_field")
    prefetch_agent_limit = CONFIG["prefetch_agent_limit"]
    if prefetch_agent_limit is not None:
        validate_positive_integer("prefetch_agent_limit", prefetch_agent_limit)

    partitions = partition_entries(
        datasets["iirc"]["entries"],
        CONFIG["iirc_partitions"],
    )
    runs = [
        interleave_entries(
            datasets["huskyqa"]["entries"],
            partition,
            CONFIG["huskyqa_queries_per_group"],
            CONFIG["iirc_queries_per_group"],
        )
        for partition in partitions
    ]

    baseline_rows = []
    prefetch_rows = []
    for device_count in device_counts:
        baseline_summaries = [
            simulate_arrivals(
                arrivals,
                datasets,
                device_count,
                cold_start_times,
                False,
                prefetch_time_field,
                prefetch_agent_limit,
            )
            for arrivals in runs
        ]
        prefetch_summaries = [
            simulate_arrivals(
                arrivals,
                datasets,
                device_count,
                cold_start_times,
                True,
                prefetch_time_field,
                prefetch_agent_limit,
            )
            for arrivals in runs
        ]
        baseline_rows.append(
            {"devices": device_count, **average_run_summaries(baseline_summaries)}
        )
        prefetch_rows.append(
            {"devices": device_count, **average_run_summaries(prefetch_summaries)}
        )

    partition_sizes = [len(partition) for partition in partitions]
    print(
        "Arrival pattern: "
        f"{CONFIG['huskyqa_queries_per_group']} HuskyQA -> "
        f"{CONFIG['iirc_queries_per_group']} IIRC"
    )
    print(f"IIRC partitions: {partition_sizes}")
    print(
        "Assignments: "
        f"HuskyQA={datasets['huskyqa']['assignment']} | "
        f"IIRC={datasets['iirc']['assignment']}"
    )
    print_crossbench_table(
        "Cross-benchmark end-to-end without planner prefetch "
        "(five-run arithmetic mean)",
        baseline_rows,
        CONFIG["seconds_precision"],
    )
    print_crossbench_table(
        "Cross-benchmark end-to-end with device-count planner prefetch "
        f"(five-run arithmetic mean, {prefetch_time_field})",
        prefetch_rows,
        CONFIG["seconds_precision"],
    )


if __name__ == "__main__":
    main()

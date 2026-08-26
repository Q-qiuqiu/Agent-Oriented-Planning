import math
from pathlib import Path

import time_report as timing


ROOT = Path(__file__).resolve().parent

# Add future benchmarks to "benchmarks", then select and order them through
# "arrival_pattern". Each tuple is (benchmark_name, queries_per_group).
CONFIG = {
    "benchmarks": {
        "huskyqa": {
            "display_name": "HuskyQA",
            "assignment": "g_q_l",
            "planner_file": "benchmarks/huskyqa/huskyqa_plans_full_llada.json",
            "results_dir": "huskyqa_test/results_1b_full_llada",
            "timings_file": "benchmarks/fastdllm_log/huskyqa_full_timings.jsonl",
        },
        "iirc": {
            "display_name": "IIRC",
            "assignment": "g_q_l",
            "planner_file": "benchmarks/iirc/iirc_plans_full_llada.json",
            "results_dir": "iirc_test/results_1b_full_llada",
            "timings_file": "benchmarks/fastdllm_log/iirc_full_timings.jsonl",
        },
        "mmlu": {
            "display_name": "MMLU-Pro",
            "assignment": "g_q_l",
            "planner_file": "benchmarks/mmlu/mmlu_plans_full_llada.json",
            "results_dir": "mmlu_test/results_1b_full_llada",
            "timings_file": "benchmarks/fastdllm_log/mmlu_full_timings.jsonl",
        },
        "chronoqa": {
            "display_name": "ChronoQA",
            "assignment": "g_q_l",
            "planner_file": "benchmarks/chronoqa/chronoqa_plans_full_llada.json",
            "results_dir": "chronoqa_test/results_1b_full_llada",
            "timings_file": "benchmarks/fastdllm_log/chronoqa_full_timings.jsonl",
        },
    },
    # One round of ("huskyqa", 1), ("iirc", 5), ("mmlu", 2) means:
    # 1 HuskyQA -> 5 IIRC -> 2 MMLU. Rounds continue until all are exhausted.
    # Remove an item to exclude that benchmark from this report.
    "arrival_pattern": [
        ("huskyqa", 1),
        #("iirc", 1),
        ("mmlu", 1),
        # ("chronoqa", 1),
    ],
    # Usually leave this empty for one run over every complete benchmark.
    # To reproduce the old five-run setup, use {"iirc": 5}; benchmarks with
    # partition count 1 are repeated in full in each run. If several benchmarks
    # are partitioned, they must use the same count.
    "partition_counts": {},
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


def partition_entries(entries, partition_count, benchmark_name):
    validate_positive_integer(
        f"partition_counts[{benchmark_name!r}]", partition_count
    )
    if partition_count > len(entries):
        raise ValueError(
            f"Partition count for {benchmark_name!r} cannot exceed its query count: "
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


def interleave_entries(entries_by_benchmark, arrival_pattern):
    arrivals = []
    positions = {name: 0 for name, _ in arrival_pattern}
    while any(
        positions[name] < len(entries_by_benchmark[name])
        for name, _ in arrival_pattern
    ):
        for name, queries_per_group in arrival_pattern:
            entries = entries_by_benchmark[name]
            start = positions[name]
            end = min(start + queries_per_group, len(entries))
            arrivals.extend(entries[start:end])
            positions[name] = end
    return arrivals


def validate_arrival_pattern(benchmark_configs):
    raw_pattern = CONFIG.get("arrival_pattern")
    if not isinstance(raw_pattern, (list, tuple)) or not raw_pattern:
        raise ValueError("CONFIG['arrival_pattern'] must be a non-empty list")

    pattern = []
    seen = set()
    for index, item in enumerate(raw_pattern):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "Each arrival_pattern item must be "
                "(benchmark_name, queries_per_group)"
            )
        name, queries_per_group = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"arrival_pattern item {index} has no benchmark name")
        name = name.strip()
        if name not in benchmark_configs:
            known = ", ".join(benchmark_configs)
            raise ValueError(
                f"Unknown benchmark {name!r} in arrival_pattern; configured: {known}"
            )
        if name in seen:
            raise ValueError(f"Benchmark {name!r} appears more than once in arrival_pattern")
        validate_positive_integer(
            f"arrival_pattern[{index}].queries_per_group", queries_per_group
        )
        seen.add(name)
        pattern.append((name, queries_per_group))

    if len(pattern) < 2:
        raise ValueError("Cross-benchmark reporting requires at least two benchmarks")
    return pattern


def build_arrival_runs(datasets, arrival_pattern):
    configured_counts = CONFIG.get("partition_counts", {})
    if not isinstance(configured_counts, dict):
        raise ValueError("CONFIG['partition_counts'] must be a dictionary")

    selected_names = {name for name, _ in arrival_pattern}
    unknown_names = set(configured_counts) - selected_names
    if unknown_names:
        raise ValueError(
            "partition_counts contains benchmarks not selected by arrival_pattern: "
            + ", ".join(sorted(unknown_names))
        )

    partition_counts = {
        name: configured_counts.get(name, 1)
        for name, _ in arrival_pattern
    }
    for name, count in partition_counts.items():
        validate_positive_integer(f"partition_counts[{name!r}]", count)

    run_count = max(partition_counts.values())
    incompatible = {
        name: count
        for name, count in partition_counts.items()
        if count not in {1, run_count}
    }
    if incompatible:
        raise ValueError(
            "Every partition count must be 1 or the common maximum run count "
            f"{run_count}; got {incompatible}"
        )

    partitions = {
        name: partition_entries(datasets[name]["entries"], count, name)
        for name, count in partition_counts.items()
    }
    runs = []
    for run_index in range(run_count):
        entries_by_benchmark = {
            name: benchmark_partitions[
                run_index if len(benchmark_partitions) > 1 else 0
            ]
            for name, benchmark_partitions in partitions.items()
        }
        runs.append(interleave_entries(entries_by_benchmark, arrival_pattern))
    return runs, partitions


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
    averaged = {}
    for field in fields:
        values = [
            summary[field]
            for summary in summaries
            if summary[field] is not None
        ]
        averaged[field] = sum(values) / len(values) if values else None
    return averaged


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
    if not isinstance(benchmark_configs, dict) or not benchmark_configs:
        raise ValueError("CONFIG['benchmarks'] must be a non-empty dictionary")
    arrival_pattern = validate_arrival_pattern(benchmark_configs)

    datasets = {
        name: load_benchmark(name, benchmark_configs[name])
        for name, _ in arrival_pattern
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

    runs, partitions = build_arrival_runs(datasets, arrival_pattern)

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

    display_names = {
        name: benchmark_configs[name].get("display_name", name)
        for name, _ in arrival_pattern
    }
    pattern_text = " -> ".join(
        f"{queries_per_group} {display_names[name]}"
        for name, queries_per_group in arrival_pattern
    )
    partition_text = " | ".join(
        f"{display_names[name]}={[len(partition) for partition in partitions[name]]}"
        for name, _ in arrival_pattern
    )
    assignment_text = " | ".join(
        f"{display_names[name]}={datasets[name]['assignment']}"
        for name, _ in arrival_pattern
    )
    run_description = (
        "single run" if len(runs) == 1 else f"{len(runs)}-run arithmetic mean"
    )
    print(f"Arrival pattern: {pattern_text}")
    print(f"Partitions: {partition_text}")
    print(f"Assignments: {assignment_text}")
    print_crossbench_table(
        "Cross-benchmark end-to-end without planner prefetch "
        f"({run_description})",
        baseline_rows,
        CONFIG["seconds_precision"],
    )
    print_crossbench_table(
        "Cross-benchmark end-to-end with device-count planner prefetch "
        f"({run_description}, {prefetch_time_field})",
        prefetch_rows,
        CONFIG["seconds_precision"],
    )


if __name__ == "__main__":
    main()

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Normally you only need to edit assignments. Four aliases select HuskyQA;
# five aliases select IIRC. Add multiple values for batch reporting.
CONFIG = {
    "assignments": [
        "q_q_q_q",
        # "g_g_g_g_g",
        # "q_q_q_q",
        # "g_g_g_g",
    ],
    "benchmarks": {
        "iirc": {
            "role_count": 5,
            "planner_file": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
                "iirc/iirc_plans_llada_now.json"
            ),
            "results_dir": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/"
                "iirc_test/results_1b_llada_now"
            ),
            "timings_file": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
                "fastdllm_log/iirc_timings.jsonl"
            ),
        },
        "huskyqa": {
            "role_count": 4,
            "planner_file": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
                "huskyqa/huskyqa_plans_llada_now.json"
            ),
            "results_dir": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/"
                "huskyqa_test/results_1b_llada_now"
            ),
            "timings_file": (
                "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
                "fastdllm_log/huskyqa_timings.jsonl"
            ),
        },
    },
    "cold_start_file": (
        "/data/home/yzx/Agent-Oriented-Planning/yzx_test/benchmarks/"
        "fastdllm_log/model_start_time.json"
    ),
    "device_counts": [2, 3, 4],
    "prefetch_agent_limit": 4,
    "prefetch_time_field": "decision_seconds",
    "seconds_precision": 4,
}


RESPONSE_PREFIX = "subtask_hetro_responses_"
SUMMARY_PREFIX = "summary_result_"


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            records.append(record)
    return records


def valid_time(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def record_key(record):
    source_index = record.get("source_index")
    if source_index is not None:
        return f"source:{source_index}"
    query = record.get("query")
    if query:
        return f"query:{query}"
    return None


def keyed_records(records, path):
    if not isinstance(records, list):
        raise ValueError(f"Top-level JSON value must be an array: {path}")
    result = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {position} is not an object: {path}")
        key = record_key(record)
        if key is None:
            raise ValueError(f"Record {position} has no source_index or query: {path}")
        if key in result:
            raise ValueError(f"Duplicate query key {key!r}: {path}")
        result[key] = record
    return result


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values, expected_count):
    return {
        "count": len(values),
        "missing": max(expected_count - len(values), 0),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def planner_times(records):
    return {
        key: float(record["time"])
        for key, record in records.items()
        if record.get("error") is None and valid_time(record.get("time"))
    }


def subtask_times(records):
    times = {}
    for key, record in records.items():
        steps = record.get("steps")
        if record.get("error") is not None or not isinstance(steps, list) or not steps:
            continue
        if not all(
            isinstance(step, dict) and step.get("error") is None for step in steps
        ):
            continue
        step_times = [step.get("time") for step in steps]
        if not all(valid_time(value) for value in step_times):
            continue
        times[key] = float(sum(step_times))
    return times


def summary_times(records):
    return {
        key: float(record["summary_time"])
        for key, record in records.items()
        if record.get("summary_error") is None
        and valid_time(record.get("summary_time"))
    }


def canonical_model_name(value):
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def load_cold_start_times(path):
    data = load_json(path)
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No model cold-start values found in {path}")

    result = {}
    for model, value in models.items():
        seconds = value.get("cold_start_seconds") if isinstance(value, dict) else value
        if not valid_time(seconds):
            raise ValueError(f"Invalid cold-start time for model {model!r}: {seconds!r}")
        result[str(model)] = float(seconds)
    return result


def cold_start_seconds(model, cold_start_times):
    candidates = (str(model).rstrip("/"), canonical_model_name(model))
    for candidate in candidates:
        if candidate in cold_start_times:
            return cold_start_times[candidate]
    known = ", ".join(sorted(cold_start_times))
    raise ValueError(
        f"No cold-start time configured for model {model!r}. Known models: {known}"
    )


def timing_records_for_plans(plans, path):
    records_by_query = {}
    for record in load_jsonl(path):
        query = record.get("query")
        if not query:
            continue
        previous = records_by_query.get(query)
        current_attempt = record.get("attempt_count") or 0
        previous_attempt = (
            (previous.get("attempt_count") or 0) if previous else -1
        )
        if previous is None or current_attempt >= previous_attempt:
            records_by_query[query] = record

    result = {}
    for key, plan in plans.items():
        timing_query = plan.get("agent_context") or plan.get("query")
        timing = records_by_query.get(timing_query)
        if timing is not None:
            result[key] = timing
    return result


def agent_models_from_responses(records):
    result = {}
    for record in records.values():
        for step in record.get("steps") or []:
            agent = step.get("agent")
            model = step.get("model")
            if not agent or not model:
                continue
            model = str(model).rstrip("/")
            previous = result.get(agent)
            if previous is not None and previous != model:
                raise ValueError(
                    f"Agent {agent!r} uses multiple models in one response file: "
                    f"{previous!r} and {model!r}"
                )
            result[agent] = model
    return result


def prepare_simulation_tasks(record):
    steps = record.get("steps") or []
    indices_by_id = {}
    for index, step in enumerate(steps):
        step_id = str(step.get("id"))
        if step_id in indices_by_id:
            raise ValueError(
                f"Duplicate step id {step_id!r} at source_index="
                f"{record.get('source_index')}"
            )
        indices_by_id[step_id] = index

    tasks = []
    for index, step in enumerate(steps):
        model = step.get("model")
        if not model:
            raise ValueError(
                f"Step {step.get('id')} has no model at source_index="
                f"{record.get('source_index')}"
            )
        dependencies = []
        for dependency in step.get("dep") or []:
            dependency_id = str(
                dependency.get("id") if isinstance(dependency, dict) else dependency
            )
            if dependency_id not in indices_by_id:
                raise ValueError(
                    f"Step {step.get('id')} references missing dependency "
                    f"{dependency!r} at source_index={record.get('source_index')}"
                )
            dependency_index = indices_by_id[dependency_id]
            if dependency_index not in dependencies:
                dependencies.append(dependency_index)
        tasks.append(
            {
                "index": index,
                "agent": step.get("agent"),
                "model": str(model).rstrip("/"),
                "execution_time": float(step["time"]),
                "dependencies": dependencies,
            }
        )
    return tasks


def assign_ready_tasks(
    ready,
    free_devices,
    devices,
    tasks,
    future_matching_models=None,
):
    available_devices = set(free_devices)
    remaining_tasks = list(ready)
    assignments = []
    future_matching_models = dict(future_matching_models or {})

    # Maximize reuse before loading or replacing any model.
    for task_index in list(remaining_tasks):
        model = tasks[task_index]["model"]
        matching = [
            device_index
            for device_index in available_devices
            if devices[device_index]["model"] == model
        ]
        if not matching:
            continue
        device_index = max(
            matching,
            key=lambda value: (devices[value]["last_used"], -value),
        )
        assignments.append((task_index, device_index))
        remaining_tasks.remove(task_index)
        available_devices.remove(device_index)
        if not available_devices:
            return assignments

    for task_index in remaining_tasks:
        if not available_devices:
            break
        model = tasks[task_index]["model"]
        if future_matching_models.get(model, 0) > 0:
            future_matching_models[model] -= 1
            continue
        empty_devices = [
            device_index
            for device_index in available_devices
            if devices[device_index]["model"] is None
        ]
        if empty_devices:
            device_index = min(empty_devices)
        else:
            device_index = min(
                available_devices,
                key=lambda value: (devices[value]["last_used"], value),
            )
        assignments.append((task_index, device_index))
        available_devices.remove(device_index)
    return assignments


def prepare_planner_prefetch(
    timing,
    devices,
    agent_models,
    cold_start_times,
    clock,
    time_field,
    agent_limit,
):
    device_ready_times = [0.0] * len(devices)
    if (
        not isinstance(timing, dict)
        or timing.get("status") != "ok"
        or timing.get("error") is not None
    ):
        return device_ready_times, clock

    generation_seconds = timing.get("generation_seconds")
    if not valid_time(generation_seconds):
        return device_ready_times, clock

    events = sorted(
        (
            event
            for event in (timing.get("agents") or [])[:agent_limit]
            if isinstance(event, dict)
        ),
        key=lambda event: event.get("slot", 0),
    )
    available_devices = set(range(len(devices)))
    for event in events:
        if not available_devices:
            break
        model = agent_models.get(event.get("agent"))
        detected_at = event.get(time_field)
        if model is None or not valid_time(detected_at):
            continue

        matching_devices = [
            device_index
            for device_index in available_devices
            if devices[device_index]["model"] == model
        ]
        if matching_devices:
            device_index = max(
                matching_devices,
                key=lambda value: (devices[value]["last_used"], -value),
            )
            remaining_load = 0.0
        else:
            empty_devices = [
                device_index
                for device_index in available_devices
                if devices[device_index]["model"] is None
            ]
            if empty_devices:
                device_index = min(empty_devices)
            else:
                device_index = min(
                    available_devices,
                    key=lambda value: (devices[value]["last_used"], value),
                )
            load_seconds = cold_start_seconds(model, cold_start_times)
            overlap_seconds = max(float(generation_seconds) - detected_at, 0.0)
            remaining_load = max(load_seconds - overlap_seconds, 0.0)

        clock += 1
        devices[device_index]["model"] = model
        devices[device_index]["last_used"] = clock
        device_ready_times[device_index] = remaining_load
        available_devices.remove(device_index)

    return device_ready_times, clock


def simulate_query_time(
    record,
    devices,
    cold_start_times,
    clock,
    initial_device_ready_times=None,
):
    tasks = prepare_simulation_tasks(record)
    if not tasks:
        raise ValueError(f"No executable steps at source_index={record.get('source_index')}")

    pending = set(range(len(tasks)))
    completed = set()
    running = {}
    now = 0.0
    device_ready_times = list(initial_device_ready_times or [0.0] * len(devices))
    if len(device_ready_times) != len(devices):
        raise ValueError("Initial device-ready times do not match device count")

    while len(completed) < len(tasks):
        finished_devices = [
            device_index
            for device_index, run in running.items()
            if run["finish_time"] <= now
        ]
        for device_index in finished_devices:
            completed.add(running[device_index]["task_index"])
            del running[device_index]

        ready = [
            task_index
            for task_index in pending
            if all(
                dependency in completed
                for dependency in tasks[task_index]["dependencies"]
            )
        ]
        ready.sort(
            key=lambda task_index: (
                bool(tasks[task_index]["dependencies"]),
                task_index,
            )
        )
        free_devices = [
            device_index
            for device_index in range(len(devices))
            if device_index not in running
            and device_ready_times[device_index] <= now
        ]
        future_matching_models = {}
        for device_index in range(len(devices)):
            if (
                device_index not in running
                and device_ready_times[device_index] > now
                and devices[device_index]["model"] is not None
            ):
                model = devices[device_index]["model"]
                future_matching_models[model] = (
                    future_matching_models.get(model, 0) + 1
                )
        assignments = assign_ready_tasks(
            ready,
            free_devices,
            devices,
            tasks,
            future_matching_models=future_matching_models,
        )

        if assignments:
            clock += 1
        for task_index, device_index in assignments:
            task = tasks[task_index]
            load_time = 0.0
            if devices[device_index]["model"] != task["model"]:
                load_time = cold_start_seconds(task["model"], cold_start_times)
            devices[device_index]["model"] = task["model"]
            devices[device_index]["last_used"] = clock
            running[device_index] = {
                "task_index": task_index,
                "finish_time": now + load_time + task["execution_time"],
            }
            pending.remove(task_index)

        if len(completed) == len(tasks):
            break
        next_events = [run["finish_time"] for run in running.values()]
        next_events.extend(
            ready_at
            for ready_at in device_ready_times
            if ready_at > now
        )
        if not next_events:
            raise ValueError(
                f"Dependency cycle at source_index={record.get('source_index')}"
            )
        now = min(next_events)

    return now, clock


def simulate_real_subtask_times(
    response_records,
    ordered_keys,
    device_count,
    cold_start_times,
):
    devices = [
        {"model": None, "last_used": -1}
        for _ in range(device_count)
    ]
    clock = 0
    times = {}
    for key in ordered_keys:
        times[key], clock = simulate_query_time(
            response_records[key],
            devices,
            cold_start_times,
            clock,
        )
    return times


def simulate_prefetched_subtask_times(
    response_records,
    timing_records,
    ordered_keys,
    device_count,
    agent_models,
    cold_start_times,
    time_field,
    agent_limit,
):
    devices = [
        {"model": None, "last_used": -1}
        for _ in range(device_count)
    ]
    clock = 0
    times = {}
    for key in ordered_keys:
        device_ready_times, clock = prepare_planner_prefetch(
            timing_records.get(key),
            devices,
            agent_models,
            cold_start_times,
            clock,
            time_field,
            agent_limit,
        )
        times[key], clock = simulate_query_time(
            response_records[key],
            devices,
            cold_start_times,
            clock,
            initial_device_ready_times=device_ready_times,
        )
    return times


def normalize_assignment(value):
    assignment = str(value).strip()
    if assignment.endswith(".json"):
        assignment = assignment[:-5]
    for prefix in (RESPONSE_PREFIX, SUMMARY_PREFIX):
        if assignment.startswith(prefix):
            assignment = assignment[len(prefix):]
    if not assignment:
        raise ValueError("Empty model-assignment suffix")
    return assignment


def benchmark_for_assignment(assignment, benchmark_configs):
    role_count = len(assignment.split("_"))
    matches = [
        benchmark
        for benchmark, config in benchmark_configs.items()
        if config.get("role_count") == role_count
    ]
    if len(matches) != 1:
        expected = ", ".join(
            f"{benchmark}={config.get('role_count')}"
            for benchmark, config in benchmark_configs.items()
        )
        raise ValueError(
            f"Cannot infer benchmark for assignment {assignment!r} with "
            f"{role_count} aliases; configured role counts: {expected}"
        )
    return matches[0]


def format_number(value, precision):
    return "N/A" if value is None else f"{value:.{precision}f}"


def print_table(rows, precision):
    headers = [
        "Stage",
        "Successful",
        "Excluded",
        "Average (s)",
        "Min (s)",
        "Max (s)",
        "P95 (s)",
    ]
    rendered_rows = []
    for stage, stats in rows:
        rendered_rows.append(
            [
                stage,
                str(stats["count"]),
                str(stats["missing"]),
                format_number(stats["mean"], precision),
                format_number(stats["min"], precision),
                format_number(stats["max"], precision),
                format_number(stats["p95"], precision),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered_rows))
        for index in range(len(headers))
    ]

    def render(row):
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    print(render(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in rendered_rows:
        print(render(row))


def report_run(
    benchmark,
    planner_path,
    result_dir,
    assignment,
    precision,
    cold_start_path,
    timings_path,
    device_counts,
    prefetch_time_field,
    prefetch_agent_limit,
):
    response_path = result_dir / f"{RESPONSE_PREFIX}{assignment}.json"
    summary_path = result_dir / f"{SUMMARY_PREFIX}{assignment}.json"

    missing_files = [
        path
        for path in (
            planner_path,
            response_path,
            summary_path,
            cold_start_path,
            timings_path,
        )
        if not path.exists()
    ]
    print(f"\n{benchmark.upper()} | {result_dir.name} | {assignment}")
    if missing_files:
        for path in missing_files:
            print(f"Missing: {path}")
        return

    plans = keyed_records(load_json(planner_path), planner_path)
    responses = keyed_records(load_json(response_path), response_path)
    summaries = keyed_records(load_json(summary_path), summary_path)
    cold_start_times = load_cold_start_times(cold_start_path)
    timing_records = timing_records_for_plans(plans, timings_path)
    agent_models = agent_models_from_responses(responses)
    expected_keys = set(plans)

    planner_by_query = planner_times(plans)
    subtask_by_query = {
        key: value for key, value in subtask_times(responses).items() if key in expected_keys
    }
    summary_by_query = {
        key: value for key, value in summary_times(summaries).items() if key in expected_keys
    }
    planner_by_query = {
        key: value for key, value in planner_by_query.items() if key in expected_keys
    }

    complete_keys = (
        set(planner_by_query) & set(subtask_by_query) & set(summary_by_query)
    )
    ordered_complete_keys = [key for key in plans if key in complete_keys]
    end_to_end = [
        planner_by_query[key] + subtask_by_query[key] + summary_by_query[key]
        for key in ordered_complete_keys
    ]
    expected_count = len(expected_keys)
    rows = [
        ("Planner decomposition", summarize(list(planner_by_query.values()), expected_count)),
        ("Subtask execution (sum)", summarize(list(subtask_by_query.values()), expected_count)),
        ("Summary generation", summarize(list(summary_by_query.values()), expected_count)),
        ("Computed end-to-end", summarize(end_to_end, expected_count)),
    ]
    prefetch_rows = []
    for device_count in device_counts:
        simulated_subtask_by_query = simulate_real_subtask_times(
            responses,
            ordered_complete_keys,
            device_count,
            cold_start_times,
        )
        real_end_to_end = [
            planner_by_query[key]
            + simulated_subtask_by_query[key]
            + summary_by_query[key]
            for key in ordered_complete_keys
        ]
        rows.append(
            (
                f"Real end-to-end ({device_count} devices)",
                summarize(real_end_to_end, expected_count),
            )
        )
        prefetched_subtask_by_query = simulate_prefetched_subtask_times(
            responses,
            timing_records,
            ordered_complete_keys,
            device_count,
            agent_models,
            cold_start_times,
            prefetch_time_field,
            prefetch_agent_limit,
        )
        prefetched_end_to_end = [
            planner_by_query[key]
            + prefetched_subtask_by_query[key]
            + summary_by_query[key]
            for key in ordered_complete_keys
        ]
        prefetch_rows.append(
            (
                f"Prefetch end-to-end ({device_count} devices)",
                summarize(prefetched_end_to_end, expected_count),
            )
        )

    print(f"Planner:  {planner_path}")
    print(f"Responses: {response_path}")
    print(f"Summary:   {summary_path}")
    print(f"Cold start: {cold_start_path}")
    print(f"Timings:   {timings_path}")
    print_table(rows, precision)
    print(
        f"\nPlanner-time first-{prefetch_agent_limit} agent prefetch "
        f"({prefetch_time_field})"
    )
    print_table(prefetch_rows, precision)


def main():
    assignments = [
        normalize_assignment(value) for value in CONFIG.get("assignments") or []
    ]
    assignments = list(dict.fromkeys(assignments))
    if not assignments:
        raise ValueError("CONFIG['assignments'] must contain at least one assignment")
    benchmark_configs = CONFIG.get("benchmarks") or {}
    if not benchmark_configs:
        raise ValueError("CONFIG['benchmarks'] must not be empty")
    precision = int(CONFIG.get("seconds_precision", 4))
    if precision < 0:
        raise ValueError("seconds_precision must be non-negative")
    cold_start_path = resolve_path(CONFIG["cold_start_file"])
    raw_device_counts = CONFIG.get("device_counts") or []
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in raw_device_counts
    ):
        raise ValueError("CONFIG['device_counts'] must contain positive integers")
    device_counts = list(dict.fromkeys(raw_device_counts))
    if not device_counts:
        raise ValueError("CONFIG['device_counts'] must not be empty")
    prefetch_time_field = CONFIG.get("prefetch_time_field", "decision_seconds")
    if prefetch_time_field not in {"decision_seconds", "confirmation_seconds"}:
        raise ValueError(
            "CONFIG['prefetch_time_field'] must be decision_seconds or "
            "confirmation_seconds"
        )
    prefetch_agent_limit = CONFIG.get("prefetch_agent_limit", 4)
    if (
        not isinstance(prefetch_agent_limit, int)
        or isinstance(prefetch_agent_limit, bool)
        or prefetch_agent_limit <= 0
    ):
        raise ValueError("CONFIG['prefetch_agent_limit'] must be a positive integer")

    for assignment in assignments:
        benchmark = benchmark_for_assignment(assignment, benchmark_configs)
        benchmark_config = benchmark_configs[benchmark]
        planner_path = resolve_path(benchmark_config["planner_file"])
        result_dir = resolve_path(benchmark_config["results_dir"])
        timings_path = resolve_path(benchmark_config["timings_file"])
        report_run(
            benchmark,
            planner_path,
            result_dir,
            assignment,
            precision,
            cold_start_path,
            timings_path,
            device_counts,
            prefetch_time_field,
            prefetch_agent_limit,
        )


if __name__ == "__main__":
    main()

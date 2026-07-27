import argparse
import json
from collections import Counter
from pathlib import Path


# Edit these defaults directly before running the script.
CONFIG = {
    "huskyqa_plans": "benchmarks/huskyqa/huskyqa_plans_llama3.json",
    "iirc_plans": "benchmarks/iirc/iirc_plans_llama3.json",
    "iirc_queries_per_huskyqa": 4,
    "device_counts": [2, 3],
    "summary_output": "multi_collision_summary_lru.json",
}

DATASET_MODEL_CONFIGS = {
    "huskyqa": {
        "label": "g_q_g_l",
        "agents": {
            "code_agent": "gemma3-4B",
            "math_agent": "qwen3-4B",
            "search_agent": "gemma3-4B",
            "commonsense_agent": "llama3-3B",
        },
    },
    "iirc": {
        "label": "l_g_q_q",
        "agents": {
            "code_agent": "llama3-3B",
            "math_agent": "gemma3-4B",
            "search_agent": "qwen3-4B",
            "commonsense_agent": "qwen3-4B",
        },
    },
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(output)
    return output.resolve()


def normalize_id(value):
    return str(value)


def interleave_plans(huskyqa_plans, iirc_plans, iirc_per_huskyqa):
    if iirc_per_huskyqa < 1:
        raise ValueError("iirc_queries_per_huskyqa must be at least 1")

    arrivals = []
    huskyqa_index = 0
    iirc_index = 0
    while huskyqa_index < len(huskyqa_plans) or iirc_index < len(iirc_plans):
        if huskyqa_index < len(huskyqa_plans):
            arrivals.append(
                {
                    "dataset": "huskyqa",
                    "record": huskyqa_plans[huskyqa_index],
                }
            )
            huskyqa_index += 1

        for _ in range(iirc_per_huskyqa):
            if iirc_index >= len(iirc_plans):
                break
            arrivals.append(
                {
                    "dataset": "iirc",
                    "record": iirc_plans[iirc_index],
                }
            )
            iirc_index += 1

    return arrivals


def prepare_tasks(dataset, plan):
    agent_models = DATASET_MODEL_CONFIGS[dataset]["agents"]
    raw_tasks = plan.get("plan") or []
    indices_by_id = {
        normalize_id(task.get("id")): index
        for index, task in enumerate(raw_tasks)
    }
    tasks = []
    dependency_issues = []

    for task in raw_tasks:
        agent = task.get("agent") or task.get("name") or task.get("name_1")
        if agent not in agent_models:
            raise ValueError(
                f"Unknown agent {agent!r} in {dataset} "
                f"source_index={plan.get('source_index')}"
            )

        dependencies = []
        for dependency in task.get("dep") or []:
            dependency_value = (
                dependency.get("id")
                if isinstance(dependency, dict)
                else dependency
            )
            dependency_id = normalize_id(dependency_value)
            if dependency_id not in indices_by_id:
                dependency_issues.append(
                    {
                        "dataset": dataset,
                        "source_index": plan.get("source_index"),
                        "step_id": task.get("id"),
                        "dependency": dependency,
                        "reason": "dependency does not reference a top-level plan step",
                    }
                )
                continue
            dependencies.append(indices_by_id[dependency_id])

        tasks.append(
            {
                "agent": agent,
                "model": agent_models[agent],
                "dependencies": dependencies,
            }
        )
    return tasks, dependency_issues


def select_ready_tasks(tasks, completed, device_count):
    ready_roots = []
    ready_dependents = []
    for index, task in enumerate(tasks):
        if completed & (1 << index):
            continue
        if not all(
            completed & (1 << dependency)
            for dependency in task["dependencies"]
        ):
            continue
        if task["dependencies"]:
            ready_dependents.append(index)
        else:
            ready_roots.append(index)

    selected = []
    for queue in (ready_roots, ready_dependents):
        for task_index in queue:
            if len(selected) >= device_count:
                break
            selected.append(task_index)
    return selected


def assign_lru(devices, tasks, selected, clock):
    available_devices = set(range(len(devices)))
    assignments = {}
    unresolved = []
    collision_devices = []
    cold_starts = 0

    for task_index in selected:
        model = tasks[task_index]["model"]
        matching = [
            device_index
            for device_index in available_devices
            if devices[device_index]["model"] == model
        ]
        if not matching:
            unresolved.append(task_index)
            continue
        device_index = max(
            matching,
            key=lambda index: (devices[index]["last_used"], -index),
        )
        assignments[task_index] = device_index
        available_devices.remove(device_index)

    still_unresolved = []
    for task_index in unresolved:
        empty_devices = [
            device_index
            for device_index in available_devices
            if devices[device_index]["model"] is None
        ]
        if not empty_devices:
            still_unresolved.append(task_index)
            continue
        device_index = min(empty_devices)
        assignments[task_index] = device_index
        available_devices.remove(device_index)
        cold_starts += 1

    for task_index in still_unresolved:
        device_index = min(
            available_devices,
            key=lambda index: (devices[index]["last_used"], index),
        )
        assignments[task_index] = device_index
        available_devices.remove(device_index)
        collision_devices.append(device_index)

    for task_index, device_index in assignments.items():
        devices[device_index]["model"] = tasks[task_index]["model"]
        devices[device_index]["last_used"] = clock

    return assignments, collision_devices, cold_starts


def collect_statistics(arrivals):
    dataset_queries = Counter()
    dataset_agent_calls = {}
    dataset_model_calls = {}
    dependency_issues = []

    for dataset in DATASET_MODEL_CONFIGS:
        dataset_agent_calls[dataset] = Counter()
        dataset_model_calls[dataset] = Counter()

    for arrival in arrivals:
        dataset = arrival["dataset"]
        plan = arrival["record"]
        dataset_queries[dataset] += 1
        tasks, issues = prepare_tasks(dataset, plan)
        dependency_issues.extend(issues)
        for task in tasks:
            dataset_agent_calls[dataset][task["agent"]] += 1
            dataset_model_calls[dataset][task["model"]] += 1

    combined_model_calls = Counter()
    for calls in dataset_model_calls.values():
        combined_model_calls.update(calls)

    return {
        "dataset_queries": dict(dataset_queries),
        "dataset_agent_calls": {
            dataset: dict(calls)
            for dataset, calls in dataset_agent_calls.items()
        },
        "dataset_model_calls": {
            dataset: dict(calls)
            for dataset, calls in dataset_model_calls.items()
        },
        "combined_model_calls": dict(combined_model_calls),
        "dependency_issues": dependency_issues,
    }


def simulate(arrivals, device_count):
    devices = [
        {
            "model": None,
            "last_used": -1,
            "requests": 0,
            "collisions": 0,
        }
        for _ in range(device_count)
    ]
    collisions_by_dataset = Counter()
    queries_with_collision_by_dataset = Counter()
    total_collisions = 0
    total_cold_starts = 0
    total_slots = 0
    clock = 0

    for arrival in arrivals:
        dataset = arrival["dataset"]
        plan = arrival["record"]
        tasks, _ = prepare_tasks(dataset, plan)
        completed = 0
        full_mask = (1 << len(tasks)) - 1
        query_collisions = 0

        while completed != full_mask:
            selected = select_ready_tasks(tasks, completed, device_count)
            if not selected:
                raise ValueError(
                    f"Plan has a dependency cycle in {dataset} at "
                    f"source_index={plan.get('source_index')}"
                )

            clock += 1
            total_slots += 1
            assignments, collision_devices, cold_starts = assign_lru(
                devices,
                tasks,
                selected,
                clock,
            )
            collisions = len(collision_devices)
            total_collisions += collisions
            query_collisions += collisions
            collisions_by_dataset[dataset] += collisions
            total_cold_starts += cold_starts

            for task_index, device_index in assignments.items():
                completed |= 1 << task_index
                devices[device_index]["requests"] += 1
            for device_index in collision_devices:
                devices[device_index]["collisions"] += 1

        if query_collisions:
            queries_with_collision_by_dataset[dataset] += 1

    return {
        "model_switch_events": total_collisions,
        "collisions_by_dataset": dict(collisions_by_dataset),
        "queries_with_collision_by_dataset": dict(
            queries_with_collision_by_dataset
        ),
        "cold_starts": total_cold_starts,
        "execution_slots": total_slots,
        "final_models": [device["model"] for device in devices],
        "device_summaries": [
            {
                "device": index + 1,
                "requests": device["requests"],
                "collisions": device["collisions"],
                "final_model": device["model"],
            }
            for index, device in enumerate(devices)
        ],
    }


def print_table(rows):
    headers = [
        "Devices",
        "Collisions",
        "Collision rate",
        "Avg collisions/query",
        "HuskyQA collisions",
        "IIRC collisions",
    ]
    display_rows = [
        [
            str(row["devices"]),
            str(row["model_switch_events"]),
            f"{row['collision_rate']:.2%}",
            f"{row['average_collisions_per_query']:.4f}",
            str(row["collisions_by_dataset"].get("huskyqa", 0)),
            str(row["collisions_by_dataset"].get("iirc", 0)),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in display_rows))
        for index in range(len(headers))
    ]

    def format_row(values):
        return "| " + " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ) + " |"

    print(format_row(headers))
    print(
        "|-"
        + "-|-".join("-" * width for width in widths)
        + "-|"
    )
    for row in display_rows:
        print(format_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Simulate LRU collisions for interleaved HuskyQA and IIRC queries."
    )
    parser.add_argument("--huskyqa-plans", default=CONFIG["huskyqa_plans"])
    parser.add_argument("--iirc-plans", default=CONFIG["iirc_plans"])
    parser.add_argument(
        "--iirc-per-huskyqa",
        type=int,
        default=CONFIG["iirc_queries_per_huskyqa"],
    )
    parser.add_argument("--output", default=CONFIG["summary_output"])
    args = parser.parse_args()

    huskyqa_plans = load_json(args.huskyqa_plans)
    iirc_plans = load_json(args.iirc_plans)
    arrivals = interleave_plans(
        huskyqa_plans,
        iirc_plans,
        args.iirc_per_huskyqa,
    )
    statistics = collect_statistics(arrivals)
    total_queries = len(arrivals)
    total_agent_calls = sum(
        statistics["combined_model_calls"].values()
    )
    rows = []
    device_results = {}

    for device_count in CONFIG["device_counts"]:
        result = simulate(arrivals, device_count)
        collisions = result["model_switch_events"]
        result["collision_rate"] = (
            collisions / total_agent_calls if total_agent_calls else 0.0
        )
        result["average_collisions_per_query"] = (
            collisions / total_queries if total_queries else 0.0
        )
        device_results[str(device_count)] = result
        rows.append({"devices": device_count, **result})

    summary = {
        "replacement_policy": "LRU",
        "arrival_pattern": {
            "huskyqa_queries": 1,
            "iirc_queries": args.iirc_per_huskyqa,
            "tail_policy": "continue the remaining dataset in source order",
        },
        "dataset_model_configurations": DATASET_MODEL_CONFIGS,
        "total_queries": total_queries,
        "total_agent_calls": total_agent_calls,
        **statistics,
        "device_results": device_results,
    }
    output = save_json(args.output, summary)

    print(
        f"Arrival pattern: 1 HuskyQA -> {args.iirc_per_huskyqa} IIRC"
    )
    print(f"Total queries: {total_queries}")
    print(f"Total agent calls: {total_agent_calls}")
    print(f"Combined model calls: {statistics['combined_model_calls']}")
    print(
        "Invalid dependency references: "
        f"{len(statistics['dependency_issues'])}\n"
    )
    print_table(rows)
    print(f"\nSummary JSON: {output}")


if __name__ == "__main__":
    main()

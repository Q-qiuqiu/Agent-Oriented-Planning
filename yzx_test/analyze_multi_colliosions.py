import argparse
import json
from collections import Counter
from pathlib import Path


# Edit these defaults directly before running the script.
CONFIG = {
    "huskyqa_plans": "benchmarks/huskyqa/huskyqa_plans_llama3.json",
    "iirc_plans": "benchmarks/iirc/iirc_plans_llama3.json",
    # Each round appends this many HuskyQA queries, then this many IIRC queries.
    # For example, 1:5 produces H, I, I, I, I, I, H, ...
    "huskyqa_queries_per_group": 1,
    "iirc_queries_per_group": 1,
    "iirc_partitions": 5,
    "device_counts": [1, 2, 3, 4],
    "summary_output": "multi_collision_summary_lru.json",
}

DATASET_MODEL_CONFIGS = {
    "huskyqa": {
        "label": "g_q_m_g",
        "agents": {
            "code_agent": "gemma3-4B",
            "math_agent": "qwen3-4B",
            "search_agent": "minicpm3-4B",
            "commonsense_agent": "gemma3-4B",
        },
    },
    "iirc": {
        "label": "q_q_q_q_m",
        "agents": {
            "context_agent": "qwen3-4B",
            "retrieval_agent": "qwen3-4B",
            "reasoning_agent": "qwen3-4B",
            "calculation_agent": "qwen3-4B",
            "answerability_agent": "minicpm3-4B",
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


def interleave_plans(
    huskyqa_plans,
    iirc_plans,
    huskyqa_per_group,
    iirc_per_group,
):
    if huskyqa_per_group < 1:
        raise ValueError("huskyqa_queries_per_group must be at least 1")
    if iirc_per_group < 1:
        raise ValueError("iirc_queries_per_group must be at least 1")

    arrivals = []
    huskyqa_index = 0
    iirc_index = 0
    while huskyqa_index < len(huskyqa_plans) or iirc_index < len(iirc_plans):
        for _ in range(huskyqa_per_group):
            if huskyqa_index >= len(huskyqa_plans):
                break
            arrivals.append(
                {
                    "dataset": "huskyqa",
                    "record": huskyqa_plans[huskyqa_index],
                }
            )
            huskyqa_index += 1

        for _ in range(iirc_per_group):
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


def partition_plans(plans, partition_count):
    if partition_count < 1:
        raise ValueError("iirc_partitions must be at least 1")
    if partition_count > len(plans):
        raise ValueError(
            "iirc_partitions cannot exceed the number of IIRC plans: "
            f"partitions={partition_count}, plans={len(plans)}"
        )

    base_size, larger_partition_count = divmod(len(plans), partition_count)
    partitions = []
    start = 0
    for partition_index in range(partition_count):
        size = base_size + (partition_index < larger_partition_count)
        end = start + size
        partitions.append(
            {
                "partition": partition_index + 1,
                "start_offset": start,
                "end_offset_exclusive": end,
                "plans": plans[start:end],
            }
        )
        start = end
    return partitions


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


def collect_ready_unique_agent_distribution(arrivals):
    distributions = {
        dataset: Counter()
        for dataset in DATASET_MODEL_CONFIGS
    }
    for arrival in arrivals:
        dataset = arrival["dataset"]
        tasks, _ = prepare_tasks(dataset, arrival["record"])
        unique_agents = {
            task["agent"]
            for task in tasks
            if not task["dependencies"]
        }
        distributions[dataset][len(unique_agents)] += 1
    return {
        dataset: dict(sorted(distribution.items()))
        for dataset, distribution in distributions.items()
    }


def collect_ready_agent_calls(arrivals):
    calls_by_dataset = Counter()
    for arrival in arrivals:
        dataset = arrival["dataset"]
        tasks, _ = prepare_tasks(dataset, arrival["record"])
        calls_by_dataset[dataset] += sum(
            not task["dependencies"]
            for task in tasks
        )
    return dict(calls_by_dataset)


def collect_unique_agent_roles(arrivals):
    role_counts_by_dataset = Counter()
    for arrival in arrivals:
        dataset = arrival["dataset"]
        tasks, _ = prepare_tasks(dataset, arrival["record"])
        role_counts_by_dataset[dataset] += len(
            {task["agent"] for task in tasks}
        )
    return dict(role_counts_by_dataset)


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
    first_batch_agent_calls_by_dataset = Counter()
    first_batch_unique_agent_calls_by_agent = Counter()
    dispatched_unique_agent_distribution_by_dataset = {
        dataset: Counter()
        for dataset in DATASET_MODEL_CONFIGS
    }
    total_collisions = 0
    total_first_batch_agent_calls = 0
    total_first_batch_unique_agent_calls = 0
    total_first_batch_cold_starts = 0
    total_slots = 0
    clock = 0

    for arrival in arrivals:
        dataset = arrival["dataset"]
        plan = arrival["record"]
        tasks, _ = prepare_tasks(dataset, plan)
        completed = 0
        full_mask = (1 << len(tasks)) - 1
        query_collisions = 0
        slot_index = 0

        if not tasks:
            dispatched_unique_agent_distribution_by_dataset[dataset][0] += 1

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

            # Only the independent tasks dispatched in the first scheduling
            # slot contribute to the collision metrics. Later tasks still run
            # so their model placements affect the next query.
            if slot_index == 0:
                first_batch_calls = len(assignments)
                unique_agents = {
                    tasks[task_index]["agent"]
                    for task_index in assignments
                }
                collisions = len(collision_devices)
                total_first_batch_agent_calls += first_batch_calls
                total_first_batch_unique_agent_calls += len(unique_agents)
                first_batch_unique_agent_calls_by_agent.update(unique_agents)
                dispatched_unique_agent_distribution_by_dataset[dataset][
                    len(unique_agents)
                ] += 1
                first_batch_agent_calls_by_dataset[dataset] += first_batch_calls
                total_collisions += collisions
                query_collisions += collisions
                collisions_by_dataset[dataset] += collisions
                total_first_batch_cold_starts += cold_starts

            for task_index, device_index in assignments.items():
                completed |= 1 << task_index
                devices[device_index]["requests"] += 1
            if slot_index == 0:
                for device_index in collision_devices:
                    devices[device_index]["collisions"] += 1
            slot_index += 1

        if query_collisions:
            queries_with_collision_by_dataset[dataset] += 1

    return {
        "model_switch_events": total_collisions,
        "first_batch_model_switch_events": total_collisions,
        "first_batch_agent_calls": total_first_batch_agent_calls,
        "first_batch_unique_agent_calls": (
            total_first_batch_unique_agent_calls
        ),
        "first_batch_unique_agent_calls_by_agent": dict(
            first_batch_unique_agent_calls_by_agent
        ),
        "dispatched_unique_agent_distribution_by_dataset": {
            dataset: dict(sorted(distribution.items()))
            for dataset, distribution in (
                dispatched_unique_agent_distribution_by_dataset.items()
            )
        },
        "first_batch_agent_calls_by_dataset": dict(
            first_batch_agent_calls_by_dataset
        ),
        "collisions_by_dataset": dict(collisions_by_dataset),
        "queries_with_collision_by_dataset": dict(
            queries_with_collision_by_dataset
        ),
        "first_batch_cold_starts": total_first_batch_cold_starts,
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
        "First-batch calls",
        "Collisions",
        "Collision rate",
        "Avg collisions/query",
        "HuskyQA collisions",
        "IIRC collisions",
    ]
    def format_count(value):
        if isinstance(value, int):
            return str(value)
        return f"{value:.2f}"

    display_rows = [
        [
            str(row["devices"]),
            format_count(row["first_batch_agent_calls"]),
            format_count(row["model_switch_events"]),
            f"{row['collision_rate']:.2%}",
            f"{row['average_collisions_per_query']:.4f}",
            format_count(row["collisions_by_dataset"].get("huskyqa", 0)),
            format_count(row["collisions_by_dataset"].get("iirc", 0)),
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


def print_agent_distribution_table(distributions_by_device):
    headers = [
        "Devices",
        "Dataset",
        "Ready unique agents",
        "Avg query count",
        "Dispatched unique agents",
        "Avg query count",
    ]
    rows = []
    for device_count in CONFIG["device_counts"]:
        distributions = distributions_by_device[str(device_count)]
        for dataset in DATASET_MODEL_CONFIGS:
            ready = distributions[dataset]["ready"]
            dispatched = distributions[dataset]["dispatched"]
            agent_counts = sorted(set(ready) | set(dispatched))
            for agent_count in agent_counts:
                rows.append(
                    [
                        str(device_count),
                        dataset,
                        str(agent_count),
                        f"{ready.get(agent_count, 0.0):.2f}",
                        str(agent_count),
                        f"{dispatched.get(agent_count, 0.0):.2f}",
                    ]
                )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(values):
        return "| " + " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ) + " |"

    print("\nFive-run average unique-agent distribution")
    print(format_row(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in rows:
        print(format_row(row))


def print_first_batch_call_average_table(rows):
    headers = [
        "Devices",
        "Dataset",
        "Ready calls/query",
        "Dispatched calls/query",
    ]
    display_rows = [
        [
            str(row["devices"]),
            row["dataset"],
            f"{row['ready_calls_per_query']:.4f}",
            f"{row['dispatched_calls_per_query']:.4f}",
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

    print("\nFive-run average first-batch sub-agent calls")
    print(format_row(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in display_rows:
        print(format_row(row))


def print_benchmark_agent_usage_table(rows):
    headers = [
        "Dataset",
        "Avg agent calls/query",
        "Avg unique agent roles/query",
    ]
    display_rows = [
        [
            row["dataset"],
            f"{row['agent_calls_per_query']:.4f}",
            f"{row['unique_agent_roles_per_query']:.4f}",
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

    print("\nFive-run average agent usage per query")
    print(format_row(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in display_rows:
        print(format_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Simulate LRU collisions for interleaved HuskyQA and IIRC queries."
    )
    parser.add_argument("--huskyqa-plans", default=CONFIG["huskyqa_plans"])
    parser.add_argument("--iirc-plans", default=CONFIG["iirc_plans"])
    parser.add_argument(
        "--huskyqa-per-group",
        type=int,
        default=CONFIG["huskyqa_queries_per_group"],
        help="HuskyQA queries appended in each interleaving round.",
    )
    parser.add_argument(
        "--iirc-per-group",
        type=int,
        default=CONFIG["iirc_queries_per_group"],
        help="IIRC queries appended in each interleaving round.",
    )
    parser.add_argument(
        "--iirc-partitions",
        type=int,
        default=CONFIG["iirc_partitions"],
        help="Split all IIRC queries evenly into this many independent runs.",
    )
    parser.add_argument("--output", default=CONFIG["summary_output"])
    args = parser.parse_args()

    huskyqa_plans = load_json(args.huskyqa_plans)
    all_iirc_plans = load_json(args.iirc_plans)
    if not huskyqa_plans:
        raise ValueError("The HuskyQA plan file contains no queries.")
    if not all_iirc_plans:
        raise ValueError("The IIRC plan file contains no queries.")

    iirc_partitions = partition_plans(
        all_iirc_plans,
        args.iirc_partitions,
    )
    experiments = []

    for partition in iirc_partitions:
        arrivals = interleave_plans(
            huskyqa_plans,
            partition["plans"],
            args.huskyqa_per_group,
            args.iirc_per_group,
        )
        run_statistics = collect_statistics(arrivals)
        ready_unique_agent_distribution = (
            collect_ready_unique_agent_distribution(arrivals)
        )
        ready_agent_calls_by_dataset = collect_ready_agent_calls(arrivals)
        unique_agent_roles_by_dataset = collect_unique_agent_roles(arrivals)
        total_queries = len(arrivals)
        total_agent_calls = sum(
            run_statistics["combined_model_calls"].values()
        )
        rows = []
        device_results = {}

        for device_count in CONFIG["device_counts"]:
            result = simulate(arrivals, device_count)
            collisions = result["model_switch_events"]
            result["collision_rate"] = (
                collisions / result["first_batch_agent_calls"]
                if result["first_batch_agent_calls"]
                else 0.0
            )
            result["average_collisions_per_query"] = (
                collisions / total_queries if total_queries else 0.0
            )
            device_results[str(device_count)] = result
            rows.append({"devices": device_count, **result})

        experiments.append(
            {
                "partition": partition["partition"],
                "iirc_start_offset": partition["start_offset"],
                "iirc_end_offset_exclusive": partition["end_offset_exclusive"],
                "iirc_query_count": len(partition["plans"]),
                "total_queries": total_queries,
                "total_agent_calls": total_agent_calls,
                "ready_unique_agent_distribution_by_dataset": (
                    ready_unique_agent_distribution
                ),
                "ready_agent_calls_by_dataset": ready_agent_calls_by_dataset,
                "unique_agent_roles_by_dataset": (
                    unique_agent_roles_by_dataset
                ),
                **run_statistics,
                "device_results": device_results,
                "table_rows": rows,
            }
        )

    average_rows = []
    average_device_results = {}
    for device_count in CONFIG["device_counts"]:
        results = [
            experiment["device_results"][str(device_count)]
            for experiment in experiments
        ]
        total_collisions = sum(
            result["model_switch_events"] for result in results
        )
        total_first_batch_calls = sum(
            result["first_batch_agent_calls"] for result in results
        )
        total_first_batch_unique_agent_calls = sum(
            result["first_batch_unique_agent_calls"]
            for result in results
        )
        total_run_queries = sum(
            experiment["total_queries"] for experiment in experiments
        )
        run_count = len(results)
        mean_collisions_by_dataset = {
            dataset: sum(
                result["collisions_by_dataset"].get(dataset, 0)
                for result in results
            ) / run_count
            for dataset in DATASET_MODEL_CONFIGS
        }
        average_result = {
            "runs": run_count,
            "mean_first_batch_agent_calls": (
                total_first_batch_calls / run_count
            ),
            "mean_first_batch_unique_agent_calls": (
                total_first_batch_unique_agent_calls / run_count
            ),
            "mean_model_switch_events": total_collisions / run_count,
            "mean_collision_rate": sum(
                result["collision_rate"] for result in results
            ) / run_count,
            "mean_average_collisions_per_query": sum(
                result["average_collisions_per_query"]
                for result in results
            ) / run_count,
            "mean_collisions_by_dataset": mean_collisions_by_dataset,
            "pooled_collision_rate": (
                total_collisions / total_first_batch_calls
                if total_first_batch_calls
                else 0.0
            ),
            "pooled_average_collisions_per_query": (
                total_collisions / total_run_queries
                if total_run_queries
                else 0.0
            ),
        }
        average_device_results[str(device_count)] = average_result
        average_rows.append(
            {
                "devices": device_count,
                "first_batch_agent_calls": (
                    average_result["mean_first_batch_agent_calls"]
                ),
                "model_switch_events": (
                    average_result["mean_model_switch_events"]
                ),
                "collision_rate": average_result["mean_collision_rate"],
                "average_collisions_per_query": (
                    average_result["mean_average_collisions_per_query"]
                ),
                "collisions_by_dataset": mean_collisions_by_dataset,
            }
        )

    average_agent_distributions = {}
    for device_count in CONFIG["device_counts"]:
        device_distributions = {}
        for dataset in DATASET_MODEL_CONFIGS:
            ready_agent_counts = set()
            dispatched_agent_counts = set()
            for experiment in experiments:
                ready_agent_counts.update(
                    experiment[
                        "ready_unique_agent_distribution_by_dataset"
                    ][dataset]
                )
                dispatched_agent_counts.update(
                    experiment["device_results"][str(device_count)][
                        "dispatched_unique_agent_distribution_by_dataset"
                    ][dataset]
                )

            ready_distribution = {
                agent_count: sum(
                    experiment[
                        "ready_unique_agent_distribution_by_dataset"
                    ][dataset].get(agent_count, 0)
                    for experiment in experiments
                ) / len(experiments)
                for agent_count in sorted(ready_agent_counts)
            }
            dispatched_distribution = {
                agent_count: sum(
                    experiment["device_results"][str(device_count)][
                        "dispatched_unique_agent_distribution_by_dataset"
                    ][dataset].get(agent_count, 0)
                    for experiment in experiments
                ) / len(experiments)
                for agent_count in sorted(dispatched_agent_counts)
            }
            device_distributions[dataset] = {
                "ready": ready_distribution,
                "dispatched": dispatched_distribution,
            }
        average_agent_distributions[str(device_count)] = device_distributions

    average_first_batch_call_rows = []
    average_first_batch_calls = {}
    for device_count in CONFIG["device_counts"]:
        device_averages = {}
        for dataset in DATASET_MODEL_CONFIGS:
            ready_rates = []
            dispatched_rates = []
            for experiment in experiments:
                query_count = experiment["dataset_queries"].get(dataset, 0)
                if not query_count:
                    continue
                ready_rates.append(
                    experiment["ready_agent_calls_by_dataset"].get(dataset, 0)
                    / query_count
                )
                dispatched_rates.append(
                    experiment["device_results"][str(device_count)][
                        "first_batch_agent_calls_by_dataset"
                    ].get(dataset, 0)
                    / query_count
                )
            ready_average = (
                sum(ready_rates) / len(ready_rates) if ready_rates else 0.0
            )
            dispatched_average = (
                sum(dispatched_rates) / len(dispatched_rates)
                if dispatched_rates
                else 0.0
            )
            device_averages[dataset] = {
                "ready_calls_per_query": ready_average,
                "dispatched_calls_per_query": dispatched_average,
            }
            average_first_batch_call_rows.append(
                {
                    "devices": device_count,
                    "dataset": dataset,
                    "ready_calls_per_query": ready_average,
                    "dispatched_calls_per_query": dispatched_average,
                }
            )
        average_first_batch_calls[str(device_count)] = device_averages

    average_benchmark_agent_usage = {}
    benchmark_agent_usage_rows = []
    for dataset in DATASET_MODEL_CONFIGS:
        call_rates = []
        unique_role_rates = []
        for experiment in experiments:
            query_count = experiment["dataset_queries"].get(dataset, 0)
            if not query_count:
                continue
            total_calls = sum(
                experiment["dataset_agent_calls"][dataset].values()
            )
            call_rates.append(total_calls / query_count)
            unique_role_rates.append(
                experiment["unique_agent_roles_by_dataset"].get(dataset, 0)
                / query_count
            )
        usage = {
            "agent_calls_per_query": (
                sum(call_rates) / len(call_rates) if call_rates else 0.0
            ),
            "unique_agent_roles_per_query": (
                sum(unique_role_rates) / len(unique_role_rates)
                if unique_role_rates
                else 0.0
            ),
        }
        average_benchmark_agent_usage[dataset] = usage
        benchmark_agent_usage_rows.append(
            {"dataset": dataset, **usage}
        )

    summary = {
        "replacement_policy": "LRU",
        "collision_definition": {
            "counted_scope": (
                "Only model replacements caused by independent tasks actually "
                "dispatched in each query's first scheduling slot."
            ),
            "excluded_scope": (
                "Queued independent tasks and all later dependent tasks. They "
                "still execute and update model residency and LRU state."
            ),
            "initial_empty_device_load": "cold start, not a collision",
            "collision_rate": (
                "first-batch model switch events / first-batch agent calls"
            ),
            "average_collisions_per_query": (
                "first-batch model switch events / total queries"
            ),
            "unique_agent_calls": (
                "For each query's dispatched first batch, repeated calls to the "
                "same agent type count once; these per-query counts are summed."
            ),
        },
        "dataset_selection": {
            "huskyqa_available_queries": len(huskyqa_plans),
            "iirc_available_queries": len(all_iirc_plans),
            "huskyqa_queries_per_run": len(huskyqa_plans),
            "iirc_partition_count": len(iirc_partitions),
            "iirc_partition_sizes": [
                len(partition["plans"])
                for partition in iirc_partitions
            ],
            "iirc_selection": (
                "All IIRC queries are split into contiguous, balanced partitions."
            ),
        },
        "arrival_pattern": {
            "huskyqa_queries_per_group": args.huskyqa_per_group,
            "iirc_queries_per_group": args.iirc_per_group,
            "tail_policy": "continue the remaining dataset in source order",
        },
        "dataset_model_configurations": DATASET_MODEL_CONFIGS,
        "experiments": [
            {
                key: value
                for key, value in experiment.items()
                if key != "table_rows"
            }
            for experiment in experiments
        ],
        "five_run_average": average_device_results,
        "five_run_average_agent_distributions": (
            average_agent_distributions
        ),
        "five_run_average_first_batch_calls": (
            average_first_batch_calls
        ),
        "five_run_average_benchmark_agent_usage": (
            average_benchmark_agent_usage
        ),
    }
    # JSON output is temporarily disabled.
    # output = save_json(args.output, summary)

    print(
        "Arrival pattern: "
        f"{args.huskyqa_per_group} HuskyQA -> "
        f"{args.iirc_per_group} IIRC"
    )
    print(
        f"IIRC partitions: {[len(partition['plans']) for partition in iirc_partitions]}"
    )

    # Per-partition collision tables are temporarily hidden. All five
    # experiments still contribute to the arithmetic mean below.

    print("\nFive-run arithmetic mean")
    print_table(average_rows)
    print(
        "\nFirst-batch call rules: Ready includes all initially independent "
        "subtasks; Dispatched includes only subtasks assigned in the first "
        "slot. Repeated calls to the same agent count separately."
    )
    print_first_batch_call_average_table(
        average_first_batch_call_rows
    )
    print_benchmark_agent_usage_table(benchmark_agent_usage_rows)
    print(
        "\nUnique-agent distribution rules: Ready includes all initially "
        "independent tasks; Dispatched includes only the first tasks assigned "
        "to devices. Repeated agent types count once per query."
    )
    print_agent_distribution_table(average_agent_distributions)
    # print(f"\nSummary JSON: {output}")


if __name__ == "__main__":
    main()

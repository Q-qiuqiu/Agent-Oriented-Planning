import argparse
import json
from collections import Counter
from pathlib import Path


# Edit these defaults directly before running the script.
CONFIG = {
    "plans": "benchmarks/huskyqa/huskyqa_plans_llama3.json",
    "device_counts": [2, 3],
    "summary_output": "huskyqa_test/results/device_collision_summary_lru.json",
}

MODEL_CONFIGS = {
    "4 agents / 3 models": {
        "code_agent": "gemma3-4B",
        "math_agent": "qwen3-4B",
        "search_agent": "gemma3-4B",
        "commonsense_agent": "llama3-3B",
    },
    "4 agents / 4 independent models": {
        "code_agent": "code-model",
        "math_agent": "math-model",
        "search_agent": "search-model",
        "commonsense_agent": "commonsense-model",
    },
}


def normalize_id(value):
    return str(value)


def configured_agents():
    agent_sets = [set(config) for config in MODEL_CONFIGS.values()]
    if not agent_sets or any(agents != agent_sets[0] for agents in agent_sets[1:]):
        raise ValueError("Every MODEL_CONFIGS entry must contain the same agents")
    return agent_sets[0]


def load_plans(path):
    with Path(path).open("r", encoding="utf-8") as file:
        plans = json.load(file)

    valid_agents = configured_agents()
    agent_counts = Counter()
    for plan in plans:
        for step in plan.get("plan") or []:
            agent = step.get("agent") or step.get("name") or step.get("name_1")
            if agent not in valid_agents:
                raise ValueError(
                    f"Unknown agent {agent!r} in source_index={plan.get('source_index')}"
                )
            agent_counts[agent] += 1
    return plans, agent_counts


def prepare_tasks(plan, agent_models):
    raw_tasks = plan.get("plan") or []
    indices_by_id = {
        normalize_id(task.get("id")): index
        for index, task in enumerate(raw_tasks)
    }
    tasks = []

    for task in raw_tasks:
        agent = task.get("agent") or task.get("name") or task.get("name_1")
        dependencies = []
        for dependency in task.get("dep") or []:
            dependency_id = normalize_id(dependency)
            if dependency_id not in indices_by_id:
                raise ValueError(
                    f"Missing dependency {dependency!r} in "
                    f"source_index={plan.get('source_index')}"
                )
            dependencies.append(indices_by_id[dependency_id])
        tasks.append(
            {
                "agent": agent,
                "model": agent_models[agent],
                "dependencies": dependencies,
            }
        )
    return tasks


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

    # Reuse matching model instances before loading or replacing anything.
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

    # An initial load onto an empty device is not a model-switch collision.
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

    # Replace the least recently used instance for every unresolved request.
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


def simulate(plans, device_count, agent_models):
    devices = [
        {
            "model": None,
            "last_used": -1,
            "requests": 0,
            "collisions": 0,
        }
        for _ in range(device_count)
    ]
    total_collisions = 0
    total_cold_starts = 0
    total_slots = 0
    clock = 0

    for plan in plans:
        tasks = prepare_tasks(plan, agent_models)
        completed = 0
        full_mask = (1 << len(tasks)) - 1

        while completed != full_mask:
            selected = select_ready_tasks(tasks, completed, device_count)
            if not selected:
                raise ValueError(
                    f"Plan has a dependency cycle at "
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
            total_collisions += len(collision_devices)
            total_cold_starts += cold_starts

            for task_index, device_index in assignments.items():
                completed |= 1 << task_index
                devices[device_index]["requests"] += 1
            for device_index in collision_devices:
                devices[device_index]["collisions"] += 1

    return {
        "model_switch_events": total_collisions,
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


def model_call_counts(agent_counts, agent_models):
    counts = Counter()
    for agent, count in agent_counts.items():
        counts[agent_models[agent]] += count
    return dict(counts)


def print_markdown_table(rows):
    headers = [
        "Model configuration",
        "Devices",
        "Collisions",
        "Collision rate",
        "Avg collisions/query",
    ]
    display_rows = [
        [
            row["model_configuration"],
            str(row["devices"]),
            str(row["model_switch_events"]),
            f"{row['collision_rate']:.2%}",
            f"{row['average_collisions_per_query']:.4f}",
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
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in display_rows:
        print(format_row(row))


def save_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)
    return output_path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate dependency-aware online LRU model replacement."
    )
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--output", default=CONFIG["summary_output"])
    args = parser.parse_args()

    plans, agent_counts = load_plans(args.plans)
    total_queries = len(plans)
    total_agent_calls = sum(agent_counts.values())
    rows = []
    configurations = {}

    for configuration_name, agent_models in MODEL_CONFIGS.items():
        device_results = {}
        for device_count in CONFIG["device_counts"]:
            result = simulate(plans, device_count, agent_models)
            collisions = result["model_switch_events"]
            collision_rate = (
                collisions / total_agent_calls if total_agent_calls else 0.0
            )
            average_collisions = (
                collisions / total_queries if total_queries else 0.0
            )
            result.update(
                {
                    "collision_rate": collision_rate,
                    "average_collisions_per_query": average_collisions,
                }
            )
            device_results[str(device_count)] = result
            rows.append(
                {
                    "model_configuration": configuration_name,
                    "devices": device_count,
                    **result,
                }
            )

        configurations[configuration_name] = {
            "agent_models": agent_models,
            "model_calls": model_call_counts(agent_counts, agent_models),
            "device_results": device_results,
        }

    summary = {
        "replacement_policy": "LRU",
        "plans": args.plans,
        "total_queries": total_queries,
        "total_agent_calls": total_agent_calls,
        "agent_calls": dict(agent_counts),
        "model_configurations": configurations,
    }
    output_path = save_json(args.output, summary)

    print(f"Total queries: {total_queries}")
    print(f"Total agent calls: {total_agent_calls}\n")
    print_markdown_table(rows)
    print(f"\nSummary JSON: {output_path}")


if __name__ == "__main__":
    main()

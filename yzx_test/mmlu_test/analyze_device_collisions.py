import argparse
import json
from collections import Counter
from pathlib import Path


# Edit these defaults directly before running the script.
CONFIG = {
    "plans": "benchmarks/mmlu_pro/mmlu_pro_plans_llada.json",
    "device_counts": [1, 2, 3],
    "summary_output": "mmlu_test/results/device_collision_summary_lru.json",
}

MODEL_CONFIGS = {
    "3 agents / 2 models": {
        "knowledge_agent": "knowledge-model",
        "reasoning_agent": "reasoning-model",
        "elimination_agent": "reasoning-model",
    },
    "3 agents / 3 independent models": {
        "knowledge_agent": "knowledge-model",
        "reasoning_agent": "reasoning-model",
        "elimination_agent": "elimination-model",
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
    dependency_issues = []

    for task in raw_tasks:
        agent = task.get("agent") or task.get("name") or task.get("name_1")
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
                        "source_index": plan.get("source_index"),
                        "query": plan.get("query"),
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


def build_query_statistics(plans):
    per_query = []
    call_distribution = Counter()
    dependency_issues = []
    validation_models = next(iter(MODEL_CONFIGS.values()))

    for plan in plans:
        raw_tasks = plan.get("plan") or []
        agent_calls = Counter(
            task.get("agent") or task.get("name") or task.get("name_1")
            for task in raw_tasks
        )
        total_calls = len(raw_tasks)
        call_distribution[total_calls] += 1
        _, issues = prepare_tasks(plan, validation_models)
        dependency_issues.extend(issues)
        per_query.append(
            {
                "source_index": plan.get("source_index"),
                "query": plan.get("query"),
                "total_agent_calls": total_calls,
                "agent_calls": {
                    agent: agent_calls.get(agent, 0)
                    for agent in sorted(configured_agents())
                },
                "planner_error": plan.get("error"),
                "invalid_dependency_count": len(issues),
            }
        )

    total_calls = sum(row["total_agent_calls"] for row in per_query)
    query_count = len(per_query)
    return {
        "per_query": per_query,
        "call_count_distribution": {
            str(call_count): query_count
            for call_count, query_count in sorted(call_distribution.items())
        },
        "average_agent_calls_per_query": (
            total_calls / query_count if query_count else 0.0
        ),
        "queries_with_empty_plan": sum(
            row["total_agent_calls"] == 0 for row in per_query
        ),
        "queries_with_planner_error": sum(
            bool(row["planner_error"]) for row in per_query
        ),
        "dependency_issues": dependency_issues,
    }


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
    total_first_batch_agent_calls = 0
    total_first_batch_cold_starts = 0
    total_slots = 0
    clock = 0

    for plan in plans:
        tasks, _ = prepare_tasks(plan, agent_models)
        completed = 0
        full_mask = (1 << len(tasks)) - 1
        slot_index = 0

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

            # Count only independent tasks actually dispatched in this query's
            # first slot. Later tasks still update model residency and LRU state.
            if slot_index == 0:
                total_first_batch_agent_calls += len(assignments)
                total_collisions += len(collision_devices)
                total_first_batch_cold_starts += cold_starts

            for task_index, device_index in assignments.items():
                completed |= 1 << task_index
                devices[device_index]["requests"] += 1
            if slot_index == 0:
                for device_index in collision_devices:
                    devices[device_index]["collisions"] += 1
            slot_index += 1

    return {
        "model_switch_events": total_collisions,
        "first_batch_model_switch_events": total_collisions,
        "first_batch_agent_calls": total_first_batch_agent_calls,
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


def model_call_counts(agent_counts, agent_models):
    counts = Counter()
    for agent, count in agent_counts.items():
        counts[agent_models[agent]] += count
    return dict(counts)


def print_markdown_table(rows):
    headers = [
        "Model configuration",
        "Devices",
        "First-batch calls",
        "Collisions",
        "Collision rate",
        "Avg collisions/query",
    ]
    display_rows = [
        [
            row["model_configuration"],
            str(row["devices"]),
            str(row["first_batch_agent_calls"]),
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


def print_query_distribution(distribution):
    print("| Sub-agent calls/query | Query count |")
    print("|----------------------:|------------:|")
    for call_count, query_count in distribution.items():
        print(f"| {call_count} | {query_count} |")


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
    query_statistics = build_query_statistics(plans)
    rows = []
    configurations = {}

    for configuration_name, agent_models in MODEL_CONFIGS.items():
        device_results = {}
        for device_count in CONFIG["device_counts"]:
            result = simulate(plans, device_count, agent_models)
            collisions = result["model_switch_events"]
            collision_rate = (
                collisions / result["first_batch_agent_calls"]
                if result["first_batch_agent_calls"]
                else 0.0
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
        },
        "plans": args.plans,
        "total_queries": total_queries,
        "total_agent_calls": total_agent_calls,
        "agent_calls": dict(agent_counts),
        "query_call_statistics": query_statistics,
        "model_configurations": configurations,
    }
    output_path = save_json(args.output, summary)

    print(f"Total queries: {total_queries}")
    print(f"Total agent calls: {total_agent_calls}")
    print(
        "Average agent calls/query: "
        f"{query_statistics['average_agent_calls_per_query']:.4f}"
    )
    print(
        "Invalid dependency references: "
        f"{len(query_statistics['dependency_issues'])}\n"
    )
    print_query_distribution(query_statistics["call_count_distribution"])
    print()
    print_markdown_table(rows)
    print(f"\nSummary JSON: {output_path}")


if __name__ == "__main__":
    main()

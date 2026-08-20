import argparse
import json
import math
import statistics
from pathlib import Path


# Change only these values for a result family.
MODEL_SIZE = "1b"
PLAN_VARIANT = "llada"
ASSIGNMENTS = ["q_qm_m"]
CONFIG = {
    "plans": f"benchmarks/mmlu_pro/mmlu_pro_plans_{PLAN_VARIANT}.json",
    "results_dir": f"mmlu_test/results_{MODEL_SIZE}_{PLAN_VARIANT}",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


def fmt(value):
    return "-" if value is None else f"{value:.4f}s"


def report(assignment, plans, responses, summaries):
    plans_by_id = {str(row.get("source_index")): row for row in plans}
    responses_by_id = {str(row.get("source_index")): row for row in responses}
    summaries_by_id = {str(row.get("source_index")): row for row in summaries}

    planner_times = [
        row["time"] for row in plans
        if row.get("error") is None and isinstance(row.get("time"), (int, float))
    ]
    subtask_times = [
        row["subtask_wall_time"] for row in responses
        if row.get("error") is None
        and isinstance(row.get("subtask_wall_time"), (int, float))
    ]
    summary_times = [
        row["summary_time"] for row in summaries
        if row.get("summary_error") is None
        and isinstance(row.get("summary_time"), (int, float))
    ]
    end_to_end = []
    for key in plans_by_id.keys() & responses_by_id.keys() & summaries_by_id.keys():
        plan = plans_by_id[key]
        response = responses_by_id[key]
        summary = summaries_by_id[key]
        if plan.get("error") or response.get("error") or summary.get("summary_error"):
            continue
        values = [plan.get("time"), response.get("subtask_wall_time"), summary.get("summary_time")]
        if all(isinstance(value, (int, float)) for value in values):
            end_to_end.append(sum(values))

    rows = [
        ("planner", stats(planner_times)),
        ("parallel subtasks", stats(subtask_times)),
        ("summary", stats(summary_times)),
        ("end-to-end", stats(end_to_end)),
    ]
    print(f"\nAssignment: {assignment}")
    print("| Stage             | Success | Mean       | Min        | Max        | P95        |")
    print("|-------------------|--------:|-----------:|-----------:|-----------:|-----------:|")
    for name, values in rows:
        print(
            f"| {name:<17} | {values['count']:>7} | {fmt(values['mean']):>10} "
            f"| {fmt(values['min']):>10} | {fmt(values['max']):>10} "
            f"| {fmt(values['p95']):>10} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Print MMLU-Pro timing reports.")
    parser.add_argument("--assignments", nargs="+", default=ASSIGNMENTS)
    args = parser.parse_args()
    plans = load_json(CONFIG["plans"])
    for assignment in args.assignments:
        base = Path(CONFIG["results_dir"])
        responses = load_json(base / f"subtask_hetro_responses_{assignment}.json")
        summaries = load_json(base / f"summary_result_{assignment}.json")
        report(assignment, plans, responses, summaries)


if __name__ == "__main__":
    main()

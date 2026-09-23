#!/usr/bin/env python3
"""Fail fast on fixed-token corruption or unfinished fixed regions."""

import argparse
import json
import statistics
from pathlib import Path


def records(path):
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-per-method", type=int, default=10)
    args = parser.parse_args()
    failures = []
    summary = {}
    methods = ("dual_vanilla", "fixed_canvas_vanilla", "fixed_canvas_plan_first")
    per_benchmark = {}
    for method in methods:
        rows = []
        for bench in ("huskyqa", "mmlu"):
            rows.extend(records(args.root / bench / method / "timings.jsonl"))
        fixed = [row.get("fixed_canvas") for row in rows if row.get("fixed_canvas")]
        summary[method] = {
            "records": len(rows),
            "fixed_records": len(fixed),
            "parseable": sum(bool(item.get("final_plan_parse_success")) for item in fixed),
            "unresolved": sum(int(item.get("unresolved_mask_count") or 0) for item in fixed),
            "fixed_corruption": sum(int(item.get("fixed_token_corruption_count") or 0) for item in fixed),
            "reasoning_nonempty": sum(bool(item.get("reasoning_nonempty")) for item in fixed),
            "plan_json_complete": sum(bool(item.get("plan_json_complete")) for item in fixed),
            "plan_capacity_overflow": sum(bool(item.get("plan_capacity_overflow")) for item in fixed),
            "plan_effective_tokens": [
                item.get("plan_effective_tokens") for item in fixed
                if item.get("plan_effective_tokens") is not None
            ],
            "unused_plan_capacity": [
                item.get("unused_plan_capacity") for item in fixed
                if item.get("unused_plan_capacity") is not None
            ],
        }
        if len(rows) < args.expected_per_method:
            failures.append(f"{method}: expected {args.expected_per_method} records, got {len(rows)}")
        if fixed:
            if summary[method]["unresolved"]:
                failures.append(f"{method}: unresolved fixed-region masks")
            if summary[method]["fixed_corruption"]:
                failures.append(f"{method}: structural delimiter corruption")
            if summary[method]["parseable"] == 0:
                failures.append(f"{method}: no fixed PLAN was parseable")
            if summary[method]["reasoning_nonempty"] == 0:
                failures.append(f"{method}: every reasoning region was empty")
            if summary[method]["plan_json_complete"] == 0:
                failures.append(f"{method}: no PLAN triggered schema-valid early stop")

    # Validate the benchmark parser, not merely the permissive online JSON
    # observer. Also catch the fixed-region failure mode where the model fills
    # the entire PLAN budget by appending repeated subtasks.
    for bench in ("huskyqa", "mmlu"):
        per_benchmark[bench] = {}
        for method in methods:
            path = args.root / bench / method / "plans.json"
            plans = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            valid = [row for row in plans if row.get("error") is None and row.get("plan")]
            counts = [len(row["plan"]) for row in valid]
            per_benchmark[bench][method] = {
                "total": len(plans),
                "parse_success": len(valid) / len(plans) if plans else 0.0,
                "agent_count_median": statistics.median(counts) if counts else None,
                "agent_counts": counts,
                "plans_by_index": {
                    str(row.get("source_index")): row.get("plan")
                    for row in valid
                },
            }
        base = per_benchmark[bench]["dual_vanilla"]
        for method in methods[1:]:
            item = per_benchmark[bench][method]
            if item["parse_success"] < base["parse_success"] - 0.05:
                failures.append(
                    f"{bench}/{method}: benchmark parse success "
                    f"{item['parse_success']:.1%} is >5pp below baseline "
                    f"{base['parse_success']:.1%}"
                )
            if (
                item["agent_count_median"] is not None
                and base["agent_count_median"] is not None
                and item["agent_count_median"] > base["agent_count_median"] * 1.25
            ):
                failures.append(
                    f"{bench}/{method}: median Agent count inflated from "
                    f"{base['agent_count_median']} to {item['agent_count_median']}"
                )
            base_plans = base["plans_by_index"]
            method_plans = item["plans_by_index"]
            shared = set(base_plans) & set(method_plans)
            if shared:
                same = sum(
                    [step.get("agent") for step in method_plans[index][:3]]
                    == [step.get("agent") for step in base_plans[index][:3]]
                    for index in shared
                ) / len(shared)
                item["first3_same_rate"] = same
                if same < 0.90:
                    failures.append(
                        f"{bench}/{method}: First-3 Same {same:.1%} is below 90%"
                    )
        for item in per_benchmark[bench].values():
            item.pop("plans_by_index", None)
    summary["benchmark_parser"] = per_benchmark
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("SANITY FAILED:\n- " + "\n- ".join(failures))
    print("SANITY PASSED")


if __name__ == "__main__":
    main()

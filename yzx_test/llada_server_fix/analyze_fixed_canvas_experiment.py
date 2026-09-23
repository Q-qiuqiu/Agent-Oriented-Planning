#!/usr/bin/env python3
"""Summarize fixed structural canvas experiments without model inference."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METHODS = (
    "dual_vanilla",
    "fixed_canvas_vanilla",
    "fixed_canvas_plan_first",
)
LABELS = {
    "dual_vanilla": "Dual Vanilla",
    "fixed_canvas_vanilla": "Fixed Canvas Vanilla",
    "fixed_canvas_plan_first": "Fixed Canvas Plan-First",
}


def percentile(values, q):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    rank = (len(values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - rank) + values[upper] * (rank - lower)


def fmt(value, percent=False):
    if value is None:
        return "—"
    return f"{100 * value:.1f}%" if percent else f"{value:.3f}"


def load_json(path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row):
    return str(row.get("source_index"))


def plan_agents(plan):
    return tuple(
        str(item.get("agent"))
        for item in (plan or [])
        if isinstance(item, dict) and item.get("agent")
    )


def valid_structure(plan):
    return bool(plan) and all(
        isinstance(item, dict)
        and all(field in item for field in ("agent", "id", "task", "reason", "dep"))
        and isinstance(item.get("dep"), list)
        for item in plan
    )


def combine(root, benchmarks):
    data = {bench: {} for bench in benchmarks}
    for bench in benchmarks:
        for method in METHODS:
            folder = root / bench / method
            plans = load_json(folder / "plans.json")
            timings = load_jsonl(folder / "timings.jsonl")
            timing_by_query = {str(row.get("query")): row for row in timings}
            rows = {}
            for plan_row in plans:
                timing = timing_by_query.get(str(plan_row.get("query")))
                plan = plan_row.get("plan")
                agents = plan_agents(plan)
                fixed = (timing or {}).get("fixed_canvas") or {}
                if method == "dual_vanilla":
                    slots = (timing or {}).get("agents") or []
                    natural = [slot.get("materialized_seconds") for slot in slots[: min(3, len(agents))]]
                    first3 = (timing or {}).get("first3_agent_seconds")
                    if first3 is None:
                        first3 = max(natural) if natural and all(value is not None for value in natural) else None
                    first = (timing or {}).get("first_agent_seconds")
                    if first is None:
                        first = natural[0] if natural else None
                    parseable = bool(plan) and plan_row.get("error") is None
                    plan_parseable = (timing or {}).get("plan_parseable_seconds") or (timing or {}).get("plan_complete_seconds")
                else:
                    first3 = fixed.get("first3_agent_seconds")
                    first = fixed.get("first_agent_seconds")
                    parseable = bool(plan) and plan_row.get("error") is None and bool(fixed.get("final_plan_parse_success"))
                    plan_parseable = fixed.get("plan_parseable_seconds")
                rows[key(plan_row)] = {
                    "source_index": plan_row.get("source_index"),
                    "query": plan_row.get("query"),
                    "plan": plan,
                    "agents": agents,
                    "parse": parseable,
                    "structure": valid_structure(plan),
                    "first": first,
                    "first3": first3,
                    "plan_parseable": plan_parseable,
                    "output_tokens": (timing or {}).get("returned_tokens"),
                    "nfe": (timing or {}).get("nfe"),
                    "generation_seconds": (timing or {}).get("generation_seconds"),
                    "raw_hash": (timing or {}).get("raw_output_sha256"),
                    "plan_effective_tokens": (
                        fixed.get("plan_effective_tokens")
                        if fixed else (timing or {}).get("plan_effective_tokens")
                    ),
                    "plan_capacity_overflow": bool(
                        fixed.get("plan_capacity_overflow")
                        if fixed else (timing or {}).get("plan_capacity_overflow")
                    ),
                    "fixed": fixed,
                }
            data[bench][method] = rows
    return data


def subset_rows(data, benchmark, method):
    if benchmark == "combined":
        return [row for bench in data for row in data[bench][method].values()]
    return list(data[benchmark][method].values())


def method_summary(data, benchmark, method):
    rows = subset_rows(data, benchmark, method)
    baseline = {}
    benches = list(data) if benchmark == "combined" else [benchmark]
    for bench in benches:
        for item_key, row in data[bench]["dual_vanilla"].items():
            baseline[(bench, item_key)] = row
    matched = []
    same3 = []
    samefull = []
    for bench in benches:
        for item_key, row in data[bench][method].items():
            base = baseline.get((bench, item_key))
            if base and row["parse"] and base["parse"]:
                same3.append(row["agents"][:3] == base["agents"][:3])
                samefull.append(row["agents"] == base["agents"])
                if row["first3"] is not None and base["first3"] is not None:
                    matched.append(float(base["first3"]) - float(row["first3"]))
    times = [row["first3"] for row in rows if row["first3"] is not None]
    first = [row["first"] for row in rows if row["first"] is not None]
    return {
        "n": len(rows),
        "parse": sum(row["parse"] for row in rows) / len(rows) if rows else None,
        "structure": sum(row["structure"] for row in rows) / len(rows) if rows else None,
        "first3_same": sum(same3) / len(same3) if same3 else None,
        "full_same": sum(samefull) / len(samefull) if samefull else None,
        "mean_tokens": statistics.mean(row["output_tokens"] for row in rows if row["output_tokens"] is not None) if any(row["output_tokens"] is not None for row in rows) else None,
        "first_p50": percentile(first, .5),
        "first3_mean": statistics.mean(times) if times else None,
        "first3_p50": percentile(times, .5),
        "first3_p95": percentile(times, .95),
        "parseable_p50": percentile([row["plan_parseable"] for row in rows], .5),
        "matched": len(matched),
        "lead_mean": statistics.mean(matched) if matched else None,
        "lead_p50": percentile(matched, .5),
        "lead_2": sum(value >= 2 for value in matched) / len(matched) if matched else None,
        "lead_5": sum(value >= 5 for value in matched) / len(matched) if matched else None,
        "lead_10": sum(value >= 10 for value in matched) / len(matched) if matched else None,
        "mean_nfe": statistics.mean(row["nfe"] for row in rows if row["nfe"] is not None) if any(row["nfe"] is not None for row in rows) else None,
        "mean_generation": statistics.mean(row["generation_seconds"] for row in rows if row["generation_seconds"] is not None) if any(row["generation_seconds"] is not None for row in rows) else None,
        "unresolved_rate": sum(bool(row["fixed"].get("unresolved_mask_count")) for row in rows) / len(rows) if rows and method != "dual_vanilla" else None,
        "corruption": sum(int(row["fixed"].get("fixed_token_corruption_count") or 0) for row in rows) if method != "dual_vanilla" else None,
        "reasoning_nonempty": sum(bool(row["fixed"].get("reasoning_nonempty")) for row in rows) / len(rows) if rows and method != "dual_vanilla" else None,
        "reasoning_leak": sum(bool(row["fixed"].get("reasoning_json_leak")) for row in rows) / len(rows) if rows and method != "dual_vanilla" else None,
        "plan_effective_p50": percentile(
            [row["plan_effective_tokens"] for row in rows], .5
        ),
        "plan_overflow": (
            sum(row["plan_capacity_overflow"] for row in rows) / len(rows)
            if rows else None
        ),
        "reasoning_effective_p50": percentile(
            [row["fixed"].get("reasoning_effective_tokens") for row in rows], .5
        ) if method != "dual_vanilla" else None,
        "mean_task_count": (
            statistics.mean(len(row["plan"]) for row in rows if row["parse"])
            if any(row["parse"] for row in rows) else None
        ),
        "reasoning_end_success": (
            sum(bool(row["fixed"].get("reasoning_end_natural_success")) for row in rows) / len(rows)
            if rows and method != "dual_vanilla" else None
        ),
        "plan_end_success": (
            sum(bool(row["fixed"].get("plan_end_natural_success")) for row in rows) / len(rows)
            if rows and method != "dual_vanilla" else None
        ),
        "reasoning_exhausted": (
            sum(bool(row["fixed"].get("reasoning_capacity_exhausted")) for row in rows) / len(rows)
            if rows and method != "dual_vanilla" else None
        ),
        "plan_exhausted": (
            sum(bool(row["fixed"].get("plan_capacity_exhausted")) for row in rows) / len(rows)
            if rows and method != "dual_vanilla" else None
        ),
        "reasoning_tail_removed_p50": percentile([
            (row["fixed"].get("layout") or {}).get("reasoning_budget", 0)
            - ((row["fixed"].get("layout") or {}).get("reasoning_end", 0)
               - (row["fixed"].get("layout") or {}).get("reasoning_start", 0))
            for row in rows
            if (row["fixed"].get("layout") or {}).get("reasoning_end") is not None
        ], .5) if method != "dual_vanilla" else None,
        "plan_tail_removed_p50": percentile([
            (row["fixed"].get("layout") or {}).get("plan_budget", 0)
            - ((row["fixed"].get("layout") or {}).get("plan_end", 0)
               - (row["fixed"].get("layout") or {}).get("plan_start", 0))
            for row in rows
            if (row["fixed"].get("layout") or {}).get("plan_end") is not None
        ], .5) if method != "dual_vanilla" else None,
        "reasoning_clean": (
            sum(bool(row["fixed"].get("reasoning_ends_cleanly")) for row in rows) / len(rows)
            if rows and method != "dual_vanilla" else None
        ),
        "reasoning_repeat": (
            statistics.mean(
                row["fixed"].get("reasoning_repeated_sentence_ratio")
                for row in rows
                if row["fixed"].get("reasoning_repeated_sentence_ratio") is not None
            )
            if method != "dual_vanilla" and any(
                row["fixed"].get("reasoning_repeated_sentence_ratio") is not None
                for row in rows
            ) else None
        ),
    }


def markdown(data):
    lines = []
    for bench in ("huskyqa", "mmlu", "combined"):
        lines += [f"## {bench}", "", "### Primary: PLAN validity, Agent order, paired First-3 lead", "",
                  "| Method | Parse | First-3 Same | Full Agent Seq Same | Paired N | Mean Lead3 | Median Lead3 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        summaries = {method: method_summary(data, bench, method) for method in METHODS}
        for method in METHODS:
            item = summaries[method]
            lines.append(f"| {LABELS[method]} | {fmt(item['parse'], True)} | {fmt(1.0 if method == 'dual_vanilla' else item['first3_same'], True)} | {fmt(1.0 if method == 'dual_vanilla' else item['full_same'], True)} | {item['matched'] if method != 'dual_vanilla' else '—'} | {fmt(item['lead_mean']) if method != 'dual_vanilla' else '0.000'} | {fmt(item['lead_p50']) if method != 'dual_vanilla' else '0.000'} |")
        lines += ["", "### Table 2: Agent timing", "",
                  "| Method | First-Agent P50 | First-3 P50 | Mean First-3 | P95 First-3 | Plan Parseable P50 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            item = summaries[method]
            lines.append(f"| {LABELS[method]} | {fmt(item['first_p50'])} | {fmt(item['first3_p50'])} | {fmt(item['first3_mean'])} | {fmt(item['first3_p95'])} | {fmt(item['parseable_p50'])} |")
        lines += ["", "### Table 3: paired lead vs Dual Vanilla", "",
                  "| Method | N matched | Mean Lead3 | Median Lead3 | ≥2s | ≥5s | ≥10s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS[1:]:
            item = summaries[method]
            lines.append(f"| {LABELS[method]} | {item['matched']} | {fmt(item['lead_mean'])} | {fmt(item['lead_p50'])} | {fmt(item['lead_2'], True)} | {fmt(item['lead_5'], True)} | {fmt(item['lead_10'], True)} |")
        lines.append("")

    # Explicitly split the two causal effects on jointly available queries.
    lines += ["## Effect split (Combined)", "",
              "| Effect | N matched | Mean lead | Median lead |",
              "|---|---:|---:|---:|"]
    pairs = [
        ("Structural-position", "dual_vanilla", "fixed_canvas_vanilla"),
        ("PLAN-first scheduling", "fixed_canvas_vanilla", "fixed_canvas_plan_first"),
    ]
    for label, left, right in pairs:
        leads = []
        for bench in data:
            shared = set(data[bench][left]) & set(data[bench][right])
            for item_key in shared:
                a, b = data[bench][left][item_key], data[bench][right][item_key]
                if a["parse"] and b["parse"] and a["first3"] is not None and b["first3"] is not None:
                    leads.append(float(a["first3"]) - float(b["first3"]))
        lines.append(f"| {label} | {len(leads)} | {fmt(statistics.mean(leads) if leads else None)} | {fmt(percentile(leads, .5))} |")

    # Husky duplicate ordered-tuple sanity check.
    target = ("search_agent", "search_agent", "calculation_agent")
    base = data.get("huskyqa", {}).get("dual_vanilla", {})
    target_keys = {item_key for item_key, row in base.items() if row["agents"][:3] == target}
    lines += ["", "## HuskyQA duplicate ordered tuple", "",
              "| Method | Baseline target N | Same target N | Same rate | P50 First-3 | P50 lead |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = data.get("huskyqa", {}).get(method, {})
        same = [item_key for item_key in target_keys if item_key in rows and rows[item_key]["agents"][:3] == target]
        times = [rows[item_key]["first3"] for item_key in same if rows[item_key]["first3"] is not None]
        leads = [base[item_key]["first3"] - rows[item_key]["first3"] for item_key in same if base[item_key]["first3"] is not None and rows[item_key]["first3"] is not None]
        rate = len(same) / len(target_keys) if target_keys else None
        lines.append(f"| {LABELS[method]} | {len(target_keys)} | {len(same)} | {fmt(rate, True)} | {fmt(percentile(times, .5))} | {fmt(percentile(leads, .5))} |")
    lines += ["", "## Health checks (Combined)", "",
              "| Method | Structural validity | Mean NFE | Mean plan time | Unresolved rate | Fixed corruption | Reasoning effective P50 | Reasoning clean end | Repeated sentence ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = method_summary(data, "combined", method)
        lines.append(
            f"| {LABELS[method]} | {fmt(item['structure'], True)} | {fmt(item['mean_nfe'])} | "
            f"{fmt(item['mean_generation'])} | {fmt(item['unresolved_rate'], True)} | "
            f"{fmt(item['corruption'])} | {fmt(item['reasoning_effective_p50'])} | "
            f"{fmt(item['reasoning_clean'], True)} | {fmt(item['reasoning_repeat'], True)} |"
        )
    lines += ["", "## Dynamic END diagnostics", "",
              "| Benchmark | Method | Mean tasks | END_REASONING success | END_PLAN success | Reasoning exhausted | PLAN exhausted | Reasoning tail removed P50 | PLAN tail removed P50 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for bench in ("huskyqa", "mmlu", "combined"):
        for method in METHODS[1:]:
            item = method_summary(data, bench, method)
            lines.append(
                f"| {bench} | {LABELS[method]} | {fmt(item['mean_task_count'])} | "
                f"{fmt(item['reasoning_end_success'], True)} | {fmt(item['plan_end_success'], True)} | "
                f"{fmt(item['reasoning_exhausted'], True)} | {fmt(item['plan_exhausted'], True)} | "
                f"{fmt(item['reasoning_tail_removed_p50'])} | {fmt(item['plan_tail_removed_p50'])} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = combine(args.root, ("huskyqa", "mmlu"))
    report = markdown(data)
    output = args.output or args.root / "fixed_canvas_report.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Token-length coverage for the fixed-canvas 5:5 capacity split."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def section(text, start_marker, end_marker, fallback_end_marker=None):
    start = re.search(rf"(?m)^\s*{re.escape(start_marker)}\s*:?[ \t]*$", text)
    if not start:
        return None
    tail = text[start.end() :]
    end = re.search(rf"(?m)^\s*{re.escape(end_marker)}\s*$", tail)
    if end:
        return tail[: end.start()]
    if fallback_end_marker:
        fallback = re.search(
            rf"(?m)^\s*{re.escape(fallback_end_marker)}\s*:?[ \t]*$", tail
        )
        if fallback:
            return tail[: fallback.start()]
    if start_marker == "PLAN_JSON":
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\[", tail):
            try:
                value, end_offset = decoder.raw_decode(tail[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and value:
                return tail[: match.start() + end_offset]
    return None


def stats(values, capacity):
    return {
        "n": len(values),
        "p50": percentile(values, .50),
        "p90": percentile(values, .90),
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "max": max(values) if values else None,
        "capacity": capacity,
        "coverage": sum(value <= capacity for value in values) / len(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", default="/data/labshare/Param/llada")
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--reasoning-ratio", type=float, default=.5)
    parser.add_argument("--plan-ratio", type=float, default=.5)
    parser.add_argument("--reasoning-budget", type=int)
    parser.add_argument("--plan-budget", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    delimiter_tokens = sum(
        len(encode(text))
        for text in (
            "PLANNING_REASONING\n",
            "\nPLAN_JSON\n",
        )
    )
    available = args.gen_length - delimiter_tokens
    if args.reasoning_budget is None and args.plan_budget is None:
        ratio_sum = args.reasoning_ratio + args.plan_ratio
        reasoning_budget = round(available * args.reasoning_ratio / ratio_sum)
        plan_budget = available - reasoning_budget
    elif args.reasoning_budget is None:
        plan_budget = args.plan_budget
        reasoning_budget = available - plan_budget
    elif args.plan_budget is None:
        reasoning_budget = args.reasoning_budget
        plan_budget = available - reasoning_budget
    else:
        reasoning_budget = args.reasoning_budget
        plan_budget = args.plan_budget
    if min(reasoning_budget, plan_budget) <= 0 or reasoning_budget + plan_budget > available:
        raise ValueError("Invalid reasoning/PLAN budgets for available canvas.")
    by_benchmark = {}
    combined_reasoning, combined_plan = [], []
    for benchmark in ("huskyqa", "mmlu"):
        path = args.root / benchmark / "dual_vanilla" / "plans.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        reasoning_lengths, plan_lengths = [], []
        for row in rows:
            raw = row.get("raw_plan") or ""
            reasoning = section(
                raw, "PLANNING_REASONING", "END_PLANNING_REASONING", "PLAN_JSON"
            )
            plan = section(raw, "PLAN_JSON", "END_PLAN_JSON")
            if reasoning is not None:
                reasoning_lengths.append(len(encode(reasoning)))
            if plan is not None:
                plan_lengths.append(len(encode(plan)))
        combined_reasoning.extend(reasoning_lengths)
        combined_plan.extend(plan_lengths)
        by_benchmark[benchmark] = {
            "reasoning": stats(reasoning_lengths, reasoning_budget),
            "plan": stats(plan_lengths, plan_budget),
        }
    by_benchmark["combined"] = {
        "reasoning": stats(combined_reasoning, reasoning_budget),
        "plan": stats(combined_plan, plan_budget),
    }
    payload = {
        "gen_length": args.gen_length,
        "delimiter_tokens": delimiter_tokens,
        "available_tokens": available,
        "reasoning_budget": reasoning_budget,
        "plan_budget": plan_budget,
        "benchmarks": by_benchmark,
    }
    lines = [
        "# Fixed-canvas capacity coverage", "",
        f"Delimiter={delimiter_tokens}, available={available}, reasoning={reasoning_budget}, PLAN={plan_budget}", "",
        "| Benchmark | Region | N | P50 | P90 | P95 | P99 | Max | <= capacity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("huskyqa", "mmlu", "combined"):
        for region in ("reasoning", "plan"):
            item = by_benchmark[benchmark][region]
            value = lambda key: "—" if item[key] is None else f"{item[key]:.1f}"
            coverage = "—" if item["coverage"] is None else f"{item['coverage']:.1%}"
            lines.append(
                f"| {benchmark} | {region} | {item['n']} | {value('p50')} | "
                f"{value('p90')} | {value('p95')} | {value('p99')} | "
                f"{value('max')} | {coverage} |"
            )
    report = "\n".join(lines) + "\n"
    output = args.output or args.root / "capacity_report.md"
    output.write_text(report, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

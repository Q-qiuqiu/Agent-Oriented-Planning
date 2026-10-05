#!/usr/bin/env python3
"""Compare schema-v7 Agent wall times across base and commit variants."""

import json
import math
import os
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 7
RUN_ROOT = os.environ.get("RUN_ROOT")

# Edit CONFIG/VARIANT_PATHS, or set RUN_ROOT to an output directory produced by
# run_huskyqa_base_commit_dual_gpu.sh. All input must use schema v7.
CONFIG = {
    "variants": ["base", "commit"],
    "benchmarks": ["chronoqa", "huskyqa", "iirc", "mmlu"],
}

VARIANT_PATHS = {
    "base": {
        "template": "benchmarks/fastdllm_log/full_llada_base/{b}_full_timings.jsonl",
        "method": "base",
        "event": "natural",
    },
    "commit": {
        "template": "benchmarks/fastdllm_log/full_llada_commit/{b}_full_timings.jsonl",
        "method": "commit",
        "event": "commit",
    },
    # Useful for measuring observer overhead independently from early commit.
    "commit_natural": {
        "template": "benchmarks/fastdllm_log/full_llada_commit/{b}_full_timings.jsonl",
        "method": "commit",
        "event": "natural",
    },
}

AGENTS_BY_BENCH = {
    "chronoqa": ["evidence_agent", "temporal_agent", "verification_agent"],
    "huskyqa": ["search_agent", "calculation_agent", "reasoning_agent"],
    "iirc": ["context_agent", "retrieval_agent", "reasoning_agent"],
    "mmlu": ["knowledge_agent", "reasoning_agent", "elimination_agent"],
}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def stats(values):
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def resolve_variant_path(variant, benchmark):
    spec = VARIANT_PATHS[variant]
    if RUN_ROOT:
        method_directory = "base" if spec["method"] == "base" else "commit"
        return (
            Path(RUN_ROOT).expanduser().resolve()
            / method_directory
            / "timing"
            / f"{benchmark}_full_timings.jsonl"
        )
    override = (spec.get("overrides") or {}).get(benchmark)
    relative = override or spec["template"].format(b=benchmark)
    path = Path(relative).expanduser()
    return path if path.is_absolute() else ROOT / path


def record_key(record):
    return record.get("request_key") or record.get("query_sha256") or record.get("query")


def load_latest_per_request(path, expected_method):
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema in {path}:{line_number}: expected "
                    f"{SCHEMA_VERSION}, got {record.get('schema_version')!r}"
                )
            if record.get("method") != expected_method:
                raise ValueError(
                    f"Expected method={expected_method} in {path}:{line_number}, "
                    f"got {record.get('method')!r}"
                )
            key = record_key(record)
            if key is None:
                raise ValueError(f"Missing request key in {path}:{line_number}")
            previous = records.get(key)
            attempt = int(record.get("attempt_count") or 0)
            previous_attempt = int(previous.get("attempt_count") or 0) if previous else -1
            if previous is None or attempt >= previous_attempt:
                records[key] = record
    return records


def base_natural_events(record):
    events = []
    for row in record.get("natural_agents") or []:
        slot = row.get("slot")
        agent = row.get("agent")
        seconds = row.get("seconds")
        if isinstance(slot, int) and isinstance(agent, str) and is_number(seconds):
            events.append((slot, agent, float(seconds)))
    return sorted(events)


def commit_events(record, event):
    events = []
    for row in record.get("agent_commits") or []:
        slot = row.get("slot_id")
        if event == "commit":
            agent = row.get("committed_agent")
            seconds = row.get("commit_wall_time")
        elif event == "natural":
            agent = row.get("natural_agent")
            seconds = row.get("natural_agent_wall_time")
        else:
            raise ValueError(f"Unsupported event type: {event!r}")
        if isinstance(slot, int) and isinstance(agent, str) and is_number(seconds):
            events.append((slot, agent, float(seconds)))
    return sorted(events)


def timing_events(record, method, event):
    if method == "base" and event == "natural":
        return base_natural_events(record)
    if method == "commit":
        return commit_events(record, event)
    raise ValueError(f"Unsupported method/event pair: {method}/{event}")


def collect(path, method, event):
    records = load_latest_per_request(path, method)
    per_agent = {}
    first_times = []
    all_times = []
    generation_times = []
    successful_requests = 0
    complete_requests = 0
    detected_slots = 0
    expected_slots = 0
    evaluated_commits = 0
    correct_commits = 0

    for record in records.values():
        if record.get("status") != "ok":
            continue
        successful_requests += 1
        events = timing_events(record, method, event)
        expected = record.get("agent_count")
        if not isinstance(expected, int):
            expected = len(events)
        expected_slots += expected
        detected_slots += len(events)

        seconds = []
        for _slot, agent, wall_time in events:
            per_agent.setdefault(agent, []).append(wall_time)
            seconds.append(wall_time)
        if seconds:
            first_times.append(min(seconds))
        if expected > 0 and len(events) == expected:
            complete_requests += 1
            all_times.append(max(seconds))

        if method == "commit" and event == "commit":
            for row in record.get("agent_commits") or []:
                committed = row.get("committed_agent")
                natural = row.get("natural_agent")
                if isinstance(committed, str) and isinstance(natural, str):
                    evaluated_commits += 1
                    correct_commits += int(committed == natural)

        generation = record.get("full_generation_time")
        if not is_number(generation):
            generation = record.get("generation_seconds")
        if is_number(generation):
            generation_times.append(float(generation))

    return {
        "requests": len(records),
        "successful_requests": successful_requests,
        "complete_requests": complete_requests,
        "detected_slots": detected_slots,
        "expected_slots": expected_slots,
        "coverage": detected_slots / expected_slots if expected_slots else None,
        "accuracy": (
            correct_commits / evaluated_commits if evaluated_commits else None
        ),
        "per_agent": {agent: stats(values) for agent, values in per_agent.items()},
        "first": stats(first_times),
        "all": stats(all_times),
        "generation": stats(generation_times),
    }


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def fmt_percent(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def display_width(text):
    return sum(
        2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
        for character in str(text)
    )


def pad(text, width):
    return str(text) + " " * max(width - display_width(text), 0)


def print_box_table(rows, variants):
    sub_headers = ("mean", "P50", "P95")
    headers = ["event"]
    for variant in variants:
        headers.extend(f"{variant} {sub_header}" for sub_header in sub_headers)

    rendered = []
    for label, per_variant in rows:
        cells = [label]
        for variant in variants:
            summary = per_variant.get(variant)
            if summary is None or summary["n"] == 0:
                cells.extend(["-"] * 3)
            else:
                cells.extend(
                    [fmt(summary["mean"]), fmt(summary["p50"]), fmt(summary["p95"])]
                )
        rendered.append(cells)

    widths = [
        max(display_width(headers[index]), *(display_width(row[index]) for row in rendered))
        for index in range(len(headers))
    ]

    def border(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def line(cells):
        return "│ " + " │ ".join(
            pad(cells[index], widths[index]) for index in range(len(cells))
        ) + " │"

    print(border("┌", "┬", "┐"))
    print(line(headers))
    print(border("├", "┼", "┤"))
    for row in rendered:
        print(line(row))
    print(border("└", "┴", "┘"))


def main():
    variants = CONFIG["variants"]
    benchmarks = CONFIG["benchmarks"]
    unknown = [variant for variant in variants if variant not in VARIANT_PATHS]
    if unknown:
        raise ValueError(
            f"Unknown variants in CONFIG: {unknown}; known={sorted(VARIANT_PATHS)}"
        )

    print("Agent detection/commit wall time (schema v7, all PLAN slots)")
    print(f"Variants: {', '.join(variants)}")
    for benchmark in benchmarks:
        collected = {}
        for variant in variants:
            spec = VARIANT_PATHS[variant]
            path = resolve_variant_path(variant, benchmark)
            if not path.exists():
                print(f"[{benchmark}] {variant}: missing {path}")
                continue
            collected[variant] = collect(
                path, method=spec["method"], event=spec["event"]
            )
        if not collected:
            continue

        print(f"\n{benchmark.upper()}")
        for variant in variants:
            if variant not in collected:
                continue
            data = collected[variant]
            detail = (
                f"  {variant}: requests={data['successful_requests']}, "
                f"complete={data['complete_requests']}, "
                f"slots={data['detected_slots']}/{data['expected_slots']}, "
                f"coverage={fmt_percent(data['coverage'])}"
            )
            if data["accuracy"] is not None:
                detail += f", commit_accuracy={fmt_percent(data['accuracy'])}"
            print(detail)

        agent_names = AGENTS_BY_BENCH.get(benchmark) or sorted(
            {
                agent
                for data in collected.values()
                for agent in data["per_agent"]
            }
        )
        rows = [
            (
                agent,
                {
                    variant: data["per_agent"].get(agent, stats([]))
                    for variant, data in collected.items()
                },
            )
            for agent in agent_names
        ]
        for label, key in (
            ("first Agent", "first"),
            ("all Agents", "all"),
            ("full generation", "generation"),
        ):
            rows.append((label, {variant: data[key] for variant, data in collected.items()}))
        print_box_table(rows, variants)


if __name__ == "__main__":
    main()

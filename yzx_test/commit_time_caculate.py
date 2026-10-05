#!/usr/bin/env python3
"""Report final latent-region/prefix/natural commit accuracy and lead."""

import json
import math
import os
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 7
SOURCES = ("latent_region", "prefix", "natural")

# Edit these paths, or set RUN_ROOT to an output directory produced by
# run_huskyqa_base_commit_dual_gpu.sh.
CONFIG = {
    "benchmarks": ["chronoqa", "huskyqa", "iirc", "mmlu"],
    "commit_log": (
        "benchmarks/fastdllm_log/full_llada_commit/{bench}_full_timings.jsonl"
    ),
}
RUN_ROOT = os.environ.get("RUN_ROOT")


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


def record_key(record):
    return record.get("request_key") or record.get("query_sha256") or record.get("query")


def validate_record(record, path, line_number):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema in {path}:{line_number}: "
            f"expected {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
        )
    if record.get("method") != "commit":
        raise ValueError(
            f"Expected method=commit in {path}:{line_number}, "
            f"got {record.get('method')!r}"
        )
    if not isinstance(record.get("agent_commits"), list):
        raise ValueError(f"Missing agent_commits list in {path}:{line_number}")


def load_latest(path):
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            validate_record(record, path, line_number)
            key = record_key(record)
            if key is None:
                raise ValueError(f"Missing request key in {path}:{line_number}")
            previous = records.get(key)
            attempt = int(record.get("attempt_count") or 0)
            previous_attempt = int(previous.get("attempt_count") or 0) if previous else -1
            if previous is None or attempt >= previous_attempt:
                records[key] = record
    return records


def empty_source_result():
    return {
        "slots": 0, "evaluated": 0, "correct": 0,
        "lead": [], "correct_lead": [],
    }


def analyze_records(records):
    result = {
        "requests": len(records),
        "successful_requests": 0,
        "complete_requests": 0,
        "slots": 0,
        "evaluated_slots": 0,
        "correct_slots": 0,
        "wrong_slots": 0,
        "prefetch_triggered": 0,
        "latent_triggers": 0,
        "correct_real_slot_triggers": 0,
        "sources": {source: empty_source_result() for source in SOURCES},
        "first_commit": [],
        "all_commit": [],
        "all_natural": [],
        "all_available": [],
        "generation": [],
    }

    for record in records.values():
        if record.get("status") != "ok":
            continue
        result["successful_requests"] += 1
        categories = record.get("latent_trigger_categories") or {}
        result["latent_triggers"] += int(
            record.get("latent_trigger_count") or sum(categories.values())
        )
        result["correct_real_slot_triggers"] += int(
            categories.get("correct_real_slot") or 0
        )
        rows = record["agent_commits"]
        expected_count = record.get("agent_count")
        if not isinstance(expected_count, int):
            expected_count = len(rows)

        commit_times = []
        natural_times = []
        available_times = []
        complete = len(rows) == expected_count and expected_count > 0

        for row in rows:
            source = row.get("commit_source")
            if source not in SOURCES:
                raise ValueError(f"Unsupported commit_source: {source!r}")
            slot_id = row.get("slot_id")
            if not isinstance(slot_id, int):
                raise ValueError(f"Invalid slot_id in commit row: {slot_id!r}")

            committed = row.get("committed_agent")
            natural = row.get("natural_agent")
            commit_time = row.get("commit_wall_time")
            natural_time = row.get("natural_agent_wall_time")

            result["slots"] += 1
            source_result = result["sources"][source]
            source_result["slots"] += 1
            if row.get("prefetch_triggered") is True:
                result["prefetch_triggered"] += 1

            if is_number(commit_time):
                commit_times.append(float(commit_time))
            else:
                complete = False
            if is_number(natural_time):
                natural_times.append(float(natural_time))
            else:
                complete = False

            correct = None
            if isinstance(committed, str) and isinstance(natural, str):
                correct = committed == natural
                result["evaluated_slots"] += 1
                source_result["evaluated"] += 1
                if correct:
                    result["correct_slots"] += 1
                    source_result["correct"] += 1
                else:
                    result["wrong_slots"] += 1
            else:
                complete = False

            if is_number(commit_time) and is_number(natural_time):
                lead = float(natural_time) - float(commit_time)
                source_result["lead"].append(lead)
                if correct is True:
                    source_result["correct_lead"].append(lead)
                # Wrong early commits become usable only at Natural reveal.
                available_times.append(
                    float(commit_time) if correct is True else float(natural_time)
                )
            else:
                complete = False

        if commit_times:
            result["first_commit"].append(min(commit_times))
        if complete:
            result["complete_requests"] += 1
            result["all_commit"].append(max(commit_times))
            result["all_natural"].append(max(natural_times))
            result["all_available"].append(max(available_times))
        generation = record.get("full_generation_time")
        if not is_number(generation):
            generation = record.get("generation_seconds")
        if is_number(generation):
            result["generation"].append(float(generation))

    result["accuracy"] = (
        result["correct_slots"] / result["evaluated_slots"]
        if result["evaluated_slots"] else None
    )
    latent = result["sources"]["latent_region"]
    result["latent_accuracy"] = (
        latent["correct"] / latent["evaluated"]
        if latent["evaluated"] else None
    )
    result["correct_latent_coverage"] = (
        latent["correct"] / result["evaluated_slots"]
        if result["evaluated_slots"] else None
    )
    result["latent_trigger_precision"] = (
        result["correct_real_slot_triggers"] / result["latent_triggers"]
        if result["latent_triggers"] else None
    )
    for source_result in result["sources"].values():
        source_result["accuracy"] = (
            source_result["correct"] / source_result["evaluated"]
            if source_result["evaluated"] else None
        )
        source_result["lead_stats"] = stats(source_result.pop("lead"))
        source_result["correct_lead_stats"] = stats(
            source_result.pop("correct_lead")
        )
    for field in (
        "first_commit", "all_commit", "all_natural", "all_available", "generation"
    ):
        result[f"{field}_stats"] = stats(result.pop(field))
    return result


def analyze_benchmark(path):
    return analyze_records(load_latest(path))


def display_width(text):
    return sum(
        2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
        for character in str(text)
    )


def print_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(display_width(headers[index]), *(display_width(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def border(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def render(row):
        cells = [
            value + " " * (widths[index] - display_width(value))
            for index, value in enumerate(row)
        ]
        return "│ " + " │ ".join(cells) + " │"

    print(border("┌", "┬", "┐"))
    print(render(headers))
    print(border("├", "┼", "┤"))
    for row in rows:
        print(render(row))
    print(border("└", "┴", "┘"))


def fmt_seconds(value):
    return "-" if value is None else f"{value:.3f}"


def fmt_percent(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def main():
    results = {}
    for benchmark in CONFIG["benchmarks"]:
        path = (
            Path(RUN_ROOT).expanduser().resolve()
            / "commit" / "timing" / f"{benchmark}_full_timings.jsonl"
            if RUN_ROOT
            else ROOT / CONFIG["commit_log"].format(bench=benchmark)
        )
        if not path.exists():
            print(f"[{benchmark}] missing commit log: {path}")
            continue
        results[benchmark] = analyze_benchmark(path)

    if not results:
        return

    print("Commit summary (all PLAN slots, schema v7)")
    print_table(
        [
            "benchmark", "requests", "complete", "slots", "correct", "wrong",
            "first-valid acc", "latent acc", "correct latent coverage",
            "trigger precision", "latent", "prefix", "natural",
        ],
        [
            [
                benchmark,
                data["successful_requests"],
                data["complete_requests"],
                data["slots"],
                data["correct_slots"],
                data["wrong_slots"],
                fmt_percent(data["accuracy"]),
                fmt_percent(data["latent_accuracy"]),
                fmt_percent(data["correct_latent_coverage"]),
                fmt_percent(data["latent_trigger_precision"]),
                data["sources"]["latent_region"]["slots"],
                data["sources"]["prefix"]["slots"],
                data["sources"]["natural"]["slots"],
            ]
            for benchmark, data in results.items()
        ],
    )

    print("\nPer-source correctness, raw lead, and correct-only lead")
    print_table(
        [
            "benchmark", "source", "slots", "evaluated", "accuracy",
            "raw lead P50", "correct lead mean", "correct P50", "correct P95",
        ],
        [
            [
                benchmark,
                source,
                source_result["slots"],
                source_result["evaluated"],
                fmt_percent(source_result["accuracy"]),
                fmt_seconds(source_result["lead_stats"]["p50"]),
                fmt_seconds(source_result["correct_lead_stats"]["mean"]),
                fmt_seconds(source_result["correct_lead_stats"]["p50"]),
                fmt_seconds(source_result["correct_lead_stats"]["p95"]),
            ]
            for benchmark, data in results.items()
            for source, source_result in data["sources"].items()
        ],
    )

    print(
        "\nRequest wall time: available uses commit time when correct, "
        "otherwise Natural reveal time"
    )
    labels = (
        ("first commit", "first_commit_stats"),
        ("all commits", "all_commit_stats"),
        ("all Natural", "all_natural_stats"),
        ("all available", "all_available_stats"),
        ("full generation", "generation_stats"),
    )
    print_table(
        ["benchmark", "metric", "n", "mean (s)", "P50 (s)", "P95 (s)"],
        [
            [
                benchmark,
                label,
                data[field]["n"],
                fmt_seconds(data[field]["mean"]),
                fmt_seconds(data[field]["p50"]),
                fmt_seconds(data[field]["p95"]),
            ]
            for benchmark, data in results.items()
            for label, field in labels
        ],
    )


if __name__ == "__main__":
    main()

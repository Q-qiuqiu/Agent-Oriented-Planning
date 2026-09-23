#!/usr/bin/env python3
"""Per-agent detection-time comparison across timing-log variants.

Edit CONFIG below to choose which variants (log sets) and benchmarks to
compare. Each variant maps to one timing JSONL per benchmark; paths may use
"{b}" for the benchmark name, and per-benchmark overrides are supported.
"""

import json
import math
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    # Variants to compare, in display order. Every name must exist in
    # VARIANT_PATHS below.
    "variants": ["full_lladav2", "base_lladav2"],
    # Benchmarks to report, in display order.
    "benchmarks": ["chronoqa", "huskyqa", "iirc", "mmlu"],
    # Timing field to analyze: "decision_seconds" or "confirmation_seconds".
    "field": "decision_seconds",
}

# Path templates for every known log set. Use "{b}" for the benchmark name;
# "overrides" pins a concrete path for a benchmark that does not follow the
# template. Missing files are reported and skipped.
VARIANT_PATHS = {
    "base_lladav1": {
        "template": "benchmarks/fastdllm_log/base_lladav1/base_{b}_full_timings.jsonl",
    },
    "base_lladav2": {
        "template": "benchmarks/fastdllm_log/base_lladav2/base_{b}_full_timings.jsonl",
    },
    "full_llada": {
        "template": "benchmarks/fastdllm_log/full_llada/{b}_full_timings.jsonl",
    },
    "full_lladav2": {
        "template": "benchmarks/fastdllm_log/full_lladav2/{b}_full_timings.jsonl",
        "overrides": {
            # file name typo in the original run
            #"mmlu": "benchmarks/fastdllm_log/full_lladav2/mmlu_ful_timings.jsonl",
        },
    },
    "full_lladav3": {
        "template": "llada_server/{b}_full_lladav3_timings.jsonl",
    },
}

# Agent roles differ per benchmark.
AGENTS_BY_BENCH = {
    "chronoqa": ["evidence_agent", "temporal_agent", "verification_agent"],
    "huskyqa": ["search_agent", "calculation_agent", "reasoning_agent"],
    "iirc": ["context_agent", "retrieval_agent", "reasoning_agent"],
    "mmlu": ["knowledge_agent", "reasoning_agent", "elimination_agent"],
}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def stats(values):
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "p95": None, "p99": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def resolve_variant_path(variant, bench):
    spec = VARIANT_PATHS[variant]
    override = (spec.get("overrides") or {}).get(bench)
    relative = override if override else spec["template"].format(b=bench)
    path = Path(relative).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_latest_per_request(path):
    by_key = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("request_key") or record.get("query")
            if key is None:
                continue
            prev = by_key.get(key)
            attempt = record.get("attempt_count") or 0
            if prev is None or attempt >= (prev.get("attempt_count") or 0):
                by_key[key] = record
    return by_key


def collect(path, field):
    by_key = load_latest_per_request(path)
    per_agent = {}
    first, all_dec, gen = [], [], []
    for record in by_key.values():
        if record.get("status") != "ok":
            continue
        events = []
        for e in record.get("agents") or []:
            agent = e.get("agent")
            t = e.get(field)
            if agent is None or not isinstance(t, (int, float)):
                continue
            per_agent.setdefault(agent, []).append(float(t))
            events.append(float(t))
        if events:
            first.append(min(events))
            all_dec.append(max(events))
        if isinstance(record.get("generation_seconds"), (int, float)):
            gen.append(float(record["generation_seconds"]))
    return {
        "requests": len(by_key),
        "per_agent": {a: stats(v) for a, v in per_agent.items()},
        "first": stats(first),
        "all": stats(all_dec),
        "gen": stats(gen),
    }


def fmt(v):
    return "N/A" if v is None else f"{v:.2f}"


def display_width(text):
    """Terminal display width; CJK wide/fullwidth chars count as 2."""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        for ch in text
    )


def pad(text, width):
    return text + " " * max(width - display_width(text), 0)


def print_box_table(rows, variants):
    """rows: list of (label, {variant: stats-dict}). One box table."""
    sub_headers = ["mean", "P95", "P99"]
    headers = ["agent"]
    for variant in variants:
        headers.extend(f"{variant} {sub}" for sub in sub_headers)

    rendered = []
    for label, per_variant in rows:
        cells = [label]
        for variant in variants:
            s = per_variant.get(variant)
            if s is None or s["n"] == 0:
                cells.extend(["-"] * 3)
            else:
                cells.extend([fmt(s["mean"]), fmt(s["p95"]), fmt(s["p99"])])
        rendered.append(cells)

    widths = [
        max(display_width(headers[i]), *(display_width(row[i]) for row in rendered))
        for i in range(len(headers))
    ]

    def border(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(cells):
        return "│ " + " │ ".join(
            pad(cells[i], widths[i]) for i in range(len(cells))
        ) + " │"

    print(border("┌", "┬", "┐"))
    print(line(headers))
    print(border("├", "┼", "┤"))
    for row in rendered:
        print(line(row))
    print(border("└", "┴", "┘"))


def main():
    variants = CONFIG["variants"]
    benches = CONFIG["benchmarks"]
    field = CONFIG["field"]
    unknown = [v for v in variants if v not in VARIANT_PATHS]
    if unknown:
        raise ValueError(
            f"Unknown variants in CONFIG: {unknown}. "
            f"Known: {sorted(VARIANT_PATHS)}"
        )

    print(f"Detection-time field: {field} (seconds after plan generation start)")
    print(f"Variants: {', '.join(variants)}\n")
    for bench in benches:
        agents_order = AGENTS_BY_BENCH.get(bench)
        collected = {}
        for variant in variants:
            path = resolve_variant_path(variant, bench)
            if not path.exists():
                print(f"[{bench}] {variant}: missing {path}")
                continue
            collected[variant] = collect(path, field)
        if not collected:
            continue

        counts = " / ".join(
            f"{v}={collected[v]['requests']}" for v in variants if v in collected
        )
        print(f"\n{bench.upper()}（query 数: {counts}）\n")

        agent_names = agents_order or sorted(
            {a for data in collected.values() for a in data["per_agent"]}
        )
        rows = []
        for agent in agent_names:
            rows.append((
                agent,
                {v: data["per_agent"].get(agent, stats([])) for v, data in collected.items()},
            ))
        for label, key in (("首个 agent", "first"),
                           ("全部检出", "all"),
                           ("plan 总时长", "gen")):
            rows.append((label, {v: data[key] for v, data in collected.items()}))
        print_box_table(rows, variants)


if __name__ == "__main__":
    main()

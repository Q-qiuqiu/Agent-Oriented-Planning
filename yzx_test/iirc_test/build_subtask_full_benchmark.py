import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from build_subtask_benchmark import (
    AGENTS,
    expand_plans,
    load_queries,
    normalize_plan,
    print_summary,
)
from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


FULL_PROMPT_VERSION = "iirc_compact_full_v2"
FULL_PLANNER_PROMPT = planner_prompt.replace(
    "Output only one valid JSON array in this schema. This example shows two\nindependent evidence tasks followed by one synthesis task:",
    "The PLAN_JSON block must contain the JSON plan in this schema. This example shows two\nindependent evidence tasks followed by one synthesis task:",
    1,
).replace(
    "- Do not output analysis, Markdown fences, comments, or text outside the array.",
    """- Use the marked response format below instead of returning a bare array.

PLAN_JSON
[the complete JSON plan required above]
END_PLAN_JSON

PLANNING_REASONING
Explain why each independent task can run immediately, why each dependency is
needed, and why any plan longer than five calls cannot be consolidated safely.
Do not solve the question or introduce tasks not present in PLAN_JSON.
END_PLANNING_REASONING

Use each marker exactly once. Do not use Markdown fences.""",
    1,
)

CONFIG = {
    "input": "benchmarks/iirc/iirc_dev_flat.json",
    "plans_output": "benchmarks/iirc/iirc_plans_full_llada.json",
    "benchmark_output": "benchmarks/iirc/iirc_subtask_full_llada.json",
    "planner_api_url": "http://10.137.144.97:7007/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "retry_missing_reasoning": False,
    "agents": AGENTS,
}


def request_completion(query, config):
    response = requests.post(
        chat_completions_url(config["planner_api_url"]),
        headers={
            "Content-Type": "application/json",
            **auth_header(config["planner_api_key"]),
        },
        json={
            "model": config["planner_model"],
            "messages": [
                {"role": "system", "content": FULL_PLANNER_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": config["planner_temperature"],
            "max_tokens": config["planner_max_tokens"],
        },
        timeout=config["timeout"],
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"].strip()


def extract_json_array(text):
    segment = text
    marker = re.search(r"(?m)^\s*PLAN_JSON\s*:?[ \t]*$", text)
    if marker:
        segment = text[marker.end():]
        end = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", segment)
        if end:
            segment = segment[: end.start()]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", segment):
        try:
            value, _ = decoder.raw_decode(segment[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    raise ValueError("No JSON plan array found")


def extract_reasoning(text):
    start = re.search(r"(?m)^\s*PLANNING_REASONING\s*:?[ \t]*$", text)
    if not start:
        return None
    tail = text[start.end():]
    end = re.search(r"(?m)^\s*END_PLANNING_REASONING\s*$", tail)
    if not end:
        return None
    return tail[: end.start()].strip() or None


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def ordered_records(records_by_index, queries):
    return [
        records_by_index[row["source_index"]]
        for row in queries
        if row["source_index"] in records_by_index
    ]


def build_plans(queries, config):
    existing = []
    if Path(config["plans_output"]).exists():
        with Path(config["plans_output"]).open("r", encoding="utf-8") as file:
            existing = json.load(file)
    by_index = {row["source_index"]: row for row in existing}
    done = {
        key for key, row in by_index.items()
        if row.get("error") is None
        and row.get("plan")
        and row.get("planner_prompt_version") == FULL_PROMPT_VERSION
        and (not config["retry_missing_reasoning"] or row.get("planning_reasoning"))
    }
    if existing:
        print(
            f"resume | loaded={len(existing)} | completed={len(done)} "
            f"| prompt_version={FULL_PROMPT_VERSION}",
            flush=True,
        )

    selected = queries[: config["limit"]] if config["limit"] else queries
    for row in selected:
        if row["source_index"] in done:
            continue
        raw_output = None
        started = time.perf_counter()
        record = {
            **row,
            "planner_model": config["planner_model"],
            "planner_mode": "plan_json_then_reasoning",
            "planner_prompt_version": FULL_PROMPT_VERSION,
        }
        record.pop("planner_input", None)
        try:
            raw_output = request_completion(row["planner_input"], config)
            reasoning = extract_reasoning(raw_output)
            plan = normalize_plan(extract_json_array(raw_output))
            record.update(
                {
                    "planning_reasoning": reasoning,
                    "raw_plan": raw_output,
                    "plan": plan,
                    "plan_call_count": len(plan),
                    "exceeds_recommended_calls": len(plan) > 5,
                    "format_warnings": [] if reasoning else ["missing planning reasoning"],
                    "error": None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "planning_reasoning": extract_reasoning(raw_output) if raw_output else None,
                    "raw_plan": raw_output,
                    "plan": None,
                    "plan_call_count": None,
                    "exceeds_recommended_calls": False,
                    "error": str(exc),
                }
            )
        record["time"] = time.perf_counter() - started
        by_index[row["source_index"]] = record
        save_json(config["plans_output"], ordered_records(by_index, queries))
        print(
            f"planned {row['source_index']} | reasoning_chars="
            f"{len(record.get('planning_reasoning') or '')} "
            f"| subtasks={len(record.get('plan') or [])} | error={record['error']}",
            flush=True,
        )
    return ordered_records(by_index, queries)


def main():
    parser = argparse.ArgumentParser(description="Build IIRC full plans.")
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument("--planner-temperature", type=float, default=CONFIG["planner_temperature"])
    parser.add_argument("--planner-max-tokens", type=int, default=CONFIG["planner_max_tokens"])
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=CONFIG["agents"])
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(vars(args))
    queries = load_queries(config["input"])
    plans = build_plans(queries, config)
    save_json(config["plans_output"], plans)
    benchmark = expand_plans(plans, config["agents"])
    save_json(config["benchmark_output"], benchmark)
    print(f"Saved plans: {config['plans_output']} ({len(plans)} queries)")
    print(f"Saved benchmark: {config['benchmark_output']} ({len(benchmark)} rows)")
    print_summary(plans)


if __name__ == "__main__":
    main()

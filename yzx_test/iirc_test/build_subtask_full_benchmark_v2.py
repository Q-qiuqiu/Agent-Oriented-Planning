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


# v3 = base_llada detection-slowdown variant: identical rules, markers and
# JSON schema, but the PLANNING_REASONING instruction now demands one detailed
# paragraph per selected agent plus a synthesis paragraph, so the PLAN_JSON
# block (and therefore the "agent":" anchors the timing monitor detects)
# starts much later in the response.
FULL_PROMPT_VERSION = "iirc_full_reasoning_first_long_v3_3sent"


def remove_json_example(prompt, introduction):
    if introduction not in prompt:
        raise ValueError("IIRC planner prompt JSON introduction was not found")
    prefix, remainder = prompt.split(introduction, 1)
    array_start = None
    array_end = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", remainder):
        try:
            value, end = decoder.raw_decode(remainder[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            array_start = match.start()
            array_end = end
            break
    if array_start is None:
        raise ValueError("IIRC planner prompt JSON example was not found")
    before_example = remainder[:array_start]
    suffix = remainder[array_start + array_end:]
    return f"{prefix.rstrip()}\n\n{before_example.strip()}\n{suffix.strip()}"


BASE_FULL_INSTRUCTIONS = remove_json_example(
    planner_prompt,
    "Output only one valid JSON array in this schema. This example shows two\n"
    "independent evidence tasks followed by one synthesis task:",
).replace(
    "- Do not output analysis, Markdown fences, comments, or text outside the array.",
    "- Follow the marked response format below exactly.",
)

FULL_PLANNER_PROMPT = BASE_FULL_INSTRUCTIONS + """

Use the same decomposition, agent selection, dependencies,
and JSON plan that you would produce under the original instructions. The only
additional requirement is to output the planning reasoning before that JSON.

PLANNING_REASONING
Explain the reasoning that led to the plan in depth. For EACH agent task you
selected, write one paragraph of about two sentences describing
the perspective it contributes, the method and kind of evidence it relies on,
and why that angle alone is insufficient without the other selected agents.
Then finish with one synthesis paragraph explaining how the selected views
complement each other and why this decomposition fits the question. This is
an additional explanation, not a different planning task. Do not solve the
subtasks in this section and do not introduce any agent-selection or
decomposition rules beyond the original instructions. Do not put JSON or
Markdown code fences in this section.
END_PLANNING_REASONING

PLAN_JSON
[
  {
    "agent": "context_agent",
    "id": 1,
    "task": "subtask description",
    "reason": "why this agent is suitable",
    "dep": []
  }
]
END_PLAN_JSON
"""

CONFIG = {
    "input": "benchmarks/iirc/iirc_dev_flat.json",
    "plans_output": "benchmarks/iirc/iirc_plans_base_llama3.json",
    "benchmark_output": "benchmarks/iirc/iirc_subtask_base_llama3.json",
    "planner_api_url": "http://10.137.144.97:7005/v1",
    "planner_api_key": "empty",
    #"planner_model": "/data/labshare/Param/llada",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
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
    decoder = json.JSONDecoder()
    value = text.strip()
    marker_matches = list(re.finditer(r"(?m)^\s*PLAN_JSON\s*:?\s*$", value))
    if marker_matches:
        segment = value[marker_matches[-1].end():]
        end_match = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", segment)
        if end_match:
            segment = segment[:end_match.start()]
        segment = re.sub(r"(?m)^\s*```(?:json)?\s*$", "", segment).strip()
        start = segment.find("[")
        if start < 0:
            raise ValueError("PLAN_JSON contains no JSON array")
        try:
            plan, _ = decoder.raw_decode(segment[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse PLAN_JSON array: {exc}") from exc
        if not isinstance(plan, list) or not all(isinstance(step, dict) for step in plan):
            raise ValueError("PLAN_JSON must be an array of plan-step objects")
        return plan

    for match in re.finditer(r"\[", value):
        try:
            plan, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(plan, list) and all(isinstance(step, dict) for step in plan):
            return plan
    raise ValueError("Cannot find plan JSON in planner output")


def extract_planning_reasoning(text):
    start_marker = "PLANNING_REASONING"
    end_marker = "END_PLANNING_REASONING"
    start = text.find(start_marker)
    if start < 0:
        return None
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        end = text.find("PLAN_JSON", start)
    if end < 0:
        return None
    return text[start:end].strip(" \n:\t") or None


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
            f"| prompt_version={FULL_PROMPT_VERSION} "
            f"| retry_missing_reasoning={config['retry_missing_reasoning']}",
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
            "planner_mode": "reasoning_long_then_json",
            "planner_prompt_version": FULL_PROMPT_VERSION,
        }
        record.pop("planner_input", None)
        try:
            raw_output = request_completion(row["planner_input"], config)
            reasoning = extract_planning_reasoning(raw_output)
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
                    "planning_reasoning": extract_planning_reasoning(raw_output) if raw_output else None,
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
    parser = argparse.ArgumentParser(
        description="Build IIRC plans with long visible reasoning and JSON subtasks."
    )
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
    parser.add_argument(
        "--retry-missing-reasoning",
        action="store_true",
        default=CONFIG["retry_missing_reasoning"],
    )
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

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from build_subtask_benchmark import AGENTS, expand_plans, load_queries, normalize_plan
from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


FULL_PROMPT_VERSION = "mmlu_pro_three_parallel_perspectives_full_v1"
FULL_PLANNER_PROMPT = planner_prompt.replace(
    "Output only one valid JSON array containing exactly three tasks.",
    "Produce the same three-task plan, but use the output structure below.",
).replace(
    "Do not solve the question in the plan\nand do not output analysis, Markdown, or extra text.",
    """Do not solve the question in the plan.

PLANNING_REASONING
Briefly explain why the three independent perspectives cover the question.
Do not solve the question and do not place JSON in this section.
END_PLANNING_REASONING

PLAN_JSON
[
  {"id": 1, "task": "...", "agent": "knowledge_agent", "reason": "...", "dep": []},
  {"id": 2, "task": "...", "agent": "reasoning_agent", "reason": "...", "dep": []},
  {"id": 3, "task": "...", "agent": "elimination_agent", "reason": "...", "dep": []}
]
END_PLAN_JSON""",
)

CONFIG = {
    "input": "benchmarks/mmlu_pro/mmlu_pro_sampled.json",
    "plans_output": "benchmarks/mmlu_pro/mmlu_pro_plans_full_llada.json",
    "benchmark_output": "benchmarks/mmlu_pro/mmlu_pro_subtask_full_llada.json",
    "planner_api_url": "http://10.137.144.97:7004/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "retry_missing_reasoning": False,
}


def request_completion(query, config):
    headers = {"Content-Type": "application/json", **auth_header(config["planner_api_key"])}
    payload = {
        "model": config["planner_model"],
        "messages": [
            {"role": "system", "content": FULL_PLANNER_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": config["planner_temperature"],
        "max_tokens": config["planner_max_tokens"],
    }
    response = requests.post(
        chat_completions_url(config["planner_api_url"]),
        headers=headers,
        json=payload,
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
    end = re.search(r"(?m)^\s*(?:END_PLANNING_REASONING|PLAN_JSON)\s*:?[ \t]*$", tail)
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
    selected = queries[: config["limit"]] if config["limit"] else queries
    for query in selected:
        if query["source_index"] in done:
            continue
        raw = None
        started = time.perf_counter()
        record = {
            "source": "TIGER-Lab/MMLU-Pro",
            **query,
            "planner_model": config["planner_model"],
            "planner_mode": "reasoning_then_json",
            "planner_prompt_version": FULL_PROMPT_VERSION,
        }
        try:
            raw = request_completion(query["query"], config)
            reasoning = extract_reasoning(raw)
            record.update(
                {
                    "planning_reasoning": reasoning,
                    "raw_plan": raw,
                    "plan": normalize_plan(extract_json_array(raw)),
                    "format_warnings": [] if reasoning else ["missing planning reasoning"],
                    "error": None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "planning_reasoning": extract_reasoning(raw) if raw else None,
                    "raw_plan": raw,
                    "plan": None,
                    "error": str(exc),
                }
            )
        record["time"] = time.perf_counter() - started
        by_index[query["source_index"]] = record
        save_json(config["plans_output"], list(by_index.values()))
        print(
            f"planned {query['source_index']} | reasoning_chars="
            f"{len(record.get('planning_reasoning') or '')} "
            f"| subtasks={len(record.get('plan') or [])} | error={record['error']}",
            flush=True,
        )
    return list(by_index.values())


def main():
    parser = argparse.ArgumentParser(description="Build MMLU-Pro full plans.")
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument("--planner-max-tokens", type=int, default=CONFIG["planner_max_tokens"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    args = parser.parse_args()
    config = dict(CONFIG)
    config.update(vars(args))
    plans = build_plans(load_queries(config["input"]), config)
    save_json(config["plans_output"], plans)
    benchmark = expand_plans(plans, AGENTS)
    save_json(config["benchmark_output"], benchmark)
    print(f"Saved plans: {config['plans_output']} ({len(plans)} queries)")
    print(f"Saved benchmark: {config['benchmark_output']} ({len(benchmark)} rows)")


if __name__ == "__main__":
    main()

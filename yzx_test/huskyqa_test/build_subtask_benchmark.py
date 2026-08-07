import argparse
import json
import os
import re
import time
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import planner_prompt


AGENTS = ["code_agent", "math_agent", "search_agent", "commonsense_agent"]

# Edit these defaults directly when running from this file.
CONFIG = {
    "input": "benchmarks/huskyqa/huskyqa_raw.json",
    "plans_output": "benchmarks/huskyqa/huskyqa_plans_llada.json",
    "benchmark_output": "benchmarks/huskyqa/huskyqa_subtask_llada.json",
    "planner_api_url": "http://10.137.144.97:7004/v1",
    "planner_api_key": "empty",
    #"planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "timeout": 120,
    "limit": None,
    "agents": AGENTS,
}


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for index, row in enumerate(data):
        queries.append(
            {
                "source_index": row.get("index", index),
                "query": row.get("question") or row.get("query"),
                "answer": row.get("answer"),
            }
        )
    return queries


def extract_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in planner output:\n{text}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[start:])
    if not isinstance(value, list):
        raise ValueError(f"Planner output JSON is not a list:\n{text}")
    return value


def normalize_plan(plan):
    normalized = []
    for i, step in enumerate(plan, start=1):
        item = dict(step)
        item.setdefault("id", i)
        if "agent" not in item:
            item["agent"] = item.get("name") or item.get("name_1")
        item.setdefault("dep", [])
        normalized.append(item)
    return normalized


def build_plans(queries, api_url, api_key, model, temperature, timeout, limit=None, resume=None):
    existing_by_index = {}
    done = set()
    if resume and Path(resume).exists():
        with open(resume, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_by_index = {item["source_index"]: item for item in existing}
        done = {idx for idx, item in existing_by_index.items() if item.get("error") is None}

    plans_by_index = dict(existing_by_index)
    selected = queries[:limit] if limit else queries
    for row in selected:
        if row["source_index"] in done:
            continue
        started = time.time()
        record = {
            "source": "agent-husky/HuskyQA",
            "source_index": row["source_index"],
            "query": row["query"],
            "answer": row.get("answer"),
            "planner_model": model,
        }
        try:
            raw_output = run_chat_completion(
                model=model,
                prompt=row["query"],
                api_url=api_url,
                api_key=api_key,
                timeout=timeout,
                temperature=temperature,
                system_prompt=planner_prompt,
            )
            plan = normalize_plan(extract_json_array(raw_output))
            record.update({"raw_plan": raw_output, "plan": plan, "error": None})
        except Exception as exc:
            record.update({"raw_plan": locals().get("raw_output"), "plan": None, "error": str(exc)})
        record["time"] = time.time() - started
        plans_by_index[row["source_index"]] = record
        print(f"planned {row['source_index']} | subtasks={len(record.get('plan') or [])} | error={record['error']}", flush=True)
    return [plans_by_index[idx] for idx in sorted(plans_by_index)]


def expand_plans(plans, agents):
    benchmark = []
    for plan_record in plans:
        plan = plan_record.get("plan") or []
        for step in plan:
            task = step.get("task")
            if not task:
                continue
            for agent_name in agents:
                benchmark.append(
                    {
                        "agent": agent_name,
                        "query": plan_record["query"],
                        "task": task,
                        "history": "None",
                        "dep": step.get("dep", []),
                        "subtask_id": step.get("id"),
                        "planner_agent": step.get("agent"),
                        "planner_reason": step.get("reason"),
                        "source": plan_record.get("source"),
                        "source_index": plan_record.get("source_index"),
                        "answer": plan_record.get("answer"),
                    }
                )
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Build an offline subtask benchmark with a planner.")
    parser.add_argument("--input", default=CONFIG["input"], help="Raw query JSON file.")
    parser.add_argument("--plans-output", default=CONFIG["plans_output"], help="Where to save planner outputs.")
    parser.add_argument(
        "--benchmark-output",
        default=CONFIG["benchmark_output"],
        help="Where to save expanded subtask-agent benchmark.",
    )
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument("--planner-temperature", type=float, default=CONFIG["planner_temperature"])
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"], help="Optional limit for quick tests.")
    parser.add_argument("--agents", nargs="+", default=CONFIG["agents"], choices=AGENTS)
    args = parser.parse_args()

    if not args.planner_api_url:
        raise ValueError("Missing planner API URL. Pass --planner-api-url or set PLANNER_OPENAI_API_URL.")

    queries = load_queries(args.input)
    plans = build_plans(
        queries,
        args.planner_api_url,
        args.planner_api_key,
        args.planner_model,
        args.planner_temperature,
        args.timeout,
        args.limit,
        args.plans_output,
    )

    plans_path = Path(args.plans_output)
    plans_path.parent.mkdir(parents=True, exist_ok=True)
    with plans_path.open("w", encoding="utf-8") as f:
        json.dump(plans, f, ensure_ascii=False, indent=2)

    benchmark = expand_plans(plans, args.agents)
    benchmark_path = Path(args.benchmark_output)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"Saved plans: {plans_path} ({len(plans)} queries)")
    print(f"Saved benchmark: {benchmark_path} ({len(benchmark)} rows)")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import planner_prompt


AGENTS = ["evidence_agent", "temporal_agent", "verification_agent"]
PLANNER_PROMPT_VERSION = "chronoqa_v1"

# Edit these defaults directly before running.
CONFIG = {
    "input": "benchmarks/chronoqa/chronoqa_sampled.json",
    "plans_output": "benchmarks/chronoqa/chronoqa_plans_llada.json",
    "benchmark_output": "benchmarks/chronoqa/chronoqa_subtask_llada.json",
    "planner_api_url": "http://10.137.144.97:7005/v1",
    "planner_api_key": "empty",
    #"planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "timeout": 120,
    "limit": None,
    "agents": AGENTS,
}


def load_queries(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [
        {
            "source_index": row.get("sample_index", index),
            "question_id": row.get("question_id"),
            "query": row["query"],
            "question": row["question"],
            "answer": row["answer"],
            "question_date": row["question_date"],
            "temporal_type": row["temporal_type"],
            "temporal_expression_type": row["temporal_expression_type"],
            "temporal_scope": row["temporal_scope"],
            "temporal_granularity": row["temporal_granularity"],
            "answer_type": row["answer_type"],
            "reference_document_count": row["reference_document_count"],
            "golden_chunks": row["golden_chunks"],
            "golden_chunks_urls": row.get("golden_chunks_urls", []),
        }
        for index, row in enumerate(data)
    ]


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def extract_json_array(text):
    value = text.strip()
    if value.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL)
        if match:
            value = match.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", value):
        try:
            plan, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(plan, list):
            return plan
    raise ValueError(f"No JSON plan array found in planner output:\n{value}")


def normalize_plan(plan):
    if len(plan) != len(AGENTS):
        raise ValueError(f"Plan must contain exactly three steps, got {len(plan)}")

    normalized = []
    roles = []
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Plan step {index} is not a JSON object")
        item = dict(step)
        item["id"] = index
        agent_name = item.get("agent") or item.get("name") or item.get("name_1")
        if isinstance(agent_name, str):
            agent_name = agent_name.strip().lower()
        if agent_name not in AGENTS:
            raise ValueError(
                f"Plan step {index} has unsupported agent {agent_name!r}; "
                f"expected one of {AGENTS}"
            )
        task = item.get("task")
        reason = item.get("reason")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Plan step {index} has no non-empty task")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Plan step {index} has no non-empty reason")
        if item.get("dep") not in (None, []):
            raise ValueError(
                f"Plan step {index} must be independent and use dep=[], "
                f"got {item.get('dep')!r}"
            )
        item.update(
            {
                "agent": agent_name,
                "task": task.strip(),
                "reason": reason.strip(),
                "dep": [],
            }
        )
        roles.append(agent_name)
        normalized.append(item)

    if set(roles) != set(AGENTS) or len(set(roles)) != len(AGENTS):
        raise ValueError(
            f"Plan must use each role exactly once; got role counts {Counter(roles)}"
        )
    return normalized


def build_plans(queries, config):
    output = Path(config["plans_output"])
    existing_by_index = {}
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        existing_by_index = {item["source_index"]: item for item in existing}
    done = {
        index
        for index, item in existing_by_index.items()
        if item.get("error") is None
        and item.get("plan")
        and item.get("planner_prompt_version") == PLANNER_PROMPT_VERSION
    }
    if existing_by_index:
        print(
            f"resume | loaded={len(existing_by_index)} | completed={len(done)} "
            f"| prompt_version={PLANNER_PROMPT_VERSION}",
            flush=True,
        )

    selected = queries[: config["limit"]] if config["limit"] else queries
    for row in selected:
        if row["source_index"] in done:
            continue
        started = time.time()
        raw_output = None
        record = {
            "source": "czy1999/ChronoQA",
            **row,
            "planner_model": config["planner_model"],
            "planner_prompt_version": PLANNER_PROMPT_VERSION,
        }
        try:
            raw_output = run_chat_completion(
                model=config["planner_model"],
                prompt=row["query"],
                api_url=config["planner_api_url"],
                api_key=config["planner_api_key"],
                timeout=config["timeout"],
                temperature=config["planner_temperature"],
                system_prompt=planner_prompt,
            )
            plan = normalize_plan(extract_json_array(raw_output))
            record.update({"raw_plan": raw_output, "plan": plan, "error": None})
        except Exception as exc:
            record.update({"raw_plan": raw_output, "plan": None, "error": str(exc)})
        record["time"] = time.time() - started
        existing_by_index[row["source_index"]] = record
        save_json(
            output,
            [existing_by_index[index] for index in sorted(existing_by_index)],
        )
        print(
            f"planned {row['source_index']} | temporal_type={row['temporal_type']} "
            f"| subtasks={len(record.get('plan') or [])} | error={record['error']}",
            flush=True,
        )
    return [existing_by_index[index] for index in sorted(existing_by_index)]


def expand_plans(plans, agents):
    rows = []
    for record in plans:
        for step in record.get("plan") or []:
            for agent_name in agents:
                rows.append(
                    {
                        "agent": agent_name,
                        "query": record["query"],
                        "task": step["task"],
                        "history": "None",
                        "dep": [],
                        "subtask_id": step["id"],
                        "planner_agent": step["agent"],
                        "planner_reason": step["reason"],
                        "source": record["source"],
                        "source_index": record["source_index"],
                        "question_id": record["question_id"],
                        "answer": record["answer"],
                        "question_date": record["question_date"],
                        "temporal_type": record["temporal_type"],
                        "temporal_expression_type": record["temporal_expression_type"],
                        "temporal_scope": record["temporal_scope"],
                        "answer_type": record["answer_type"],
                        "reference_document_count": record["reference_document_count"],
                    }
                )
    return rows


def print_summary(plans):
    valid = [record for record in plans if record.get("plan")]
    calls = Counter(
        step["agent"] for record in valid for step in record["plan"]
    )
    temporal_types = Counter(record["temporal_type"] for record in plans)
    print("Planner summary")
    print(f"  records={len(plans)} | valid={len(valid)} | errors={len(plans) - len(valid)}")
    print(f"  agent_calls={dict(calls)}")
    print(f"  temporal_types={dict(sorted(temporal_types.items()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Build three-way parallel ChronoQA plans and role-fit rows."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument(
        "--planner-temperature", type=float, default=CONFIG["planner_temperature"]
    )
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=CONFIG["agents"])
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(vars(args))
    plans = build_plans(load_queries(config["input"]), config)
    save_json(config["plans_output"], plans)
    benchmark = expand_plans(plans, config["agents"])
    save_json(config["benchmark_output"], benchmark)
    print(f"Saved plans: {config['plans_output']} ({len(plans)} queries)")
    print(f"Saved benchmark: {config['benchmark_output']} ({len(benchmark)} rows)")
    print_summary(plans)


if __name__ == "__main__":
    main()

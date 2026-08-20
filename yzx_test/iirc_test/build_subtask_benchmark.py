import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import planner_prompt


AGENTS = ["context_agent", "retrieval_agent", "reasoning_agent"]
PLANNER_PROMPT_VERSION = "iirc_compact_v2"

# Edit these defaults directly before running.
CONFIG = {
    "input": "benchmarks/iirc/iirc_dev_flat.json",
    "plans_output": "benchmarks/iirc/iirc_plans_llama3.json",
    "benchmark_output": "benchmarks/iirc/iirc_subtask_llama3.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    #"planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "timeout": 180,
    "limit": None,
    "agents": AGENTS,
}


def load_queries(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    queries = []
    for position, row in enumerate(data):
        query = row.get("question") or row.get("query")
        if not query:
            continue
        queries.append(
            {
                "source_index": row.get("index", position),
                "query": query,
                "answer": row.get("answer"),
                "source": row.get("source", "allenai/IIRC"),
                "planner_input": row.get("agent_context") or query,
                "agent_context": row.get("agent_context") or query,
                "answer_type": row.get("answer_type"),
                "original_answer": row.get("original_answer"),
                "article_pid": row.get("article_pid"),
                "article_title": row.get("article_title"),
                "available_links": row.get("available_links") or [],
                "gold_question_links": row.get("gold_question_links") or [],
                "gold_context": row.get("gold_context") or [],
            }
        )
    return queries


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def extract_json_array(text):
    value = text.strip()
    plan_marker = re.search(r"(?m)^\s*PLAN_JSON\s*:?[ \t]*$", value)
    if plan_marker:
        value = value[plan_marker.end():]
        end_marker = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", value)
        if end_marker:
            value = value[: end_marker.start()]
    if value.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL)
        if fenced:
            value = fenced.group(1).strip()
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
    if not plan:
        raise ValueError("Plan must contain at least one step")

    prepared = []
    original_ids = {}
    for position, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Plan step {position} is not a JSON object")
        item = dict(step)
        original_id = item.get("id", position)
        original_key = str(original_id)
        if original_key in original_ids:
            raise ValueError(f"Plan has duplicate step id {original_id!r}")
        original_ids[original_key] = position

        agent = item.get("agent") or item.get("name") or item.get("name_1")
        if isinstance(agent, str):
            agent = agent.strip().lower()
        if agent not in AGENTS:
            raise ValueError(
                f"Plan step {position} has unsupported agent {agent!r}; "
                f"expected one of {AGENTS}"
            )
        task = item.get("task")
        reason = item.get("reason")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Plan step {position} has no non-empty task")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Plan step {position} has no non-empty reason")
        item.update(
            {
                "agent": agent,
                "id": position,
                "task": task.strip(),
                "reason": reason.strip(),
            }
        )
        prepared.append(item)

    normalized = []
    for position, item in enumerate(prepared, start=1):
        dependencies = item.get("dep")
        if dependencies is None:
            dependencies = []
        elif not isinstance(dependencies, list):
            dependencies = [dependencies]

        normalized_dependencies = []
        for dependency in dependencies:
            if isinstance(dependency, dict):
                dependency = dependency.get("id")
            dependency_key = str(dependency)
            if dependency_key not in original_ids:
                raise ValueError(
                    f"Plan step {position} references missing dependency "
                    f"{dependency!r}"
                )
            normalized_dependency = original_ids[dependency_key]
            if normalized_dependency >= position:
                raise ValueError(
                    f"Plan step {position} dependency {dependency!r} must refer "
                    "to an earlier step"
                )
            if normalized_dependency not in normalized_dependencies:
                normalized_dependencies.append(normalized_dependency)
        item["dep"] = normalized_dependencies
        normalized.append(item)

    return normalized


def ordered_records(records_by_index, queries):
    order = [row["source_index"] for row in queries]
    return [records_by_index[index] for index in order if index in records_by_index]


def build_plans(queries, config):
    output = Path(config["plans_output"])
    existing = []
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    by_index = {row["source_index"]: row for row in existing}
    done = {
        key for key, row in by_index.items()
        if row.get("error") is None
        and row.get("plan")
        and row.get("planner_prompt_version") == PLANNER_PROMPT_VERSION
    }
    if existing:
        print(
            f"resume | loaded={len(existing)} | completed={len(done)} "
            f"| prompt_version={PLANNER_PROMPT_VERSION}",
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
            "planner_prompt_version": PLANNER_PROMPT_VERSION,
        }
        record.pop("planner_input", None)
        try:
            raw_output = run_chat_completion(
                model=config["planner_model"],
                prompt=row["planner_input"],
                api_url=config["planner_api_url"],
                api_key=config["planner_api_key"],
                timeout=config["timeout"],
                temperature=config["planner_temperature"],
                system_prompt=planner_prompt,
            )
            plan = normalize_plan(extract_json_array(raw_output))
            record.update(
                {
                    "raw_plan": raw_output,
                    "plan": plan,
                    "plan_call_count": len(plan),
                    "exceeds_recommended_calls": len(plan) > 5,
                    "error": None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "raw_plan": raw_output,
                    "plan": None,
                    "plan_call_count": None,
                    "exceeds_recommended_calls": False,
                    "error": str(exc),
                }
            )
        record["time"] = time.perf_counter() - started
        by_index[row["source_index"]] = record
        save_json(output, ordered_records(by_index, queries))
        print(
            f"planned {row['source_index']} | subtasks={len(record.get('plan') or [])} "
            f"| error={record['error']}",
            flush=True,
        )
    return ordered_records(by_index, queries)


def expand_plans(plans, agents):
    benchmark = []
    metadata_fields = (
        "source", "source_index", "answer", "agent_context", "answer_type",
        "original_answer", "article_pid", "article_title", "available_links",
        "gold_question_links", "gold_context",
    )
    for plan_record in plans:
        for step in plan_record.get("plan") or []:
            for agent in agents:
                benchmark.append(
                    {
                        "agent": agent,
                        "query": plan_record["query"],
                        "task": step["task"],
                        "history": "None",
                        "dep": step["dep"],
                        "subtask_id": step["id"],
                        "planner_agent": step["agent"],
                        "planner_reason": step["reason"],
                        **{
                            field: plan_record.get(field)
                            for field in metadata_fields
                        },
                    }
                )
    return benchmark


def print_summary(plans):
    valid = [row for row in plans if row.get("plan")]
    calls = Counter(
        step["agent"] for row in valid for step in row["plan"]
    )
    answer_types = Counter(row.get("answer_type") for row in plans)
    call_distribution = Counter(len(row["plan"]) for row in valid)
    over_five = sum(len(row["plan"]) > 5 for row in valid)
    print("Planner summary")
    print(f"  records={len(plans)} | valid={len(valid)} | errors={len(plans) - len(valid)}")
    print(f"  agent_calls={dict(calls)}")
    print(f"  calls_per_query={dict(sorted(call_distribution.items()))}")
    print(f"  plans_over_5_calls={over_five}")
    print(f"  answer_types={dict(sorted(answer_types.items(), key=lambda item: str(item[0])))}")


def main():
    parser = argparse.ArgumentParser(
        description="Build compact plans using three IIRC agent roles."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument("--planner-temperature", type=float, default=CONFIG["planner_temperature"])
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

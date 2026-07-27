import argparse
import json
import time
from pathlib import Path

from evaluate_agent_fit import (
    build_prompt,
    format_for_scorer,
    parse_scores,
    summarize,
)
from openai_compat import run_chat_completion
from prompt import scorer_prompt

# "model": "/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct",
# "api_url": "http://10.137.144.97:7001/v1",

# "model": "/data/labshare/Param/gemma-3-4b-it",
# "api_url": "http://10.137.144.97:7002/v1",

# "model": "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
# "api_url": "http://10.137.144.97:7003/v1",

AGENT_CONFIG = {
    "code_agent": {
        "model": "/data/labshare/Param/gemma-3-4b-it",
        "api_url": "http://10.137.144.97:7002/v1",
        "api_key": "empty",
        "temperature": 0.0,
        "timeout": 120,
    },
    "math_agent": {
        "model": "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
        "api_url": "http://10.137.144.97:7003/v1",
        "api_key": "empty",
        "temperature": 0.0,
        "timeout": 120,
    },
    "search_agent": {
        "model": "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
        "api_url": "http://10.137.144.97:7003/v1",
        "api_key": "empty",
        "temperature": 0.0,
        "timeout": 120,
    },
    "commonsense_agent": {
        "model": "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
        "api_url": "http://10.137.144.97:7003/v1",
        "api_key": "empty",
        "temperature": 0.0,
        "timeout": 120,
    },
}


# Edit these defaults directly before running the script.
CONFIG = {
    "mode": "respond",
    "plans": "benchmarks/iirc/iirc_plans_llama3.json",
    "responses": "iirc_test/results/subtask_hetro_responses_g_q_q_q.json",
    "output": "iirc_test/results/subtask_hetro_scores_g_q_q_q.json",
    "limit": None,
    "force": False,
    "retry_errors": True,
    "search_backend": "iirc_sqlite",
    "search_top_k": 5,
    "iirc_sqlite_path": "benchmarks/iirc/context_articles.sqlite3",
    "judge_api_url": "http://10.137.144.97:7001/v1",
    "judge_api_key": "empty",
    "judge_model": "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "judge_temperature": 0.0,
    "judge_timeout": 120,
}


def load_json(path, default=None):
    if not path or not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temporary.replace(output)


def normalize_step_id(value):
    return str(value)


def ordered_records(records_by_index, plans):
    order = [record.get("source_index") for record in plans]
    seen = set()
    records = []
    for source_index in order:
        if source_index in records_by_index and source_index not in seen:
            records.append(records_by_index[source_index])
            seen.add(source_index)
    records.extend(record for key, record in records_by_index.items() if key not in seen)
    return records


def answer_for_history(step_record):
    return step_record.get("response") or step_record.get("raw_output") or ""


def build_history(step, completed_by_id):
    dependencies = step.get("dep") or []
    if not dependencies:
        return "None"

    history = []
    missing = []
    failed = []
    for dependency in dependencies:
        dependency_id = normalize_step_id(dependency)
        record = completed_by_id.get(dependency_id)
        if record is None:
            missing.append(dependency)
        elif record.get("error"):
            failed.append(dependency)
        else:
            history.append(f"Subtask {dependency}: {answer_for_history(record)}")

    if missing or failed:
        details = []
        if missing:
            details.append(f"missing dependencies={missing}")
        if failed:
            details.append(f"failed dependencies={failed}")
        raise RuntimeError(", ".join(details))
    return "\n".join(history) or "None"


def validate_agent_config(agent_name):
    if agent_name not in AGENT_CONFIG:
        raise ValueError(f"No AGENT_CONFIG entry for {agent_name!r}")
    config = AGENT_CONFIG[agent_name]
    for key in ["model", "api_url"]:
        if not config.get(key):
            raise ValueError(f"Missing AGENT_CONFIG[{agent_name!r}][{key!r}]")
    return config


def execute_step(plan_record, step, history):
    agent_name = step.get("agent") or step.get("name") or step.get("name_1")
    agent_config = validate_agent_config(agent_name)
    task = step.get("task")
    if not task:
        raise ValueError("Plan step has no task")

    prompt = build_prompt(
        agent_name,
        plan_record["query"],
        task,
        history,
        plan_record.get("agent_context"),
    )
    raw_output = run_chat_completion(
        agent_config["model"],
        prompt,
        agent_config["api_url"],
        agent_config.get("api_key", ""),
        agent_config.get("timeout", 120),
        agent_config.get("temperature", 0.0),
    )
    scorer_response, metadata = format_for_scorer(
        agent_name,
        raw_output,
        task,
        agent_config["model"],
        agent_config["api_url"],
        agent_config.get("api_key", ""),
        agent_config.get("timeout", 120),
        agent_config.get("temperature", 0.0),
        search_backend=CONFIG["search_backend"],
        search_top_k=CONFIG["search_top_k"],
        iirc_sqlite_path=CONFIG["iirc_sqlite_path"],
    )
    response = metadata.get("rewritten_response") or scorer_response or raw_output
    return {
        "id": step.get("id"),
        "task": task,
        "agent": agent_name,
        "reason": step.get("reason"),
        "dep": step.get("dep") or [],
        "model": agent_config["model"],
        "api_url": agent_config["api_url"],
        "history": history,
        "raw_output": raw_output,
        "response": response,
        "scorer_response": scorer_response,
        "metadata": metadata,
        "error": None,
    }


def base_response_record(plan_record):
    return {
        "source": plan_record.get("source"),
        "source_index": plan_record.get("source_index"),
        "query": plan_record.get("query"),
        "answer": plan_record.get("answer"),
        "agent_context": plan_record.get("agent_context"),
        "answer_type": plan_record.get("answer_type"),
        "original_answer": plan_record.get("original_answer"),
        "article_pid": plan_record.get("article_pid"),
        "article_title": plan_record.get("article_title"),
        "available_links": plan_record.get("available_links") or [],
        "gold_question_links": plan_record.get("gold_question_links") or [],
        "gold_context": plan_record.get("gold_context") or [],
        "planner_model": plan_record.get("planner_model"),
        "steps": [],
        "error": None,
    }


def execute_plans(plans, responses_path, limit=None, force=False, retry_errors=True):
    existing = [] if force else load_json(responses_path, []) or []
    records_by_index = {record.get("source_index"): record for record in existing}
    selected = plans[:limit] if limit else plans

    for plan_record in selected:
        source_index = plan_record.get("source_index")
        response_record = records_by_index.get(source_index) or base_response_record(plan_record)
        records_by_index[source_index] = response_record

        if plan_record.get("error") or not plan_record.get("plan"):
            response_record["error"] = plan_record.get("error") or "planner returned no steps"
            save_json(responses_path, ordered_records(records_by_index, plans))
            continue

        step_records = {
            normalize_step_id(record.get("id")): record
            for record in response_record.get("steps", [])
        }

        for step in plan_record["plan"]:
            step_id = normalize_step_id(step.get("id"))
            previous = step_records.get(step_id)
            if previous and previous.get("error") is None and previous.get("response") and not force:
                continue
            if previous and previous.get("error") and not retry_errors and not force:
                continue

            started = time.time()
            try:
                history = build_history(step, step_records)
                step_record = execute_step(plan_record, step, history)
            except Exception as exc:
                agent_name = step.get("agent") or step.get("name") or step.get("name_1")
                agent_config = AGENT_CONFIG.get(agent_name, {})
                step_record = {
                    "id": step.get("id"),
                    "task": step.get("task"),
                    "agent": agent_name,
                    "reason": step.get("reason"),
                    "dep": step.get("dep") or [],
                    "model": agent_config.get("model"),
                    "api_url": agent_config.get("api_url"),
                    "history": None,
                    "raw_output": None,
                    "response": None,
                    "scorer_response": None,
                    "metadata": {},
                    "error": str(exc),
                }
            step_record["time"] = time.time() - started
            step_records[step_id] = step_record
            response_record["steps"] = [
                step_records[normalize_step_id(item.get("id"))]
                for item in plan_record["plan"]
                if normalize_step_id(item.get("id")) in step_records
            ]
            response_record["error"] = (
                "one or more subtask executions failed"
                if any(record.get("error") for record in response_record["steps"])
                else None
            )
            save_json(responses_path, ordered_records(records_by_index, plans))
            print(
                f"respond source={source_index} | step={step.get('id')} | agent={step_record.get('agent')} "
                f"| model={step_record.get('model')} | error={step_record.get('error')}",
                flush=True,
            )

    return ordered_records(records_by_index, plans)


def flatten_steps(response_records):
    rows = []
    for record in response_records:
        for step in record.get("steps", []):
            row = dict(step)
            row.update(
                {
                    "source": record.get("source"),
                    "source_index": record.get("source_index"),
                    "query": record.get("query"),
                    "answer": record.get("answer"),
                    "agent_context": record.get("agent_context"),
                    "answer_type": record.get("answer_type"),
                    "original_answer": record.get("original_answer"),
                    "article_pid": record.get("article_pid"),
                    "article_title": record.get("article_title"),
                    "available_links": record.get("available_links") or [],
                    "gold_question_links": record.get("gold_question_links") or [],
                    "gold_context": record.get("gold_context") or [],
                    "subtask_id": step.get("id"),
                }
            )
            rows.append(row)
    return rows


def judge_key(row):
    return (
        row.get("source_index"),
        normalize_step_id(row.get("subtask_id")),
        row.get("agent"),
        row.get("model"),
    )


def judge_rows(response_records, output_path, force=False):
    rows = flatten_steps(response_records)
    existing_output = {} if force else load_json(output_path, {}) or {}
    existing_by_key = {judge_key(row): row for row in existing_output.get("rows", [])}
    judged = []

    for row in rows:
        previous = existing_by_key.get(judge_key(row))
        if (
            previous
            and previous.get("judge_output")
            and previous.get("scores")
            and previous.get("scorer_response") == row.get("scorer_response")
        ):
            judged.append(previous)
            continue

        record = dict(row)
        started = time.time()
        if record.get("error") or not record.get("scorer_response"):
            record.update(
                {
                    "judge_output": None,
                    "scores": {"correctness": 0, "relevance": 0, "completeness": 0, "total": 0},
                    "judge_error": record.get("error") or "missing scorer_response",
                    "judge_time": 0,
                }
            )
        else:
            try:
                prompt = scorer_prompt % (record["agent"], record["task"], record["scorer_response"])
                judge_output = run_chat_completion(
                    CONFIG["judge_model"],
                    prompt,
                    CONFIG["judge_api_url"],
                    CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"],
                    CONFIG["judge_temperature"],
                )
                record.update(
                    {
                        "judge_output": judge_output,
                        "scores": parse_scores(judge_output),
                        "judge_error": None,
                        "judge_time": time.time() - started,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "judge_output": None,
                        "scores": {"correctness": 0, "relevance": 0, "completeness": 0, "total": 0},
                        "judge_error": str(exc),
                        "judge_time": time.time() - started,
                    }
                )

        judged.append(record)
        result = {"rows": judged, "summary": summarize(judged)}
        save_json(output_path, result)
        print(
            f"judge source={record.get('source_index')} | step={record.get('subtask_id')} "
            f"| agent={record.get('agent')} | score={record['scores']['total']} "
            f"| error={record.get('judge_error')}",
            flush=True,
        )

    result = {"rows": judged, "summary": summarize(judged)}
    save_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Execute planner-selected subtasks with heterogeneous agent APIs, then judge offline results."
    )
    parser.add_argument("--mode", choices=["respond", "judge", "all"], default=CONFIG["mode"])
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    plans = load_json(args.plans, []) or []
    if not plans:
        raise ValueError(f"No planner records found in {args.plans}")

    if args.mode in {"respond", "all"}:
        response_records = execute_plans(
            plans,
            args.responses,
            args.limit,
            args.force,
            CONFIG["retry_errors"],
        )
        print(f"Saved heterogeneous subtask responses to {args.responses}", flush=True)
    else:
        response_records = load_json(args.responses, []) or []

    if args.mode in {"judge", "all"}:
        if not response_records:
            raise ValueError(f"No response records found in {args.responses}")
        result = judge_rows(response_records, args.output, args.force)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved judged results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

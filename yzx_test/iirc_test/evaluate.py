import argparse
import json
import time
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import plan_detector_prompt


# Edit these values directly before running.
CONFIG = {
    "plans": "benchmarks/iirc/iirc_plans_parallel_llama3.json",
    "query": None,
    "source_index": None,
    "limit": None,
    "output": "iirc_test/results/plan_evaluate.json",
    "force": False,
    "judge_api_url": "http://10.137.144.97:7001/v1",
    "judge_api_key": "empty",
    "judge_model": "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "judge_temperature": 0.0,
    "judge_timeout": 120,
}


def load_json(path, default=None):
    if not path or not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(output)


def normalize_plan(plan):
    normalized = []
    for position, step in enumerate(plan or [], start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Plan step {position} is not an object")
        item = dict(step)
        item.setdefault("id", position)
        item["agent"] = item.get("agent") or item.get("name") or item.get("name_1")
        item["dep"] = item.get("dep") or []
        if not item.get("task"):
            raise ValueError(f"Plan step {position} has no task")
        normalized.append(item)
    return normalized


def is_bare_plan(data):
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(item, dict) and "task" in item for item in data)
    )


def load_plan_records(path, query=None):
    data = load_json(path)
    if data is None:
        raise ValueError(f"Plan file does not exist: {path}")

    if is_bare_plan(data):
        if not query:
            raise ValueError("A bare plan JSON requires CONFIG['query'] or --query.")
        records = [{"source_index": 0, "query": query, "plan": data}]
    elif isinstance(data, dict) and "plan" in data:
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Plan JSON must contain a plan list or planner records.")

    normalized = []
    for position, record in enumerate(records):
        item = dict(record)
        item.setdefault("source_index", position)
        item["query"] = item.get("query") or item.get("question")
        if query and item.get("query") != query:
            continue
        if not item.get("query"):
            raise ValueError(f"Plan record {position} has no query")
        item["plan"] = normalize_plan(item.get("plan"))
        normalized.append(item)
    return normalized


def select_records(records, source_index=None, limit=None):
    selected = records
    if source_index is not None:
        selected = [
            record
            for record in selected
            if str(record.get("source_index")) == str(source_index)
        ]
    return selected[:limit] if limit else selected


def record_key(record):
    return str(record.get("source_index")), record.get("query")


def plan_signature(record):
    return json.dumps(record.get("plan") or [], ensure_ascii=False, sort_keys=True)


def format_plan(record):
    task_input = record.get("agent_context") or record["query"]
    lines = [f"Task: {task_input}"]
    for position, step in enumerate(record["plan"], start=1):
        step_id = step.get("id", position)
        lines.append(
            f"Subtask {step_id}: {step.get('task')}    "
            f"Dependency: {step.get('dep') or []}"
        )
    return "\n".join(lines)


def plan_passed(text):
    normalized = " ".join((text or "").lower().split()).rstrip(".")
    expected = "the plan satisfies completeness and non-redundancy"
    return normalized == expected


def summarize(rows):
    count = len(rows)
    pass_count = sum(row.get("plan_pass") is True for row in rows)
    return {
        "count": count,
        "judged_count": sum(row.get("judge_error") is None for row in rows),
        "plan_pass_count": pass_count,
        "plan_pass_rate": pass_count / count if count else 0,
        "judge_failure_count": sum(bool(row.get("judge_error")) for row in rows),
    }


def evaluate_plans(records, output_path, force=False):
    existing_output = {} if force else load_json(output_path, {}) or {}
    selected_keys = {record_key(record) for record in records}
    retained = [
        row
        for row in existing_output.get("rows", [])
        if record_key(row) not in selected_keys
    ]
    existing_by_key = {
        record_key(row): row for row in existing_output.get("rows", [])
    }
    rows = []

    for record in records:
        key = record_key(record)
        signature = plan_signature(record)
        previous = existing_by_key.get(key)
        if (
            previous
            and previous.get("plan_signature") == signature
            and previous.get("detector_output") is not None
            and not force
        ):
            rows.append(previous)
            continue

        result = {
            "source": record.get("source"),
            "source_index": record.get("source_index"),
            "query": record["query"],
            "answer": record.get("answer"),
            "answer_type": record.get("answer_type"),
            "article_pid": record.get("article_pid"),
            "planner_model": record.get("planner_model"),
            "plan": record["plan"],
            "plan_signature": signature,
        }
        started = time.time()
        if not record["plan"]:
            result.update(
                {
                    "detector_output": None,
                    "plan_pass": False,
                    "plan_score": 0,
                    "judge_error": record.get("error") or "empty plan",
                }
            )
        else:
            try:
                prompt = f"{plan_detector_prompt}\n---\n{format_plan(record)}"
                detector_output = run_chat_completion(
                    CONFIG["judge_model"],
                    prompt,
                    CONFIG["judge_api_url"],
                    CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"],
                    CONFIG["judge_temperature"],
                )
                passed = plan_passed(detector_output)
                result.update(
                    {
                        "detector_output": detector_output,
                        "plan_pass": passed,
                        "plan_score": int(passed),
                        "judge_error": None,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "detector_output": None,
                        "plan_pass": False,
                        "plan_score": 0,
                        "judge_error": str(exc),
                    }
                )
        result["judge_time"] = time.time() - started
        rows.append(result)
        all_rows = retained + rows
        output = {"rows": all_rows, "summary": summarize(all_rows)}
        save_json(output_path, output)
        print(
            f"evaluate source={result.get('source_index')} "
            f"| pass={result['plan_pass']} | error={result['judge_error']}",
            flush=True,
        )

    all_rows = retained + rows
    output = {"rows": all_rows, "summary": summarize(all_rows)}
    save_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate plan completeness and non-redundancy with the original plan detector prompt."
    )
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--query", default=CONFIG["query"])
    parser.add_argument("--source-index", default=CONFIG["source_index"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    records = load_plan_records(args.plans, args.query)
    records = select_records(records, args.source_index, args.limit)
    if not records:
        raise ValueError("No plan records matched the selected query/source index.")
    result = evaluate_plans(records, args.output, args.force)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved plan evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()

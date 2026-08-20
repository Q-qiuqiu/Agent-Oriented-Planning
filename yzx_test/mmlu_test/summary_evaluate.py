import argparse
import json
import time
from pathlib import Path

from answer_utils import accuracy_summary, extract_answer_choice
from openai_compat import run_chat_completion
from prompt import summarization_agent_prompt


# Keep these values aligned with subtask_hetro.py.
MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "q_qm_m"
PLAN_VARIANT = "llada"
RESULTS_DIR = f"mmlu_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"

CONFIG = {
    "responses": f"{RESULTS_DIR}/subtask_hetro_responses_{AGENT_ASSIGNMENT}.json",
    "output": f"{RESULTS_DIR}/summary_result_{AGENT_ASSIGNMENT}.json",
    "query": None,
    "source_index": None,
    "limit": None,
    "force": False,
    "retry_errors": True,
    "summary_api_url": "http://10.137.144.97:7007/v1",
    "summary_api_key": "empty",
    "summary_model": "/data/labshare/Param/llada",
    "summary_temperature": 0.0,
    "summary_timeout": 120,
}


def load_json(path, default=None):
    if not path or not Path(path).exists():
        return default
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(output)


def record_key(record):
    return str(record.get("source_index"))


def response_signature(record):
    return json.dumps(
        [
            {
                "id": step.get("id"),
                "agent": step.get("agent"),
                "response": step.get("response"),
                "error": step.get("error"),
            }
            for step in record.get("steps", [])
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def build_summary_prompt(record):
    plan = [
        {
            "id": step.get("id"),
            "agent": step.get("agent"),
            "task": step.get("task"),
            "reason": step.get("reason"),
            "dep": step.get("dep") or [],
        }
        for step in record.get("steps", [])
    ]
    responses = [
        {
            "id": step.get("id"),
            "agent": step.get("agent"),
            "response": step.get("response"),
        }
        for step in record.get("steps", [])
    ]
    return summarization_agent_prompt % (
        record["query"],
        json.dumps(plan, ensure_ascii=False, indent=2),
        json.dumps(responses, ensure_ascii=False, indent=2),
    )


def select_records(records, query=None, source_index=None, limit=None):
    if query:
        records = [row for row in records if row.get("query") == query]
    if source_index is not None:
        records = [
            row for row in records
            if str(row.get("source_index")) == str(source_index)
        ]
    return records[:limit] if limit else records


def summarize(records, output_path, force=False, retry_errors=True):
    existing = [] if force else load_json(output_path, []) or []
    by_key = {record_key(row): row for row in existing}

    for record in records:
        key = record_key(record)
        signature = response_signature(record)
        previous = by_key.get(key)
        if (
            previous
            and previous.get("response_signature") == signature
            and previous.get("summary_error") is None
            and previous.get("final_answer")
            and not force
        ):
            continue
        if previous and previous.get("summary_error") and not retry_errors and not force:
            continue

        result = {
            "source": record.get("source"),
            "source_index": record.get("source_index"),
            "question_id": record.get("question_id"),
            "category": record.get("category"),
            "query": record.get("query"),
            "question": record.get("question"),
            "options": record.get("options"),
            "answer": record.get("answer"),
            "answer_index": record.get("answer_index"),
            "planner_model": record.get("planner_model"),
            "response_signature": signature,
            "subtasks": record.get("steps", []),
        }
        started = time.perf_counter()
        if record.get("error") or len(record.get("steps", [])) != 3:
            result.update(
                {
                    "final_answer": None,
                    "predicted_answer": None,
                    "correct": False,
                    "summary_error": record.get("error") or "expected three responses",
                }
            )
        else:
            try:
                final_answer = run_chat_completion(
                    CONFIG["summary_model"],
                    build_summary_prompt(record),
                    CONFIG["summary_api_url"],
                    CONFIG["summary_api_key"],
                    CONFIG["summary_timeout"],
                    CONFIG["summary_temperature"],
                )
                prediction = extract_answer_choice(final_answer)
                result.update(
                    {
                        "final_answer": final_answer,
                        "predicted_answer": prediction,
                        "correct": prediction == record.get("answer"),
                        "summary_error": None,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "final_answer": None,
                        "predicted_answer": None,
                        "correct": False,
                        "summary_error": str(exc),
                    }
                )
        result["summary_time"] = time.perf_counter() - started
        by_key[key] = result
        save_json(output_path, list(by_key.values()))
        print(
            f"summary source={result['source_index']} "
            f"| prediction={result.get('predicted_answer')} "
            f"| error={result['summary_error']}",
            flush=True,
        )
    output = list(by_key.values())
    save_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Summarize three independent MMLU-Pro agent responses."
    )
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--query", default=CONFIG["query"])
    parser.add_argument("--source-index", default=CONFIG["source_index"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    records = select_records(
        load_json(args.responses, []) or [], args.query, args.source_index, args.limit
    )
    if not records:
        raise ValueError("No subtask response records matched the selection")
    output = summarize(records, args.output, args.force, CONFIG["retry_errors"])
    print(json.dumps(accuracy_summary(output), ensure_ascii=False, indent=2))
    print(f"Saved {len(output)} summaries to {args.output}")


if __name__ == "__main__":
    main()

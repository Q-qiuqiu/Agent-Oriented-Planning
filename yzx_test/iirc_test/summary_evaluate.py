import argparse
import json
import time
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import summarization_agent_prompt


MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "f_q_m"
PLAN_VARIANT = "llada"
RESULTS_DIR = f"iirc_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"

# Keep these values aligned with subtask_hetro.py.
CONFIG = {
    "responses": f"{RESULTS_DIR}/subtask_hetro_responses_{AGENT_ASSIGNMENT}.json",
    "query": None,
    "source_index": None,
    "limit": None,
    "output": f"{RESULTS_DIR}/summary_result_{AGENT_ASSIGNMENT}.json",
    "force": False,
    "summary_api_url": "http://10.137.144.97:7007/v1",
    "summary_api_key": "empty",
    #"summary_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "summary_model": "/data/labshare/Param/llada",
    "summary_temperature": 0.0,
    "summary_timeout": 180,
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


def record_key(record):
    return str(record.get("source_index")), record.get("query")


def select_records(records, query=None, source_index=None, limit=None):
    selected = records
    if query:
        selected = [record for record in selected if record.get("query") == query]
    if source_index is not None:
        selected = [
            record
            for record in selected
            if str(record.get("source_index")) == str(source_index)
        ]
    return selected[:limit] if limit else selected


def response_signature(record):
    content = [
        {
            "id": step.get("id"),
            "task": step.get("task"),
            "agent": step.get("agent"),
            "response": step.get("response"),
            "error": step.get("error"),
        }
        for step in record.get("steps", [])
    ]
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def build_summary_prompt(record):
    tasks = [step.get("task") for step in record.get("steps", [])]
    responses = [
        step.get("response")
        if step.get("response") is not None
        else f"[ERROR] {step.get('error') or 'missing response'}"
        for step in record.get("steps", [])
    ]
    original_input = record.get("agent_context") or record["query"]
    return summarization_agent_prompt % (original_input, tasks, responses)


def summarize_responses(records, output_path, force=False):
    existing = [] if force else load_json(output_path, []) or []
    existing_by_key = {record_key(row): row for row in existing}
    selected_keys = {record_key(record) for record in records}
    retained = [row for row in existing if record_key(row) not in selected_keys]
    results = []

    for record in records:
        if not record.get("query"):
            raise ValueError(f"Response record {record.get('source_index')} has no query")
        signature = response_signature(record)
        previous = existing_by_key.get(record_key(record))
        if (
            previous
            and previous.get("response_signature") == signature
            and previous.get("final_answer") is not None
            and previous.get("summary_error") is None
            and not force
        ):
            results.append(previous)
            continue

        result = {
            "source": record.get("source"),
            "source_index": record.get("source_index"),
            "query": record["query"],
            "answer": record.get("answer"),
            "agent_context": record.get("agent_context"),
            "answer_type": record.get("answer_type"),
            "original_answer": record.get("original_answer"),
            "article_pid": record.get("article_pid"),
            "article_title": record.get("article_title"),
            "available_links": record.get("available_links") or [],
            "gold_question_links": record.get("gold_question_links") or [],
            "gold_context": record.get("gold_context") or [],
            "planner_model": record.get("planner_model"),
            "response_signature": signature,
            "subtasks": [
                {
                    "id": step.get("id"),
                    "task": step.get("task"),
                    "agent": step.get("agent"),
                    "dep": step.get("dep") or [],
                    "response": step.get("response"),
                    "error": step.get("error"),
                }
                for step in record.get("steps", [])
            ],
        }
        started = time.time()
        if record.get("error") or not record.get("steps"):
            result.update(
                {
                    "final_answer": None,
                    "summary_error": record.get("error") or "no subtask responses",
                }
            )
        else:
            try:
                result["final_answer"] = run_chat_completion(
                    CONFIG["summary_model"],
                    build_summary_prompt(record),
                    CONFIG["summary_api_url"],
                    CONFIG["summary_api_key"],
                    CONFIG["summary_timeout"],
                    CONFIG["summary_temperature"],
                )
                result["summary_error"] = None
            except Exception as exc:
                result["final_answer"] = None
                result["summary_error"] = str(exc)
        result["summary_time"] = time.time() - started
        results.append(result)
        save_json(output_path, retained + results)
        print(
            f"summary source={result.get('source_index')} "
            f"| error={result.get('summary_error')}",
            flush=True,
        )

    output = retained + results
    save_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Summarize saved heterogeneous sub-agent responses into final answers."
    )
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--query", default=CONFIG["query"])
    parser.add_argument("--source-index", default=CONFIG["source_index"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    records = load_json(args.responses, []) or []
    records = select_records(records, args.query, args.source_index, args.limit)
    if not records:
        raise ValueError("No subtask response records matched the selection.")
    output = summarize_responses(records, args.output, args.force)
    print(f"Saved {len(output)} summarized answers to {args.output}", flush=True)


if __name__ == "__main__":
    main()

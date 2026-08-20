import argparse
import json
import re
import time
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import evaluate_prompt


# Keep these values aligned with subtask_hetro.py and summary_evaluate.py.
MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "f_q_m"
PLAN_VARIANT = "llada"
RESULTS_DIR = f"huskyqa_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"

# Edit judge API settings directly before running.
CONFIG = {
    "input": f"{RESULTS_DIR}/summary_result_{AGENT_ASSIGNMENT}.json",
    "query": None,
    "source_index": None,
    "limit": None,
    "output": f"{RESULTS_DIR}/summary_score_{AGENT_ASSIGNMENT}.json",
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


def parse_yes_no(text):
    match = re.search(r"\b(yes|no)\b", text or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def summarize(rows):
    eligible = sum(bool(row.get("answer")) for row in rows)
    correct = sum(row.get("correct") is True for row in rows)
    return {
        "count": len(rows),
        "eligible_count": eligible,
        "correct_count": correct,
        "accuracy": correct / eligible if eligible else 0,
        "judge_failure_count": sum(bool(row.get("judge_error")) for row in rows),
    }


def score_summaries(records, output_path, force=False):
    existing_output = {} if force else load_json(output_path, {}) or {}
    existing = existing_output.get("rows", [])
    existing_by_key = {record_key(row): row for row in existing}
    selected_keys = {record_key(record) for record in records}
    retained = [row for row in existing if record_key(row) not in selected_keys]
    rows = []

    for record in records:
        key = record_key(record)
        previous = existing_by_key.get(key)
        if (
            previous
            and previous.get("final_answer") == record.get("final_answer")
            and previous.get("judge_output") is not None
            and not force
        ):
            rows.append(previous)
            continue

        result = dict(record)
        started = time.time()
        if not result.get("answer"):
            result.update(
                {
                    "judge_output": None,
                    "correct": None,
                    "score": None,
                    "judge_error": "missing reference answer",
                }
            )
        elif not result.get("final_answer"):
            result.update(
                {
                    "judge_output": None,
                    "correct": False,
                    "score": 0,
                    "judge_error": result.get("summary_error") or "missing final answer",
                }
            )
        else:
            try:
                judge_output = run_chat_completion(
                    CONFIG["judge_model"],
                    evaluate_prompt
                    % (result["query"], result["answer"], result["final_answer"]),
                    CONFIG["judge_api_url"],
                    CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"],
                    CONFIG["judge_temperature"],
                )
                correct = parse_yes_no(judge_output)
                result.update(
                    {
                        "judge_output": judge_output,
                        "correct": correct if correct is not None else False,
                        "score": int(correct is True),
                        "judge_error": None if correct is not None else "judge output has no yes/no",
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "judge_output": None,
                        "correct": False,
                        "score": 0,
                        "judge_error": str(exc),
                    }
                )
        result["judge_time"] = time.time() - started
        rows.append(result)
        all_rows = retained + rows
        output = {"rows": all_rows, "summary": summarize(all_rows)}
        save_json(output_path, output)
        print(
            f"score source={result.get('source_index')} "
            f"| correct={result.get('correct')} | error={result.get('judge_error')}",
            flush=True,
        )

    all_rows = retained + rows
    output = {"rows": all_rows, "summary": summarize(all_rows)}
    save_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Score summarized final answers with the original CompareGPT prompt."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--query", default=CONFIG["query"])
    parser.add_argument("--source-index", default=CONFIG["source_index"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    records = load_json(args.input, []) or []
    records = select_records(records, args.query, args.source_index, args.limit)
    if not records:
        raise ValueError("No summarized records matched the selection.")
    output = score_summaries(records, args.output, args.force)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Saved final-answer evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()

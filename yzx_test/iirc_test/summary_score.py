import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

from openai_compat import run_chat_completion
from prompt import evaluate_prompt


# Edit these values directly before running.
CONFIG = {
    "input": "iirc_test/results/summary_result_parallel_roles.json",
    "query": None,
    "source_index": None,
    "limit": None,
    "output": "iirc_test/results/summary_score_parallel_roles.json",
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


def normalize_answer(text):
    value = str(text or "").lower()
    value = "".join(character if character not in string.punctuation else " " for character in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def is_no_answer(text):
    normalized = normalize_answer(text)
    patterns = [
        "not enough information",
        "insufficient information",
        "cannot be determined",
        "can not be determined",
        "unknown",
        "no answer",
    ]
    return any(pattern in normalized for pattern in patterns)


def lexical_scores(prediction, reference, answer_type=None):
    if answer_type == "none":
        exact_match = float(is_no_answer(prediction))
        return exact_match, exact_match

    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    exact_match = float(prediction_tokens == reference_tokens)
    if not prediction_tokens or not reference_tokens:
        return exact_match, float(prediction_tokens == reference_tokens)

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return exact_match, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return exact_match, 2 * precision * recall / (precision + recall)


def summarize(rows):
    eligible = sum(bool(row.get("answer")) for row in rows)
    correct = sum(row.get("correct") is True for row in rows)
    lexical_rows = [row for row in rows if row.get("answer") is not None]
    return {
        "count": len(rows),
        "eligible_count": eligible,
        "correct_count": correct,
        "accuracy": correct / eligible if eligible else 0,
        "iirc_exact_match": (
            sum(row.get("iirc_exact_match", 0) for row in lexical_rows) / len(lexical_rows)
            if lexical_rows
            else 0
        ),
        "iirc_token_f1": (
            sum(row.get("iirc_token_f1", 0) for row in lexical_rows) / len(lexical_rows)
            if lexical_rows
            else 0
        ),
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
        exact_match, token_f1 = lexical_scores(
            result.get("final_answer"),
            result.get("answer"),
            result.get("answer_type"),
        )
        result["iirc_exact_match"] = exact_match
        result["iirc_token_f1"] = token_f1
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

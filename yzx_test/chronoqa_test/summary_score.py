import argparse
import json
import time
from pathlib import Path

from answer_utils import accuracy_summary, extract_eval_score
from openai_compat import run_chat_completion
from prompt import judge_prompt


MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "s_s_s"
PLAN_VARIANT = "full_llada"
RESULTS_DIR = f"chronoqa_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"
CONFIG = {
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
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(output)


def score(records, output_path, force=False):
    previous = {} if force else load_json(output_path, {}) or {}
    by_key = {str(row.get("source_index")): row for row in previous.get("rows", [])}
    for record in records:
        key = str(record.get("source_index"))
        old = by_key.get(key)
        if (
            old
            and old.get("final_answer") == record.get("final_answer")
            and old.get("eval_score") in (0, 1)
            and old.get("judge_model") == CONFIG["judge_model"]
            and not force
        ):
            continue
        result = dict(record)
        result["judge_model"] = CONFIG["judge_model"]
        result["judge_api_url"] = CONFIG["judge_api_url"]
        started = time.perf_counter()
        if not result.get("answer"):
            result.update({"eval_score": None, "judge_output": None, "judge_error": "missing reference answer"})
        elif not result.get("final_answer"):
            result.update({"eval_score": 0, "judge_output": None, "judge_error": result.get("summary_error") or "missing final answer"})
        else:
            try:
                judge_output = run_chat_completion(
                    CONFIG["judge_model"],
                    judge_prompt % (result["query"], result["answer"], result["final_answer"]),
                    CONFIG["judge_api_url"], CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"], CONFIG["judge_temperature"],
                )
                result.update({"eval_score": extract_eval_score(judge_output), "judge_output": judge_output, "judge_error": None})
            except Exception as exc:
                result.update({"eval_score": None, "judge_output": None, "judge_error": str(exc)})
        result["judge_time"] = time.perf_counter() - started
        by_key[key] = result
        rows = list(by_key.values())
        save_json(output_path, {"rows": rows, "summary": accuracy_summary(rows)})
        print(f"score source={key} | eval_score={result.get('eval_score')} | error={result.get('judge_error')}", flush=True)
    rows = list(by_key.values())
    return {"rows": rows, "summary": accuracy_summary(rows)}


def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser(description="Score ChronoQA final answers with the official binary LLM-judge rule.")
    parser.add_argument("--assignment", default=AGENT_ASSIGNMENT)
    parser.add_argument("--model-size", choices=("1b", "3b"), default=MODEL_SIZE)
    parser.add_argument("--plan-variant", default=PLAN_VARIANT)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    parser.add_argument("--judge-api-url", default=CONFIG["judge_api_url"])
    parser.add_argument("--judge-api-key", default=CONFIG["judge_api_key"])
    parser.add_argument("--judge-model", default=CONFIG["judge_model"])
    parser.add_argument(
        "--judge-temperature", type=float, default=CONFIG["judge_temperature"]
    )
    parser.add_argument("--judge-timeout", type=int, default=CONFIG["judge_timeout"])
    args = parser.parse_args()
    RESULTS_DIR = f"chronoqa_test/results_{args.model_size}_{args.plan_variant}"
    CONFIG.update(
        {
            "judge_api_url": args.judge_api_url,
            "judge_api_key": args.judge_api_key,
            "judge_model": args.judge_model,
            "judge_temperature": args.judge_temperature,
            "judge_timeout": args.judge_timeout,
        }
    )
    args.input = args.input or f"{RESULTS_DIR}/summary_result_{args.assignment}.json"
    args.output = args.output or f"{RESULTS_DIR}/summary_score_{args.assignment}.json"
    records = load_json(args.input, []) or []
    if not records:
        raise ValueError(f"No summary records found in {args.input}")
    result = score(records, args.output, args.force)
    save_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved final-answer evaluation to {args.output}")


if __name__ == "__main__":
    main()

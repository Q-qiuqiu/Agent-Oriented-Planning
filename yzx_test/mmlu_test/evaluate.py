import argparse
import json
import time
from pathlib import Path

from build_subtask_benchmark import AGENTS
from openai_compat import run_chat_completion
from prompt import plan_detector_prompt


CONFIG = {
    "plans": "benchmarks/mmlu_pro/mmlu_pro_plans_llada.json",
    "output": "mmlu_test/results/plan_evaluate_llada.json",
    "source_index": None,
    "limit": None,
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


def structural_violations(record):
    plan = record.get("plan") or []
    violations = []
    if len(plan) != 3:
        violations.append(f"expected 3 steps, got {len(plan)}")
    roles = [step.get("agent") for step in plan if isinstance(step, dict)]
    if sorted(roles) != sorted(AGENTS):
        violations.append(f"expected each role once, got {roles}")
    for position, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            violations.append(f"step {position} is not an object")
            continue
        if not str(step.get("task") or "").strip():
            violations.append(f"step {position} has no task")
        if not str(step.get("reason") or "").strip():
            violations.append(f"step {position} has no reason")
        if step.get("dep") not in (None, []):
            violations.append(f"step {position} is not parallel: dep={step.get('dep')}")
    return violations


def plan_signature(record):
    return json.dumps(record.get("plan") or [], ensure_ascii=False, sort_keys=True)


def judge_prompt(record):
    return (
        f"{plan_detector_prompt}\n\nQuestion:\n{record['query']}\n\n"
        f"Plan:\n{json.dumps(record.get('plan') or [], ensure_ascii=False, indent=2)}"
    )


def plan_passed(text):
    normalized = " ".join((text or "").lower().split()).rstrip(".")
    return normalized == "the plan satisfies completeness and non-redundancy"


def summarize(rows):
    count = len(rows)
    return {
        "count": count,
        "structural_pass_count": sum(row["structural_pass"] for row in rows),
        "structural_pass_rate": (
            sum(row["structural_pass"] for row in rows) / count if count else 0
        ),
        "judge_pass_count": sum(row["judge_pass"] for row in rows),
        "judge_pass_rate": sum(row["judge_pass"] for row in rows) / count if count else 0,
        "judge_failure_count": sum(bool(row.get("judge_error")) for row in rows),
    }


def evaluate(records, output_path, force=False):
    existing = {} if force else load_json(output_path, {}) or {}
    by_key = {str(row.get("source_index")): row for row in existing.get("rows", [])}
    for record in records:
        key = str(record.get("source_index"))
        signature = plan_signature(record)
        previous = by_key.get(key)
        if (
            previous
            and previous.get("plan_signature") == signature
            and previous.get("judge_error") is None
            and not force
        ):
            continue

        violations = structural_violations(record)
        result = {
            "source": record.get("source"),
            "source_index": record.get("source_index"),
            "question_id": record.get("question_id"),
            "category": record.get("category"),
            "query": record.get("query"),
            "planner_model": record.get("planner_model"),
            "plan": record.get("plan"),
            "plan_signature": signature,
            "structural_violations": violations,
            "structural_pass": not violations,
        }
        started = time.perf_counter()
        if violations:
            result.update(
                {
                    "detector_output": None,
                    "judge_pass": False,
                    "judge_error": record.get("error") or "; ".join(violations),
                }
            )
        else:
            try:
                detector_output = run_chat_completion(
                    CONFIG["judge_model"],
                    judge_prompt(record),
                    CONFIG["judge_api_url"],
                    CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"],
                    CONFIG["judge_temperature"],
                )
                result.update(
                    {
                        "detector_output": detector_output,
                        "judge_pass": plan_passed(detector_output),
                        "judge_error": None,
                    }
                )
            except Exception as exc:
                result.update(
                    {"detector_output": None, "judge_pass": False, "judge_error": str(exc)}
                )
        result["judge_time"] = time.perf_counter() - started
        by_key[key] = result
        rows = list(by_key.values())
        save_json(output_path, {"rows": rows, "summary": summarize(rows)})
        print(
            f"evaluate source={key} | structural={result['structural_pass']} "
            f"| judge={result['judge_pass']} | error={result['judge_error']}",
            flush=True,
        )
    rows = list(by_key.values())
    return {"rows": rows, "summary": summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMLU-Pro plan quality.")
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--source-index", default=CONFIG["source_index"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    records = load_json(args.plans, []) or []
    if args.source_index is not None:
        records = [
            row for row in records
            if str(row.get("source_index")) == str(args.source_index)
        ]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("No plan records matched the selection")
    result = evaluate(records, args.output, args.force)
    save_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved plan evaluation to {args.output}")


if __name__ == "__main__":
    main()

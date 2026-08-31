import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from answer_utils import accuracy_summary, extract_eval_score, extract_final_answer
from evaluate_agent_fit import build_prompt
from openai_compat import run_chat_completion
from prompt import judge_prompt


# Assignment order: evidence_agent, temporal_agent, verification_agent.
MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "s_s_s"
PLAN_VARIANT = "full_llada"

AGENT_ORDER = ("evidence_agent", "temporal_agent", "verification_agent")
MODEL_PRESETS = {
    "1b": {
        "l": ("/data/labshare/Param/llama/llama3/Llama-3.2-1B-Instruct", "http://10.137.144.97:7021/v1"),
        "g": ("/data/labshare/Param/gemma-3-1b-it", "http://10.137.144.97:7022/v1"),
        "q": ("/data/labshare/Param/Qwen/Qwen3-1.7B", "http://10.137.144.97:7023/v1"),
        "h": ("/data/labshare/Param/Hunyuan-1.8B-Instruct", "http://10.137.144.97:7024/v1"),
        "f": ("/data/labshare/Param/LFM2.5-1.2B-Instruct", "http://10.137.144.97:7025/v1"),
        "m": ("/data/labshare/Param/MiniCPM5-1B", "http://10.137.144.97:7026/v1"),
        "d": ("/data/labshare/Param/DeepSeek-R1-Distill-Qwen-1.5B", "http://10.137.144.97:7027/v1"),
        "qm": ("/data/labshare/Param/Qwen/Qwen2.5-Math-1.5B-Instruct", "http://10.137.144.97:7028/v1"),
        "qc": ("/data/labshare/Param/Qwen/Qwen2.5-Coder-1.5B-Instruct", "http://10.137.144.97:7029/v1"),
        "i": ("/data/labshare/Param/internlm2_5-1_8b-chat", "http://10.137.144.97:7030/v1"),
        "s": ("/data/labshare/Param/SmolLM2-1.7B-Instruct", "http://10.137.144.97:7031/v1"),
    },
    "3b": {
        "l": ("/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct", "http://10.137.144.97:7011/v1"),
        "g": ("/data/labshare/Param/gemma-3-4b-it", "http://10.137.144.97:7012/v1"),
        "q": ("/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507", "http://10.137.144.97:7013/v1"),
        "p": ("/data/labshare/Param/Phi-4-mini-instruct", "http://10.137.144.97:7014/v1"),
        "m": ("/data/labshare/Param/MiniCPM3-4B", "http://10.137.144.97:7015/v1"),
    },
}


def build_agent_config(model_size, assignment):
    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Unknown MODEL_SIZE {model_size!r}")
    aliases = assignment.split("_")
    if len(aliases) != len(AGENT_ORDER):
        raise ValueError(
            "AGENT_ASSIGNMENT must contain three aliases in "
            "evidence_temporal_verification order"
        )
    pool = MODEL_PRESETS[model_size]
    unknown = sorted(set(aliases) - set(pool))
    if unknown:
        raise ValueError(f"Unknown aliases for {model_size}: {unknown}")
    return {
        agent: {
            "alias": alias,
            "model": pool[alias][0],
            "api_url": pool[alias][1],
            "api_key": "empty",
            "temperature": 0.0,
            "timeout": 120,
        }
        for agent, alias in zip(AGENT_ORDER, aliases)
    }


AGENT_CONFIG = build_agent_config(MODEL_SIZE, AGENT_ASSIGNMENT)
RESULTS_DIR = f"chronoqa_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"
CONFIG = {
    "mode": "respond",
    "plans": f"benchmarks/chronoqa/chronoqa_plans_{PLAN_VARIANT}.json",
    "responses": f"{RESULTS_DIR}/subtask_hetro_responses_{AGENT_ASSIGNMENT}.json",
    "output": f"{RESULTS_DIR}/subtask_hetro_scores_{AGENT_ASSIGNMENT}.json",
    "limit": None,
    "force": False,
    "retry_errors": True,
    "max_workers": 3,
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


def execute_step(plan_record, step):
    agent_name = step["agent"]
    config = AGENT_CONFIG[agent_name]
    started = time.perf_counter()
    try:
        response = run_chat_completion(
            config["model"],
            build_prompt(agent_name, plan_record["query"], step["task"]),
            config["api_url"],
            config["api_key"],
            config["timeout"],
            config["temperature"],
        )
        prediction = extract_final_answer(response)
        error = None
    except Exception as exc:
        response = None
        prediction = None
        error = str(exc)
    return {
        **step,
        "model": config["model"],
        "api_url": config["api_url"],
        "history": "None",
        "response": response,
        "predicted_answer": prediction,
        "error": error,
        "time": time.perf_counter() - started,
    }


def reusable_step(previous, force, retry_errors):
    if not previous or force:
        return False
    if previous.get("error"):
        return not retry_errors
    return bool(previous.get("response"))


def execute_plans(plans, output_path, limit=None, force=False, retry_errors=True):
    existing = [] if force else load_json(output_path, []) or []
    by_index = {str(row.get("source_index")): row for row in existing}
    selected = plans[:limit] if limit else plans

    for plan_record in selected:
        source_index = str(plan_record.get("source_index"))
        previous_record = by_index.get(source_index, {})
        previous_steps = {
            str(step.get("id")): step for step in previous_record.get("steps", [])
        }
        record = {
            "source": plan_record.get("source"),
            "source_index": plan_record.get("source_index"),
            "question_id": plan_record.get("question_id"),
            "query": plan_record.get("query"),
            "question": plan_record.get("question"),
            "answer": plan_record.get("answer"),
            "question_date": plan_record.get("question_date"),
            "temporal_type": plan_record.get("temporal_type"),
            "temporal_expression_type": plan_record.get("temporal_expression_type"),
            "temporal_scope": plan_record.get("temporal_scope"),
            "answer_type": plan_record.get("answer_type"),
            "reference_document_count": plan_record.get("reference_document_count"),
            "planner_model": plan_record.get("planner_model"),
            "steps": [],
            "error": None,
        }
        if plan_record.get("error") or not plan_record.get("plan"):
            record["error"] = plan_record.get("error") or "planner returned no steps"
            by_index[source_index] = record
            save_json(output_path, list(by_index.values()))
            continue

        pending = []
        completed = {}
        for step in plan_record["plan"]:
            previous = previous_steps.get(str(step.get("id")))
            if reusable_step(previous, force, retry_errors):
                completed[str(step["id"])] = previous
            else:
                pending.append(step)

        if not pending and previous_record.get("error") is None and not force:
            continue

        started = time.perf_counter()
        if pending:
            with ThreadPoolExecutor(max_workers=min(CONFIG["max_workers"], len(pending))) as pool:
                futures = {
                    pool.submit(execute_step, plan_record, step): step for step in pending
                }
                for future in as_completed(futures):
                    step_result = future.result()
                    completed[str(step_result["id"])] = step_result
                    print(
                        f"respond source={source_index} | step={step_result['id']} "
                        f"| agent={step_result['agent']} | error={step_result['error']}",
                        flush=True,
                    )

        record["steps"] = [
            completed[str(step["id"])] for step in plan_record["plan"]
            if str(step["id"]) in completed
        ]
        record["subtask_wall_time"] = time.perf_counter() - started
        record["error"] = (
            "one or more subtask executions failed"
            if len(record["steps"]) != len(plan_record["plan"])
            or any(step.get("error") for step in record["steps"])
            else None
        )
        by_index[source_index] = record
        save_json(output_path, list(by_index.values()))
    return list(by_index.values())


def judge_records(records, output_path, force=False):
    existing = {} if force else load_json(output_path, {}) or {}
    previous = {
        (str(row.get("source_index")), str(row.get("id")), row.get("model")): row
        for row in existing.get("rows", [])
    }
    rows = []
    for record in records:
        for step in record.get("steps", []):
            row = {**step, "source_index": record.get("source_index"), "question_id": record.get("question_id"), "query": record.get("query"), "answer": record.get("answer"), "temporal_type": record.get("temporal_type")}
            key = (str(row.get("source_index")), str(row.get("id")), row.get("model"))
            old = previous.get(key)
            if (
                old
                and old.get("response") == row.get("response")
                and old.get("eval_score") in (0, 1)
                and old.get("judge_model") == CONFIG["judge_model"]
                and not force
            ):
                rows.append(old)
                continue
            row["judge_model"] = CONFIG["judge_model"]
            row["judge_api_url"] = CONFIG["judge_api_url"]
            if row.get("error") or not row.get("response"):
                row.update({"eval_score": 0, "judge_output": None, "judge_error": row.get("error") or "missing response"})
            else:
                try:
                    output = run_chat_completion(
                        CONFIG["judge_model"],
                        judge_prompt % (row["query"], row["answer"], row["response"]),
                        CONFIG["judge_api_url"], CONFIG["judge_api_key"],
                        CONFIG["judge_timeout"], CONFIG["judge_temperature"],
                    )
                    row.update({"eval_score": extract_eval_score(output), "judge_output": output, "judge_error": None})
                except Exception as exc:
                    row.update({"eval_score": None, "judge_output": None, "judge_error": str(exc)})
            rows.append(row)
            save_json(output_path, {"rows": rows, "summary": accuracy_summary(rows)})
            print(
                f"judge source={row.get('source_index')} | step={row.get('id')} "
                f"| agent={row.get('agent')} | score={row.get('eval_score')} "
                f"| error={row.get('judge_error')}",
                flush=True,
            )
    overall = accuracy_summary(rows)
    by_agent = {}
    for agent in AGENT_ORDER:
        agent_summary = accuracy_summary(
            [row for row in rows if row.get("agent") == agent]
        )
        by_agent[agent] = {
            "count": agent_summary["count"],
            "correct": agent_summary["correct"],
            "accuracy": agent_summary["accuracy"],
            "judge_failure_count": agent_summary["judge_failure_count"],
        }

    summary = {
        "by_agent": by_agent,
        "count": overall["count"],
        "eligible_count": overall["eligible_count"],
        "correct": overall["correct"],
        "accuracy": overall["accuracy"],
        "judge_failure_count": overall["judge_failure_count"],
    }
    return {"rows": rows, "summary": summary}


def main():
    global AGENT_CONFIG, RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Run the three ChronoQA sub-agents concurrently with heterogeneous APIs."
    )
    parser.add_argument("--mode", choices=["respond", "judge", "all"], default=CONFIG["mode"])
    parser.add_argument(
        "--assignment",
        default=AGENT_ASSIGNMENT,
        help="Model aliases in evidence_temporal_verification order (for example: g_q_l).",
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default=MODEL_SIZE)
    parser.add_argument("--plan-variant", default=PLAN_VARIANT)
    parser.add_argument("--plans", default=None)
    parser.add_argument("--responses", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    parser.add_argument("--judge-api-url", default=CONFIG["judge_api_url"])
    parser.add_argument("--judge-api-key", default=CONFIG["judge_api_key"])
    parser.add_argument("--judge-model", default=CONFIG["judge_model"])
    parser.add_argument(
        "--judge-temperature", type=float, default=CONFIG["judge_temperature"]
    )
    parser.add_argument("--judge-timeout", type=int, default=CONFIG["judge_timeout"])
    args = parser.parse_args()

    CONFIG.update(
        {
            "judge_api_url": args.judge_api_url,
            "judge_api_key": args.judge_api_key,
            "judge_model": args.judge_model,
            "judge_temperature": args.judge_temperature,
            "judge_timeout": args.judge_timeout,
        }
    )
    RESULTS_DIR = f"chronoqa_test/results_{args.model_size}_{args.plan_variant}"
    AGENT_CONFIG = build_agent_config(args.model_size, args.assignment)
    args.plans = args.plans or (
        f"benchmarks/chronoqa/chronoqa_plans_{args.plan_variant}.json"
    )
    args.responses = args.responses or (
        f"{RESULTS_DIR}/subtask_hetro_responses_{args.assignment}.json"
    )
    args.output = args.output or (
        f"{RESULTS_DIR}/subtask_hetro_scores_{args.assignment}.json"
    )

    print(f"Model assignment: size={args.model_size} | {args.assignment}")
    for agent in AGENT_ORDER:
        config = AGENT_CONFIG[agent]
        print(f"  {agent}: {config['alias']} | {config['model']} | {config['api_url']}")

    records = None
    if args.mode in {"respond", "all"}:
        plans = load_json(args.plans, []) or []
        if not plans:
            raise ValueError(f"No plans found in {args.plans}")
        records = execute_plans(
            plans, args.responses, args.limit, args.force, CONFIG["retry_errors"]
        )
        print(f"Saved responses to {args.responses}")
    if args.mode in {"judge", "all"}:
        records = records or load_json(args.responses, []) or []
        result = judge_records(records, args.output, args.force)
        save_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved judged subtask scores to {args.output}")


if __name__ == "__main__":
    main()

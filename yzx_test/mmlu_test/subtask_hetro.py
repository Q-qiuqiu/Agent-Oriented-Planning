import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from answer_utils import accuracy_summary, extract_answer_choice
from evaluate_agent_fit import build_prompt
from openai_compat import run_chat_completion


# Assignment order: knowledge_agent, reasoning_agent, elimination_agent.
MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "q_qm_m"
PLAN_VARIANT = "llada"

AGENT_ORDER = ("knowledge_agent", "reasoning_agent", "elimination_agent")
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
            "knowledge_reasoning_elimination order"
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
RESULTS_DIR = f"mmlu_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"
CONFIG = {
    "mode": "respond",
    "plans": f"benchmarks/mmlu_pro/mmlu_pro_plans_{PLAN_VARIANT}.json",
    "responses": f"{RESULTS_DIR}/subtask_hetro_responses_{AGENT_ASSIGNMENT}.json",
    "output": f"{RESULTS_DIR}/subtask_hetro_scores_{AGENT_ASSIGNMENT}.json",
    "limit": None,
    "force": False,
    "retry_errors": True,
    "max_workers": 3,
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
        prediction = extract_answer_choice(response)
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
        "correct": prediction == plan_record.get("answer"),
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
            "category": plan_record.get("category"),
            "query": plan_record.get("query"),
            "question": plan_record.get("question"),
            "options": plan_record.get("options"),
            "answer": plan_record.get("answer"),
            "answer_index": plan_record.get("answer_index"),
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


def score_records(records):
    rows = []
    for record in records:
        for step in record.get("steps", []):
            rows.append(
                {
                    **step,
                    "source_index": record.get("source_index"),
                    "question_id": record.get("question_id"),
                    "category": record.get("category"),
                    "answer": record.get("answer"),
                }
            )
    return {"rows": rows, "summary": accuracy_summary(rows)}


def main():
    parser = argparse.ArgumentParser(
        description="Run the three MMLU-Pro sub-agents concurrently with heterogeneous APIs."
    )
    parser.add_argument("--mode", choices=["respond", "score", "all"], default=CONFIG["mode"])
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    print(f"Model assignment: size={MODEL_SIZE} | {AGENT_ASSIGNMENT}")
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
    if args.mode in {"score", "all"}:
        records = records or load_json(args.responses, []) or []
        result = score_records(records)
        save_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved subtask scores to {args.output}")


if __name__ == "__main__":
    main()

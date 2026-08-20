import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from answer_utils import accuracy_summary, extract_answer_choice
from openai_compat import run_chat_completion
from prompt import (
    elimination_agent_prompt,
    knowledge_agent_prompt,
    reasoning_agent_prompt,
)


AGENTS = ["knowledge_agent", "reasoning_agent", "elimination_agent"]

# Edit these defaults directly before running.
CONFIG = {
    "mode": "respond",
    "models": ["/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507"],
    "agents": AGENTS,
    "benchmarks": "benchmarks/mmlu_pro/mmlu_pro_subtask_llada.json",
    "responses": "mmlu_test/results/agent_fit_responses_qwen3.json",
    "output": "mmlu_test/results/agent_fit_scores_qwen3.json",
    "local_api_url": "http://10.137.144.97:7003/v1",
    "local_api_key": "empty",
    "local_temperature": 0.0,
    "timeout": 120,
    "limit": None,
    "force": False,
    "retry_errors": True,
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


def build_prompt(agent_name, query, task, history="None"):
    del history
    templates = {
        "knowledge_agent": knowledge_agent_prompt,
        "reasoning_agent": reasoning_agent_prompt,
        "elimination_agent": elimination_agent_prompt,
    }
    try:
        return templates[agent_name] % (query, task)
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {agent_name}") from exc


def row_key(row):
    return (
        str(row.get("source_index")),
        str(row.get("subtask_id")),
        row.get("agent"),
        row.get("model"),
    )


def execute_rows(benchmarks, models, agents, output_path, config):
    existing = [] if config["force"] else load_json(output_path, []) or []
    by_key = {row_key(row): row for row in existing}
    selected = benchmarks[: config["limit"]] if config["limit"] else benchmarks

    for benchmark in selected:
        agent_name = benchmark.get("agent")
        if agent_name not in agents:
            continue
        for model in models:
            key = (
                str(benchmark.get("source_index")),
                str(benchmark.get("subtask_id")),
                agent_name,
                model,
            )
            previous = by_key.get(key)
            if previous and previous.get("error") is None and not config["force"]:
                continue
            if previous and previous.get("error") and not config["retry_errors"]:
                continue

            record = {
                **benchmark,
                "model": model,
                "api_url": config["local_api_url"],
            }
            started = time.perf_counter()
            try:
                raw_output = run_chat_completion(
                    model,
                    build_prompt(agent_name, benchmark["query"], benchmark["task"]),
                    config["local_api_url"],
                    config["local_api_key"],
                    config["timeout"],
                    config["local_temperature"],
                )
                prediction = extract_answer_choice(raw_output)
                record.update(
                    {
                        "response": raw_output,
                        "predicted_answer": prediction,
                        "correct": prediction == benchmark.get("answer"),
                        "error": None,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "response": None,
                        "predicted_answer": None,
                        "correct": False,
                        "error": str(exc),
                    }
                )
            record["time"] = time.perf_counter() - started
            by_key[key] = record
            save_json(output_path, list(by_key.values()))
            print(
                f"respond {model} | {agent_name} | source={record.get('source_index')} "
                f"| prediction={record.get('predicted_answer')} | error={record['error']}",
                flush=True,
            )
    return list(by_key.values())


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("model"), row.get("agent"))].append(row)

    by_model = defaultdict(dict)
    for (model, agent), group in sorted(grouped.items()):
        by_model[model][agent] = accuracy_summary(group)

    for model, agents in by_model.items():
        ranked = sorted(
            (
                {"agent": agent, "accuracy": values["accuracy"]}
                for agent, values in agents.items()
            ),
            key=lambda item: item["accuracy"],
            reverse=True,
        )
        agents["recommended_agents"] = ranked
    return dict(by_model)


def score_responses(responses_path, output_path):
    rows = load_json(responses_path, []) or []
    result = {"rows": rows, "summary": summarize(rows)}
    save_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local models as the three independent MMLU-Pro agents."
    )
    parser.add_argument("--mode", choices=["respond", "score", "all"], default=CONFIG["mode"])
    parser.add_argument("--models", nargs="+", default=CONFIG["models"])
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=CONFIG["agents"])
    parser.add_argument("--benchmarks", default=CONFIG["benchmarks"])
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(vars(args))
    if args.mode in {"respond", "all"}:
        benchmarks = load_json(args.benchmarks, []) or []
        if not benchmarks:
            raise ValueError(f"No benchmark rows found in {args.benchmarks}")
        execute_rows(benchmarks, args.models, args.agents, args.responses, config)
        print(f"Saved responses to {args.responses}")

    if args.mode in {"score", "all"}:
        result = score_responses(args.responses, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()

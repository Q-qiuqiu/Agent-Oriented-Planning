import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from build_subtask_benchmark import extract_json_array, normalize_plan
from openai_compat import run_chat_completion
from prompt import planner_prompt, summarization_agent_prompt
from subtask_hetro import AGENT_CONFIG, AGENT_ORDER
from evaluate_agent_fit import build_prompt


CONFIG = {
    "planner_api_url": "http://10.137.144.97:7004/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llada",
    "summary_api_url": "http://10.137.144.97:7007/v1",
    "summary_api_key": "empty",
    "summary_model": "/data/labshare/Param/llada",
    "temperature": 0.0,
    "timeout": 120,
}


def execute_agent(query, step):
    config = AGENT_CONFIG[step["agent"]]
    response = run_chat_completion(
        config["model"],
        build_prompt(step["agent"], query, step["task"]),
        config["api_url"],
        config["api_key"],
        config["timeout"],
        config["temperature"],
    )
    return {**step, "model": config["model"], "response": response}


def run_query(query):
    raw_plan = run_chat_completion(
        CONFIG["planner_model"],
        query,
        CONFIG["planner_api_url"],
        CONFIG["planner_api_key"],
        CONFIG["timeout"],
        CONFIG["temperature"],
        system_prompt=planner_prompt,
    )
    plan = normalize_plan(extract_json_array(raw_plan))
    with ThreadPoolExecutor(max_workers=len(AGENT_ORDER)) as pool:
        responses = list(pool.map(lambda step: execute_agent(query, step), plan))
    final_prompt = summarization_agent_prompt % (
        query,
        json.dumps(plan, ensure_ascii=False, indent=2),
        json.dumps(responses, ensure_ascii=False, indent=2),
    )
    final_answer = run_chat_completion(
        CONFIG["summary_model"],
        final_prompt,
        CONFIG["summary_api_url"],
        CONFIG["summary_api_key"],
        CONFIG["timeout"],
        CONFIG["temperature"],
    )
    return {
        "query": query,
        "raw_plan": raw_plan,
        "plan": plan,
        "subtask_responses": responses,
        "final_answer": final_answer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run one MMLU-Pro question through planner, agents, and summary."
    )
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(run_query(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

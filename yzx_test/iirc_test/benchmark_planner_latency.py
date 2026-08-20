import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from build_subtask_benchmark import extract_json_array, normalize_plan
from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


CONFIG = {
    "dataset_path": "benchmarks/iirc/iirc_dev_flat.json",
    "query_index": 0,
    "repeat": 1,
    "api_url": "http://10.137.144.97:7004/v1",
    "api_key": "empty",
    "model": "/data/labshare/Param/llada",
    "temperature": 0.0,
    "max_tokens": 1024,
    "timeout": 600,
}


def load_query(path, index):
    with Path(path).open("r", encoding="utf-8") as file:
        rows = json.load(file)
    row = rows[index]
    return {
        "source_index": row.get("index", index),
        "query": row.get("query") or row.get("question"),
        "planner_input": row.get("agent_context") or row.get("query") or row.get("question"),
    }


def request_plan(query):
    response = requests.post(
        chat_completions_url(CONFIG["api_url"]),
        headers={"Content-Type": "application/json", **auth_header(CONFIG["api_key"])},
        json={
            "model": CONFIG["model"],
            "messages": [
                {"role": "system", "content": planner_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": CONFIG["temperature"],
            "max_tokens": CONFIG["max_tokens"],
        },
        timeout=CONFIG["timeout"],
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Measure one IIRC planner query.")
    parser.add_argument("--dataset-path", default=CONFIG["dataset_path"])
    parser.add_argument("--query-index", type=int, default=CONFIG["query_index"])
    parser.add_argument("--repeat", type=int, default=CONFIG["repeat"])
    args = parser.parse_args()

    query = load_query(args.dataset_path, args.query_index)
    measurements = []
    for index in range(args.repeat):
        started = time.perf_counter()
        usage = {}
        metrics = {}
        error = None
        steps = 0
        try:
            response = request_plan(query["planner_input"])
            raw = response["choices"][0]["message"]["content"].strip()
            steps = len(normalize_plan(extract_json_array(raw)))
            usage = response.get("usage") or {}
            metrics = response.get("fastdllm") or {}
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - started
        completion_tokens = usage.get("completion_tokens")
        tps = completion_tokens / latency if completion_tokens and latency else None
        measurements.append(latency)
        repair = metrics.get("plan_json_repair") or {}
        print(
            f"measure {index + 1}/{args.repeat} | latency={latency:.4f}s "
            f"| prompt_tokens={usage.get('prompt_tokens')} "
            f"| generated_tokens={completion_tokens} "
            f"| decode_tps={f'{tps:.2f}' if tps is not None else None} "
            f"| steps={steps} | json_repair={repair.get('method')} | error={error}",
            flush=True,
        )

    print("\nPlanner latency summary")
    print(
        f"mean={statistics.mean(measurements):.4f}s "
        f"| median={statistics.median(measurements):.4f}s "
        f"| min={min(measurements):.4f}s | max={max(measurements):.4f}s"
    )


if __name__ == "__main__":
    main()

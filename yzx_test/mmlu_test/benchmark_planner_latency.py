import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from build_subtask_benchmark import extract_json_array, load_queries, normalize_plan
from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


CONFIG = {
    "input": "benchmarks/mmlu_pro/mmlu_pro_sampled.json",
    "query_index": 0,
    "repeat": 1,
    "planner_api_url": "http://10.137.144.97:7004/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llada",
    "temperature": 0.0,
    "max_tokens": 512,
    "timeout": 600,
}


def request(query):
    response = requests.post(
        chat_completions_url(CONFIG["planner_api_url"]),
        headers={"Content-Type": "application/json", **auth_header(CONFIG["planner_api_key"])},
        json={
            "model": CONFIG["planner_model"],
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
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content, data.get("usage", {}), data.get("fastdllm", {})


def main():
    parser = argparse.ArgumentParser(description="Measure one MMLU-Pro planner query.")
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--query-index", type=int, default=CONFIG["query_index"])
    parser.add_argument("--repeat", type=int, default=CONFIG["repeat"])
    args = parser.parse_args()
    records = load_queries(args.input)
    query = records[args.query_index]
    measurements = []
    for index in range(args.repeat):
        started = time.perf_counter()
        error = None
        steps = 0
        usage = {}
        metrics = {}
        try:
            content, usage, metrics = request(query["query"])
            steps = len(normalize_plan(extract_json_array(content)))
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - started
        completion_tokens = usage.get("completion_tokens")
        tps = completion_tokens / latency if completion_tokens and latency else None
        measurements.append(latency)
        print(
            f"measure {index + 1}/{args.repeat} | latency={latency:.4f}s "
            f"| prompt_tokens={usage.get('prompt_tokens')} "
            f"| generated_tokens={completion_tokens} | decode_tps="
            f"{f'{tps:.2f}' if tps is not None else None} | steps={steps} "
            f"| nfe={metrics.get('nfe')} | error={error}",
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

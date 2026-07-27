import json
import re
import statistics
import time
from pathlib import Path

import requests

from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


# Edit these values directly before running the script.
CONFIG = {
    "api_url": "http://127.0.0.1:7002/v1",
    "api_key": "empty",
    "model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "dataset_path": "benchmarks/huskyqa/huskyqa_raw.json",
    "source_index": 0,
    "custom_query": None,
    "temperature": 0.0,
    "max_tokens": 1024,
    "timeout": 600,
    "warmup_requests": 0,
    "measured_requests": 1,
    "output": "huskyqa_test/results/planner_latency_single.json",
}


def load_query():
    if CONFIG["custom_query"]:
        return {
            "source_index": "custom",
            "query": CONFIG["custom_query"],
            "planner_input": CONFIG["custom_query"],
        }

    with Path(CONFIG["dataset_path"]).open("r", encoding="utf-8") as file:
        rows = json.load(file)

    for row in rows:
        if str(row.get("index")) != str(CONFIG["source_index"]):
            continue
        query = row.get("query") or row.get("question")
        return {
            "source_index": row.get("index"),
            "query": query,
            "planner_input": query,
        }
    raise ValueError(
        f"No query with source_index={CONFIG['source_index']!r} in "
        f"{CONFIG['dataset_path']}"
    )


def extract_json_array(text):
    value = text.strip()
    if value.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL)
        if match:
            value = match.group(1).strip()
    start = value.find("[")
    if start < 0:
        raise ValueError("planner output contains no JSON array")
    plan, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(plan, list):
        raise ValueError("planner output JSON is not a list")
    return plan


def request_plan(query_record):
    headers = {"Content-Type": "application/json"}
    headers.update(auth_header(CONFIG["api_key"]))
    payload = {
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": query_record["planner_input"]},
        ],
        "temperature": CONFIG["temperature"],
        "max_tokens": CONFIG["max_tokens"],
    }

    started = time.perf_counter()
    response = requests.post(
        chat_completions_url(CONFIG["api_url"]),
        headers=headers,
        json=payload,
        timeout=CONFIG["timeout"],
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    raw_output = data["choices"][0]["message"]["content"].strip()

    parse_started = time.perf_counter()
    parse_error = None
    try:
        plan = extract_json_array(raw_output)
    except Exception as exc:
        plan = None
        parse_error = str(exc)
    parse_seconds = time.perf_counter() - parse_started

    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return {
        "http_round_trip_seconds": elapsed,
        "parse_seconds": parse_seconds,
        "total_seconds": elapsed + parse_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "completion_tokens_per_second": (
            completion_tokens / elapsed
            if completion_tokens is not None and elapsed
            else None
        ),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "plan": plan,
        "raw_output": raw_output,
        "parse_error": parse_error,
    }


def timing_summary(measurements):
    values = [item["http_round_trip_seconds"] for item in measurements]
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def save_result(result):
    output = Path(CONFIG["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    temporary.replace(output)
    return output.resolve()


def main():
    if CONFIG["measured_requests"] < 1:
        raise ValueError("measured_requests must be at least 1")

    query_record = load_query()
    for index in range(CONFIG["warmup_requests"]):
        result = request_plan(query_record)
        print(
            f"warmup {index + 1}/{CONFIG['warmup_requests']} "
            f"| latency={result['http_round_trip_seconds']:.4f}s",
            flush=True,
        )

    measurements = []
    for index in range(CONFIG["measured_requests"]):
        measurement = request_plan(query_record)
        measurements.append(measurement)
        print(
            f"measure {index + 1}/{CONFIG['measured_requests']} "
            f"| latency={measurement['http_round_trip_seconds']:.4f}s "
            f"| steps={len(measurement['plan'] or [])} "
            f"| error={measurement['parse_error']}",
            flush=True,
        )

    result = {
        "api_url": chat_completions_url(CONFIG["api_url"]),
        "model": CONFIG["model"],
        "source_index": query_record["source_index"],
        "query": query_record["query"],
        "warmup_requests": CONFIG["warmup_requests"],
        "timing_summary": timing_summary(measurements),
        "measurements": measurements,
    }
    output = save_result(result)

    print("\nPlanner latency summary")
    print(
        f"mean={result['timing_summary']['mean_seconds']:.4f}s "
        f"| median={result['timing_summary']['median_seconds']:.4f}s "
        f"| min={result['timing_summary']['min_seconds']:.4f}s "
        f"| max={result['timing_summary']['max_seconds']:.4f}s"
    )
    print(f"Result JSON: {output}")


if __name__ == "__main__":
    main()

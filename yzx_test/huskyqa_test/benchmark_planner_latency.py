import json
import re
import statistics
import time
from pathlib import Path

import requests

from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


STANDARD_OUTPUT_BLOCK = """Output only valid JSON in this format:
[
  {
    "id": 1,
    "task": "subtask description",
    "agent": "math_agent",
    "reason": "why this agent is suitable",
    "dep": []
  }
]"""

LATENCY_TEST_OUTPUT_BLOCK = """For this single-query latency measurement,
produce exactly two sections in the following order. The first section must be
detailed enough to make the response substantially longer than the normal
planner output.

PLANNING_REASONING
Explain in detailed prose:
1. which facts, entities, numbers, constraints, and dates must be preserved;
2. which operations or external evidence are needed to answer the query;
3. which subtasks can run independently in the first batch;
4. which later subtasks depend on earlier results;
5. why every selected agent role is appropriate; and
6. why the plan is complete without unnecessary or duplicate work.

Make this reasoning section roughly as detailed as the JSON plan. Do not put
JSON, square brackets, braces, or Markdown code fences in this section.
END_PLANNING_REASONING

PLAN_JSON
[
  {
    "id": 1,
    "task": "A detailed, self-contained, executable subtask that preserves all relevant entities, numbers, constraints, and expected output.",
    "agent": "math_agent",
    "reason": "A detailed explanation of why this role is the best fit and how its output supports the final answer.",
    "dep": []
  }
]
END_PLAN_JSON

The PLAN_JSON section must contain one valid JSON array using the original
schema. Make every task and reason detailed and self-contained. Do not add
comments, trailing commas, extra fields, or prose inside the JSON array.
"""

if STANDARD_OUTPUT_BLOCK not in planner_prompt:
    raise RuntimeError("Cannot locate the standard planner output instructions")
LATENCY_TEST_PLANNER_PROMPT = planner_prompt.replace(
    STANDARD_OUTPUT_BLOCK,
    LATENCY_TEST_OUTPUT_BLOCK,
    1,
)


# Edit these values directly before running the script.
CONFIG = {
    # "api_url": "http://10.137.144.97:7002/v1",
    # "model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "api_url": "http://10.137.144.97:7004/v1",
    "model": "/data/labshare/Param/llada",
    "api_key": "empty",
    "dataset_path": "benchmarks/huskyqa/huskyqa_raw.json",
    "source_index": 2,
    "custom_query": None,
    "temperature": 0.0,
    # The server's configured generation length must be at least this value.
    "max_tokens": 1024,
    "timeout": 600,
    "warmup_requests": 0,
    "measured_requests": 1,
    "output": "huskyqa_test/planner_latency_single.json",
}


def load_query():
    if CONFIG["custom_query"]:
        return {
            "source_index": "custom",
            "query": CONFIG["custom_query"],
        }

    with Path(CONFIG["dataset_path"]).open("r", encoding="utf-8") as file:
        rows = json.load(file)

    for row in rows:
        if str(row.get("index")) != str(CONFIG["source_index"]):
            continue
        return {
            "source_index": row.get("index"),
            "query": row.get("query") or row.get("question"),
        }
    raise ValueError(
        f"No query with source_index={CONFIG['source_index']!r} in "
        f"{CONFIG['dataset_path']}"
    )


def extract_json_array(text):
    decoder = json.JSONDecoder()
    value = text.strip()
    marker_matches = list(
        re.finditer(r"(?m)^\s*PLAN_JSON\s*:?\s*$", value)
    )
    if marker_matches:
        segment = value[marker_matches[-1].end():]
        end_match = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", segment)
        if end_match:
            segment = segment[:end_match.start()]
        segment = re.sub(
            r"(?m)^\s*```(?:json)?\s*$",
            "",
            segment,
        ).strip()
        start = segment.find("[")
        if start < 0:
            raise ValueError("PLAN_JSON contains no JSON array")
        try:
            plan, _ = decoder.raw_decode(segment[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse PLAN_JSON array: {exc}") from exc
        if not isinstance(plan, list) or not all(
            isinstance(step, dict) for step in plan
        ):
            raise ValueError("PLAN_JSON must be an array of plan-step objects")
        return plan

    for match in re.finditer(r"\[", value):
        try:
            plan, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(plan, list) and all(
            isinstance(step, dict) for step in plan
        ):
            return plan
    raise ValueError("Cannot find plan JSON in planner output")


def extract_planning_reasoning(text):
    start_marker = "PLANNING_REASONING"
    end_marker = "END_PLANNING_REASONING"
    start = text.find(start_marker)
    if start < 0:
        return None
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        end = text.find("PLAN_JSON", start)
    if end < 0:
        return None
    reasoning = text[start:end].strip(" \n:\t")
    return reasoning or None


def request_plan(query_record):
    headers = {"Content-Type": "application/json"}
    headers.update(auth_header(CONFIG["api_key"]))
    payload = {
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": LATENCY_TEST_PLANNER_PROMPT},
            {"role": "user", "content": query_record["query"]},
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
    if not response.ok:
        raise RuntimeError(
            f"Planner request failed with HTTP {response.status_code}: "
            f"{response.text}"
        )
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
    effective_tps = (
        completion_tokens / elapsed
        if completion_tokens is not None and elapsed
        else None
    )
    server_metrics = data.get("fastdllm") or {}
    return {
        "http_round_trip_seconds": elapsed,
        "parse_seconds": parse_seconds,
        "total_seconds": elapsed + parse_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "effective_tps": effective_tps,
        "server_generation_seconds": server_metrics.get("generation_time"),
        "server_generation_tps": server_metrics.get("generation_tps"),
        "server_nfe": server_metrics.get("nfe"),
        "server_steps": server_metrics.get("steps"),
        "server_block_size": server_metrics.get("block_size"),
        "server_cache_mode": server_metrics.get("cache_mode"),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "planning_reasoning": extract_planning_reasoning(raw_output),
        "plan": plan,
        "raw_output": raw_output,
        "parse_error": parse_error,
    }


def timing_summary(measurements):
    latencies = [item["http_round_trip_seconds"] for item in measurements]
    completion_tokens = [
        item["completion_tokens"]
        for item in measurements
        if item["completion_tokens"] is not None
    ]
    effective_tps = [
        item["effective_tps"]
        for item in measurements
        if item["effective_tps"] is not None
    ]
    server_tps = [
        item["server_generation_tps"]
        for item in measurements
        if item["server_generation_tps"] is not None
    ]
    return {
        "count": len(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "mean_generated_tokens": (
            statistics.mean(completion_tokens) if completion_tokens else None
        ),
        "mean_effective_tps": (
            statistics.mean(effective_tps) if effective_tps else None
        ),
        "mean_server_generation_tps": (
            statistics.mean(server_tps) if server_tps else None
        ),
    }


def format_metric(value):
    return f"{value:.2f}" if value is not None else "N/A"


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
            f"| prompt_tokens={measurement['prompt_tokens']} "
            f"| generated_tokens={measurement['completion_tokens']} "
            f"| effective_tps={format_metric(measurement['effective_tps'])} "
            f"| server_tps={format_metric(measurement['server_generation_tps'])} "
            f"| nfe={measurement['server_nfe']} "
            f"| reasoning_chars={len(measurement['planning_reasoning'] or '')} "
            f"| steps={len(measurement['plan'] or [])} "
            f"| error={measurement['parse_error']}",
            flush=True,
        )

    summary = timing_summary(measurements)
    result = {
        "planner_mode": "latency_test_reasoning_then_json",
        "api_url": chat_completions_url(CONFIG["api_url"]),
        "model": CONFIG["model"],
        "source_index": query_record["source_index"],
        "query": query_record["query"],
        "warmup_requests": CONFIG["warmup_requests"],
        "timing_summary": summary,
        "measurements": measurements,
    }
    output = save_result(result)

    print("\nPlanner latency summary")
    print(
        f"mean={summary['mean_seconds']:.4f}s "
        f"| median={summary['median_seconds']:.4f}s "
        f"| min={summary['min_seconds']:.4f}s "
        f"| max={summary['max_seconds']:.4f}s"
    )
    print(
        f"generated_tokens={summary['mean_generated_tokens']} "
        f"| effective_tps={format_metric(summary['mean_effective_tps'])} "
        f"| server_tps={format_metric(summary['mean_server_generation_tps'])}"
    )
    print(f"Result JSON: {output}")


if __name__ == "__main__":
    main()

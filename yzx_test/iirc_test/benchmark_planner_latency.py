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
    "task": "detailed executable subtask",
    "agent": "context_agent",
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
1. what useful information is already present in the initial passage;
2. what evidence is missing and which linked articles may contain it;
3. which evidence tasks can run independently in the first batch;
4. which later tasks depend on earlier results;
5. why every selected agent role is appropriate; and
6. why the resulting plan is complete without redundant work.

Make this reasoning section roughly as detailed as the JSON plan. Do not put
JSON, square brackets, braces, or Markdown code fences in this section.
END_PLANNING_REASONING

PLAN_JSON
[
  {
    "id": 1,
    "task": "A detailed, self-contained, executable subtask that preserves all relevant entities, constraints, and expected evidence.",
    "agent": "context_agent",
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
    "api_url": "http://127.0.0.1:7002/v1",
    "api_key": "empty",
    "model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "dataset_path": "benchmarks/iirc/iirc_dev_flat.json",
    "source_index": "q_10839",
    "custom_query": None,
    "use_agent_context": True,
    "temperature": 0.0,
    "max_tokens": 1024,
    "timeout": 600,
    "warmup_requests": 0,
    "measured_requests": 1,
    "output": "iirc_test/planner_latency_single.json",
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
        planner_input = (
            row.get("agent_context")
            if CONFIG["use_agent_context"]
            else query
        )
        return {
            "source_index": row.get("index"),
            "query": query,
            "planner_input": planner_input,
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
        "planning_reasoning": extract_planning_reasoning(raw_output),
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
            f"| reasoning_chars={len(measurement['planning_reasoning'] or '')} "
            f"| steps={len(measurement['plan'] or [])} "
            f"| error={measurement['parse_error']}",
            flush=True,
        )

    result = {
        "api_url": chat_completions_url(CONFIG["api_url"]),
        "model": CONFIG["model"],
        "source_index": query_record["source_index"],
        "query": query_record["query"],
        "planner_mode": "latency_test_reasoning_then_json",
        "used_agent_context": CONFIG["use_agent_context"],
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

import argparse
import json
import re
import time
from pathlib import Path

import requests

from openai_compat import auth_header, chat_completions_url
from prompt import planner_prompt


AGENTS = [
    "context_agent",
    "retrieval_agent",
    "reasoning_agent",
    "calculation_agent",
    "answerability_agent",
]
FULL_PROMPT_VERSION = "iirc_full_shared_planner_prompt_v2"

AGENT_ALIASES = {
    "retriev_agent": "retrieval_agent",
    "retrival_agent": "retrieval_agent",
    "retrieve_agent": "retrieval_agent",
    "retriever_agent": "retrieval_agent",
    "reasonability_agent": "reasoning_agent",
    "calculability_agent": "calculation_agent",
}

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

FULL_OUTPUT_BLOCK = """Use the same decomposition, agent selection, dependencies,
and JSON plan that you would produce under the original instructions. The only
additional requirement is to output the planning reasoning before that JSON.

PLANNING_REASONING
Explain the reasoning that led to the plan. This is an additional explanation,
not a different planning task. Do not solve the subtasks in this section and do
not introduce any agent-selection, evidence-gathering, or decomposition rules
beyond the original instructions. Do not put JSON or Markdown code fences in
this section.
END_PLANNING_REASONING

PLAN_JSON
[
  {
    "id": 1,
    "task": "detailed executable subtask",
    "agent": "context_agent",
    "reason": "why this agent is suitable",
    "dep": []
  }
]
END_PLAN_JSON
"""

if STANDARD_OUTPUT_BLOCK not in planner_prompt:
    raise RuntimeError("Cannot locate the standard planner output instructions")
FULL_PLANNER_PROMPT = planner_prompt.replace(
    STANDARD_OUTPUT_BLOCK,
    FULL_OUTPUT_BLOCK,
    1,
)


# Edit these defaults directly before running the script.
CONFIG = {
    "input": "benchmarks/iirc/iirc_dev_flat.json",
    "plans_output": "benchmarks/iirc/iirc_plans_full_llama3.json",
    "benchmark_output": "benchmarks/iirc/iirc_subtask_full_llama3.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    #"planner_model": "/data/labshare/Param/llada",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "retry_missing_reasoning": False,
    "agents": AGENTS,
}


def load_queries(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    queries = []
    for index, row in enumerate(data):
        queries.append(
            {
                "source_index": row.get("index", index),
                "query": row.get("question") or row.get("query"),
                "answer": row.get("answer"),
                "source": row.get("source", "allenai/IIRC"),
                "planner_input": row.get("agent_context")
                or row.get("planner_input")
                or row.get("question")
                or row.get("query"),
                "agent_context": row.get("agent_context"),
                "answer_type": row.get("answer_type"),
                "original_answer": row.get("original_answer"),
                "article_pid": row.get("article_pid"),
                "article_title": row.get("article_title"),
                "available_links": row.get("available_links") or [],
                "gold_question_links": row.get("gold_question_links") or [],
                "gold_context": row.get("gold_context") or [],
            }
        )
    return queries


def request_completion(prompt, api_url, api_key, model, temperature, max_tokens, timeout):
    headers = {"Content-Type": "application/json"}
    headers.update(auth_header(api_key))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": FULL_PLANNER_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        chat_completions_url(api_url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Planner request failed with HTTP {response.status_code}: "
            f"{response.text}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def extract_json_array(text):
    decoder = json.JSONDecoder()
    value = text.strip()
    marker_matches = list(re.finditer(r"(?m)^\s*PLAN_JSON\s*:?\s*$", value))
    if marker_matches:
        segment = value[marker_matches[-1].end():]
        end_match = re.search(r"(?m)^\s*END_PLAN_JSON\s*$", segment)
        if end_match:
            segment = segment[:end_match.start()]
        segment = re.sub(r"(?m)^\s*```(?:json)?\s*$", "", segment).strip()
        start = segment.find("[")
        if start < 0:
            raise ValueError("PLAN_JSON contains no JSON array")
        try:
            plan, _ = decoder.raw_decode(segment[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse PLAN_JSON array: {exc}") from exc
        if not isinstance(plan, list) or not all(isinstance(step, dict) for step in plan):
            raise ValueError("PLAN_JSON must be an array of plan-step objects")
        return plan

    for match in re.finditer(r"\[", value):
        try:
            plan, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(plan, list) and all(isinstance(step, dict) for step in plan):
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
    return text[start:end].strip(" \n:\t") or None


def normalize_plan(plan):
    normalized = []
    repairs = []
    for index, step in enumerate(plan, start=1):
        item = dict(step)
        item.setdefault("id", index)
        if "agent" not in item:
            item["agent"] = item.get("name") or item.get("name_1")
        raw_agent = item.get("agent")
        if isinstance(raw_agent, str):
            candidate = raw_agent.strip().lower()
            canonical_agent = AGENT_ALIASES.get(candidate, candidate)
            if canonical_agent != raw_agent:
                repairs.append(
                    {
                        "step_id": item["id"],
                        "field": "agent",
                        "original": raw_agent,
                        "normalized": canonical_agent,
                    }
                )
            item["agent"] = canonical_agent
        if item.get("agent") not in AGENTS:
            raise ValueError(
                f"Plan step {item['id']} has unsupported agent "
                f"{item.get('agent')!r}; expected one of {AGENTS}"
            )
        item.setdefault("dep", [])
        normalized.append(item)
    return normalized, repairs


def ordered_records(records_by_index):
    return [records_by_index[index] for index in sorted(records_by_index)]


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(output)


def build_plans(queries, config):
    output = Path(config["plans_output"])
    existing_by_index = {}
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        existing_by_index = {item["source_index"]: item for item in existing}
    done = {
        index
        for index, item in existing_by_index.items()
        if item.get("error") is None
        and item.get("plan")
        and item.get("planner_prompt_version") == FULL_PROMPT_VERSION
        and (
            not config["retry_missing_reasoning"]
            or item.get("planning_reasoning")
        )
    }
    if existing_by_index:
        print(
            f"resume | loaded={len(existing_by_index)} "
            f"| completed={len(done)} "
            f"| prompt_version={FULL_PROMPT_VERSION} "
            f"| retry_missing_reasoning="
            f"{config['retry_missing_reasoning']}",
            flush=True,
        )

    selected = queries[:config["limit"]] if config["limit"] else queries
    for row in selected:
        if row["source_index"] in done:
            continue
        started = time.time()
        raw_output = None
        record = {
            "source": row.get("source", "allenai/IIRC"),
            "source_index": row["source_index"],
            "query": row["query"],
            "answer": row.get("answer"),
            "agent_context": row.get("agent_context"),
            "answer_type": row.get("answer_type"),
            "original_answer": row.get("original_answer"),
            "article_pid": row.get("article_pid"),
            "article_title": row.get("article_title"),
            "available_links": row.get("available_links") or [],
            "gold_question_links": row.get("gold_question_links") or [],
            "gold_context": row.get("gold_context") or [],
            "planner_model": config["planner_model"],
            "planner_mode": "reasoning_then_json",
            "planner_prompt_version": FULL_PROMPT_VERSION,
        }
        try:
            raw_output = request_completion(
                row["planner_input"],
                config["planner_api_url"],
                config["planner_api_key"],
                config["planner_model"],
                config["planner_temperature"],
                config["planner_max_tokens"],
                config["timeout"],
            )
            plan, repairs = normalize_plan(extract_json_array(raw_output))
            reasoning = extract_planning_reasoning(raw_output)
            warnings = [] if reasoning else ["missing planning reasoning section"]
            record.update(
                {
                    "planning_reasoning": reasoning,
                    "raw_plan": raw_output,
                    "plan": plan,
                    "normalization_repairs": repairs,
                    "format_warnings": warnings,
                    "error": None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "planning_reasoning": (
                        extract_planning_reasoning(raw_output) if raw_output else None
                    ),
                    "raw_plan": raw_output,
                    "plan": None,
                    "error": str(exc),
                }
            )
        record["time"] = time.time() - started
        existing_by_index[row["source_index"]] = record
        save_json(output, ordered_records(existing_by_index))
        print(
            f"planned {row['source_index']} "
            f"| reasoning_chars={len(record.get('planning_reasoning') or '')} "
            f"| subtasks={len(record.get('plan') or [])} "
            f"| repairs={len(record.get('normalization_repairs') or [])} "
            f"| error={record['error']}",
            flush=True,
        )
    return ordered_records(existing_by_index)


def expand_plans(plans, agents):
    benchmark = []
    for plan_record in plans:
        for step in plan_record.get("plan") or []:
            task = step.get("task")
            if not task:
                continue
            for agent_name in agents:
                benchmark.append(
                    {
                        "agent": agent_name,
                        "query": plan_record["query"],
                        "task": task,
                        "history": "None",
                        "dep": step.get("dep", []),
                        "subtask_id": step.get("id"),
                        "planner_agent": step.get("agent"),
                        "planner_reason": step.get("reason"),
                        "source": plan_record.get("source"),
                        "source_index": plan_record.get("source_index"),
                        "answer": plan_record.get("answer"),
                        "agent_context": plan_record.get("agent_context"),
                        "answer_type": plan_record.get("answer_type"),
                        "original_answer": plan_record.get("original_answer"),
                        "article_pid": plan_record.get("article_pid"),
                        "article_title": plan_record.get("article_title"),
                        "available_links": plan_record.get("available_links") or [],
                        "gold_question_links": plan_record.get("gold_question_links") or [],
                        "gold_context": plan_record.get("gold_context") or [],
                    }
                )
    return benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Build IIRC plans with visible reasoning and JSON subtasks."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--plans-output", default=CONFIG["plans_output"])
    parser.add_argument("--benchmark-output", default=CONFIG["benchmark_output"])
    parser.add_argument("--planner-api-url", default=CONFIG["planner_api_url"])
    parser.add_argument("--planner-api-key", default=CONFIG["planner_api_key"])
    parser.add_argument("--planner-model", default=CONFIG["planner_model"])
    parser.add_argument(
        "--planner-temperature", type=float, default=CONFIG["planner_temperature"]
    )
    parser.add_argument(
        "--planner-max-tokens", type=int, default=CONFIG["planner_max_tokens"]
    )
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--agents", nargs="+", default=CONFIG["agents"], choices=AGENTS)
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(vars(args))
    if not config["planner_api_url"]:
        raise ValueError("Missing planner API URL")
    if config["planner_max_tokens"] < 1:
        raise ValueError("planner_max_tokens must be at least 1")

    plans = build_plans(load_queries(config["input"]), config)
    save_json(config["plans_output"], plans)
    benchmark = expand_plans(plans, config["agents"])
    save_json(config["benchmark_output"], benchmark)
    print(f"Saved plans: {config['plans_output']} ({len(plans)} queries)")
    print(
        f"Saved benchmark: {config['benchmark_output']} "
        f"({len(benchmark)} rows)"
    )


if __name__ == "__main__":
    main()

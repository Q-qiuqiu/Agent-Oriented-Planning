import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

from llada_server.planner_json_repair import repair_plan_json_response


STRATEGY = "base_llama3"
PREFETCH_START_MARKER = "PREFETCH_AGENTS"
PREFETCH_END_MARKER = "END_PREFETCH_AGENTS"


def _extract_marked_json_array(text, start_marker, end_marker):
    starts = list(re.finditer(
        rf"(?m)^\s*{re.escape(start_marker)}\s*:?[ \t]*$", text
    ))
    if not starts:
        raise ValueError(f"Missing {start_marker} marker")
    if len(starts) != 1:
        raise ValueError(f"Multiple {start_marker} sections found")
    segment = text[starts[0].end():]
    end = re.search(
        rf"(?m)^\s*{re.escape(end_marker)}\s*$", segment
    )
    if end is not None:
        segment = segment[:end.start()]

    # Decode only the first array after the marker. Searching later opening
    # brackets can mistake a nested dep=[] for the complete plan when the
    # outer JSON array has a syntax error.
    array_start = segment.find("[")
    if array_start < 0:
        raise ValueError(f"No JSON array found after {start_marker}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(segment[array_start:])
    if not isinstance(value, list):
        raise ValueError(f"JSON value after {start_marker} is not an array")
    return value


def _repair_plan_json_syntax_only(text, allowed_agents):
    repaired, report = repair_plan_json_response(text, allowed_agents)
    operations = report.get("operations") or []
    syntax_method = report.get("method") in {"minimal_syntax", "schema_rebuild"}
    changed_agent = any(
        str(operation).startswith("canonicalize_agent_name:")
        for operation in operations
    )
    if report.get("applied") and syntax_method and not changed_agent:
        return repaired
    return text


def extract_planning_reasoning(text):
    start = re.search(
        r"(?m)^\s*PLANNING_REASONING\s*:?[ \t]*$", text
    )
    if start is None:
        return None
    segment = text[start.end():]
    end = re.search(
        r"(?m)^\s*(?:END_PLANNING_REASONING|PLAN_JSON)\s*:?[ \t]*$",
        segment,
    )
    if end is None:
        return None
    return segment[:end.start()].strip() or None


def build_prefetch_reasoning_plan_prompt(full_prompt, language="en"):
    """Prepend Agent assignments while preserving the full prompt verbatim."""
    example_plan = _extract_marked_json_array(
        full_prompt, "PLAN_JSON", "END_PLAN_JSON"
    )
    example_agents = [
        {"id": step.get("id"), "agent": step.get("agent")}
        for step in example_plan
    ]
    example = json.dumps(example_agents, ensure_ascii=False, indent=2)
    if language == "zh":
        contract = f"""在执行下方原始规划指令之前，额外先输出一个 Agent 分配区段。
该区段必须是整个响应的第一部分：

{PREFETCH_START_MARKER}
{example}
{PREFETCH_END_MARKER}

其中必须按最终执行顺序为 PLAN_JSON 的每个任务输出一项，只能包含 `id` 和
`agent`。随后逐字遵循下方原始指令要求的响应结构，依次输出完整的
PLANNING_REASONING 和 PLAN_JSON。PREFETCH_AGENTS 与最终 PLAN_JSON 的任务数量、
id 和 Agent 序列必须逐项完全一致；否则整个计划视为失败。

以下原始规划指令除增加上述前置区段外保持不变："""
    else:
        contract = f"""Before following the original planning instructions below,
output one additional Agent-assignment section as the first part of the response:

{PREFETCH_START_MARKER}
{example}
{PREFETCH_END_MARKER}

Emit exactly one entry for every PLAN_JSON task in final execution order. Each
entry must contain only `id` and `agent`. Then follow the original response
instructions below verbatim, producing the complete PLANNING_REASONING followed
by PLAN_JSON. PREFETCH_AGENTS and PLAN_JSON must have exactly matching task
counts, ids, and Agent sequences; otherwise the entire plan is invalid.

Original planning instructions, unchanged except for the added prefix above:"""
    return f"{contract}\n\n{full_prompt}"


def parse_prefetch_reasoning_plan(text, allowed_agents):
    parseable_text = _repair_plan_json_syntax_only(text, allowed_agents)
    prefetch_rows = _extract_marked_json_array(
        parseable_text, PREFETCH_START_MARKER, PREFETCH_END_MARKER
    )
    raw_plan = _extract_marked_json_array(
        parseable_text, "PLAN_JSON", "END_PLAN_JSON"
    )
    if not prefetch_rows:
        raise ValueError("PREFETCH_AGENTS must be a non-empty JSON array")
    if not raw_plan or not all(isinstance(step, dict) for step in raw_plan):
        raise ValueError("PLAN_JSON must be a non-empty array of task objects")

    names = []
    ids = []
    for position, row in enumerate(prefetch_rows, start=1):
        if not isinstance(row, dict) or set(row) != {"id", "agent"}:
            raise ValueError(
                f"PREFETCH_AGENTS[{position - 1}] must contain only id and agent"
            )
        agent = row.get("agent")
        if not isinstance(agent, str):
            raise ValueError(
                f"PREFETCH_AGENTS[{position - 1}].agent must be a string"
            )
        agent = agent.strip().lower()
        if agent not in allowed_agents:
            raise ValueError(
                f"Unsupported prefetched agent {agent!r} at position {position}; "
                f"expected one of {allowed_agents}"
            )
        ids.append(row.get("id"))
        names.append(agent)
    return names, ids, raw_plan, extract_planning_reasoning(parseable_text)


def validate_prefetch_plan_match(prefetch_names, prefetch_ids, plan):
    planned_names = [step["agent"] for step in plan]
    planned_ids = [step.get("id") for step in plan]
    mismatch = (
        len(prefetch_names) != len(planned_names)
        or [str(value) for value in prefetch_ids]
        != [str(value) for value in planned_ids]
        or prefetch_names != planned_names
    )
    if mismatch:
        prefetched = list(zip(prefetch_ids, prefetch_names))
        planned = list(zip(planned_ids, planned_names))
        raise ValueError(
            "PREFETCH_AGENTS does not exactly match PLAN_JSON: "
            f"prefetch={prefetched}, plan={planned}"
        )
    return planned_names


def replace_json_array_example(base_prompt, introduction, output_contract):
    """Replace the original plan example while preserving its planning policy."""
    if introduction not in base_prompt:
        raise ValueError("Planner prompt JSON introduction was not found")
    prefix, remainder = base_prompt.split(introduction, 1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", remainder):
        try:
            value, end = decoder.raw_decode(remainder[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value and all(
            isinstance(item, dict) for item in value
        ):
            suffix = remainder[match.start() + end:]
            return (
                f"{prefix.rstrip()}\n\n{output_contract.strip()}\n\n"
                f"{suffix.strip()}"
            )
    raise ValueError("Planner prompt JSON example was not found")


def _extract_json_object(text):
    value = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            payload, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            "agents" in payload
            or "agent_names" in payload
            or "prefetch_agents" in payload
            or "plan" in payload
            or "tasks" in payload
            or "task_details" in payload
        ):
            return payload
    raise ValueError("No agent-first JSON object found in planner output")


def parse_agent_first_plan(text, allowed_agents, task_details_format=None):
    payload = _extract_json_object(text)
    keys = list(payload)
    if "prefetch_agents" in payload or "plan" in payload:
        if keys != ["prefetch_agents", "plan"]:
            raise ValueError(
                "Planner JSON must contain prefetch_agents first and plan second"
            )
        prefetch_rows = payload["prefetch_agents"]
        plan = payload["plan"]
        if not isinstance(prefetch_rows, list) or not prefetch_rows:
            raise ValueError("prefetch_agents must be a non-empty JSON array")
        if not isinstance(plan, list) or not plan:
            raise ValueError("plan must be a non-empty JSON array")

        normalized_names = []
        for position, agent_row in enumerate(prefetch_rows, start=1):
            if not isinstance(agent_row, dict):
                raise ValueError(
                    f"prefetch_agents[{position - 1}] must be a JSON object"
                )
            if set(agent_row) != {"id", "agent"}:
                raise ValueError(
                    f"prefetch_agents[{position - 1}] must contain only id "
                    "and agent"
                )
            if str(agent_row.get("id")) != str(position):
                raise ValueError(
                    f"prefetch_agents[{position - 1}] must use id={position}, "
                    f"got {agent_row.get('id')!r}"
                )
            agent_name = agent_row.get("agent")
            if not isinstance(agent_name, str):
                raise ValueError(
                    f"prefetch_agents[{position - 1}].agent must be a string"
                )
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                raise ValueError(
                    f"Unsupported prefetched agent {agent_name!r} at position "
                    f"{position}; expected one of {allowed_agents}"
                )
            normalized_names.append(agent_name)
        if not all(isinstance(step, dict) for step in plan):
            raise ValueError("plan must contain only JSON task objects")
        return normalized_names, plan

    if "agents" in payload:
        if keys != ["agents", "tasks"]:
            raise ValueError(
                "Planner JSON must contain agents first and tasks second"
            )
        agent_rows = payload["agents"]
        tasks = payload["tasks"]
        if not isinstance(agent_rows, list) or not agent_rows:
            raise ValueError("agents must be a non-empty JSON array")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks must be a non-empty JSON array")
        if len(agent_rows) != len(tasks):
            raise ValueError(
                "agents and tasks must have the same length: "
                f"{len(agent_rows)} != {len(tasks)}"
            )

        normalized_names = []
        combined = []
        for position, (agent_row, task) in enumerate(
            zip(agent_rows, tasks), start=1
        ):
            if not isinstance(agent_row, dict):
                raise ValueError(
                    f"agents[{position - 1}] must be a JSON object"
                )
            if set(agent_row) != {"id", "agent"}:
                raise ValueError(
                    f"agents[{position - 1}] must contain only id and agent"
                )
            if str(agent_row.get("id")) != str(position):
                raise ValueError(
                    f"agents[{position - 1}] must use id={position}, got "
                    f"{agent_row.get('id')!r}"
                )
            agent_name = agent_row.get("agent")
            if not isinstance(agent_name, str):
                raise ValueError(
                    f"agents[{position - 1}].agent must be a string"
                )
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                raise ValueError(
                    f"Unsupported agent {agent_name!r} at position {position}; "
                    f"expected one of {allowed_agents}"
                )
            if not isinstance(task, dict):
                raise ValueError(f"tasks[{position - 1}] must be a JSON object")
            if str(task.get("id")) != str(position):
                raise ValueError(
                    f"tasks[{position - 1}] must use id={position}, got "
                    f"{task.get('id')!r}"
                )
            if any(key in task for key in ("agent", "name", "name_1")):
                raise ValueError(
                    f"tasks[{position - 1}] repeats an agent field; agent names "
                    "must appear only in the leading agents array"
                )
            normalized_names.append(agent_name)
            combined.append({"agent": agent_name, **task})
        return normalized_names, combined

    if "task_details" in payload:
        if not keys or keys[-1] != "task_details":
            raise ValueError(
                "task_details must be the final field after all numbered agent "
                "assignments"
            )
        assignment_ids = keys[:-1]
        if not assignment_ids:
            raise ValueError("At least one numbered agent assignment is required")
        expected_ids = [str(index) for index in range(1, len(assignment_ids) + 1)]
        if assignment_ids != expected_ids:
            raise ValueError(
                "Agent assignment fields must be consecutive task ids starting "
                f"at 1; got {assignment_ids}"
            )
        task_details = payload["task_details"]
        if (
            task_details_format == "numbered_list"
            and not isinstance(task_details, list)
        ):
            raise ValueError(
                "task_details must be a JSON array with an explicit id in "
                "every task object"
            )
        if (
            task_details_format == "numbered_object"
            and not isinstance(task_details, dict)
        ):
            raise ValueError(
                "task_details must be a JSON object keyed by task id"
            )
        if isinstance(task_details, dict):
            task_ids = list(task_details)
            if task_ids != assignment_ids:
                raise ValueError(
                    "Numbered agent assignments and task_details ids must match "
                    f"in order: {assignment_ids} != {task_ids}"
                )
            tasks = []
            for task_id in assignment_ids:
                detail = task_details[task_id]
                if not isinstance(detail, dict):
                    raise ValueError(
                        f"task_details[{task_id!r}] must be a JSON object"
                    )
                detail = dict(detail)
                explicit_id = detail.pop("id", task_id)
                if str(explicit_id) != task_id:
                    raise ValueError(
                        f"task_details[{task_id!r}] has mismatched id "
                        f"{explicit_id!r}"
                    )
                tasks.append({"id": int(task_id), **detail})
        elif isinstance(task_details, list) and task_details:
            tasks = task_details
            task_ids = [
                str(task.get("id")) if isinstance(task, dict) else None
                for task in tasks
            ]
            if task_ids != assignment_ids:
                raise ValueError(
                    "Numbered agent assignments and task_details ids must match "
                    f"in order: {assignment_ids} != {task_ids}"
                )
        else:
            raise ValueError(
                "task_details must be a non-empty JSON object or array"
            )

        normalized_names = []
        combined = []
        for task_id, task in zip(assignment_ids, tasks):
            agent_name = payload[task_id]
            if not isinstance(agent_name, str):
                raise ValueError(f"Agent assignment {task_id} must be a string")
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                raise ValueError(
                    f"Unsupported agent {agent_name!r} for task {task_id}; "
                    f"expected one of {allowed_agents}"
                )
            if any(key in task for key in ("agent", "name", "name_1")):
                raise ValueError(
                    f"Task {task_id} repeats an agent field; agent names must "
                    "appear only in the numbered prefix"
                )
            task_text = task.get("task")
            if (
                task_details_format == "numbered_list"
                and isinstance(task_text, str)
                and re.fullmatch(r"\s*\d+\s*", task_text)
            ):
                raise ValueError(
                    f"Task {task_id} contains only a numeric id instead of an "
                    "executable instruction"
                )
            normalized_names.append(agent_name)
            combined.append({"agent": agent_name, **task})
        return normalized_names, combined

    if not keys or keys[0] != "agent_names":
        raise ValueError(
            "agent_names must be the first field in the planner JSON object"
        )
    if set(keys) != {"agent_names", "tasks"}:
        raise ValueError(
            "Planner JSON object must contain only agent_names and tasks; "
            f"got fields {keys}"
        )
    agent_names = payload.get("agent_names")
    tasks = payload.get("tasks")
    if not isinstance(agent_names, list) or not agent_names:
        raise ValueError("agent_names must be a non-empty JSON array")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty JSON array")
    if len(agent_names) != len(tasks):
        raise ValueError(
            "agent_names and tasks must have the same length: "
            f"{len(agent_names)} != {len(tasks)}"
        )

    normalized_names = []
    combined = []
    for position, (agent_name, task) in enumerate(
        zip(agent_names, tasks), start=1
    ):
        if not isinstance(agent_name, str):
            raise ValueError(f"agent_names[{position - 1}] must be a string")
        agent_name = agent_name.strip().lower()
        if agent_name not in allowed_agents:
            raise ValueError(
                f"Unsupported agent {agent_name!r} at position {position}; "
                f"expected one of {allowed_agents}"
            )
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{position - 1}] must be a JSON object")
        if any(key in task for key in ("agent", "name", "name_1")):
            raise ValueError(
                f"tasks[{position - 1}] repeats an agent field; agent names "
                "must appear only in the leading agent_names array"
            )
        normalized_names.append(agent_name)
        combined.append({"agent": agent_name, **task})
    return normalized_names, combined


def extract_prefetch_agent_names(text, allowed_agents):
    """Extract an emitted prefix even when later task validation fails."""
    payload = _extract_json_object(text)
    keys = list(payload)
    if "prefetch_agents" in payload:
        if keys != ["prefetch_agents", "plan"]:
            return None
        agent_rows = payload.get("prefetch_agents")
        if not isinstance(agent_rows, list) or not agent_rows:
            return None
        normalized = []
        for position, agent_row in enumerate(agent_rows, start=1):
            if not isinstance(agent_row, dict):
                return None
            if str(agent_row.get("id")) != str(position):
                return None
            agent_name = agent_row.get("agent")
            if not isinstance(agent_name, str):
                return None
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                return None
            normalized.append(agent_name)
        return normalized

    if "agents" in payload:
        if keys != ["agents", "tasks"]:
            return None
        agent_rows = payload.get("agents")
        if not isinstance(agent_rows, list) or not agent_rows:
            return None
        normalized = []
        for position, agent_row in enumerate(agent_rows, start=1):
            if not isinstance(agent_row, dict):
                return None
            if str(agent_row.get("id")) != str(position):
                return None
            agent_name = agent_row.get("agent")
            if not isinstance(agent_name, str):
                return None
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                return None
            normalized.append(agent_name)
        return normalized

    if "task_details" in payload:
        if not keys or keys[-1] != "task_details":
            return None
        assignment_ids = keys[:-1]
        expected_ids = [str(index) for index in range(1, len(assignment_ids) + 1)]
        if not assignment_ids or assignment_ids != expected_ids:
            return None
        values = [payload[task_id] for task_id in assignment_ids]
        normalized = []
        for agent_name in values:
            if not isinstance(agent_name, str):
                return None
            agent_name = agent_name.strip().lower()
            if agent_name not in allowed_agents:
                return None
            normalized.append(agent_name)
        return normalized

    if not keys or keys[0] != "agent_names":
        return None
    agent_names = payload.get("agent_names")
    if not isinstance(agent_names, list) or not agent_names:
        return None
    normalized = []
    for agent_name in agent_names:
        if not isinstance(agent_name, str):
            return None
        agent_name = agent_name.strip().lower()
        if agent_name not in allowed_agents:
            return None
        normalized.append(agent_name)
    return normalized


def _chat_completions_url(api_url):
    value = api_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def request_completion(query, config):
    headers = {"Content-Type": "application/json"}
    if config.get("planner_api_key") is not None:
        headers["Authorization"] = f"Bearer {config['planner_api_key']}"
    payload = {
        "model": config["planner_model"],
        "messages": [
            {"role": "system", "content": config["planner_prompt"]},
            {"role": "user", "content": query},
        ],
        "temperature": config["planner_temperature"],
        "max_tokens": config["planner_max_tokens"],
    }
    response = requests.post(
        _chat_completions_url(config["planner_api_url"]),
        headers=headers,
        json=payload,
        timeout=config["timeout"],
    )
    if not response.ok:
        raise RuntimeError(
            f"Planner request failed with HTTP {response.status_code}: "
            f"{response.text}"
        )
    return response.json()["choices"][0]["message"]["content"].strip()


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def ordered_records(records_by_index, queries):
    records = []
    seen = set()
    for query in queries:
        source_index = query["source_index"]
        if source_index in records_by_index:
            records.append(records_by_index[source_index])
            seen.add(source_index)
    records.extend(
        record
        for source_index, record in records_by_index.items()
        if source_index not in seen
    )
    return records


def build_plans(queries, config, allowed_agents, normalize_plan):
    planner_mode = config.get(
        "planner_mode", "agent_names_then_task_details"
    )
    output = Path(config["plans_output"])
    existing = []
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    for row in existing:
        for field in (
            "prefetch_position_matches",
            "prefetch_precision",
            "prefetch_coverage",
            "prefetch_exact_match",
        ):
            row.pop(field, None)
        if row.get("error") is None and row.get("plan"):
            planned_agent_names = [
                step.get("agent") for step in row["plan"]
            ]
            row["planned_agent_names"] = planned_agent_names
            if isinstance(row.get("prefetch_agent_names"), list):
                row["prefetch_plan_mismatch"] = (
                    row["prefetch_agent_names"] != planned_agent_names
                )
    by_index = {row["source_index"]: row for row in existing}
    done = {
        source_index
        for source_index, row in by_index.items()
        if row.get("error") is None
        and row.get("plan")
        and row.get("planner_prompt_version") == config["prompt_version"]
        and row.get("planner_mode") == planner_mode
    }
    if existing:
        print(
            f"resume | loaded={len(existing)} | completed={len(done)} "
            f"| prompt_version={config['prompt_version']}",
            flush=True,
        )

    selected = queries[: config["limit"]] if config["limit"] else queries
    for row in selected:
        source_index = row["source_index"]
        if source_index in done:
            continue
        started = time.perf_counter()
        raw_output = None
        agent_names = None
        prefetch_ids = None
        planned_agent_names = None
        planning_reasoning = None
        prefetch_plan_mismatch = None
        record = dict(row)
        planner_input = record.pop("planner_input", None) or record["query"]
        record.setdefault("source", config["source"])
        record.update(
            {
                "planner_model": config["planner_model"],
                "planner_mode": planner_mode,
                "planner_prompt_version": config["prompt_version"],
            }
        )
        try:
            raw_output = request_completion(planner_input, config)
            if planner_mode == "prefetch_reasoning_plan":
                (
                    agent_names,
                    prefetch_ids,
                    raw_plan,
                    planning_reasoning,
                ) = parse_prefetch_reasoning_plan(raw_output, allowed_agents)
            else:
                agent_names = extract_prefetch_agent_names(
                    raw_output, allowed_agents
                )
                agent_names, raw_plan = parse_agent_first_plan(
                    raw_output,
                    allowed_agents,
                    config.get("task_details_format"),
                )
            plan = normalize_plan(raw_plan)
            planned_agent_names = [step["agent"] for step in plan]
            if planner_mode == "prefetch_reasoning_plan":
                try:
                    planned_agent_names = validate_prefetch_plan_match(
                        agent_names, prefetch_ids, plan
                    )
                    prefetch_plan_mismatch = False
                except ValueError:
                    prefetch_plan_mismatch = True
                    raise
            else:
                prefetch_plan_mismatch = agent_names != planned_agent_names
            record.update(
                {
                    "prefetch_agent_names": agent_names,
                    "prefetch_agent_ids": prefetch_ids,
                    "planned_agent_names": planned_agent_names,
                    "prefetch_plan_mismatch": prefetch_plan_mismatch,
                    "planning_reasoning": planning_reasoning,
                    "raw_plan": raw_output,
                    "plan": plan,
                    "plan_call_count": len(plan),
                    "exceeds_recommended_calls": len(plan) > 5,
                    "format_warnings": (
                        []
                        if planning_reasoning
                        else ["missing planning reasoning section"]
                    ),
                    "error": None,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "prefetch_agent_names": agent_names,
                    "prefetch_agent_ids": prefetch_ids,
                    "planned_agent_names": planned_agent_names,
                    "prefetch_plan_mismatch": prefetch_plan_mismatch,
                    "planning_reasoning": planning_reasoning,
                    "raw_plan": raw_output,
                    "plan": None,
                    "plan_call_count": None,
                    "exceeds_recommended_calls": False,
                    "error": str(exc),
                }
            )
        record["time"] = time.perf_counter() - started
        by_index[source_index] = record
        save_json(output, ordered_records(by_index, queries))
        print(
            f"planned {source_index} | prefetched="
            f"{len(record.get('prefetch_agent_names') or [])} "
            f"| reasoning_chars="
            f"{len(record.get('planning_reasoning') or '')} "
            f"| subtasks={len(record.get('plan') or [])} "
            f"| mismatch={record.get('prefetch_plan_mismatch')} "
            f"| error={record['error']}",
            flush=True,
        )
    return ordered_records(by_index, queries)


def run_builder(
    *,
    config,
    agents,
    load_queries,
    normalize_plan,
    expand_plans,
    print_summary,
    description,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", default=config["input"])
    parser.add_argument("--plans-output", default=config["plans_output"])
    parser.add_argument("--benchmark-output", default=config["benchmark_output"])
    parser.add_argument("--planner-api-url", default=config["planner_api_url"])
    parser.add_argument("--planner-api-key", default=config["planner_api_key"])
    parser.add_argument("--planner-model", default=config["planner_model"])
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=config["planner_temperature"],
    )
    parser.add_argument(
        "--planner-max-tokens",
        type=int,
        default=config["planner_max_tokens"],
    )
    parser.add_argument("--timeout", type=int, default=config["timeout"])
    parser.add_argument("--limit", type=int, default=config["limit"])
    parser.add_argument("--agents", nargs="+", choices=agents, default=agents)
    args = parser.parse_args()

    runtime_config = dict(config)
    runtime_config.update(vars(args))
    plans = build_plans(
        load_queries(runtime_config["input"]),
        runtime_config,
        agents,
        normalize_plan,
    )
    save_json(runtime_config["plans_output"], plans)
    benchmark = expand_plans(plans, runtime_config["agents"])
    save_json(runtime_config["benchmark_output"], benchmark)
    print(
        f"Saved plans: {runtime_config['plans_output']} ({len(plans)} queries)"
    )
    print(
        f"Saved benchmark: {runtime_config['benchmark_output']} "
        f"({len(benchmark)} rows)"
    )
    print_summary(plans)
    successful = [
        row for row in plans
        if row.get("error") is None and row.get("plan")
    ]
    error_count = len(plans) - len(successful)
    mismatch_count = sum(
        row.get("prefetch_plan_mismatch") is True for row in plans
    )
    print("Base Llama3 generation summary")
    print(
        f"  records={len(plans)} | success={len(successful)} "
        f"| errors={error_count} | mismatches={mismatch_count}"
    )

import re
import subprocess
import sys
import time

from planner import agent, response_text
from prompt import (
    code_agent_prompt,
    commonsense_agent_prompt,
    math_agent_prompt,
    rewrite_code_agent_prompt,
    search_agent_prompt,
    summarization_agent_prompt,
)
from search import DDGSSearch
from utils import simplify_answer


def _extract_python_code(text):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _call_agent(prompt, model="gpt-4o"):
    return response_text(agent(prompt, model=model))


def get_response(agent_name, subtask, query, history="", model="gpt-4o"):
    start_time = time.time()
    history = history or "None"

    if agent_name == "math_agent":
        prompt = math_agent_prompt % (query, subtask, history)
        answer = _call_agent(prompt, model=model)
        return {
            "task": subtask,
            "agent": agent_name,
            "response": answer,
            "time": time.time() - start_time,
        }

    if agent_name == "commonsense_agent":
        prompt = commonsense_agent_prompt % (query, subtask, history)
        answer = _call_agent(prompt, model=model)
        return {
            "task": subtask,
            "agent": agent_name,
            "response": answer,
            "time": time.time() - start_time,
        }

    if agent_name == "code_agent":
        original_answer = _call_agent(code_agent_prompt % (query, subtask, history), model=model)
        code = _extract_python_code(original_answer)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        code_output = simplify_answer(result.stdout.strip(), convert_to_str=True)
        if result.stderr.strip():
            code_output = f"[ERROR] {result.stderr.strip()}"
        rewritten = _call_agent(
            rewrite_code_agent_prompt % (subtask, code, code_output),
            model=model,
        )
        return {
            "task": subtask,
            "agent": agent_name,
            "code": code,
            "code_output": code_output,
            "response": rewritten,
            "time": time.time() - start_time,
        }

    if agent_name == "search_agent":
        search_query = _call_agent(search_agent_prompt % (subtask, history), model=model)
        answer = DDGSSearch().search(search_query)
        return {
            "task": subtask,
            "agent": agent_name,
            "search_query": search_query,
            "response": answer,
            "time": time.time() - start_time,
        }

    raise ValueError(f"Unknown agent: {agent_name}")


def normalize_plan(plan):
    normalized = []
    for index, step in enumerate(plan, start=1):
        item = dict(step)
        item.setdefault("id", index)
        if "agent" not in item:
            item["agent"] = item.get("name") or item.get("name_1")
        if not item.get("agent"):
            raise ValueError(f"Plan step {index} has no agent/name/name_1 field: {step}")
        item.setdefault("dep", [])
        normalized.append(item)
    return normalized


def _history_for_step(history, deps):
    if not deps:
        return "\n".join(history)
    selected = []
    for dep in deps:
        dep_index = int(dep) - 1
        if 0 <= dep_index < len(history):
            selected.append(history[dep_index])
    return "\n".join(selected)


def plan_execution(query, plan, dep="dep", model="gpt-4o"):
    plan = normalize_plan(plan)
    history = []
    subtasks_response = []

    for step in plan:
        history_text = _history_for_step(history, step.get("dep", [])) if dep == "dep" else "\n".join(history)
        output = get_response(step["agent"], step["task"], query, history_text, model=model)
        history.append(output["response"])
        subtasks_response.append(output)

    extract_plan = [item["task"] for item in subtasks_response]
    extract_responses = [item["response"] for item in subtasks_response]
    final_answer = _call_agent(
        summarization_agent_prompt % (query, extract_plan, extract_responses),
        model=model,
    )
    return query, subtasks_response, final_answer

import time

from planner import agent, response_text
from prompt import (
    calculation_agent_prompt,
    reasoning_agent_prompt,
    search_agent_prompt,
    summarization_agent_prompt,
)
from search import DDGSSearch


def _call_agent(prompt, model="gpt-4o"):
    return response_text(agent(prompt, model=model))


def get_response(agent_name, subtask, query, history="", model="gpt-4o"):
    start_time = time.time()
    history = history or "None"

    if agent_name == "calculation_agent":
        prompt = calculation_agent_prompt % (query, subtask, history)
        answer = _call_agent(prompt, model=model)
        return {
            "task": subtask,
            "agent": agent_name,
            "response": answer,
            "time": time.time() - start_time,
        }

    if agent_name == "reasoning_agent":
        prompt = reasoning_agent_prompt % (query, subtask, history)
        answer = _call_agent(prompt, model=model)
        return {
            "task": subtask,
            "agent": agent_name,
            "response": answer,
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
    allowed_agents = {"search_agent", "calculation_agent", "reasoning_agent"}
    if not plan:
        raise ValueError("A HuskyQA plan must contain at least one step")
    normalized = []
    for index, step in enumerate(plan, start=1):
        item = dict(step)
        item.setdefault("id", index)
        if "agent" not in item:
            item["agent"] = item.get("name") or item.get("name_1")
        if not item.get("agent"):
            raise ValueError(f"Plan step {index} has no agent/name/name_1 field: {step}")
        if item["agent"] not in allowed_agents:
            raise ValueError(f"Unknown agent: {item['agent']}")
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

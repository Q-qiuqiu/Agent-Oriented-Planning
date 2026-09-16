"""Prompt policy helpers shared by the OpenAI planner server."""

from __future__ import annotations

from typing import Dict, List, Sequence


def apply_planner_prompt_policy(
    messages: Sequence[Dict[str, str]],
    policy: str,
    agent_names: Sequence[str],
) -> List[Dict[str, str]]:
    """Return messages for the requested planner output-order policy.

    ``reasonplan`` intentionally returns an equivalent copy without injecting
    any output-format instruction. ``planreason`` is the former ``now`` policy
    and retains its plan-first, Agent-field-first prompt. ``mid`` keeps the same
    plan-first prompt for its non-priority-decoding baseline.
    """

    result = [dict(message) for message in messages]
    if policy not in {"mid", "planreason"}:
        return result

    registry = ", ".join(agent_names)
    planner_instruction = {
        "role": "system",
        "content": (
            "Preserve all planning, role-selection, dependency, and evidence "
            "rules from the caller. Change only the output order: write "
            "PLAN_JSON before PLANNING_REASONING. Use compact JSON without "
            "Markdown fences and put agent first in every plan object, followed "
            "by id, task, reason, and dep. Agent must be one of: "
            f"{registry}. Use exactly these standalone markers in order: "
            "PLAN_JSON, END_PLAN_JSON, PLANNING_REASONING, "
            "END_PLANNING_REASONING. Return no other text."
        ),
    }
    first_user = next(
        (
            index
            for index, message in enumerate(result)
            if message["role"] == "user"
        ),
        len(result),
    )
    result.insert(first_user, planner_instruction)
    return result

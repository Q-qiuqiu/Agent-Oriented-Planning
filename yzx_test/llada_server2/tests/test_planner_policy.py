from planner_policy import apply_planner_prompt_policy


BASE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Output PLANNING_REASONING, END_PLANNING_REASONING, then "
            "PLAN_JSON and END_PLAN_JSON."
        ),
    },
    {"role": "user", "content": "decompose this query"},
]
AGENTS = ["search_agent", "math_agent"]


def test_reasonplan_preserves_original_reasoning_then_plan_prompt_exactly():
    result = apply_planner_prompt_policy(BASE_MESSAGES, "reasonplan", AGENTS)

    assert result == BASE_MESSAGES
    assert result is not BASE_MESSAGES
    assert all(new is not old for new, old in zip(result, BASE_MESSAGES))


def test_planreason_injects_plan_first_agent_first_instruction():
    result = apply_planner_prompt_policy(BASE_MESSAGES, "planreason", AGENTS)

    assert len(result) == 3
    assert result[0] == BASE_MESSAGES[0]
    assert result[2] == BASE_MESSAGES[1]
    instruction = result[1]
    assert instruction["role"] == "system"
    content = instruction["content"]
    assert "write PLAN_JSON before PLANNING_REASONING" in content
    assert "put agent first" in content
    assert "search_agent, math_agent" in instruction["content"]
    assert len(content) < 600


def test_mid_keeps_planreason_prompt_and_raw_is_unchanged():
    mid = apply_planner_prompt_policy(BASE_MESSAGES, "mid", AGENTS)
    planreason = apply_planner_prompt_policy(BASE_MESSAGES, "planreason", AGENTS)
    raw = apply_planner_prompt_policy(BASE_MESSAGES, "raw", AGENTS)

    assert mid == planreason
    assert raw == BASE_MESSAGES

import pytest

from base_llama3_builder import (
    build_plans,
    parse_prefetch_reasoning_plan,
    validate_prefetch_plan_match,
)


ALLOWED_AGENTS = ("search_agent", "calculation_agent", "reasoning_agent")


def planner_response(prefetch_agent="search_agent", prefetch_id=1):
    return f"""PREFETCH_AGENTS
[
  {{"id": {prefetch_id}, "agent": "{prefetch_agent}"}}
]
END_PREFETCH_AGENTS

PLANNING_REASONING
Retrieve the required external fact before answering.
END_PLANNING_REASONING

PLAN_JSON
[
  {{"agent": "search_agent", "id": 1, "task": "Retrieve the fact", "reason": "External evidence is required", "dep": []}}
]
END_PLAN_JSON"""


def test_prefetch_reasoning_plan_parses_matching_sections():
    names, ids, plan, reasoning = parse_prefetch_reasoning_plan(
        planner_response(), ALLOWED_AGENTS
    )

    assert names == ["search_agent"]
    assert ids == [1]
    assert reasoning.startswith("Retrieve")
    assert validate_prefetch_plan_match(names, ids, plan) == ["search_agent"]


def test_missing_prefetch_end_marker_is_accepted_when_array_is_unambiguous():
    response = planner_response().replace("END_PREFETCH_AGENTS\n", "")

    names, ids, plan, _ = parse_prefetch_reasoning_plan(
        response, ALLOWED_AGENTS
    )

    assert validate_prefetch_plan_match(names, ids, plan) == ["search_agent"]


def test_plan_json_with_minor_syntax_damage_is_repaired():
    response = planner_response().replace('"id": 1, "task"', '"id": 1", "task"')

    names, ids, plan, _ = parse_prefetch_reasoning_plan(
        response, ALLOWED_AGENTS
    )

    assert validate_prefetch_plan_match(names, ids, plan) == ["search_agent"]


def test_multiple_plan_sections_are_not_silently_truncated():
    response = planner_response() + "\n\n" + planner_response()

    with pytest.raises(ValueError, match="Multiple PREFETCH_AGENTS"):
        parse_prefetch_reasoning_plan(response, ALLOWED_AGENTS)


@pytest.mark.parametrize(
    ("prefetch_agent", "prefetch_id"),
    (("reasoning_agent", 1), ("search_agent", 2)),
)
def test_prefetch_agent_or_id_mismatch_invalidates_plan(
    prefetch_agent, prefetch_id
):
    names, ids, plan, _ = parse_prefetch_reasoning_plan(
        planner_response(prefetch_agent, prefetch_id), ALLOWED_AGENTS
    )

    with pytest.raises(
        ValueError, match="does not exactly match PLAN_JSON"
    ):
        validate_prefetch_plan_match(names, ids, plan)


def test_build_plans_saves_mismatch_as_failed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "base_llama3_builder.request_completion",
        lambda query, config: planner_response("reasoning_agent", 1),
    )
    output = tmp_path / "plans.json"
    config = {
        "planner_mode": "prefetch_reasoning_plan",
        "plans_output": str(output),
        "prompt_version": "test-v1",
        "limit": None,
        "source": "test",
        "planner_model": "llama",
    }

    records = build_plans(
        [{"source_index": 1, "query": "question", "answer": "answer"}],
        config,
        ALLOWED_AGENTS,
        lambda plan: plan,
    )

    assert records[0]["prefetch_plan_mismatch"] is True
    assert records[0]["planned_agent_names"] == ["search_agent"]
    assert records[0]["plan"] is None
    assert "does not exactly match PLAN_JSON" in records[0]["error"]

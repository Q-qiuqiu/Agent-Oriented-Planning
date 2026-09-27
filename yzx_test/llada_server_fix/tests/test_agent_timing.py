import json

from agent_timing import AgentTimingRecorder


def record(recorder, query="q", method="commit", slots=None, priority=None, **kwargs):
    agent_priority = {"policy": method, "agent_slots": slots or []}
    agent_priority.update(priority or {})
    return recorder.record(
        completion_id="completion",
        created_unix=1,
        query=query,
        model="llada",
        temperature=0.0,
        requested_max_tokens=1024,
        metrics={
            "method": method,
            "generation_seconds": 12.5,
            "nfe": 1024,
            "agent_priority": agent_priority,
        },
        **kwargs,
    )


def test_compact_log_contains_predictions_natural_and_generation_time(tmp_path):
    path = tmp_path / "timings.jsonl"
    recorder = AgentTimingRecorder(str(path))
    result = record(recorder, slots=[
        {
            "slot": 0,
            "prefetched_agent": "search_agent",
            "prefetch_source": "global",
            "T_slot_prefetch": 4.0,
            "materialized_candidate": "search_agent",
            "materialized_seconds": 8.0,
        },
        {
            "slot": 1,
            "agent": "calculation_agent",
            "confirmation_seconds": 9.0,
        },
    ])
    assert result["predicted_agents"] == [{
        "slot": 0,
        "agent": "search_agent",
        "seconds": 4.0,
        "source": "global",
        "probability": None,
        "margin": None,
        "correct": True,
    }]
    assert [row["agent"] for row in result["natural_agents"]] == [
        "search_agent", "calculation_agent"
    ]
    assert result["generation_seconds"] == 12.5
    assert result["nfe"] == 1024
    assert "agent_observe" not in result
    assert "fusion_prefetch" not in result
    assert "agent_fusion" not in result
    assert "first_agent_seconds" not in result
    assert "first3_prediction_seconds" not in result
    assert "first3_natural_seconds" not in result
    assert "all_natural_seconds" not in result
    assert "T_first3_plan" not in result
    assert "T_first3_all" not in result
    assert json.loads(path.read_text()) == result


def test_base_has_only_natural_agents(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "base.jsonl"))
    result = record(recorder, method="base", slots=[{
        "slot": 0,
        "agent": "knowledge_agent",
        "confirmation_seconds": 7.0,
    }])
    assert result["predicted_agents"] == []
    assert result["natural_agents"][0]["agent"] == "knowledge_agent"


def test_plan_log_keeps_only_core_natural_and_health_metrics(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "plan.jsonl"))
    result = record(
        recorder,
        method="plan",
        slots=[{
            "slot": 0,
            "materialized_candidate": "knowledge_agent",
            "materialized_seconds": 2.5,
        }],
        priority={
            "final_plan_parse_success": True,
            "plan_end_natural_success": True,
            "reasoning_end_natural_success": True,
            "plan_capacity_exhausted": False,
            "reasoning_capacity_exhausted": False,
            "unresolved_mask_count": 0,
        },
    )
    assert "predicted_agents" not in result
    assert "agent_fusion" not in result
    assert "T_first3_all" not in result
    assert result["agent_count"] == 1
    assert "first_agent_seconds" not in result
    assert "first3_natural_seconds" not in result
    assert "T_first3_plan" not in result
    assert "all_natural_seconds" not in result
    assert result["final_plan_parse_success"] is True
    assert result["unresolved_mask_count"] == 0


def test_all_log_keeps_local_natural_fusion_and_first3_metrics(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "all.jsonl"))
    result = record(
        recorder,
        method="all",
        slots=[{
            "slot": 0,
            "T_local_prediction": 3.0,
            "T_natural_decode": 5.0,
            "T_fused_prefetch": 3.0,
            "predicted_agent": "search_agent",
            "predicted_seconds": 3.0,
            "natural_agent": "search_agent",
            "materialized_candidate": "search_agent",
            "materialized_seconds": 5.0,
            "prediction_correct": True,
            "source": "local",
            "prefetched_agent": "search_agent",
            "prefetch_source": "local",
            "T_slot_prefetch": 3.0,
            "local_lead": 2.0,
            "fused_lead": 2.0,
        }],
        priority={
            "T_first3_plan": 6.0,
            "T_first3_all": 4.0,
            "all_first3_speculative_exact": True,
            "correct_first3_lead": 2.0,
            "prediction_incremental_lead": 2.0,
            "wrong_speculative_rate": 0.0,
            "natural_fallback_rate": 0.0,
            "all_not_later_than_plan": True,
        },
    )
    assert result["agent_fusion"][0]["T_local_prediction"] == 3.0
    assert result["agent_fusion"][0]["T_natural_decode"] == 5.0
    assert result["T_first3_plan"] == 6.0
    assert result["T_first3_all"] == 4.0
    assert result["all_not_later_than_plan"] is True


def test_same_query_different_methods_are_distinct_and_retries_upsert(tmp_path):
    path = tmp_path / "methods.jsonl"
    recorder = AgentTimingRecorder(str(path))
    record(recorder, method="base")
    record(recorder, method="plan")
    retry = record(recorder, method="base")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"base", "plan"}
    assert retry["attempt_count"] == 2


def test_legacy_log_is_backed_up_and_compacted(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({
        "schema_version": 2,
        "session_id": "old",
        "request_index": 1,
        "completion_id": "old-completion",
        "created_unix": 1,
        "query": "legacy query",
        "model": "llada",
        "policy": "base",
        "status": "ok",
        "agents": [{
            "slot": 0,
            "agent": "search_agent",
            "confirmation_seconds": 3.0,
        }],
        "generation_seconds": 5.0,
        "nfe": 32,
    }) + "\n")
    recorder = AgentTimingRecorder(str(path))
    row = json.loads(path.read_text())
    assert row["schema_version"] == 4
    assert row["natural_agents"][0]["agent"] == "search_agent"
    assert "agents" not in row
    assert recorder._migration_backup_path is not None

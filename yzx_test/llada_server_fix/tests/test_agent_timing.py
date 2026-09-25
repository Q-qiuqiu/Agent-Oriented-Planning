import json

from agent_timing import AgentTimingRecorder


def record(recorder, query="q", method="commit", slots=None, **kwargs):
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
            "agent_priority": {"policy": method, "agent_slots": slots or []},
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
    assert row["schema_version"] == 3
    assert row["natural_agents"][0]["agent"] == "search_agent"
    assert "agents" not in row
    assert recorder._migration_backup_path is not None

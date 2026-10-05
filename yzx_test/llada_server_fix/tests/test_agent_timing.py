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


def test_commit_log_contains_per_slot_commits_and_request_summary(tmp_path):
    path = tmp_path / "timings.jsonl"
    recorder = AgentTimingRecorder(str(path))
    result = record(recorder, slots=[
        {
            "slot_id": 0,
            "commit_source": "semantic",
            "committed_agent": "search_agent",
            "natural_agent": "search_agent",
            "commit_wall_time": 4.0,
            "natural_agent_wall_time": 8.0,
            "prefix": None,
            "semantic_confidence": 0.95,
            "semantic_margin": 0.5,
            "stable_count": 2,
            "correct": True,
            "strict_semantic_early": True,
            "lead_seconds": 4.0,
            "prefetch_triggered": True,
            "prefetch_reused": False,
        },
        {
            "slot_id": 1,
            "commit_source": "prefix",
            "committed_agent": "calculation_agent",
            "natural_agent": "calculation_agent",
            "commit_wall_time": 6.0,
            "natural_agent_wall_time": 9.0,
            "prefix": "calc",
            "semantic_confidence": None,
            "semantic_margin": None,
            "stable_count": None,
            "correct": True,
            "strict_semantic_early": None,
            "lead_seconds": 3.0,
            "prefetch_triggered": True,
            "prefetch_reused": False,
        },
    ], priority={
        "agent_count": 2,
        "semantic_commit_count": 1,
        "prefix_commit_count": 1,
        "natural_commit_count": 0,
        "semantic_observation_stride": 8,
        "semantic_scorer_call_count": 3,
        "semantic_scorer_total_time": 0.12,
        "semantic_scorer_mean_time": 0.04,
        "semantic_skipped_same_evidence_count": 7,
        "semantic_skipped_evidence_not_ready_count": 5,
        "semantic_skipped_stride_count": 9,
        "first_agent_commit_time": 4.0,
        "all_agents_commit_time": 6.0,
        "final_plan_parse_success": True,
    })
    assert [row["commit_source"] for row in result["agent_commits"]] == [
        "semantic", "prefix"
    ]
    assert result["semantic_commit_count"] == 1
    assert result["prefix_commit_count"] == 1
    assert result["natural_commit_count"] == 0
    assert result["semantic_observation_stride"] == 8
    assert result["semantic_scorer_call_count"] == 3
    assert result["semantic_scorer_total_time"] == 0.12
    assert result["semantic_scorer_mean_time"] == 0.04
    assert result["semantic_skipped_same_evidence_count"] == 7
    assert result["semantic_skipped_evidence_not_ready_count"] == 5
    assert result["semantic_skipped_stride_count"] == 9
    assert result["first_agent_commit_time"] == 4.0
    assert result["all_agents_commit_time"] == 6.0
    assert result["full_generation_time"] == 12.5
    assert result["generation_seconds"] == 12.5
    assert result["nfe"] == 1024
    assert "step_id" not in result["agent_commits"][0]
    assert "commit_iteration" not in result["agent_commits"][0]
    assert "predicted_agent" not in result["agent_commits"][0]
    assert "final_agent" not in result["agent_commits"][0]
    assert "predicted_agents" not in result
    assert "natural_agents" not in result
    assert "agent_fusion" not in result
    assert "first_agent_seconds" not in result
    assert "first3_prediction_seconds" not in result
    assert "first3_natural_seconds" not in result
    assert "all_natural_seconds" not in result
    assert "T_first3_plan" not in result
    assert "T_first3_all" not in result
    assert json.loads(path.read_text()) == result


def test_commit_log_persists_final_latent_region_policy_metrics(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "latent.jsonl"))
    result = record(recorder, slots=[{
        "slot_id": 0,
        "commit_source": "latent_region",
        "committed_agent": "search_agent",
        "natural_agent": "search_agent",
        "commit_wall_time": 1.0,
        "natural_agent_wall_time": 2.0,
        "hypothesis_id": 7,
        "region_score": -0.2,
        "region_margin": 0.3,
        "stable_count": 2,
        "correct": True,
        "lead_seconds": 1.0,
    }], priority={
        "agent_count": 1,
        "latent_region_commit_count": 1,
        "latent_region_commit_accuracy": 1.0,
        "first_valid_commit_accuracy": 1.0,
        "correct_latent_coverage": 1.0,
        "correct_latent_commit_count": 1,
        "correct_latent_lead": {"count": 1, "mean": 1.0, "p50": 1.0, "p95": 1.0},
        "latent_trigger_count": 1,
        "latent_trigger_precision": 1.0,
        "extra_model_forwards": 0,
        "final_plan_parse_success": True,
    })

    assert result["agent_commits"][0]["hypothesis_id"] == 7
    assert result["agent_commits"][0]["region_margin"] == 0.3
    assert result["latent_region_commit_count"] == 1
    assert result["first_valid_commit_accuracy"] == 1.0
    assert result["latent_region_commit_accuracy"] == 1.0
    assert result["correct_latent_coverage"] == 1.0
    assert result["correct_latent_lead"]["p50"] == 1.0
    assert result["extra_model_forwards"] == 0
    assert result["nfe"] == 1024


def test_base_has_only_natural_agents(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "base.jsonl"))
    result = record(recorder, method="base", slots=[{
        "slot": 0,
        "agent": "knowledge_agent",
        "confirmation_seconds": 7.0,
    }])
    assert result["natural_agents"][0]["agent"] == "knowledge_agent"
    assert result["natural_agents"][0]["seconds"] == 7.0
    assert result["agent_count"] == 1
    assert result["full_generation_time"] == 12.5
    assert "predicted_agents" not in result
    assert "agent_fusion" not in result
    assert "agent_commits" not in result
    assert "first3_natural_seconds" not in result


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


def test_oracle_latent_log_preserves_full_diagnostic_trajectory(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "oracle.jsonl"))
    result = record(
        recorder,
        method="oracle_latent",
        slots=[{
            "slot_id": 0,
            "final_agent": "calculation_agent",
            "agent_first_char_time": 4.0,
            "natural_agent_time": 5.0,
            "strict_stable_correct_lead": 2.0,
            "trajectory": [{
                "iteration": 32,
                "wall_time": 3.0,
                "predicted_agent": "calculation_agent",
                "correct": True,
                "candidate_scores": {"calculation_agent": -0.1},
                "candidate_raw_logit_sums": {"calculation_agent": 12.0},
            }],
        }],
        priority={
            "runtime_agent_registry": ["calculation_agent"],
            "candidate_tokenizations": [{
                "agent_name": "calculation_agent",
                "token_ids": [1, 2],
                "token_count": 2,
            }],
            "full_sequence_observation_count": 4,
            "oracle_span_count": 1,
            "diagnostic_observer_overhead_seconds": 0.5,
            "read_only": True,
            "extra_model_forwards": 0,
            "final_plan_parse_success": True,
        },
    )

    assert result["method"] == "oracle_latent"
    assert result["diagnostic_only"] is True
    assert result["extra_model_forwards"] == 0
    assert result["agent_count"] == 1
    slot = result["oracle_latent_slots"][0]
    assert slot["strict_stable_correct_lead"] == 2.0
    event = slot["trajectory"][0]
    assert event["request_id"] == "completion"
    assert event["final_agent"] == "calculation_agent"
    assert event["natural_agent_time"] == 5.0
    assert "agent_commits" not in result
    assert json.loads((tmp_path / "oracle.jsonl").read_text()) == result


def test_online_latent_log_keeps_oracle_and_online_trajectories(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "online.jsonl"))
    result = record(
        recorder,
        method="online_latent_diagnostic",
        slots=[{
            "slot_id": 0,
            "final_agent": "search_agent",
            "agent_first_char_time": 4.0,
            "natural_agent_time": 5.0,
            "oracle_stable_lead": 3.0,
            "online_stable_lead": 2.0,
            "oracle_trajectory": [{
                "observation": 1,
                "wall_time": 2.0,
                "predicted_agent": "search_agent",
            }],
            "online_trajectory": [{
                "observation": 1,
                "wall_time": 3.0,
                "online_value_start": 42,
                "oracle_value_start": 42,
                "position_error": 0,
                "predicted_agent": "search_agent",
            }],
        }],
        priority={
            "runtime_agent_registry": ["search_agent"],
            "anchor_candidate_tokenizations": [{
                "anchor_variant": 'agent": "',
                "candidate_agent": "search_agent",
            }],
            "full_sequence_observation_count": 1,
            "oracle_span_count": 1,
            "online_anchor_slot_count": 1,
            "anchor_min_logit_margin": -6.0,
            "anchor_position_tolerance": 4,
            "anchor_stable_observations": 2,
            "read_only": True,
            "extra_model_forwards": 0,
        },
    )

    assert result["method"] == "online_latent_diagnostic"
    assert result["diagnostic_only"] is True
    assert result["extra_model_forwards"] == 0
    assert "oracle_latent_slots" not in result
    slot = result["online_latent_slots"][0]
    assert slot["oracle_trajectory"][0]["request_id"] == "completion"
    assert slot["online_trajectory"][0]["position_error"] == 0
    assert result["online_anchor_slot_count"] == 1


def test_refined_tracking_log_preserves_track_diagnostics(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "tracking.jsonl"))
    result = record(
        recorder,
        method="online_latent_refine_tracking",
        slots=[{
            "slot_id": 0,
            "final_agent": "calculation_agent",
            "oracle_trajectory": [],
            "online_trajectory": [{
                "observation": 1,
                "wall_time": 2.0,
                "coarse_value_start": 40,
                "refined_value_start": 42,
                "oracle_value_start": 42,
                "position_error": 0,
                "predicted_agent": "calculation_agent",
            }],
        }],
        priority={
            "runtime_agent_registry": ["calculation_agent"],
            "persistent_tracking": True,
            "refinement_radius": 4,
            "refinement_anchor_weight": 1.0,
            "refinement_agent_weight": 1.0,
            "tracks_created": 3,
            "stable_tracks": 2,
            "provisional_tracks": 1,
            "unmatched_candidate_count": 3,
            "track_missing_count": 4,
            "track_reassociation_count": 1,
            "slot_identity_shift_count": 2,
            "tracks": [{"track_id": 0, "refined_position_history": [42]}],
            "read_only": True,
            "extra_model_forwards": 0,
        },
    )

    assert result["method"] == "online_latent_refine_tracking"
    assert result["persistent_tracking"] is True
    assert result["refinement_radius"] == 4
    assert result["tracks_created"] == 3
    assert result["track_reassociation_count"] == 1
    assert result["tracks"][0]["track_id"] == 0
    assert result["online_latent_slots"][0]["online_trajectory"][0][
        "refined_value_start"
    ] == 42


def test_region_log_preserves_region_trajectory_and_false_tracks(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "region.jsonl"))
    result = record(
        recorder,
        method="online_latent_region",
        slots=[{
            "slot_id": 0,
            "final_agent": "calculation_agent",
            "oracle_trajectory": [],
            "online_trajectory": [],
            "region_trajectory": [{
                "observation": 1,
                "wall_time": 1.0,
                "track_id": 2,
                "track_center": 41,
                "region_left": 37,
                "region_right": 45,
                "region_contains_oracle": True,
                "predicted_agent": "calculation_agent",
            }],
        }],
        priority={
            "runtime_agent_registry": ["calculation_agent"],
            "persistent_tracking": True,
            "region_radius": 4,
            "region_top_k": 2,
            "region_temperature": 1.0,
            "region_main_aggregation": "top2_mean",
            "region_main_score_variant": "agent_only",
            "region_aggregations": ["max", "top2_mean", "soft"],
            "region_score_variants": ["agent_only", "full_template"],
            "full_template_tokenizations": [{"candidate_agent": "calculation_agent"}],
            "track_false_positive_summary": {
                "tracks_created": 3,
                "tracks_mapped_to_real_slot": 1,
                "unmapped_false_tracks": 2,
            },
            "region_track_trajectory": [{
                "track_id": 2,
                "mapped_slot_id": 0,
                "observation": 1,
            }],
            "read_only": True,
            "extra_model_forwards": 0,
        },
    )

    assert result["method"] == "online_latent_region"
    assert result["region_radius"] == 4
    assert result["region_main_aggregation"] == "top2_mean"
    assert result["track_false_positive_summary"][
        "unmapped_false_tracks"
    ] == 2
    event = result["online_latent_slots"][0]["region_trajectory"][0]
    assert event["request_id"] == "completion"
    assert event["region_contains_oracle"] is True
    assert result["region_track_trajectory"][0]["request_id"] == "completion"


def test_hypothesis_log_preserves_validation_metrics(tmp_path):
    recorder = AgentTimingRecorder(str(tmp_path / "hypothesis.jsonl"))
    result = record(
        recorder,
        method="online_latent_hypothesis",
        slots=[{
            "slot_id": 0,
            "final_agent": "calculation_agent",
            "oracle_trajectory": [],
            "online_trajectory": [],
            "region_trajectory": [],
            "hypothesis_trajectory": [{
                "observation": 2,
                "hypothesis_id": 1,
                "validated": True,
                "predicted_agent": "calculation_agent",
            }],
        }],
        priority={
            "runtime_agent_registry": ["calculation_agent"],
            "hypothesis_merge_distance": 6,
            "hypothesis_merge_gap": 2,
            "hypothesis_min_seen": 2,
            "hypothesis_min_support": 0.5,
            "hypothesis_max_center_jump": 12,
            "hypothesis_duplicate_observations": 2,
            "hypothesis_evaluation": {
                "raw_tracks": 4,
                "merged_hypotheses": 2,
                "validated_hypotheses": 1,
                "covered_slots": 1,
            },
            "hypotheses": [{"hypothesis_id": 1, "validated": True}],
            "hypothesis_trajectory": [{
                "hypothesis_id": 1,
                "observation": 2,
            }],
            "read_only": True,
            "extra_model_forwards": 0,
        },
    )

    assert result["method"] == "online_latent_hypothesis"
    assert result["hypothesis_merge_distance"] == 6
    assert result["hypothesis_evaluation"]["merged_hypotheses"] == 2
    assert result["hypothesis_trajectory"][0]["request_id"] == "completion"
    slot_event = result["online_latent_slots"][0][
        "hypothesis_trajectory"
    ][0]
    assert slot_event["request_id"] == "completion"


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
    assert row["schema_version"] == 7
    assert row["natural_agents"][0]["agent"] == "search_agent"
    assert "agents" not in row
    assert recorder._migration_backup_path is not None


def test_schema5_commit_migration_keeps_wall_times_and_drops_step_fields(tmp_path):
    path = tmp_path / "schema5_commit.jsonl"
    path.write_text(json.dumps({
        "schema_version": 5,
        "session_id": "old",
        "request_index": 1,
        "completion_id": "old-completion",
        "created_unix": 1,
        "query": "legacy commit query",
        "model": "llada",
        "method": "commit",
        "status": "ok",
        "agent_commits": [{
            "step_id": 9,
            "slot_id": 0,
            "commit_source": "prefix",
            "predicted_agent": "search_agent",
            "committed_agent": "search_agent",
            "final_agent": "search_agent",
            "commit_iteration": 224,
            "commit_wall_time": 2.0,
            "natural_agent_wall_time": 3.0,
            "lead_seconds": 1.0,
            "correct": True,
        }],
        "agent_count": 1,
        "prefix_commit_count": 1,
        "first_agent_commit_time": 2.0,
        "all_agents_commit_time": 2.0,
        "full_generation_time": 5.0,
        "generation_seconds": 5.0,
        "nfe": 32,
    }) + "\n")

    recorder = AgentTimingRecorder(str(path))
    row = json.loads(path.read_text())
    commit = row["agent_commits"][0]

    assert row["schema_version"] == 7
    assert commit["slot_id"] == 0
    assert commit["commit_wall_time"] == 2.0
    assert commit["natural_agent"] == "search_agent"
    assert "predicted_agent" not in commit
    assert "final_agent" not in commit
    assert "step_id" not in commit
    assert "commit_iteration" not in commit
    assert row["nfe"] == 32
    assert recorder._migration_backup_path is not None

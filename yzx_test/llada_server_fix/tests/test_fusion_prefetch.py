from fusion_prefetch import FusionAgentPrefetchTracker, FusionGateConfig


def row(
    agent, position, probability=0.99, margin=0.9, observed=0.0,
    track_id=None,
):
    result = {
        "agent": agent,
        "relative_pos": position,
        "probability": probability,
        "margin": margin,
        "anchor_observed_ratio": observed,
    }
    if track_id is not None:
        result["track_id"] = track_id
        result["anchor_id"] = track_id
    return result


def test_natural_is_zero_lag_fallback_and_authoritative():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    for slot, agent in enumerate(("a_agent", "b_agent", "c_agent")):
        tracker.observe_natural(slot, agent, seconds=5.0 + slot, step=slot)
    metrics = tracker.metrics(["a_agent", "b_agent", "c_agent"])
    assert metrics["first_prefetch_first3_exact"] is True
    assert metrics["T_first3_fused"] == metrics["T_first3_natural"] == 7.0
    assert metrics["first3_lead"] == 0.0
    assert metrics["source_winners"] == {
        "global": 0, "fixed_schema_completion": 0,
        "local": 0, "natural": 3
    }


def test_earliest_global_prefetch_is_corrected_by_natural():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    speculative = [row("wrong_agent", 10), row("b_agent", 30), row("c_agent", 50)]
    tracker.observe_predictions(speculative, seconds=1.0, step=1)
    tracker.observe_predictions(speculative, seconds=2.0, step=2)
    tracker.observe_natural(0, "a_agent", seconds=4.0, step=4)
    metrics = tracker.metrics(["a_agent", "b_agent", "c_agent"])
    slot = metrics["agent_slots"][0]
    assert slot["prefetch_source"] == "global"
    assert slot["prefetch_agent"] == "wrong_agent"
    assert slot["authoritative_agent"] == "a_agent"
    assert slot["wrong_prefetch"] is True
    assert slot["wrong_prefetch_duration"] == 2.0
    assert slot["fused_lead"] == 2.0


def test_local_wins_a_same_time_tie_over_global():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    materialized_anchor = [
        row("a_agent", 10, observed=1.0),
        row("b_agent", 30, observed=1.0),
        row("c_agent", 50, observed=1.0),
    ]
    tracker.observe_predictions(materialized_anchor, seconds=1.0, step=1)
    tracker.observe_predictions(materialized_anchor, seconds=2.0, step=2)
    metrics = tracker.metrics(["a_agent", "b_agent", "c_agent"])
    assert all(
        slot["prefetch_source"] == "local"
        for slot in metrics["agent_slots"]
    )


def test_late_predictions_never_replace_natural_first_prefetch():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    tracker.observe_natural(0, "a_agent", seconds=1.0, step=1)
    speculative = [row("wrong_agent", 10)]
    tracker.observe_predictions(speculative, seconds=2.0, step=2)
    tracker.observe_predictions(speculative, seconds=3.0, step=3)
    metrics = tracker.metrics(["a_agent"])
    slot = metrics["agent_slots"][0]
    assert slot["prefetch_source"] == "natural"
    assert slot["T_prefetch"] == slot["T_natural_decode"] == 1.0
    assert slot["fused_lead"] == 0.0


def test_later_global_slots_require_independent_structural_evidence():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    rows = [
        row("a_agent", 10, observed=0.0, track_id=1),
        row("a_agent", 30, observed=0.0, track_id=2),
        row("c_agent", 50, observed=0.0, track_id=3),
    ]
    tracker.observe_predictions(rows, seconds=1.0, step=1)
    tracker.observe_predictions(rows, seconds=2.0, step=2)
    metrics = tracker.metrics(["a_agent", "a_agent", "c_agent"])
    assert metrics["agent_slots"][0]["global_candidate"] == "a_agent"
    assert metrics["agent_slots"][1]["global_candidate"] is None
    assert metrics["agent_slots"][2]["global_candidate"] is None
    assert metrics["agent_slots"][1]["global_events"][-1][
        "independent_slot_ready"
    ] is False


def test_repeated_agent_is_allowed_at_distinct_independent_positions():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    rows = [
        row("search_agent", 10, observed=0.5, track_id=1),
        row("search_agent", 30, observed=0.5, track_id=2),
        row("calculation_agent", 50, observed=0.5, track_id=3),
    ]
    tracker.observe_predictions(rows, seconds=1.0, step=1)
    tracker.observe_predictions(rows, seconds=2.0, step=2)
    metrics = tracker.metrics(
        ["search_agent", "search_agent", "calculation_agent"]
    )
    assert metrics["global_first3_tuple"] == [
        "search_agent", "search_agent", "calculation_agent"
    ]
    assert metrics["global_first3_exact"] is True


def test_reused_cluster_cannot_trigger_a_later_global_slot():
    tracker = FusionAgentPrefetchTracker(FusionGateConfig())
    rows = [
        row("a_agent", 10, observed=1.0, track_id=7),
        row("a_agent", 30, observed=1.0, track_id=7),
        row("c_agent", 50, observed=1.0, track_id=9),
    ]
    tracker.observe_predictions(rows, seconds=1.0, step=1)
    tracker.observe_predictions(rows, seconds=2.0, step=2)
    metrics = tracker.metrics(["a_agent", "b_agent", "c_agent"])
    assert metrics["agent_slots"][1]["global_candidate"] is None
    event = metrics["agent_slots"][1]["global_events"][-1]
    assert event["cluster_reused"] is True
    assert event["global_ready"] is False


def test_fixed_schema_completion_resolves_only_the_remaining_unique_agent():
    tracker = FusionAgentPrefetchTracker(
        FusionGateConfig(),
        benchmark="mmlu",
        catalog=("knowledge_agent", "reasoning_agent", "elimination_agent"),
    )
    rows = [
        row("knowledge_agent", 10, observed=0.5, track_id=1),
        row("reasoning_agent", 30, observed=0.5, track_id=2),
    ]
    tracker.observe_predictions(rows, seconds=1.0, step=1)
    tracker.observe_predictions(rows, seconds=2.0, step=2)
    metrics = tracker.metrics(
        ["knowledge_agent", "reasoning_agent", "elimination_agent"]
    )
    slot3 = metrics["agent_slots"][2]
    assert slot3["global_candidate"] == "elimination_agent"
    assert slot3["global_resolution_source"] == "fixed_schema_completion"
    assert slot3["prefetch_source"] == "fixed_schema_completion"

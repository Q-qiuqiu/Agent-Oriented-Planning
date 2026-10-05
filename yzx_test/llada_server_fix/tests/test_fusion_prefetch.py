from fusion_prefetch import FusionAgentPrefetchTracker, FusionGateConfig


CATALOG = ("search_agent", "calculation_agent", "reasoning_agent")


def tracker():
    return FusionAgentPrefetchTracker(FusionGateConfig(), catalog=CATALOG)


def row(agent, position, probability=0.99, margin=0.9, fully_masked=True):
    return {
        "agent": agent,
        "relative_pos": position,
        "probability": probability,
        "margin": margin,
        "anchor_observed_ratio": 0.0,
        "semantic_value_available": True,
        "semantic_evidence_ready": True,
        "agent_value_fully_masked": fully_masked,
    }


def test_semantic_commit_uses_existing_confidence_and_stability_gates():
    state = tracker()
    prediction = [row("search_agent", 10)]
    state.observe_predictions(prediction, seconds=1.0)
    assert state.metrics(["search_agent"])["agent_slots"][0][
        "commit_source"
    ] is None

    state.observe_predictions(prediction, seconds=2.0)
    state.observe_natural(0, "search_agent", seconds=5.0)
    metrics = state.metrics([{"id": 7, "agent": "search_agent"}])
    slot = metrics["agent_slots"][0]
    assert slot["slot_id"] == 0
    assert slot["commit_source"] == "semantic"
    assert slot["committed_agent"] == "search_agent"
    assert slot["natural_agent"] == "search_agent"
    assert "predicted_agent" not in slot
    assert "final_agent" not in slot
    assert slot["semantic_confidence"] == 0.99
    assert slot["semantic_margin"] == 0.9
    assert slot["stable_count"] == 2
    assert slot["strict_semantic_early"] is True
    assert slot["correct"] is True
    assert slot["lead_seconds"] == 3.0


def test_prefix_commit_requires_a_unique_continuous_prefix():
    state = FusionAgentPrefetchTracker(
        FusionGateConfig(),
        catalog=("search_agent", "search_aux_agent", "calculation_agent"),
    )
    assert state.observe_prefix(0, "", seconds=1.0) is False
    assert state.observe_prefix(0, "search_", seconds=1.1) is False
    assert state.observe_prefix(0, "sear", seconds=1.2) is False
    # A longer unique prefix commits.
    assert state.observe_prefix(0, "search_a", seconds=1.3) is False
    assert state.observe_prefix(0, "search_ag", seconds=1.4) is True
    state.observe_natural(0, "search_agent", seconds=3.0)
    slot = state.metrics(["search_agent"])["agent_slots"][0]
    assert slot["commit_source"] == "prefix"
    assert slot["prefix"] == "search_ag"
    assert slot["committed_agent"] == "search_agent"
    assert slot["natural_agent"] == "search_agent"
    assert slot["lead_seconds"] == 1.6


def test_natural_commit_is_fallback_and_first_valid_commit_wins():
    state = tracker()
    state.observe_natural(0, "calculation_agent", seconds=2.0)
    for _ in range(3):
        state.observe_predictions(
            [row("search_agent", 10)], seconds=4.0
        )
    slot = state.metrics(["calculation_agent"])["agent_slots"][0]
    assert slot["commit_source"] == "natural"
    assert slot["committed_agent"] == "calculation_agent"
    assert slot["natural_agent"] == "calculation_agent"
    assert slot["lead_seconds"] == 0.0


def test_unresolved_slots_commit_independently_with_repeated_agents():
    state = tracker()
    agents = (
        "search_agent",
        "search_agent",
        "calculation_agent",
        "calculation_agent",
    )
    rows = [row(agent, 10 + index * 20) for index, agent in enumerate(agents)]
    state.observe_predictions(rows, seconds=1.0)
    state.observe_predictions(rows, seconds=2.0)
    assert state.committed_slots == (0, 1, 2, 3)
    for index, agent in enumerate(agents):
        state.observe_natural(
            index, agent, seconds=5.0 + index
        )
    metrics = state.metrics(agents)
    assert metrics["agent_count"] == 4
    assert metrics["semantic_commit_count"] == 4
    assert metrics["all_agents_commit_time"] == 2.0
    assert [slot["committed_agent"] for slot in metrics["agent_slots"]] == list(
        agents
    )
    assert [slot["prefetch_triggered"] for slot in metrics["agent_slots"]] == [
        True,
        False,
        True,
        False,
    ]


def test_later_slot_can_commit_while_previous_slot_lacks_evidence():
    state = tracker()
    blocked = row("search_agent", 10)
    blocked["semantic_evidence_ready"] = False
    ready = row("calculation_agent", 30)
    for seconds in (1.0, 2.0):
        state.observe_predictions([blocked, ready], seconds=seconds)
    assert state.slots[0]["commit"] is None
    assert state.slots[0]["stable_count"] == 0
    assert state.slots[1]["commit"]["agent"] == "calculation_agent"

    state.observe_predictions([row("search_agent", 10), None], seconds=3.0)
    state.observe_predictions([row("search_agent", 10), None], seconds=4.0)
    assert state.slots[0]["commit"]["agent"] == "search_agent"
    assert state.slots[1]["commit"]["seconds"] == 2.0


def test_insufficient_evidence_does_not_accumulate_semantic_stability():
    state = tracker()
    blocked = row("search_agent", 10)
    blocked["semantic_evidence_ready"] = False
    for seconds in (1.0, 2.0, 3.0):
        state.observe_predictions([blocked], seconds=seconds)
    assert state.slots[0]["stable_count"] == 0

    ready = row("search_agent", 10)
    state.observe_predictions([ready], seconds=4.0)
    assert state.slots[0]["stable_count"] == 1
    assert state.slots[0]["commit"] is None
    state.observe_predictions([ready], seconds=5.0)
    assert state.slots[0]["commit"]["source"] == "semantic"


def test_visible_agent_value_prefix_cannot_feed_semantic_commit():
    state = tracker()
    partial_value = row("search_agent", 10, fully_masked=False)
    state.observe_predictions([partial_value], seconds=1.0)
    state.observe_predictions([partial_value], seconds=2.0)
    assert state.slots[0]["stable_count"] == 0
    assert state.slots[0]["commit"] is None


def test_all_agents_commit_time_requires_every_actual_plan_slot():
    state = tracker()
    state.observe_natural(0, "search_agent", seconds=1.0)
    metrics = state.metrics(["search_agent", "reasoning_agent"])
    assert metrics["first_agent_commit_time"] == 1.0
    assert metrics["all_agents_commit_time"] is None
    assert metrics["agent_count"] == 2

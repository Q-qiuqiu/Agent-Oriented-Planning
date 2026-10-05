from offline_latent_commit_simulator import (
    evaluate_request,
    simulate_causal_triggers,
)


def event(observation, hypothesis_id, center, agent, seen, support):
    return {
        "observation": observation,
        "wall_time": float(observation),
        "hypothesis_id": hypothesis_id,
        "hypothesis_center": center,
        "region_left": center - 4,
        "region_right": center + 4,
        "seen_count": seen,
        "support_ratio": support,
        "predicted_agent": agent,
        "region_margin": 1.0,
    }


def test_causal_stability_resets_and_false_tracks_are_evaluated_afterward():
    record = {
        "request_id": "request",
        "request_index": 1,
        "runtime_agent_registry": ["search_agent", "calculation_agent"],
        "online_latent_slots": [
            {
                "slot_id": 0,
                "value_start": 100,
                "final_agent": "search_agent",
                "agent_first_char_time": 5.0,
                "natural_agent_time": 6.0,
                "oracle_trajectory": [
                    {"observation": value, "wall_time": float(value)}
                    for value in range(1, 6)
                ],
            },
            {
                "slot_id": 1,
                "value_start": 200,
                "final_agent": "calculation_agent",
                "agent_first_char_time": 8.0,
                "natural_agent_time": 9.0,
            },
        ],
        "hypothesis_trajectory": [
            event(1, 0, 100, "search_agent", 1, 1.0),
            event(2, 0, 100, "search_agent", 2, 1.0),
            event(3, 0, 100, "search_agent", 3, 1.0),
            event(4, 0, 100, "search_agent", 4, 1.0),
            event(1, 1, 300, "calculation_agent", 1, 1.0),
            event(2, 1, 300, "calculation_agent", 2, 1.0),
            event(3, 1, 300, "calculation_agent", 3, 1.0),
            event(1, 2, 200, "search_agent", 1, 1.0),
            event(2, 2, 200, "search_agent", 2, 1.0),
            event(3, 2, 200, "calculation_agent", 3, 1.0),
            # Observation 4 is missing, so observation 5 cannot extend it.
            event(5, 2, 200, "calculation_agent", 4, 0.8),
        ],
    }

    causal = simulate_causal_triggers(record)
    assert len(causal["immediate"]) == 3
    assert len(causal["stable2"]) == 2
    assert len(causal["stable3"]) == 1

    slots, triggers = evaluate_request(record, causal)
    immediate = [row for row in triggers if row["variant"] == "immediate"]
    assert sum(row["category"] == "false_track" for row in immediate) == 1
    assert slots[0]["commit_source_immediate"] == "latent_region"
    assert slots[0]["correct_immediate"] is True
    assert slots[1]["committed_agent_immediate"] == "search_agent"
    assert slots[1]["correct_immediate"] is False
    assert slots[1]["commit_source_stable2"] == "prefix"

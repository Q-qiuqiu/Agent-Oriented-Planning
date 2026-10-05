import types

import torch

from region_latent_agent_observer import RegionLatentAgentObserver


class CharacterTokenizer:
    mask_token_id = 250
    all_special_ids = [0, 250]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            chr(int(token))
            for token in ids
            if not skip_special_tokens or int(token) not in self.all_special_ids
        )


def make_observer():
    return RegionLatentAgentObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=160,
        mask_id=250,
        score_chunk_size=16,
        region_radius=4,
        region_top_k=2,
    )


def candidate(position):
    return {
        "anchor_start": position - 3,
        "anchor_end": position,
        "value_start": position,
        "coarse_anchor_start": position - 3,
        "coarse_anchor_end": position,
        "coarse_value_start": position,
        "refined_value_start": position,
        "anchor_variant": 'agent": "',
        "anchor_token_ids": [1, 2, 3],
        "anchor_score": -0.1,
        "anchor_observed_ratio": 0.0,
        "best_joint_score": -0.1,
        "refined_predicted_agent": "search_agent",
        "refinement_offset": 0,
    }


def test_region_aggregation_treats_alignment_as_nuisance_variable():
    observer = make_observer()
    available = observer.gen_length
    scores = torch.full((3, available), -20.0)
    raw = torch.zeros_like(scores)
    center = 50
    # The correct Agent is supported at two different alignments; the single
    # best alignment is deliberately not the centre/exact boundary.
    scores[1, center - observer.prompt_length + 1] = -0.2
    scores[1, center - observer.prompt_length + 2] = -0.3
    scores[0, center - observer.prompt_length] = -0.1
    scores[0, center - observer.prompt_length + 1] = -4.0
    observation = {
        "scores": scores,
        "raw_sums": raw,
        "template_scores": scores.clone(),
        "template_raw_sums": raw.clone(),
    }
    snapshot = candidate(center)
    snapshot.update({"track_id": 0, "track_stable": True,
                     "track_provisional": False, "track_miss_count": 0})

    result = observer._score_region(
        track_id=0,
        snapshot=snapshot,
        left=center - 2,
        right=center + 2,
        observation=observation,
    )

    assert result["predicted_agent_max"] == "search_agent"
    assert result["predicted_agent_top2"] == "calculation_agent"
    assert result["best_alignment_position"] == center + 1
    assert result["track_center"] == center


def test_missing_track_observation_breaks_region_stability():
    observer = make_observer()

    def event(usable, strict=True):
        return {
            "track_stable": usable,
            "region_contains_oracle": usable,
            "strict_pre_materialization": strict,
            "variant_results": {
                "agent_only": {
                    "top2_mean": {"prediction_correct": usable}
                }
            },
        }

    events = [event(True), event(False), event(True)]
    stable = observer._stable_usable_event(
        events,
        score_variant="agent_only",
        aggregation="top2_mean",
    )
    assert stable is events[2]


def test_region_diagnostic_is_read_only_and_emits_all_aggregations():
    observer = make_observer()
    canvas = torch.full((1, 163), 250, dtype=torch.long)
    canvas[:, :3] = torch.tensor([[1, 2, 3]])
    observer.initialize(canvas)
    final_text = (
        '[{"task":"one","agent": "search_agent"},'
        '{"task":"two-padding","agent": "calculation_agent"}]'
    )
    pattern = tuple(observer._encode('agent": "'))
    first_key = final_text.index('"agent"') + 1
    second_key = final_text.index('"agent"', first_key + 1) + 1
    starts = [3 + first_key, 3 + second_key]
    value_starts = [start + len(pattern) for start in starts]
    logits = torch.zeros((1, 163, 256))
    logits[:, :, ord("A")] = 10.0
    for start, value, agent in zip(
        starts, value_starts, ("search_agent", "calculation_agent")
    ):
        for offset, token_id in enumerate(pattern):
            logits[0, start + offset, token_id] = 15.0
        for offset, token_id in enumerate(observer._candidate_ids[agent]):
            logits[0, value + offset, token_id] = 15.0

    coarse_calls = iter((
        [candidate(value + 2) for value in value_starts],
        [candidate(value + 1) for value in value_starts],
    ))

    def fake_candidates(self, logits, x, logits_start):
        del self, logits, x, logits_start
        return next(coarse_calls)

    observer._online_anchor_candidates = types.MethodType(
        fake_candidates, observer
    )
    original = canvas.clone()
    observer.observe(logits, canvas, 0, 0)
    observer.observe(logits, canvas, 0, 32)
    assert torch.equal(canvas, original)

    encoded = observer.tokenizer.encode(final_text)
    canvas[0, 3:3 + len(encoded)] = torch.tensor(encoded)
    observer.step_callback(2, 0, 1, canvas)
    observer.finalize(canvas)
    observer.set_evaluation_plan_text(final_text)
    metrics = observer.metrics()

    assert metrics["policy"] == "online_latent_region"
    assert metrics["diagnostic_only"] is True
    assert metrics["read_only"] is True
    assert metrics["extra_model_forwards"] == 0
    assert metrics["region_radius"] == 4
    assert metrics["region_main_aggregation"] == "top2_mean"
    assert len(metrics["agent_slots"]) == 2
    assert metrics["full_template_tokenizations"]
    assert metrics["region_track_trajectory"]
    assert all(
        "mapped_slot_id" in event
        for event in metrics["region_track_trajectory"]
    )
    for slot in metrics["agent_slots"]:
        assert slot["first_stable_track_time"] is not None
        assert slot["first_region_contains_oracle_time"] is not None
        assert slot["region_stable_correct_time"] is not None
        assert slot["region_stable_lead"] is not None
        assert slot["region_trajectory"]
        event = slot["region_trajectory"][0]
        assert set(event["variant_results"]["agent_only"]) == {
            "max", "top2_mean", "soft"
        }
        assert "full_template" in event["variant_results"]

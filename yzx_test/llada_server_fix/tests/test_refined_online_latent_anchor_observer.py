import torch
import types

from refined_online_latent_anchor_observer import (
    RefinedOnlineLatentAnchorObserver,
)


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


def make_observer(*, tracking=False):
    return RefinedOnlineLatentAnchorObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=128,
        mask_id=250,
        persistent_tracking=tracking,
        score_chunk_size=16,
    )


def test_joint_local_refinement_recovers_exact_value_start_from_plus_two():
    observer = make_observer()
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    canvas[:, :3] = torch.tensor([[1, 2, 3]])
    observer.initialize(canvas)
    logits = torch.zeros((1, 131, 256))
    logits[:, :, ord("A")] = 10.0
    pattern = tuple(observer._encode('agent": "'))
    anchor_start = 40
    value_start = anchor_start + len(pattern)
    for offset, token_id in enumerate(pattern):
        logits[0, anchor_start + offset, token_id] = 15.0
    for offset, token_id in enumerate(observer._candidate_ids["search_agent"]):
        logits[0, value_start + offset, token_id] = 15.0
    scores, raw = observer._compress_full_sequence_scores(logits)
    observation = {"scores": scores, "raw_sums": raw}
    coarse = {
        "anchor_start": anchor_start + 2,
        "anchor_end": value_start + 2,
        "value_start": value_start + 2,
        "anchor_variant": 'agent": "',
        "anchor_token_ids": list(pattern),
        "anchor_score": -0.5,
        "anchor_observed_ratio": 0.0,
        "speculative": True,
    }
    original = canvas.clone()

    refined = observer._refine_anchor(
        coarse=coarse,
        observation=observation,
        logits=logits,
        x=canvas,
        logits_start=0,
        sequence_max=logits[0].amax(dim=-1),
    )

    assert torch.equal(canvas, original)
    assert refined["coarse_value_start"] == value_start + 2
    assert refined["refined_value_start"] == value_start
    assert refined["refinement_offset"] == -2
    assert refined["refined_predicted_agent"] == "search_agent"
    assert refined["best_joint_score"] > refined["second_best_joint_score"]


def candidate(position, agent="search_agent", score=-0.1):
    return {
        "anchor_start": position - 3,
        "anchor_end": position,
        "value_start": position,
        "coarse_anchor_start": position - 3,
        "coarse_anchor_end": position,
        "coarse_value_start": position,
        "refined_value_start": position,
        "anchor_score": score,
        "best_joint_score": score,
        "refined_predicted_agent": agent,
        "refinement_offset": 0,
    }


def observation(number):
    return {
        "observation": number,
        "iteration": number * 32,
        "wall_time": float(number),
    }


def test_persistent_monotonic_tracks_survive_front_insertion_and_missing():
    observer = make_observer(tracking=True)

    first = observer._track_candidates(
        [candidate(100), candidate(200, "calculation_agent")],
        observation(1),
    )
    assert first[0]["refined_value_start"] == 100
    assert first[1]["refined_value_start"] == 200

    second = observer._track_candidates(
        [candidate(50), candidate(101), candidate(201, "calculation_agent")],
        observation(2),
    )
    assert second[0]["refined_value_start"] == 101
    assert second[1]["refined_value_start"] == 201
    assert second[2]["refined_value_start"] == 50
    assert observer._tracks[0]["stable"] is True
    assert observer._tracks[1]["stable"] is True

    observer._track_candidates(
        [candidate(202, "calculation_agent")], observation(3)
    )
    fourth = observer._track_candidates(
        [candidate(103), candidate(203, "calculation_agent")],
        observation(4),
    )
    assert fourth[0]["refined_value_start"] == 103
    assert fourth[1]["refined_value_start"] == 203
    assert observer._tracks[0]["reassociation_count"] == 1
    assert observer._tracks[0]["track_id"] == 0
    assert observer._tracks[1]["track_id"] == 1
    assert observer._track_reassociation_count >= 1
    assert observer._unmatched_candidate_count == 3


def test_tracking_observer_finalizes_joint_refined_slot_trajectories():
    observer = make_observer(tracking=True)
    canvas = torch.full((1, 131), 250, dtype=torch.long)
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
    logits = torch.zeros((1, 131, 256))
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
        [candidate(value) for value in value_starts],
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

    assert metrics["policy"] == "online_latent_refine_tracking"
    assert metrics["persistent_tracking"] is True
    assert metrics["extra_model_forwards"] == 0
    assert metrics["stable_tracks"] == 2
    assert len(metrics["agent_slots"]) == 2
    assert {slot["track_id"] for slot in metrics["agent_slots"]} == {0, 1}
    for slot in metrics["agent_slots"]:
        assert slot["first_exact_refined_time"] is not None
        assert slot["first_stable_exact_refined_time"] is not None
        assert all(
            event["refined_position_error"] == 0
            for event in slot["online_trajectory"]
        )

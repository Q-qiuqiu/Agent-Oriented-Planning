import time

import torch

from online_latent_anchor_observer import OnlineLatentAnchorObserver


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


def _logits(observer, canvas, anchors, agents):
    logits = torch.zeros((1, canvas.shape[1], 256))
    # Suppress all unintended anchor positions below the legacy -6 margin.
    logits[:, :, ord("A")] = 10.0
    pattern = tuple(observer._encode('agent": "'))
    for anchor_start, agent in zip(anchors, agents):
        for offset, token_id in enumerate(pattern):
            logits[0, anchor_start + offset, token_id] = 15.0
        value_start = anchor_start + len(pattern)
        for offset, token_id in enumerate(observer._candidate_ids[agent]):
            logits[0, value_start + offset, token_id] = 14.0
    return logits


def test_online_anchor_diagnostic_tracks_all_slots_without_mutating_canvas():
    tokenizer = CharacterTokenizer()
    catalog = ["search_agent", "calculation_agent", "reasoning_agent"]
    prompt_length = 3
    gen_length = 160
    observer = OnlineLatentAnchorObserver(
        tokenizer=tokenizer,
        catalog=catalog,
        prompt_length=prompt_length,
        gen_length=gen_length,
        mask_id=tokenizer.mask_token_id,
        score_chunk_size=16,
        anchor_min_logit_margin=-0.01,
    )
    canvas = torch.full(
        (1, prompt_length + gen_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    canvas[:, :prompt_length] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)

    final_text = (
        '[{"task":"one","agent": "search_agent"},'
        '{"task":"two-long-padding","agent": "calculation_agent"}]'
    )
    pattern = tuple(observer._encode('agent": "'))
    first_key = final_text.index('"agent"') + 1
    second_key = final_text.index('"agent"', first_key + 1) + 1
    first = prompt_length + first_key
    second = prompt_length + second_key
    original = canvas.clone()

    observer.observe(
        _logits(observer, canvas, [first + 2, second + 2], catalog[:2]),
        canvas,
        0,
        0,
    )
    time.sleep(0.001)
    observer.observe(
        _logits(observer, canvas, [first, second], catalog[:2]),
        canvas,
        0,
        32,
    )
    assert torch.equal(canvas, original)

    encoded = tokenizer.encode(final_text)
    canvas[0, prompt_length:prompt_length + len(encoded)] = torch.tensor(encoded)
    observer.step_callback(2, 0, 1, canvas)
    observer.finalize(canvas)
    observer.set_evaluation_plan_text(final_text)
    metrics = observer.metrics()

    assert metrics["diagnostic_only"] is True
    assert metrics["read_only"] is True
    assert metrics["extra_model_forwards"] == 0
    assert metrics["online_anchor_slot_count"] == 2
    assert metrics["online_anchor_observation_counts"] == [2, 2]
    assert len(metrics["agent_slots"]) == 2
    for slot_id, slot in enumerate(metrics["agent_slots"]):
        assert slot["slot_id"] == slot_id
        assert slot["anchor_found"] is True
        assert slot["correct_anchor_found"] is True
        assert slot["stable_correct_anchor"] is True
        assert slot["first_correct_anchor_time"] is not None
        assert slot["online_prediction_count"] == 2
        assert len(slot["online_trajectory"]) == 2
        assert len(slot["oracle_trajectory"]) == 2
        assert slot["online_trajectory"][0]["anchor_position_delta"] is None
        assert slot["online_trajectory"][1]["anchor_position_delta"] == -2
        assert slot["online_trajectory"][1]["position_stable_count"] == 2
        assert slot["online_trajectory"][1]["online_value_start"] == (
            [first, second][slot_id] + len(pattern)
        )


def test_online_anchor_tokenizer_diagnostics_cover_registry_and_variants():
    observer = OnlineLatentAnchorObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["search_agent", "calculation_agent"],
        prompt_length=1,
        gen_length=64,
        mask_id=250,
    )
    rows = observer.metrics()["anchor_candidate_tokenizations"]
    assert rows
    assert {row["candidate_agent"] for row in rows} == {
        "search_agent", "calculation_agent"
    }
    assert any(row["anchor_variant"] == 'agent":"' for row in rows)
    assert any(row["anchor_variant"] == 'agent": "' for row in rows)
    assert all(row["separate_equals_combined"] for row in rows)

import time
import types

import torch

from generate import generate_with_dual_cache
from oracle_latent_observer import OracleLatentAgentObserver


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


class ConstantModel:
    device = torch.device("cpu")

    def __call__(self, x, **kwargs):
        del kwargs
        logits = torch.zeros((x.shape[0], x.shape[1], 256))
        logits[:, :, ord("A")] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=((torch.zeros(1),),),
        )


def test_oracle_observer_is_read_only_and_adds_no_forward():
    tokenizer = CharacterTokenizer()
    prompt = torch.tensor([[2, 3, 4]])
    plain, plain_nfe = generate_with_dual_cache(
        ConstantModel(),
        prompt,
        steps=64,
        gen_length=64,
        block_length=32,
        threshold=0.9,
        mask_id=tokenizer.mask_token_id,
    )
    observer = OracleLatentAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=64,
        mask_id=tokenizer.mask_token_id,
        score_chunk_size=8,
    )
    observed, observed_nfe = generate_with_dual_cache(
        ConstantModel(),
        prompt,
        steps=64,
        gen_length=64,
        block_length=32,
        threshold=0.9,
        mask_id=tokenizer.mask_token_id,
        agent_controller=observer,
        step_callback=observer.step_callback,
    )

    assert torch.equal(plain, observed)
    assert plain_nfe == observed_nfe
    metrics = observer.metrics()
    assert metrics["diagnostic_only"] is True
    assert metrics["read_only"] is True
    assert metrics["extra_model_forwards"] == 0
    assert metrics["full_sequence_observation_count"] == 2


def test_oracle_replay_finds_strict_stable_calculation_after_search_flip():
    tokenizer = CharacterTokenizer()
    catalog = ["search_agent", "calculation_agent", "reasoning_agent"]
    prompt_length = 3
    gen_length = 96
    observer = OracleLatentAgentObserver(
        tokenizer=tokenizer,
        catalog=catalog,
        prompt_length=prompt_length,
        gen_length=gen_length,
        mask_id=tokenizer.mask_token_id,
        score_chunk_size=8,
    )
    canvas = torch.full(
        (1, prompt_length + gen_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    canvas[:, :prompt_length] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    final_text = '[{"agent":"calculation_agent"}]'
    value_start = prompt_length + final_text.index("calculation_agent")

    def logits_for(agent):
        logits = torch.zeros((1, canvas.shape[1], 256))
        for offset, token_id in enumerate(observer._candidate_ids[agent]):
            logits[0, value_start + offset, token_id] = 12.0
        return logits

    original = canvas.clone()
    observer.observe(logits_for("search_agent"), canvas, 0, 0)
    time.sleep(0.001)
    observer.observe(logits_for("calculation_agent"), canvas, 0, 1)
    assert torch.equal(canvas, original)

    encoded = tokenizer.encode(final_text)
    canvas[0, prompt_length:prompt_length + len(encoded)] = torch.tensor(encoded)
    observer.step_callback(2, 0, 1, canvas)
    observer.finalize(canvas)
    observer.set_evaluation_plan_text(final_text)
    metrics = observer.metrics()

    assert metrics["oracle_span_count"] == 1
    assert metrics["agent_count"] == 1
    slot = metrics["agent_slots"][0]
    assert slot["final_agent"] == "calculation_agent"
    assert slot["first_latent_prediction"] == "search_agent"
    assert slot["first_prediction_correct"] is False
    assert slot["earliest_stable_correct_latent_time"] is not None
    assert slot["strict_stable_correct_latent_time"] is not None
    assert slot["strict_stable_correct_lead"] > 0
    assert slot["top1_flip_count"] == 1
    assert slot["wrong_search_to_stable_calculation_seconds"] > 0
    assert len(slot["trajectory"]) == 2
    assert all(
        event["strict_pre_materialization"]
        and event["agent_value_visible_token_count"] == 0
        for event in slot["trajectory"]
    )
    assert all(
        row["token_count"] == len(row["token_ids"])
        for row in metrics["candidate_tokenizations"]
    )
    assert all(
        check["separate_equals_combined"]
        for row in metrics["candidate_tokenizations"]
        for check in row["boundary_checks"]
    )


def test_oracle_uses_mean_logprob_and_keeps_raw_sum():
    tokenizer = CharacterTokenizer()
    observer = OracleLatentAgentObserver(
        tokenizer=tokenizer,
        catalog=["a_agent", "much_longer_agent"],
        prompt_length=1,
        gen_length=32,
        mask_id=tokenizer.mask_token_id,
        score_chunk_size=4,
    )
    canvas = torch.full((1, 33), tokenizer.mask_token_id)
    canvas[:, 0] = 1
    observer.initialize(canvas)
    logits = torch.randn(1, 33, 256)
    normalized, raw = observer._compress_full_sequence_scores(logits)

    assert normalized.shape == (2, 32)
    assert raw.shape == (2, 32)
    assert torch.isfinite(normalized[0, 0])
    assert torch.isfinite(raw[0, 0])
    assert not torch.equal(normalized[:, 0], raw[:, 0])

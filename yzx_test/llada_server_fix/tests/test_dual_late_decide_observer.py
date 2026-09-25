import types

import torch

from dual_late_decide_observer import DualVanillaLateDecideObserver
from generate import generate_with_dual_cache


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
        replace = kwargs.get("replace_position")
        logits = torch.zeros((x.shape[0], x.shape[1], 256))
        logits[:, :, ord("A")] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=((torch.zeros(1),),),
        )


def run(observer):
    prompt = torch.tensor([[2, 3, 4]])
    output, nfe = generate_with_dual_cache(
        ConstantModel(),
        prompt,
        steps=64,
        gen_length=64,
        block_length=32,
        threshold=0.9,
        mask_id=CharacterTokenizer.mask_token_id,
        agent_controller=observer,
        step_callback=observer.step_callback if observer is not None else None,
    )
    return output, nfe


def test_dual_late_decide_observer_is_read_only_and_adds_no_forward():
    plain, plain_nfe = run(None)
    observer = DualVanillaLateDecideObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["a_agent", "b_agent", "c_agent"],
        priority_slots=3,
        tracking_slots=8,
        prompt_length=3,
        gen_length=64,
        mask_id=CharacterTokenizer.mask_token_id,
    )
    observed, observed_nfe = run(observer)
    assert torch.equal(plain, observed)
    assert plain_nfe == observed_nfe
    assert observer.has_unconfirmed_agents() is False
    metrics = observer.metrics()
    assert metrics["read_only"] is True
    assert metrics["probe_forwards"] == 0



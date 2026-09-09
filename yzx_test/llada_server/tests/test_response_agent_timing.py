import torch

from json_agent_priority import JsonAgentPriorityConfig
from response_agent_timing import PassiveJsonAgentMonitor, infer_benchmark


class CharacterTokenizer:
    mask_token_id = 255

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 1 for character in text]

    @staticmethod
    def decode(ids, **kwargs):
        del kwargs
        return "".join(chr(int(token_id) - 1) for token_id in ids)


def test_infer_benchmark_ignores_registry_order():
    assert infer_benchmark(
        ["reasoning_agent", "search_agent", "calculation_agent"]
    ) == "huskyqa"


def test_unknown_registry_is_not_assigned_to_a_benchmark():
    assert infer_benchmark(["unknown_agent", "reasoning_agent"]) is None


def test_passive_monitor_records_repeated_fields_without_mutating_response():
    tokenizer = CharacterTokenizer()
    text = (
        'agent":"search_agent","next":'
        'agent":"search_agent","last":'
        'agent":"calculation_agent"'
    )
    prompt_length = 2
    gen_length = len(text) + 8
    monitor = PassiveJsonAgentMonitor(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=(
                "search_agent",
                "calculation_agent",
                "reasoning_agent",
            ),
            priority_slots=3,
            tracking_slots=8,
        ),
        prompt_length=prompt_length,
        gen_length=gen_length,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full(
        (1, prompt_length + gen_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    encoded = tokenizer.encode(text)
    x[0, prompt_length : prompt_length + len(encoded)] = torch.tensor(encoded)
    before = x.clone()
    monitor.initialize(x)
    monitor.observe(None, x, 0, 7)
    assert torch.equal(x, before)

    metrics = monitor.metrics()
    assert metrics["timing_source"] == "passive_materialized_response"
    assert [slot["agent"] for slot in metrics["agent_slots"]] == [
        "search_agent",
        "search_agent",
        "calculation_agent",
    ]
    assert all(slot["priority"] is False for slot in metrics["agent_slots"])
    assert all(slot["recognized_step"] == 7 for slot in metrics["agent_slots"])


def test_step_callback_records_name_immediately_after_decoder_update():
    tokenizer = CharacterTokenizer()
    text = 'agent":"reasoning_agent"'
    monitor = PassiveJsonAgentMonitor(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=("search_agent", "calculation_agent", "reasoning_agent"),
            priority_slots=1,
            tracking_slots=1,
        ),
        prompt_length=1,
        gen_length=len(text),
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full(
        (1, 1 + len(text)),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    monitor.initialize(x)
    x[0, 1:] = torch.tensor(tokenizer.encode(text))
    before = x.clone()

    monitor.step_callback(3, 0, 2, x)

    assert torch.equal(x, before)
    slot = monitor.metrics()["agent_slots"][0]
    assert slot["agent"] == "reasoning_agent"
    assert slot["recognized_step"] == 3

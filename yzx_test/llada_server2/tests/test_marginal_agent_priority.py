import torch

from marginal_agent_priority import MarginalizedAgentFieldController
from json_agent_priority import JsonAgentPriorityConfig


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


def make_controller(stable_steps=2):
    tokenizer = CharacterTokenizer()
    controller = MarginalizedAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent"],
            priority_slots=3,
            tracking_slots=3,
            confirm_stable_steps=stable_steps,
        ),
        prompt_length=8,
        gen_length=220,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 228), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :8] = 1
    controller.initialize(x)
    return controller, x


def logits_for(controller, x, starts, agents):
    logits = torch.full((1, x.shape[1], 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = controller.anchor_variants[0]
    for start, agent in zip(starts, agents):
        for offset, token_id in enumerate(anchor):
            logits[0, start + offset, token_id] = 10.0
        name_start = start + len(anchor)
        for offset, token_id in enumerate(controller.padded_catalog_ids[agent]):
            logits[0, name_start + offset, token_id] = 10.0
    return logits


def materialize_plan_fields(controller, x, starts, agents):
    anchor = controller.anchor_variants[0]
    for start, agent in zip(starts, agents):
        anchor_tensor = torch.tensor(anchor, dtype=x.dtype)
        x[0, start:start + len(anchor)] = anchor_tensor
        name_start = start + len(anchor)
        value = torch.tensor(controller.catalog_value_ids[agent], dtype=x.dtype)
        x[0, name_start:name_start + len(value)] = value


def test_moving_positions_accumulate_by_order_without_modifying_output():
    controller, x = make_controller()
    agents = ["search_agent", "search_agent", "math_agent"]
    original = x.clone()

    controller.observe(logits_for(controller, x, [24, 84, 144], agents), x, 0, 0)
    assert controller.metrics()["recognized_agent_fields"] == 0
    controller.observe(logits_for(controller, x, [34, 94, 154], agents), x, 0, 32)

    metrics = controller.metrics()
    assert [slot["agent"] for slot in metrics["agent_slots"]] == agents
    assert all(
        slot["decision_source"] == "independent_slot_map"
        for slot in metrics["agent_slots"]
    )
    assert metrics["method"] == "independent_slot_map_v4"
    assert metrics["sequence_probability"] is None
    assert metrics["all_priority_agents_recognized"] is True
    assert torch.equal(x, original)


def test_final_plan_checks_prefetch_accuracy_without_replacing_prediction():
    controller, x = make_controller()
    predicted = ["search_agent", "search_agent", "math_agent"]
    starts = [24, 84, 144]
    controller.observe(logits_for(controller, x, starts, predicted), x, 0, 0)
    controller.observe(logits_for(controller, x, starts, predicted), x, 0, 32)

    final = ["search_agent", "math_agent", "math_agent"]
    materialize_plan_fields(controller, x, starts, final)
    controller.finalize(x)
    metrics = controller.metrics()

    assert [slot["agent"] for slot in metrics["agent_slots"]] == predicted
    assert [slot["final_agent"] for slot in metrics["agent_slots"]] == final
    assert [slot["prediction_correct"] for slot in metrics["agent_slots"]] == [
        True,
        False,
        True,
    ]
    assert all(
        slot["final_agent_seconds"] is not None
        for slot in metrics["agent_slots"]
    )
    assert metrics["all_final_agents_seconds"] is not None
    assert metrics["prefetch_switch_count"] == 1
    assert metrics["last_agent_correction_seconds"] is not None
    assert metrics["effective_all_agents_ready_seconds"] is not None
    assert metrics["effective_prefetch_lead_seconds"] is not None
    assert metrics["agent_slots"][1]["switch_required"] is True
    assert metrics["agent_slots"][1]["switch_seconds"] is not None
    assert metrics["prediction_accuracy"] == 2 / 3


def test_prefetch_requires_configured_consecutive_name_observations():
    controller, x = make_controller(stable_steps=4)
    starts = [24, 84, 144]
    agents = ["search_agent", "code_agent", "math_agent"]
    logits = logits_for(controller, x, starts, agents)

    for step in (0, 32, 64):
        controller.observe(logits, x, 0, step)
        assert controller.metrics()["recognized_agent_fields"] == 0

    controller.observe(logits, x, 0, 96)
    assert controller.metrics()["recognized_agent_fields"] == 3


def test_predicted_slot_missing_from_final_plan_counts_as_incorrect():
    controller, x = make_controller()
    predicted = ["search_agent", "search_agent", "math_agent"]
    starts = [24, 84, 144]
    controller.observe(logits_for(controller, x, starts, predicted), x, 0, 0)
    controller.observe(logits_for(controller, x, starts, predicted), x, 0, 32)

    materialize_plan_fields(controller, x, starts[:2], predicted[:2])
    controller.finalize(x)
    metrics = controller.metrics()

    assert [slot["prediction_correct"] for slot in metrics["agent_slots"]] == [
        True,
        True,
        False,
    ]
    assert metrics["prediction_accuracy"] == 2 / 3


def test_low_confidence_observations_do_not_force_prefetch_decisions():
    controller, x = make_controller()
    starts = [24, 84, 144]
    logits = torch.full((1, x.shape[1], 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = controller.anchor_variants[0]
    for start in starts:
        for offset, token_id in enumerate(anchor):
            logits[0, start + offset, token_id] = 10.0

    controller.observe(logits, x, 0, 0)
    controller.observe(logits, x, 0, 32)
    assert controller.metrics()["recognized_agent_fields"] == 0

    controller.observe(logits, x, 0, 64)
    metrics = controller.metrics()
    assert metrics["recognized_agent_fields"] == 0
    assert metrics["all_priority_agents_recognized"] is False

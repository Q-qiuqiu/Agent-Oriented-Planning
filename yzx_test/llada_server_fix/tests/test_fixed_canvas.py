import types

import torch

from dynamic_fixed_canvas import DynamicFixedCanvasMonitor as FixedCanvasMonitor
from generate import generate_with_fixed_canvas_dual_cache


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


class PositionedModel:
    device = torch.device("cpu")

    def __init__(self, token_by_position):
        self.token_by_position = token_by_position

    def __call__(self, x, **kwargs):
        replace = kwargs.get("replace_position")
        if replace is None:
            positions = list(range(x.shape[1]))
        else:
            positions = replace[0].nonzero(as_tuple=False).flatten().tolist()
            assert len(positions) == x.shape[1]
        logits = torch.zeros((x.shape[0], x.shape[1], 256))
        for local, absolute in enumerate(positions):
            logits[:, local, self.token_by_position.get(absolute, ord("A"))] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=((torch.zeros(1),),),
        )


def run(mode):
    prompt = torch.tensor([[2, 3, 4]])
    monitor = FixedCanvasMonitor(
        tokenizer=CharacterTokenizer(),
        catalog=["a_agent", "b_agent"],
        prompt_length=prompt.shape[1],
        gen_length=128,
        mask_id=CharacterTokenizer.mask_token_id,
        reasoning_budget=32,
        structure_mode=mode,
    )
    output, nfe = generate_with_fixed_canvas_dual_cache(
        ConstantModel(),
        prompt,
        steps=128,
        gen_length=128,
        block_length=32,
        threshold=0.9,
        mask_id=CharacterTokenizer.mask_token_id,
        agent_controller=monitor,
        structure_mode=mode,
    )
    return output, nfe, monitor.metrics()


def test_fixed_canvas_vanilla_finishes_and_protects_delimiters():
    output, nfe, metrics = run("fixed_canvas_vanilla")
    assert nfe > 0
    assert int((output == CharacterTokenizer.mask_token_id).sum()) == 0
    assert metrics["fixed_token_corruption_count"] == 0
    assert metrics["unresolved_mask_count"] == 0
    assert metrics["schedule"][0]["phase"] == "reasoning"
    assert metrics["reasoning_capacity_exhausted"] is True
    assert metrics["plan_capacity_exhausted"] is True


def test_fixed_canvas_plan_first_changes_only_region_visit_order():
    vanilla, _, vanilla_metrics = run("fixed_canvas_vanilla")
    plan_first, _, plan_metrics = run("fixed_canvas_plan_first")
    assert plan_metrics["schedule"][0]["phase"] == "plan"
    assert vanilla_metrics["schedule"][0]["phase"] == "reasoning"
    assert plan_metrics["fixed_token_corruption_count"] == 0
    assert int((plan_first == CharacterTokenizer.mask_token_id).sum()) == 0
    # A constant model proposes the same token everywhere, so changing only
    # region order must leave the generated canvas identical.
    assert torch.equal(vanilla, plan_first)


def test_schema_valid_contiguous_json_stops_and_compacts_plan_capacity():
    tokenizer = CharacterTokenizer()
    prompt = torch.tensor([[2, 3, 4]])
    monitor = FixedCanvasMonitor(
        tokenizer=tokenizer,
        catalog=["a_agent", "b_agent"],
        prompt_length=prompt.shape[1],
        gen_length=256,
        mask_id=tokenizer.mask_token_id,
        structure_mode="fixed_canvas_plan_first",
    )
    plan_text = (
        '[{"agent":"a_agent","id":1,"task":"x",'
        '"reason":"y","dep":[]}]'
    )
    emitted = plan_text + "\nEND_PLAN_JSON"
    token_by_position = {
        monitor.layout.plan_start + offset: token
        for offset, token in enumerate(tokenizer.encode(emitted))
    }
    output, _ = generate_with_fixed_canvas_dual_cache(
        PositionedModel(token_by_position),
        prompt,
        steps=256,
        gen_length=256,
        block_length=32,
        threshold=0.9,
        mask_id=tokenizer.mask_token_id,
        agent_controller=monitor,
        structure_mode="fixed_canvas_plan_first",
    )
    metrics = monitor.metrics()
    assert metrics["plan_json_complete"] is True
    assert metrics["plan_capacity_overflow"] is False
    assert metrics["plan_effective_tokens"] == len(plan_text)
    assert metrics["unused_plan_capacity"] > 0
    assert metrics["plan_tokens_after_json_before_detection"] >= 0
    assert metrics["fixed_token_corruption_count"] == 0
    assert int((output == tokenizer.mask_token_id).sum()) == 0
    assert output.shape[1] < prompt.shape[1] + 256
    decoded = tokenizer.decode(output[0, prompt.shape[1] :].tolist())
    assert plan_text + "\nEND_PLAN_JSON" in decoded
    assert metrics["plan_end_natural_success"] is True
    assert metrics["plan_capacity_exhausted"] is False


def test_both_natural_end_markers_compact_in_either_phase_order():
    tokenizer = CharacterTokenizer()
    prompt = torch.tensor([[2, 3, 4]])
    reasoning_text = "Two short sentences. Done."
    reasoning_emitted = reasoning_text + "\nEND_PLANNING_REASONING"
    plan_text = (
        '[{"agent":"a_agent","id":1,"task":"x",'
        '"reason":"y","dep":[]}]'
    )
    plan_emitted = plan_text + "\nEND_PLAN_JSON"

    for mode in ("fixed_canvas_vanilla", "fixed_canvas_plan_first"):
        monitor = FixedCanvasMonitor(
            tokenizer=tokenizer,
            catalog=["a_agent", "b_agent"],
            prompt_length=prompt.shape[1],
            gen_length=512,
            mask_id=tokenizer.mask_token_id,
            structure_mode=mode,
        )
        initial_plan_start = monitor.layout.plan_start
        compacted_plan_start = (
            monitor.layout.reasoning_start
            + len(reasoning_emitted)
            + len(monitor.layout.middle_ids)
        )
        token_by_position = {
            monitor.layout.reasoning_start + offset: token
            for offset, token in enumerate(tokenizer.encode(reasoning_emitted))
        }
        for start in (initial_plan_start, compacted_plan_start):
            token_by_position.update({
                start + offset: token
                for offset, token in enumerate(tokenizer.encode(plan_emitted))
            })

        output, _ = generate_with_fixed_canvas_dual_cache(
            PositionedModel(token_by_position),
            prompt,
            steps=512,
            gen_length=512,
            block_length=32,
            threshold=0.9,
            mask_id=tokenizer.mask_token_id,
            agent_controller=monitor,
            structure_mode=mode,
        )
        metrics = monitor.metrics()
        decoded = tokenizer.decode(output[0, prompt.shape[1] :].tolist())
        assert metrics["reasoning_end_natural_success"] is True
        assert metrics["plan_end_natural_success"] is True
        assert metrics["reasoning_capacity_exhausted"] is False
        assert metrics["plan_capacity_exhausted"] is False
        assert metrics["reasoning_effective_tokens"] == len(reasoning_text)
        assert metrics["plan_effective_tokens"] == len(plan_text)
        assert metrics["final_plan_parse_success"] is True
        assert int((output == tokenizer.mask_token_id).sum()) == 0
        assert decoded == (
            "PLANNING_REASONING\n"
            + reasoning_emitted
            + "\nPLAN_JSON\n"
            + plan_emitted
        )

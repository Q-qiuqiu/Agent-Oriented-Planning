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
        return types.SimpleNamespace(logits=logits, past_key_values=((torch.zeros(1),),))


class PositionedModel:
    device = torch.device("cpu")

    def __init__(self, token_by_position):
        self.token_by_position = token_by_position

    def __call__(self, x, **kwargs):
        replace = kwargs.get("replace_position")
        positions = (
            list(range(x.shape[1]))
            if replace is None
            else replace[0].nonzero(as_tuple=False).flatten().tolist()
        )
        assert len(positions) == x.shape[1]
        logits = torch.zeros((x.shape[0], x.shape[1], 256))
        for local, absolute in enumerate(positions):
            logits[:, local, self.token_by_position.get(absolute, ord("A"))] = 10.0
        return types.SimpleNamespace(logits=logits, past_key_values=((torch.zeros(1),),))


def run(
    mode, local_agent_observe=False, threshold=0.9,
    local_observation_stride=1, local_agent_probability=0.90,
    lightweight_tracking=False,
):
    prompt = torch.tensor([[2, 3, 4]])
    monitor = FixedCanvasMonitor(
        tokenizer=CharacterTokenizer(),
        catalog=["a_agent", "b_agent"],
        prompt_length=prompt.shape[1],
        gen_length=128,
        mask_id=CharacterTokenizer.mask_token_id,
        reasoning_budget=32,
        structure_mode=mode,
        local_agent_observe=local_agent_observe,
        local_observation_stride=local_observation_stride,
        local_agent_probability=local_agent_probability,
        lightweight_tracking=lightweight_tracking,
    )
    output, nfe = generate_with_fixed_canvas_dual_cache(
        ConstantModel(), prompt, steps=128, gen_length=128, block_length=32,
        threshold=threshold, mask_id=CharacterTokenizer.mask_token_id,
        agent_controller=monitor, structure_mode=mode,
    )
    return output, nfe, monitor.metrics()


def test_fixed_canvas_vanilla_finishes_and_protects_delimiters():
    output, nfe, metrics = run("fixed_canvas_vanilla")
    assert nfe > 0
    assert int((output == CharacterTokenizer.mask_token_id).sum()) == 0
    assert metrics["fixed_token_corruption_count"] == 0
    assert metrics["unresolved_mask_count"] == 0
    assert metrics["schedule"][0]["phase"] == "reasoning"


def test_plan_first_changes_only_region_visit_order_for_constant_model():
    vanilla, _, vanilla_metrics = run("fixed_canvas_vanilla")
    plan_first, _, plan_metrics = run("fixed_canvas_plan_first")
    assert vanilla_metrics["schedule"][0]["phase"] == "reasoning"
    assert plan_metrics["schedule"][0]["phase"] == "plan"
    assert torch.equal(vanilla, plan_first)


def test_plan_lightweight_tracking_preserves_output_nfe_and_core_metrics():
    diagnostic, diagnostic_nfe, _ = run("fixed_canvas_plan_first")
    lightweight, lightweight_nfe, metrics = run(
        "fixed_canvas_plan_first", lightweight_tracking=True
    )
    assert torch.equal(diagnostic, lightweight)
    assert diagnostic_nfe == lightweight_nfe
    assert metrics["lightweight_tracking"] is True
    assert "first_agent_seconds" not in metrics
    assert "first3_materialized_seconds" not in metrics
    assert "T_first3_plan" not in metrics
    assert "all_final_agent_seconds" not in metrics
    assert "trajectory" not in metrics
    assert "schedule" not in metrics
    assert "reasoning_effective_tokens" not in metrics


def test_all_observer_is_read_only_and_adds_no_forward():
    plan, plan_nfe, _ = run("fixed_canvas_plan_first", local_agent_observe=False)
    all_output, all_nfe, metrics = run(
        "fixed_canvas_plan_first", local_agent_observe=True
    )
    assert torch.equal(plan, all_output)
    assert plan_nfe == all_nfe
    assert metrics["agent_observe"]["prediction_only"] is True
    assert metrics["agent_observe"]["probe_forwards"] == 0
    assert metrics["commit_count"] == 0


def test_local_observation_stride_only_subsamples_existing_refinement_logits():
    monitor = FixedCanvasMonitor(
        tokenizer=CharacterTokenizer(),
        catalog=["a_agent", "b_agent"],
        prompt_length=3,
        gen_length=128,
        mask_id=CharacterTokenizer.mask_token_id,
        reasoning_budget=32,
        structure_mode="fixed_canvas_plan_first",
        local_agent_observe=True,
        local_observation_stride=4,
    )
    assert [
        step for step in range(1, 17)
        if monitor.should_observe_local_step(step)
    ] == [4, 8, 12, 16]

    stride1, nfe1, _ = run(
        "fixed_canvas_plan_first",
        local_agent_observe=True,
        threshold=None,
        local_observation_stride=1,
    )
    stride4, nfe4, metrics4 = run(
        "fixed_canvas_plan_first",
        local_agent_observe=True,
        threshold=None,
        local_observation_stride=4,
    )
    assert torch.equal(stride1, stride4)
    assert nfe1 == nfe4
    assert metrics4["local_observation_stride"] == 4
    assert metrics4["local_observer_refinement_calls"] > 0


def test_all_local_probability_is_read_only_and_configures_only_the_gate():
    probability90, nfe90, _ = run(
        "fixed_canvas_plan_first",
        local_agent_observe=True,
        threshold=None,
        local_observation_stride=8,
        local_agent_probability=0.90,
    )
    probability80, nfe80, metrics80 = run(
        "fixed_canvas_plan_first",
        local_agent_observe=True,
        threshold=None,
        local_observation_stride=8,
        local_agent_probability=0.80,
    )
    assert torch.equal(probability90, probability80)
    assert nfe90 == nfe80
    assert metrics80["local_agent_probability"] == 0.80
    assert metrics80["agent_observe"]["prediction_only"] is True


def _fusion_metrics(local_third="a_agent", local_third_seconds=4.0):
    tokenizer = CharacterTokenizer()
    monitor = FixedCanvasMonitor(
        tokenizer=tokenizer,
        catalog=["a_agent", "b_agent", "c_agent"],
        prompt_length=3,
        gen_length=128,
        mask_id=tokenizer.mask_token_id,
        reasoning_budget=32,
        structure_mode="fixed_canvas_plan_first",
        local_agent_observe=True,
    )
    monitor._final_occurrences = [
        (10, "a_agent"), (30, "b_agent"), (50, "a_agent")
    ]
    monitor._snapshots = [
        {
            "step": step,
            "nfe": step,
            "seconds": seconds,
            "phase": "plan",
            "plan_masks": 0,
            "reasoning_masks": 0,
            "reasoning_end_found": True,
            "plan_end_found": True,
            "materialized_agents": materialized,
        }
        for step, seconds, materialized in (
            (1, 4.0, [{"plan_offset": 10, "agent": "a_agent"}]),
            (2, 5.0, [
                {"plan_offset": 10, "agent": "a_agent"},
                {"plan_offset": 30, "agent": "b_agent"},
            ]),
            (3, 6.0, [
                {"plan_offset": 10, "agent": "a_agent"},
                {"plan_offset": 30, "agent": "b_agent"},
                {"plan_offset": 50, "agent": "a_agent"},
            ]),
        )
    ]
    monitor._final_ids = torch.tensor([1, 2, 3])
    monitor._reasoning_payload_ids = torch.tensor([])
    monitor._plan_payload_ids = torch.tensor([])
    monitor._reasoning_text = ""
    monitor._plan_text = ""
    monitor._json_end = None
    monitor._fixed_reference = []
    monitor._final_plan = None
    monitor.agent_observer = types.SimpleNamespace(metrics=lambda: {
        "agent_slots": [
            {
                "shadow_agent": "a_agent", "shadow_seconds": 2.0,
                "shadow_step": 1, "shadow_anchor_offset": 10,
                "shadow_probability": 0.95, "shadow_margin": 0.8,
            },
            {
                "shadow_agent": "b_agent", "shadow_seconds": 3.0,
                "shadow_step": 2, "shadow_anchor_offset": 30,
                "shadow_probability": 0.95, "shadow_margin": 0.8,
            },
            {
                "shadow_agent": local_third,
                "shadow_seconds": local_third_seconds,
                "shadow_step": 3, "shadow_anchor_offset": 50,
                "shadow_probability": 0.95, "shadow_margin": 0.8,
            },
        ]
    })
    return monitor.metrics()


def test_all_local_fusion_uses_natural_zero_lag_fallback_and_never_lags():
    metrics = _fusion_metrics(local_third_seconds=7.0)
    assert [slot["prefetch_source"] for slot in metrics["agent_slots"]] == [
        "local", "local", "natural"
    ]
    assert metrics["T_first3_plan"] == 6.0
    assert metrics["T_first3_all"] == 6.0
    assert metrics["prediction_incremental_lead"] == 0.0
    assert metrics["all_not_later_than_plan"] is True


def test_all_reports_correct_incremental_lead_and_wrong_correction_cost():
    correct = _fusion_metrics()
    assert correct["all_first3_speculative_exact"] is True
    assert correct["local_first3_coverage"] is True
    assert correct["local_first3_exact"] is True
    assert correct["T_first3_all"] == 4.0
    assert correct["correct_first3_lead"] == 2.0

    wrong = _fusion_metrics(local_third="c_agent")
    slot3 = wrong["agent_slots"][2]
    assert wrong["all_first3_speculative_exact"] is False
    assert wrong["local_first3_exact"] is False
    assert wrong["correct_first3_lead"] is None
    assert slot3["wrong_prefetch"] is True
    assert slot3["T_correction"] == 6.0
    assert slot3["wrong_prefetch_duration"] == 2.0


def test_schema_valid_json_and_dynamic_end_compact_plan_capacity():
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
    plan_text = '[{"agent":"a_agent","id":1,"task":"x","reason":"y","dep":[]}]'
    emitted = plan_text + "\nEND_PLAN_JSON"
    token_by_position = {
        monitor.layout.plan_start + offset: token
        for offset, token in enumerate(tokenizer.encode(emitted))
    }
    output, _ = generate_with_fixed_canvas_dual_cache(
        PositionedModel(token_by_position), prompt, steps=256, gen_length=256,
        block_length=32, threshold=0.9, mask_id=tokenizer.mask_token_id,
        agent_controller=monitor, structure_mode="fixed_canvas_plan_first",
    )
    metrics = monitor.metrics()
    assert metrics["plan_json_complete"] is True
    assert metrics["plan_capacity_overflow"] is False
    assert metrics["plan_effective_tokens"] == len(plan_text)
    assert metrics["unused_plan_capacity"] > 0
    assert metrics["fixed_token_corruption_count"] == 0
    assert int((output == tokenizer.mask_token_id).sum()) == 0


def test_both_natural_end_markers_compact_in_either_phase_order():
    tokenizer = CharacterTokenizer()
    prompt = torch.tensor([[2, 3, 4]])
    reasoning_text = "Two short sentences. Done."
    reasoning_emitted = reasoning_text + "\nEND_PLANNING_REASONING"
    plan_text = '[{"agent":"a_agent","id":1,"task":"x","reason":"y","dep":[]}]'
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
            PositionedModel(token_by_position), prompt, steps=512, gen_length=512,
            block_length=32, threshold=0.9, mask_id=tokenizer.mask_token_id,
            agent_controller=monitor, structure_mode=mode,
        )
        metrics = monitor.metrics()
        assert metrics["reasoning_end_natural_success"] is True
        assert metrics["plan_end_natural_success"] is True
        assert metrics["final_plan_parse_success"] is True
        assert int((output == tokenizer.mask_token_id).sum()) == 0

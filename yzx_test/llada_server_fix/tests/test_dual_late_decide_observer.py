import types

import torch

from dual_late_decide_observer import DualVanillaLateDecideObserver
from fusion_prefetch import FusionAgentPrefetchTracker, FusionGateConfig
from generate import generate_with_dual_cache
from json_agent_priority import JsonAgentSlotRuntime
from ordered_plan_observer import OrderedPlanAgentObserver


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


def test_fusion_natural_event_is_exposed_to_timing_recorder():
    observer = DualVanillaLateDecideObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["a_agent", "b_agent", "c_agent"],
        prompt_length=3,
        gen_length=64,
        mask_id=CharacterTokenizer.mask_token_id,
    )
    canvas = torch.full((1, 67), CharacterTokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    observer.fusion.observe_natural(
        0,
        "a_agent",
        seconds=4.5,
    )

    metrics = observer.metrics()
    slot = metrics["agent_slots"][0]
    assert slot["natural_decoded_agent"] == "a_agent"
    assert slot["T_natural_decode"] == 4.5
    assert "S_natural_decode" not in slot
    assert metrics["natural_agent_sequence"] == ["a_agent"]


def test_semantic_scorer_ignores_naturally_visible_agent_prefix_tokens():
    tokenizer = CharacterTokenizer()
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=64,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    canvas = torch.full((1, 67), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    runtime = JsonAgentSlotRuntime(name_start=10)
    canvas[0, runtime.name_start] = ord("s")
    logits = torch.zeros((1, 67, 256))
    target = observer.scorer.padded_catalog_ids["calculation_agent"]
    for offset, token_id in enumerate(target[1:], start=1):
        logits[0, runtime.name_start + offset, token_id] = 10.0

    first, fully_masked = observer._score_masked_agent_value(
        logits, canvas, 0, runtime
    )
    canvas[0, runtime.name_start] = ord("r")
    second, _ = observer._score_masked_agent_value(logits, canvas, 0, runtime)

    assert fully_masked is False
    assert first == second
    assert max(first, key=first.get) == "calculation_agent"


def test_natural_prefix_stops_at_first_mask():
    tokenizer = CharacterTokenizer()
    observer = DualVanillaLateDecideObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=64,
        mask_id=tokenizer.mask_token_id,
    )
    canvas = torch.full((1, 67), tokenizer.mask_token_id)
    runtime = JsonAgentSlotRuntime(name_start=10)
    for offset, character in enumerate("sear"):
        canvas[0, runtime.name_start + offset] = ord(character)
    # Characters after a mask are not a continuous prefix and must be ignored.
    for offset, character in enumerate("calculation", start=5):
        canvas[0, runtime.name_start + offset] = ord(character)

    prefix = observer._continuous_agent_prefix(canvas, runtime)

    assert prefix == "sear"
    assert observer.fusion.observe_prefix(
        0, prefix, seconds=1.0
    ) is True
    assert observer.fusion.slots[0]["commit"]["agent"] == "search_agent"


def test_production_latent_region_requires_two_causally_valid_predictions():
    tokenizer = CharacterTokenizer()
    observer = DualVanillaLateDecideObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=64,
        mask_id=tokenizer.mask_token_id,
    )
    canvas = torch.full((1, 67), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    observer._online_anchor_candidates = lambda *_: [{
        "anchor_start": 12,
        "anchor_end": 20,
        "value_start": 20,
        "anchor_score": -1.0,
    }]
    observer._track_candidates = lambda _candidates, _observation: {0: {
        "refined_value_start": 20,
    }}
    observer._score_agent_region = lambda _logits, center: {
        "predicted_agent": "search_agent",
        "top1_region_score": -0.1,
        "top2_region_score": -1.1,
        "region_margin": 1.0,
        "best_alignment_position": center,
        "region_left": center - 4,
        "region_right": center + 4,
    }
    logits = torch.zeros((1, 67, 256))

    observer.observe(logits, canvas, 0, 0)
    observer.observe(logits, canvas, 0, 32)
    assert observer._latent_triggers == []
    observer.observe(logits, canvas, 0, 64)
    assert len(observer._latent_triggers) == 1
    assert observer._latent_triggers[0]["region_stable_count"] == 2

    observer._natural_value_starts[0] = 20
    observer.fusion.observe_natural(0, "search_agent", seconds=10.0)
    observer.set_evaluation_plan_text('[{"agent":"search_agent"}]')
    metrics = observer.metrics()
    assert metrics["agent_slots"][0]["commit_source"] == "latent_region"
    assert metrics["agent_slots"][0]["correct"] is True
    assert metrics["first_valid_commit_accuracy"] == 1.0
    assert metrics["latent_region_commit_accuracy"] == 1.0
    assert metrics["correct_latent_coverage"] == 1.0
    assert metrics["extra_model_forwards"] == 0


def test_semantic_evidence_requires_task_rationale_and_materialized_agent_key():
    tokenizer = CharacterTokenizer()
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=160,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    text = (
        '{"id":1,"task":"abcdefgh","rationale":'
        '"abcdefghijklmnop","dep":[],"agent":"'
    )
    canvas = torch.full((1, 163), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    canvas[0, 3:3 + len(text)] = torch.tensor(tokenizer.encode(text))
    agent_start = 3 + text.index('agent":"')

    assert observer.semantic_evidence_ready(
        canvas,
        step_start=3,
        agent_anchor_start=agent_start,
        agent_key_observed_ratio=1.0,
    ) is True
    assert observer.semantic_evidence_ready(
        canvas,
        step_start=3,
        agent_anchor_start=agent_start,
        agent_key_observed_ratio=0.75,
    ) is False

    short_text = (
        '{"id":1,"task":"short","rationale":"too short",'
        '"dep":[],"agent":"'
    )
    short_canvas = torch.full((1, 163), tokenizer.mask_token_id)
    short_canvas[:, :3] = torch.tensor([[2, 3, 4]])
    short_canvas[0, 3:3 + len(short_text)] = torch.tensor(
        tokenizer.encode(short_text)
    )
    short_agent_start = 3 + short_text.index('agent":"')
    assert observer.semantic_evidence_ready(
        short_canvas,
        step_start=3,
        agent_anchor_start=short_agent_start,
        agent_key_observed_ratio=1.0,
    ) is False


def test_semantic_scorer_skips_same_evidence_and_obeys_own_stride():
    tokenizer = CharacterTokenizer()
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=160,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    observer.semantic_observation_stride = 2
    text = (
        '{"id":1,"task":"abcdefghi","rationale":'
        '"abcdefghijklmnopq","dep":[],"agent":"'
    )
    canvas = torch.full((1, 163), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    canvas[0, 3:3 + len(text)] = torch.tensor(tokenizer.encode(text))
    task_change = 3 + text.index("abcdefghi") + 8
    canvas[0, task_change] = tokenizer.mask_token_id
    observer.initialize(canvas)
    anchors = observer.scorer._materialized_anchor_candidates(canvas)
    anchor_start, pattern, _score, _ratio = anchors[0]
    name_start = anchor_start + len(pattern)
    logits = torch.zeros((1, canvas.shape[1], 256))
    for offset, token_id in enumerate(
        observer.scorer.padded_catalog_ids["search_agent"]
    ):
        logits[0, name_start + offset, token_id] = 10.0

    first = observer.observe(
        logits,
        canvas,
        logits_start=0,
        global_step=None,
        plan_start=3,
        plan_end=canvas.shape[1],
        phase="test",
        committed_slots=(),
        materialized_anchors=anchors,
    )
    second = observer.observe(
        logits,
        canvas,
        logits_start=0,
        global_step=None,
        plan_start=3,
        plan_end=canvas.shape[1],
        phase="test",
        committed_slots=(),
        materialized_anchors=anchors,
    )
    third = observer.observe(
        logits,
        canvas,
        logits_start=0,
        global_step=None,
        plan_start=3,
        plan_end=canvas.shape[1],
        phase="test",
        committed_slots=(),
        materialized_anchors=anchors,
    )
    canvas[0, task_change] = ord("i")
    fourth = observer.observe(
        logits,
        canvas,
        logits_start=0,
        global_step=None,
        plan_start=3,
        plan_end=canvas.shape[1],
        phase="test",
        committed_slots=(),
        materialized_anchors=anchors,
    )

    assert first["status"] == "stride"
    assert second["status"] == "scored"
    assert third["status"] == "same_evidence"
    assert fourth["status"] == "scored"
    metrics = observer.scorer_metrics()
    assert metrics["semantic_observation_stride"] == 2
    assert metrics["semantic_scorer_call_count"] == 2
    assert metrics["semantic_skipped_stride_count"] == 1
    assert metrics["semantic_skipped_same_evidence_count"] == 1
    assert metrics["semantic_scorer_total_time"] >= 0.0
    assert metrics["semantic_scorer_mean_time"] >= 0.0


def test_semantic_observer_skips_committed_slots(monkeypatch):
    tokenizer = CharacterTokenizer()
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=96,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    observer.semantic_observation_stride = 1
    canvas = torch.full((1, 99), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    pattern = observer.scorer.anchor_variants[0]
    anchors = [
        (10, pattern, 0.0, 1.0),
        (50, pattern, 0.0, 1.0),
        (75, pattern, 0.0, 1.0),
    ]
    evidence_calls = []

    def evidence(*_args, **kwargs):
        evidence_calls.append(kwargs["agent_anchor_start"])
        return True, ((1,), (2,))

    monkeypatch.setattr(observer, "_semantic_evidence", evidence)
    monkeypatch.setattr(
        observer,
        "_score_masked_agent_value",
        lambda *_args: ({
            "search_agent": 0.99,
            "calculation_agent": 0.005,
            "reasoning_agent": 0.005,
        }, True),
    )
    event = observer.observe(
        torch.zeros((1, 99, 256)),
        canvas,
        logits_start=0,
        global_step=None,
        plan_start=3,
        plan_end=99,
        phase="test",
        committed_slots=(0, 2),
        materialized_anchors=anchors,
    )

    assert event["status"] == "scored"
    assert evidence_calls == [50]
    assert len(event["slots"]) == 3
    assert event["slots"][1]["slot"] == 1


def test_semantic_evidence_signature_is_independent_per_slot(monkeypatch):
    tokenizer = CharacterTokenizer()
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=["search_agent", "calculation_agent", "reasoning_agent"],
        prompt_length=3,
        gen_length=96,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    observer.semantic_observation_stride = 1
    canvas = torch.full((1, 99), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    observer.initialize(canvas)
    pattern = observer.scorer.anchor_variants[0]
    anchors = [(10, pattern, 0.0, 1.0), (50, pattern, 0.0, 1.0)]
    signatures = {10: ((1,), (2,)), 50: ((3,), (4,))}

    monkeypatch.setattr(
        observer,
        "_semantic_evidence",
        lambda *_args, **kwargs: (
            True, signatures[kwargs["agent_anchor_start"]]
        ),
    )
    monkeypatch.setattr(
        observer,
        "_score_masked_agent_value",
        lambda *_args: ({
            "search_agent": 0.99,
            "calculation_agent": 0.005,
            "reasoning_agent": 0.005,
        }, True),
    )
    for _ in range(2):
        observer.observe(
            torch.zeros((1, 99, 256)),
            canvas,
            logits_start=0,
            global_step=None,
            plan_start=3,
            plan_end=99,
            phase="test",
            committed_slots=(),
            materialized_anchors=anchors,
        )
        signatures[50] = ((3, 5), (4,))

    assert observer._semantic_scorer_calls_by_slot == {0: 1, 1: 2}
    assert observer.scorer_metrics()["semantic_skipped_same_evidence_count"] == 1


def test_later_mature_slot_scores_before_earlier_immature_slot():
    tokenizer = CharacterTokenizer()
    catalog = ["search_agent", "calculation_agent", "reasoning_agent"]
    observer = OrderedPlanAgentObserver(
        tokenizer=tokenizer,
        catalog=catalog,
        prompt_length=3,
        gen_length=297,
        mask_id=tokenizer.mask_token_id,
        elapsed=lambda: 0.0,
    )
    observer.semantic_observation_stride = 1
    tracker = FusionAgentPrefetchTracker(FusionGateConfig(), catalog=catalog)
    first = (
        '{"id":1,"task":"abcdefghij","rationale":'
        '"abcdefghijklmnopqrst","dep":[],"agent":"'
    )
    second = (
        '{"id":2,"task":"abcdefghij","rationale":'
        '"abcdefghijklmnopqrst","dep":[],"agent":"'
    )
    canvas = torch.full((1, 300), tokenizer.mask_token_id)
    canvas[:, :3] = torch.tensor([[2, 3, 4]])
    canvas[0, 3:3 + len(first)] = torch.tensor(tokenizer.encode(first))
    canvas[0, 120:120 + len(second)] = torch.tensor(tokenizer.encode(second))
    task_start = 3 + first.index("abcdefghij")
    rationale_start = 3 + first.index("abcdefghijklmnopqrst")
    second_task_start = 120 + second.index("abcdefghij")
    canvas[0, task_start + 4:task_start + 10] = tokenizer.mask_token_id
    canvas[0, rationale_start + 8:rationale_start + 20] = tokenizer.mask_token_id
    observer.initialize(canvas)
    anchors = observer.scorer._materialized_anchor_candidates(canvas)
    assert len(anchors) == 2

    logits = torch.zeros((1, canvas.shape[1], 256))
    for index, agent in enumerate(("search_agent", "calculation_agent")):
        name_start = anchors[index][0] + len(anchors[index][1])
        for offset, token_id in enumerate(observer.scorer.padded_catalog_ids[agent]):
            logits[0, name_start + offset, token_id] = 10.0

    for observation in range(2):
        if observation:
            canvas[0, second_task_start + 9] = ord("k")
        event = observer.observe(
            logits,
            canvas,
            logits_start=0,
            global_step=None,
            plan_start=3,
            plan_end=canvas.shape[1],
            phase="test",
            committed_slots=tracker.committed_slots,
            materialized_anchors=anchors,
        )
        tracker.observe_predictions(event["slots"], seconds=event["seconds"])

    assert tracker.slots[0]["commit"] is None
    assert tracker.slots[0]["stable_count"] == 0
    assert tracker.slots[1]["commit"]["agent"] == "calculation_agent"
    assert observer._semantic_scorer_calls_by_slot == {1: 2}

    canvas[0, task_start + 4:task_start + 10] = torch.tensor(
        tokenizer.encode("efghij")
    )
    canvas[0, rationale_start + 8:rationale_start + 20] = torch.tensor(
        tokenizer.encode("ijklmnopqrst")
    )
    for observation in range(2):
        if observation:
            canvas[0, task_start + 9] = ord("k")
        event = observer.observe(
            logits,
            canvas,
            logits_start=0,
            global_step=None,
            plan_start=3,
            plan_end=canvas.shape[1],
            phase="test",
            committed_slots=tracker.committed_slots,
            materialized_anchors=anchors,
        )
        tracker.observe_predictions(event["slots"], seconds=event["seconds"])

    assert tracker.slots[0]["commit"]["agent"] == "search_agent"
    assert observer._semantic_scorer_calls_by_slot == {1: 2, 0: 2}

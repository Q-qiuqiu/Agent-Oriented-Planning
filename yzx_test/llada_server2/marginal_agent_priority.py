"""Read-only early Agent prediction for reasoning-first LLaDA plans.

The normal JSON controller associates a slot with one absolute token position.
That is a poor fit for ``reasonplan`` because the still-masked reasoning prefix
causes future JSON fields to move between denoising observations.  This module
instead marginalizes over plausible JSON-field positions, accumulates evidence
by occurrence rank, and reports the first three Agent names without modifying
the generated token canvas.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    JsonAgentSlotRuntime,
    extract_agent_registry,
)


LOGGER = logging.getLogger("fastdllm.agent_priority")

# These are method constants, not resource-scheduling parameters.  The policy
# always predicts the first ``priority_slots`` Agent calls in response order.
MAX_POSITION_CANDIDATES = 18
JOINT_LAYOUT_BEAM = 64
TEMPORAL_EMA_PREVIOUS_WEIGHT = 0.60
FORCED_PREFETCH_OBSERVATIONS = 3
MIN_ABSOLUTE_NAME_MARGIN = -4.0
EPSILON = 1e-8


@dataclass
class PositionCandidate:
    anchor_start: int
    anchor_token_ids: Tuple[int, ...]
    anchor_score: float
    observed_ratio: float
    distribution: Dict[str, float]
    name_score: float
    layout_score: float


@dataclass
class SequenceInference:
    sequence_distribution: Dict[Tuple[str, ...], float]
    slot_distributions: List[Dict[str, float]]
    best_sequence: Tuple[str, ...]
    best_sequence_probability: float
    best_sequence_margin: float
    best_layout: Tuple[PositionCandidate, ...]
    best_layout_by_sequence: Dict[
        Tuple[str, ...], Tuple[PositionCandidate, ...]
    ]
    supporting_positions: List[List[int]]


@dataclass
class MarginalizedSlotRuntime(JsonAgentSlotRuntime):
    final_agent: Optional[str] = None
    final_agent_seconds: Optional[float] = None
    final_agent_step: Optional[int] = None
    prediction_correct: Optional[bool] = None
    decision_source: Optional[str] = None
    supporting_positions: List[int] = field(default_factory=list)
    best_name_score: Optional[float] = None
    switch_required: bool = False
    switch_seconds: Optional[float] = None
    switch_step: Optional[int] = None
    switch_emitted: bool = False


class MarginalizedAgentFieldController(JsonAgentFieldController):
    """Predict ordered Agent slots while marginalizing unstable field positions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slots = [
            MarginalizedSlotRuntime() for _ in range(self.tracking_slots)
        ]
        self._temporal_sequence_distribution: Optional[
            Dict[Tuple[str, ...], float]
        ] = None
        self._sequence_candidate: Optional[Tuple[str, ...]] = None
        self._sequence_consistent_steps = 0
        self._sequence_probability = 0.0
        self._sequence_margin = 0.0
        self._last_sequence_inference: Optional[SequenceInference] = None

    def _all_anchor_candidates(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
    ) -> List[Tuple[int, Tuple[int, ...], float, float]]:
        """Find plausible anchors without assigning them to persistent positions."""

        sequence_logits = logits[0]
        sequence_max = sequence_logits.amax(dim=-1)
        absolute_end = logits_start + sequence_logits.shape[0]
        generation_start = self.prompt_length
        generation_end = self.prompt_length + self.gen_length
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}

        for pattern in self.anchor_variants:
            width = len(pattern)
            start = max(generation_start, logits_start)
            end = min(generation_end, absolute_end) - width + 1
            if end <= start:
                continue
            relative = start - logits_start
            count = end - start
            scores = torch.zeros(count, device=logits.device, dtype=torch.float32)
            compatible = torch.ones(count, device=logits.device, dtype=torch.bool)
            observed = torch.zeros(count, device=logits.device, dtype=torch.float32)

            for offset, token_id in enumerate(pattern):
                row_start = relative + offset
                row_end = row_start + count
                target = sequence_logits[row_start:row_end, token_id].float()
                scores += target - sequence_max[row_start:row_end].float()
                positions = torch.arange(
                    start + offset,
                    end + offset,
                    device=x.device,
                )
                current = x[0, positions]
                compatible &= (current == self.mask_id) | (current == token_id)
                observed += (current == token_id).float()

            scores /= float(width)
            observed /= float(width)
            if pattern in self.speculative_anchor_variants:
                eligible = compatible & (
                    (scores >= self.config.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
            else:
                eligible = compatible & (observed == 1.0)

            observed_indices = torch.nonzero(
                eligible & (observed == 1.0), as_tuple=False
            ).flatten()
            speculative_indices = torch.nonzero(
                eligible & (observed != 1.0), as_tuple=False
            ).flatten()
            selected = observed_indices.detach().cpu().tolist()
            if speculative_indices.numel() > 0:
                top_count = min(
                    int(speculative_indices.numel()),
                    MAX_POSITION_CANDIDATES,
                )
                top = torch.topk(scores[speculative_indices], k=top_count).indices
                selected.extend(speculative_indices[top].detach().cpu().tolist())

            for local_index in selected:
                absolute_start = start + int(local_index)
                candidate = (
                    absolute_start,
                    pattern,
                    float(scores[local_index].detach().cpu()),
                    float(observed[local_index].detach().cpu()),
                )
                previous = candidates.get(absolute_start)
                if previous is None or (candidate[3], candidate[2]) > (
                    previous[3], previous[2]
                ):
                    candidates[absolute_start] = candidate

        cluster_radius = max(len(pattern) for pattern in self.anchor_variants)
        clusters: List[List[Tuple[int, Tuple[int, ...], float, float]]] = []
        for candidate in sorted(candidates.values(), key=lambda item: item[0]):
            if not clusters or candidate[0] - clusters[-1][-1][0] >= cluster_radius:
                clusters.append([candidate])
            else:
                clusters[-1].append(candidate)
        collapsed = [
            max(cluster, key=lambda item: (item[3], item[2]))
            for cluster in clusters
        ]

        # Keep every materialized field and the strongest speculative fields.
        materialized = [candidate for candidate in collapsed if candidate[3] == 1.0]
        speculative = sorted(
            (candidate for candidate in collapsed if candidate[3] != 1.0),
            key=lambda item: item[2],
            reverse=True,
        )[:MAX_POSITION_CANDIDATES]
        ranked = materialized + speculative

        # Non-maximum suppression prevents whitespace variants or adjacent token
        # accidents from representing the same logical field twice.
        selected: List[Tuple[int, Tuple[int, ...], float, float]] = []
        for candidate in sorted(
            ranked, key=lambda item: (item[3], item[2]), reverse=True
        ):
            if any(
                abs(candidate[0] - existing[0]) < self.config.min_anchor_gap
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= MAX_POSITION_CANDIDATES:
                break
        return sorted(selected, key=lambda item: item[0])

    def _score_position(
        self,
        logits: torch.Tensor,
        logits_start: int,
        anchor: Tuple[int, Tuple[int, ...], float, float],
    ) -> Optional[PositionCandidate]:
        anchor_start, pattern, anchor_score, observed_ratio = anchor
        name_start = anchor_start + len(pattern)
        relative_start = name_start - logits_start
        relative_end = relative_start + self.value_width
        if relative_start < 0 or relative_end > logits.shape[1]:
            return None
        if self._catalog_target_ids is None or self._catalog_positions is None:
            raise RuntimeError("Agent controller must be initialized before scoring.")

        field_logits = logits[0, relative_start:relative_end].float()
        candidate_logits = field_logits[
            self._catalog_positions,
            self._catalog_target_ids,
        ]
        sequence_scores = candidate_logits.sum(dim=1)
        probabilities = torch.softmax(sequence_scores, dim=0)
        distribution = dict(
            zip(self.catalog_names, probabilities.detach().cpu().tolist())
        )

        target_best = candidate_logits.max(dim=0).values
        name_score = float(
            (target_best - field_logits.amax(dim=-1)).mean().detach().cpu()
        )
        # Agent identity is scored later as part of a complete sequence. Keep
        # layout validity independent from whichever catalog entry wins locally.
        layout_score = (
            anchor_score
            + 0.35 * name_score
            + (12.0 if observed_ratio == 1.0 else 0.0)
        )
        return PositionCandidate(
            anchor_start=anchor_start,
            anchor_token_ids=pattern,
            anchor_score=anchor_score,
            observed_ratio=observed_ratio,
            distribution=distribution,
            name_score=name_score,
            layout_score=layout_score,
        )

    @staticmethod
    def _logaddexp(left: Optional[float], right: float) -> float:
        if left is None:
            return right
        maximum = max(left, right)
        return maximum + math.log(
            math.exp(left - maximum) + math.exp(right - maximum)
        )

    def _joint_sequence_inference(
        self,
        candidates: Sequence[PositionCandidate],
    ) -> Optional[SequenceInference]:
        """Infer one complete Agent sequence across uncertain field layouts."""

        slot_count = self.config.priority_slots
        if len(candidates) < slot_count:
            return None

        layouts = []
        for combination in itertools.combinations(candidates, slot_count):
            if any(
                right.anchor_start - left.anchor_start < self.config.min_anchor_gap
                for left, right in zip(combination, combination[1:])
            ):
                continue
            layouts.append((sum(item.layout_score for item in combination), combination))
        if not layouts:
            return None
        layouts.sort(key=lambda item: item[0], reverse=True)
        layouts = layouts[:JOINT_LAYOUT_BEAM]

        sequences = list(itertools.product(self.catalog_names, repeat=slot_count))
        sequence_log_scores: Dict[Tuple[str, ...], float] = {}
        best_layout_by_sequence: Dict[
            Tuple[str, ...], Tuple[float, Tuple[PositionCandidate, ...]]
        ] = {}
        for layout_score, layout in layouts:
            for sequence in sequences:
                joint_score = layout_score + sum(
                    math.log(max(candidate.distribution[agent], EPSILON))
                    for candidate, agent in zip(layout, sequence)
                )
                sequence_log_scores[sequence] = self._logaddexp(
                    sequence_log_scores.get(sequence), joint_score
                )
                previous = best_layout_by_sequence.get(sequence)
                if previous is None or joint_score > previous[0]:
                    best_layout_by_sequence[sequence] = (joint_score, layout)

        maximum = max(sequence_log_scores.values())
        weights = {
            sequence: math.exp(score - maximum)
            for sequence, score in sequence_log_scores.items()
        }
        total = sum(weights.values())
        distribution = {
            sequence: weight / total for sequence, weight in weights.items()
        }
        ranked = sorted(
            distribution.items(), key=lambda item: item[1], reverse=True
        )
        best_sequence, best_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        best_layout = best_layout_by_sequence[best_sequence][1]

        slot_distributions = [
            {name: 0.0 for name in self.catalog_names}
            for _ in range(slot_count)
        ]
        for sequence, probability in distribution.items():
            for slot_index, agent in enumerate(sequence):
                slot_distributions[slot_index][agent] += probability
        supporting_positions = [
            sorted({layout[slot_index].anchor_start for _, layout in layouts})
            for slot_index in range(slot_count)
        ]
        return SequenceInference(
            sequence_distribution=distribution,
            slot_distributions=slot_distributions,
            best_sequence=best_sequence,
            best_sequence_probability=best_probability,
            best_sequence_margin=best_probability - second_probability,
            best_layout=best_layout,
            best_layout_by_sequence={
                sequence: value[1]
                for sequence, value in best_layout_by_sequence.items()
            },
            supporting_positions=supporting_positions,
        )

    def _temporal_sequence_inference(
        self, current: SequenceInference
    ) -> SequenceInference:
        if self._temporal_sequence_distribution is None:
            smoothed = dict(current.sequence_distribution)
        else:
            smoothed = {
                sequence: TEMPORAL_EMA_PREVIOUS_WEIGHT
                * self._temporal_sequence_distribution.get(sequence, 0.0)
                + (1.0 - TEMPORAL_EMA_PREVIOUS_WEIGHT) * probability
                for sequence, probability in current.sequence_distribution.items()
            }
        total = sum(smoothed.values())
        self._temporal_sequence_distribution = {
            sequence: value / total for sequence, value in smoothed.items()
        }
        ranked = sorted(
            self._temporal_sequence_distribution.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_sequence, best_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        slot_distributions = [
            {name: 0.0 for name in self.catalog_names}
            for _ in range(self.config.priority_slots)
        ]
        for sequence, probability in self._temporal_sequence_distribution.items():
            for slot_index, agent in enumerate(sequence):
                slot_distributions[slot_index][agent] += probability
        return SequenceInference(
            sequence_distribution=dict(self._temporal_sequence_distribution),
            slot_distributions=slot_distributions,
            best_sequence=best_sequence,
            best_sequence_probability=best_probability,
            best_sequence_margin=best_probability - second_probability,
            # Position metadata comes from the current observation. Agent
            # identity is the only state accumulated across observations.
            best_layout=current.best_layout_by_sequence.get(
                best_sequence, current.best_layout
            ),
            best_layout_by_sequence=current.best_layout_by_sequence,
            supporting_positions=current.supporting_positions,
        )

    def _recognize(
        self,
        slot_index: int,
        distribution: Dict[str, float],
        global_step: int,
        source: str,
        force: bool = False,
        candidate_override: Optional[str] = None,
    ) -> None:
        runtime = self.slots[slot_index]
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        candidate = candidate_override or ranked[0][0]
        probability = distribution[candidate]
        second = max(
            (value for name, value in distribution.items() if name != candidate),
            default=0.0,
        )
        margin = probability - second
        now = self._elapsed()

        if runtime.first_observed_seconds is None:
            runtime.first_observed_seconds = now
            runtime.first_observed_step = global_step
        if runtime.candidate == candidate:
            runtime.candidate_consistent_steps += 1
        else:
            runtime.candidate = candidate
            runtime.candidate_consistent_steps = 1
        runtime.candidate_probability = probability
        runtime.candidate_margin = margin
        runtime.last_distribution = dict(distribution)

        absolute_evidence_ready = (
            runtime.best_name_score is not None
            and runtime.best_name_score >= MIN_ABSOLUTE_NAME_MARGIN
        )
        reliable = force or (
            runtime.candidate_consistent_steps >= 2
            and absolute_evidence_ready
            and probability >= self.config.tentative_probability
            and margin >= self.config.tentative_margin
        )
        if reliable and runtime.recognized_candidate is None:
            runtime.recognized_candidate = candidate
            runtime.recognized_probability = probability
            runtime.recognized_margin = margin
            runtime.recognized_seconds = now
            runtime.recognized_step = global_step
            runtime.decision_source = source
            self.logger.info(
                "agent_prefetch_decision %s",
                json.dumps(
                    {
                        "slot": slot_index,
                        "agent": candidate,
                        "seconds": now,
                        "step": global_step,
                        "probability": probability,
                        "margin": margin,
                        "source": source,
                    },
                    sort_keys=True,
                ),
            )

    def _observe_materialized(self, x: torch.Tensor, global_step: int) -> None:
        anchors = self._materialized_anchor_candidates(x)
        for slot_index, anchor in enumerate(anchors[:self.tracking_slots]):
            runtime = self.slots[slot_index]
            runtime.anchor_start = anchor[0]
            runtime.anchor_token_ids = anchor[1]
            runtime.name_start = anchor[0] + len(anchor[1])
            runtime.anchor_score = anchor[2]
            runtime.anchor_observed_ratio = 1.0
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is None:
                continue
            now = self._elapsed()
            if runtime.final_agent is None:
                runtime.final_agent = observed_name
                runtime.final_agent_seconds = now
                runtime.final_agent_step = global_step
            if runtime.recognized_candidate is None:
                distribution = {
                    name: 1.0 if name == observed_name else 0.0
                    for name in self.catalog_names
                }
                self._recognize(
                    slot_index,
                    distribution,
                    global_step,
                    source="materialized_json",
                    force=True,
                )
            runtime.prediction_correct = runtime.recognized_candidate == observed_name
            if runtime.prediction_correct and not runtime.confirmed:
                runtime.confirmed = True
                runtime.confirmed_seconds = now
                runtime.confirmed_step = global_step
            elif not runtime.prediction_correct and not runtime.switch_emitted:
                runtime.switch_required = True
                runtime.switch_seconds = now
                runtime.switch_step = global_step
                runtime.switch_emitted = True
                self.logger.info(
                    "agent_prefetch_switch %s",
                    json.dumps(
                        {
                            "slot": slot_index,
                            "from_agent": runtime.recognized_candidate,
                            "to_agent": observed_name,
                            "seconds": now,
                            "step": global_step,
                        },
                        sort_keys=True,
                    ),
                )

    def _update_sequence_candidate(self, inference: SequenceInference) -> None:
        candidate = inference.best_sequence
        if self._sequence_candidate == candidate:
            self._sequence_consistent_steps += 1
        else:
            self._sequence_candidate = candidate
            self._sequence_consistent_steps = 1
        self._sequence_probability = inference.best_sequence_probability
        self._sequence_margin = inference.best_sequence_margin

    def _commit_sequence(
        self,
        inference: SequenceInference,
        global_step: int,
        force: bool,
    ) -> None:
        absolute_evidence_ready = all(
            candidate.name_score >= MIN_ABSOLUTE_NAME_MARGIN
            for candidate in inference.best_layout
        )
        reliable = force or (
            self._sequence_consistent_steps >= 2
            and absolute_evidence_ready
            and inference.best_sequence_probability
            >= self.config.tentative_probability
            and inference.best_sequence_margin >= self.config.tentative_margin
        )
        if not reliable:
            return

        for slot_index, agent in enumerate(inference.best_sequence):
            runtime = self.slots[slot_index]
            if runtime.recognized_candidate is not None:
                continue
            self._recognize(
                slot_index,
                inference.slot_distributions[slot_index],
                global_step,
                source="joint_sequence_map",
                force=True,
                candidate_override=agent,
            )

    def _force_catalog_fallback(self, global_step: int) -> None:
        """Honor the fixed three-slot prefetch contract when no layout survives."""

        fallback = self.catalog_names[0]
        distribution = {
            name: 1.0 if name == fallback else 0.0
            for name in self.catalog_names
        }
        for slot_index in range(self.config.priority_slots):
            if self.slots[slot_index].recognized_candidate is not None:
                continue
            self._recognize(
                slot_index,
                distribution,
                global_step,
                source="forced_catalog_fallback",
                force=True,
                candidate_override=fallback,
            )

    def observe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        global_step: int,
        is_last_agent_step: bool = False,
    ) -> None:
        del is_last_agent_step
        self._observed_steps += 1
        self._observe_materialized(x, global_step)
        if all(
            runtime.recognized_candidate is not None
            for runtime in self.slots[:self.config.priority_slots]
        ):
            return

        covers_full_sequence = (
            logits_start <= self.prompt_length
            and logits_start + logits.shape[1]
            >= self.prompt_length + self.gen_length
        )
        if not covers_full_sequence:
            return
        self._full_sequence_observations += 1

        anchors = self._all_anchor_candidates(logits, x, logits_start)
        position_candidates = [
            candidate
            for anchor in anchors
            if (candidate := self._score_position(logits, logits_start, anchor))
            is not None
        ]
        current = self._joint_sequence_inference(position_candidates)
        if current is not None:
            inference = self._temporal_sequence_inference(current)
            self._last_sequence_inference = inference
            self._update_sequence_candidate(inference)
            for slot_index, best in enumerate(inference.best_layout):
                runtime = self.slots[slot_index]
                runtime.anchor_start = best.anchor_start
                runtime.anchor_token_ids = best.anchor_token_ids
                runtime.name_start = best.anchor_start + len(best.anchor_token_ids)
                runtime.anchor_score = best.anchor_score
                runtime.anchor_observed_ratio = best.observed_ratio
                runtime.supporting_positions = inference.supporting_positions[
                    slot_index
                ]
                runtime.best_name_score = best.name_score
            self._commit_sequence(
                inference,
                global_step,
                force=(
                    self._full_sequence_observations
                    >= FORCED_PREFETCH_OBSERVATIONS
                ),
            )

        if (
            self._full_sequence_observations >= FORCED_PREFETCH_OBSERVATIONS
            and any(
                runtime.recognized_candidate is None
                for runtime in self.slots[:self.config.priority_slots]
            )
        ):
            if self._last_sequence_inference is not None:
                self._commit_sequence(
                    self._last_sequence_inference,
                    global_step,
                    force=True,
                )
            else:
                self._force_catalog_fallback(global_step)

    def decoder_mask(
        self, mask_index: torch.Tensor, mask_start: int = 0
    ) -> torch.Tensor:
        del mask_start
        # Prediction is observational only. It cannot alter normal decoding.
        return mask_index

    def has_unconfirmed_agents(self) -> bool:
        return False

    def finalize(self, x: torch.Tensor) -> None:
        self._observe_materialized(x, self._observed_steps)
        # A speculative slot that does not exist in the completed plan is a
        # false prefetch, not an unchecked prediction.
        for runtime in self.slots[:self.config.priority_slots]:
            if (
                runtime.recognized_candidate is not None
                and runtime.final_agent is None
                and runtime.prediction_correct is None
            ):
                runtime.prediction_correct = False

    def metrics(self) -> Dict[str, object]:
        slots = []
        for index, runtime in enumerate(self.slots):
            if runtime.recognized_candidate is None and runtime.final_agent is None:
                continue
            slots.append(
                {
                    "slot": index,
                    "priority": index < self.config.priority_slots,
                    "agent": runtime.recognized_candidate,
                    "final_agent": runtime.final_agent,
                    "final_agent_seconds": runtime.final_agent_seconds,
                    "final_agent_step": runtime.final_agent_step,
                    "prediction_correct": runtime.prediction_correct,
                    "decision_source": runtime.decision_source,
                    "switch_required": runtime.switch_required,
                    "switch_seconds": runtime.switch_seconds,
                    "switch_step": runtime.switch_step,
                    "anchor_start": runtime.anchor_start,
                    "anchor_score": (
                        runtime.anchor_score
                        if math.isfinite(runtime.anchor_score)
                        else None
                    ),
                    "anchor_observed_ratio": runtime.anchor_observed_ratio,
                    "supporting_positions": runtime.supporting_positions,
                    "best_name_score": runtime.best_name_score,
                    "first_observed_seconds": runtime.first_observed_seconds,
                    "recognized_seconds": runtime.recognized_seconds,
                    "confirmed_seconds": runtime.confirmed_seconds,
                    "first_observed_step": runtime.first_observed_step,
                    "recognized_step": runtime.recognized_step,
                    "confirmed_step": runtime.confirmed_step,
                    "probability": runtime.recognized_probability,
                    "margin": runtime.recognized_margin,
                    "confirmed": runtime.confirmed,
                    "fuzzy_matched_from": runtime.fuzzy_matched_from,
                }
            )

        priority = slots[:self.config.priority_slots]
        recognized = [
            slot["recognized_seconds"]
            for slot in priority
            if slot["recognized_seconds"] is not None
        ]
        correctness = [
            slot["prediction_correct"]
            for slot in priority
            if slot["prediction_correct"] is not None
        ]
        final_agent_times = [
            slot["final_agent_seconds"]
            for slot in priority
            if slot["final_agent_seconds"] is not None
        ]
        return {
            "method": "joint_sequence_map_v3",
            "priority_slots": self.config.priority_slots,
            "tracking_slots": self.tracking_slots,
            "catalog": list(self.config.catalog),
            "observed_steps": self._observed_steps,
            "full_sequence_observations": self._full_sequence_observations,
            "discovered_agent_fields": sum(
                slot["final_agent"] is not None for slot in slots
            ),
            "recognized_agent_fields": len(recognized),
            "all_priority_agents_recognized": (
                len(recognized) == self.config.priority_slots
            ),
            "all_tracked_agents_recognized": (
                bool(slots)
                and all(slot["recognized_seconds"] is not None for slot in slots)
            ),
            "prediction_checked_count": len(correctness),
            "prediction_correct_count": sum(value is True for value in correctness),
            "prediction_accuracy": (
                sum(value is True for value in correctness) / len(correctness)
                if correctness
                else None
            ),
            "predicted_agent_sequence": (
                list(self._sequence_candidate)
                if self._sequence_candidate is not None
                else None
            ),
            "sequence_probability": self._sequence_probability,
            "sequence_margin": self._sequence_margin,
            "sequence_consistent_steps": self._sequence_consistent_steps,
            "prefetch_switch_count": sum(
                slot["switch_required"] for slot in priority
            ),
            "all_final_agents_seconds": (
                max(final_agent_times)
                if len(final_agent_times) == self.config.priority_slots
                else None
            ),
            "agent_slots": slots,
            "all_recognized_seconds": (
                max(recognized)
                if len(recognized) == self.config.priority_slots
                else None
            ),
            "partial_recognized_seconds": max(recognized) if recognized else None,
        }


__all__ = [
    "JsonAgentPriorityConfig",
    "MarginalizedAgentFieldController",
    "extract_agent_registry",
]

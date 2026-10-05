"""Compact read-only ordered Agent observer for existing Dual warmups."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    JsonAgentSlotRuntime,
)
SEMANTIC_MIN_TASK_VISIBLE_TOKENS = 8
SEMANTIC_MIN_RATIONALE_VISIBLE_TOKENS = 16
SEMANTIC_OBSERVATION_STRIDE = 8


class OrderedPlanAgentObserver:
    """Track all ordered Agent fields without mutating decoding."""

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        elapsed: Callable[[], float],
        anchor_min_logit_margin: float = -6.0,
        candidate_limit: Optional[int] = None,
    ) -> None:
        self.elapsed = elapsed
        # Production commit policy selected from test_03. These are fixed so
        # diagnostic ablations cannot accidentally leak into real runs.
        self.min_task_visible_tokens = SEMANTIC_MIN_TASK_VISIBLE_TOKENS
        self.min_rationale_visible_tokens = SEMANTIC_MIN_RATIONALE_VISIBLE_TOKENS
        self.semantic_observation_stride = SEMANTIC_OBSERVATION_STRIDE
        resolved_candidate_limit = int(
            candidate_limit
            if candidate_limit is not None
            else max(1, gen_length // 12)
        )
        self.scorer = JsonAgentFieldController(
            tokenizer=tokenizer,
            config=JsonAgentPriorityConfig(
                catalog=list(catalog),
                priority_slots=resolved_candidate_limit,
                tracking_slots=resolved_candidate_limit,
                anchor_min_logit_margin=anchor_min_logit_margin,
                tentative_probability=0.90,
                tentative_margin=0.40,
                probe_period=0,
            ),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
        )
        self._task_anchor_variants = self._field_anchor_variants(
            "task", quoted_value=True
        )
        self._rationale_anchor_variants = self._field_anchor_variants(
            "rationale", quoted_value=True
        )
        self._dep_anchor_variants = self._field_anchor_variants(
            "dep", quoted_value=False
        )
        self._observation = 0
        self._events: List[Dict[str, object]] = []
        self._last_scored_evidence_signatures: Dict[
            int, Tuple[Tuple[int, ...], ...]
        ] = {}
        self._evidence_regions: Dict[
            int, Tuple[int, int, int, int, int]
        ] = {}
        self._text_token_cache: Dict[int, bool] = {}
        self._semantic_scorer_call_count = 0
        self._semantic_scorer_calls_by_slot: Dict[int, int] = {}
        self._semantic_scorer_total_time = 0.0
        self._semantic_skipped_same_evidence_count = 0
        self._semantic_skipped_evidence_not_ready_count = 0
        self._semantic_skipped_stride_count = 0

    @property
    def events(self) -> List[Dict[str, object]]:
        return self._events

    def initialize(self, x: torch.Tensor) -> None:
        self.scorer.initialize(x)
        self._observation = 0
        self._events.clear()
        self._last_scored_evidence_signatures.clear()
        self._evidence_regions.clear()
        self._semantic_scorer_call_count = 0
        self._semantic_scorer_calls_by_slot.clear()
        self._semantic_scorer_total_time = 0.0
        self._semantic_skipped_same_evidence_count = 0
        self._semantic_skipped_evidence_not_ready_count = 0
        self._semantic_skipped_stride_count = 0

    def _field_anchor_variants(
        self, field: str, *, quoted_value: bool
    ) -> Tuple[Tuple[int, ...], ...]:
        suffix = '"' if quoted_value else ""
        # The planner schema uses canonical JSON scalar/list formatting. A
        # missing match keeps the semantic gate closed, so limiting this to
        # the two normal forms is both conservative and inexpensive.
        texts = [
            f'{field}":{suffix}',
            f'{field}": {suffix}',
        ]
        return tuple(
            dict.fromkeys(
                tuple(self.scorer._encode(text)) for text in texts
            )
        )

    def _materialized_field_anchors(
        self,
        x: torch.Tensor,
        patterns: Sequence[Tuple[int, ...]],
        *,
        start: int,
        end: int,
    ) -> List[Tuple[int, Tuple[int, ...]]]:
        matches: List[Tuple[int, Tuple[int, ...]]] = []
        sequence = x[0]
        for pattern in patterns:
            width = len(pattern)
            stop = end - width + 1
            if stop <= start:
                continue
            target = torch.tensor(
                pattern, device=x.device, dtype=x.dtype
            )
            windows = sequence[start:end].unfold(0, width, 1)
            indices = torch.nonzero(
                torch.all(windows == target, dim=1), as_tuple=False
            ).flatten()
            matches.extend(
                (start + int(index), pattern)
                for index in indices.detach().cpu().tolist()
            )
        return sorted(matches, key=lambda item: item[0])

    def semantic_evidence_ready(
        self,
        x: torch.Tensor,
        *,
        step_start: int,
        agent_anchor_start: int,
        agent_key_observed_ratio: float,
        slot_id: Optional[int] = None,
    ) -> bool:
        """Return whether task/rationale evidence has naturally matured."""
        ready, _signature = self._semantic_evidence(
            x,
            step_start=step_start,
            agent_anchor_start=agent_anchor_start,
            agent_key_observed_ratio=agent_key_observed_ratio,
            slot_id=slot_id,
        )
        return ready

    def _semantic_evidence(
        self,
        x: torch.Tensor,
        *,
        step_start: int,
        agent_anchor_start: int,
        agent_key_observed_ratio: float,
        slot_id: Optional[int] = None,
    ) -> Tuple[
        bool, Optional[Tuple[Tuple[int, ...], ...]]
    ]:
        """Evaluate the cheap gate and return its exact visible-token state."""
        if agent_key_observed_ratio < 1.0:
            return False, None

        cached = self._evidence_regions.get(slot_id) if slot_id is not None else None
        if cached is not None and cached[0] == agent_anchor_start:
            (
                _cached_agent_start,
                task_value_start,
                rationale_start,
                rationale_value_start,
                dep_start,
            ) = cached
        else:
            task_anchors = self._materialized_field_anchors(
                x,
                self._task_anchor_variants,
                start=step_start,
                end=agent_anchor_start,
            )
            rationale_anchors = self._materialized_field_anchors(
                x,
                self._rationale_anchor_variants,
                start=step_start,
                end=agent_anchor_start,
            )
            dep_anchors = self._materialized_field_anchors(
                x,
                self._dep_anchor_variants,
                start=step_start,
                end=agent_anchor_start,
            )
            if not task_anchors or not rationale_anchors or not dep_anchors:
                return False, None

            rationale_start, rationale_pattern = rationale_anchors[-1]
            task_choices = [
                row for row in task_anchors if row[0] < rationale_start
            ]
            dep_choices = [
                row for row in dep_anchors if row[0] > rationale_start
            ]
            if not task_choices or not dep_choices:
                return False, None
            task_start, task_pattern = task_choices[-1]
            dep_start, _dep_pattern = dep_choices[0]
            if not task_start < rationale_start < dep_start < agent_anchor_start:
                return False, None

            task_value_start = task_start + len(task_pattern)
            rationale_value_start = rationale_start + len(rationale_pattern)
            if slot_id is not None:
                self._evidence_regions[slot_id] = (
                    agent_anchor_start,
                    task_value_start,
                    rationale_start,
                    rationale_value_start,
                    dep_start,
                )
        task_tokens = self._region_token_ids(
            x, task_value_start, rationale_start
        )
        rationale_tokens = self._region_token_ids(
            x, rationale_value_start, dep_start
        )
        task_signature = self._visible_text_signature(task_tokens)
        rationale_signature = self._visible_text_signature(rationale_tokens)
        signature = (task_signature, rationale_signature)
        if (
            slot_id is not None
            and self._last_scored_evidence_signatures.get(slot_id) == signature
        ):
            # An identical signature was ready when it was scored, so its
            # visible-token counts cannot have regressed.
            return True, signature
        task_visible = len(task_signature) // 2
        rationale_visible = len(rationale_signature) // 2
        ready = (
            task_visible >= self.min_task_visible_tokens
            and rationale_visible >= self.min_rationale_visible_tokens
        )
        return ready, signature if ready else None

    @staticmethod
    def _region_token_ids(
        x: torch.Tensor, start: int, end: int
    ) -> Tuple[int, ...]:
        return tuple(
            int(value)
            for value in x[0, start:end].detach().cpu().tolist()
        )

    def _visible_text_signature(
        self, token_ids: Sequence[int]
    ) -> Tuple[int, ...]:
        signature = []
        for position, token_id in enumerate(token_ids):
            if token_id == self.scorer.mask_id:
                continue
            has_text = self._text_token_cache.get(token_id)
            if has_text is None:
                text = self.scorer.tokenizer.decode(
                    [int(token_id)], skip_special_tokens=True
                )
                has_text = any(character.isalnum() for character in text)
                self._text_token_cache[token_id] = has_text
            if has_text:
                signature.extend((position, int(token_id)))
        return tuple(signature)

    def scorer_metrics(self) -> Dict[str, object]:
        calls = self._semantic_scorer_call_count
        total = self._semantic_scorer_total_time
        return {
            "semantic_observation_stride": self.semantic_observation_stride,
            "semantic_scorer_call_count": calls,
            "semantic_scorer_total_time": total,
            "semantic_scorer_mean_time": total / calls if calls else 0.0,
            "semantic_skipped_same_evidence_count": (
                self._semantic_skipped_same_evidence_count
            ),
            "semantic_skipped_evidence_not_ready_count": (
                self._semantic_skipped_evidence_not_ready_count
            ),
            "semantic_skipped_stride_count": (
                self._semantic_skipped_stride_count
            ),
            "semantic_min_task_visible_tokens": self.min_task_visible_tokens,
            "semantic_min_rationale_visible_tokens": self.min_rationale_visible_tokens,
            "semantic_require_materialized_agent_key": True,
        }

    def _agent_value_fully_masked(
        self, x: torch.Tensor, runtime: JsonAgentSlotRuntime
    ) -> bool:
        if runtime.name_start is None:
            return False
        value_end = runtime.name_start + self.scorer.value_width
        if value_end > x.shape[1]:
            return False
        return bool(
            torch.all(
                x[0, runtime.name_start:value_end] == self.scorer.mask_id
            ).item()
        )

    @staticmethod
    def _rank_distribution(distribution: Dict[str, float]):
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        agent, probability = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return agent, float(probability), float(probability - second)

    def _score_masked_agent_value(self, logits, x, logits_start, runtime):
        """Score only value positions that remain MASK in the natural canvas."""
        if runtime.name_start is None:
            return None, False
        relative_start = runtime.name_start - logits_start
        relative_end = relative_start + self.scorer.value_width
        value_end = runtime.name_start + self.scorer.value_width
        if (
            relative_start < 0
            or relative_end > logits.shape[1]
            or value_end > x.shape[1]
        ):
            return None, False
        value_tokens = x[0, runtime.name_start:value_end]
        masked_positions = torch.nonzero(
            value_tokens == self.scorer.mask_id, as_tuple=False
        ).flatten()
        fully_masked = int(masked_positions.numel()) == self.scorer.value_width
        if masked_positions.numel() == 0:
            return None, fully_masked
        target_ids = self.scorer._catalog_target_ids
        if target_ids is None:
            raise RuntimeError("Agent scorer must be initialized before use")
        field_logits = logits[0, relative_start:relative_end].float()
        selected_logits = field_logits[masked_positions]
        selected_targets = target_ids[:, masked_positions]
        candidate_count = selected_targets.shape[0]
        gathered = selected_logits.unsqueeze(0).expand(
            candidate_count, -1, -1
        ).gather(2, selected_targets.unsqueeze(-1)).squeeze(-1)
        probabilities = torch.softmax(gathered.sum(dim=1), dim=0)
        return (
            dict(zip(
                self.scorer.catalog_names,
                probabilities.detach().cpu().tolist(),
            )),
            fully_masked,
        )

    def observe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        *,
        logits_start: int,
        global_step: Optional[int],
        plan_start: int,
        plan_end: int,
        phase: Optional[str],
        committed_slots: Sequence[int] = (),
        materialized_anchors: Optional[
            Sequence[Tuple[int, Tuple[int, ...], float, float]]
        ] = None,
    ) -> Dict[str, object]:
        del global_step
        full_sequence_logits = (
            logits_start <= plan_start
            and logits_start + logits.shape[1] >= plan_end
        )
        if not full_sequence_logits:
            self._semantic_skipped_evidence_not_ready_count += 1
            return {"status": "evidence_not_ready", "slots": []}
        self._observation += 1
        self.scorer.set_search_region(plan_start, plan_end)
        absolute = list(materialized_anchors or ())

        if not absolute:
            self._semantic_skipped_evidence_not_ready_count += 1
            return {"status": "evidence_not_ready", "slots": []}
        committed = set(committed_slots)
        slots: List[Optional[Dict[str, object]]] = [None] * len(absolute)
        statuses = []
        candidate_rows = [
            (
                slot,
                plan_start if slot == 0 else absolute[slot - 1][0] + 1,
                row,
            )
            for slot, row in enumerate(absolute)
        ]
        for slot, step_start, (
            anchor_start, pattern, score, observed_ratio
        ) in candidate_rows:
            if slot in committed:
                continue
            evidence_ready, evidence_signature = self._semantic_evidence(
                x,
                step_start=step_start,
                agent_anchor_start=anchor_start,
                agent_key_observed_ratio=float(observed_ratio),
                slot_id=slot,
            )
            runtime = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=anchor_start + len(pattern),
            )
            fully_masked = (
                self._agent_value_fully_masked(x, runtime)
                if evidence_ready else False
            )
            if not evidence_ready or not fully_masked:
                slots[slot] = {"semantic_evidence_ready": False}
                statuses.append("evidence_not_ready")
                continue
            if self._last_scored_evidence_signatures.get(slot) == evidence_signature:
                statuses.append("same_evidence")
                continue
            if self._observation % self.semantic_observation_stride != 0:
                statuses.append("stride")
                continue

            scorer_started = time.perf_counter()
            distribution, _fully_masked = self._score_masked_agent_value(
                logits, x, logits_start, runtime
            )
            scorer_finished = time.perf_counter()
            if distribution is None:
                slots[slot] = {"semantic_evidence_ready": False}
                statuses.append("evidence_not_ready")
                continue
            self._semantic_scorer_call_count += 1
            self._semantic_scorer_calls_by_slot[slot] = (
                self._semantic_scorer_calls_by_slot.get(slot, 0) + 1
            )
            self._semantic_scorer_total_time += scorer_finished - scorer_started
            self._last_scored_evidence_signatures[slot] = evidence_signature
            agent, probability, margin = self._rank_distribution(distribution)
            prediction_seconds = float(self.elapsed())
            slots[slot] = {
                "slot": slot,
                "seconds": prediction_seconds,
                "relative_pos": int(anchor_start - plan_start),
                "agent": agent,
                "probability": probability,
                "margin": margin,
                "anchor_score": float(score),
                "anchor_observed_ratio": float(observed_ratio),
                "semantic_value_available": True,
                "semantic_evidence_ready": True,
                "agent_value_fully_masked": self._agent_value_fully_masked(x, runtime),
                "entropy": -sum(
                    float(value) * math.log(max(float(value), 1e-12))
                    for value in distribution.values()
                ),
                "posterior": {
                    str(name): float(value)
                    for name, value in distribution.items()
                },
            }
            statuses.append("scored")
        # Keep the existing request-level skip counters observation-based.
        self._semantic_skipped_evidence_not_ready_count += int(
            "evidence_not_ready" in statuses
        )
        self._semantic_skipped_same_evidence_count += int(
            "same_evidence" in statuses
        )
        self._semantic_skipped_stride_count += int("stride" in statuses)
        status = next(
            (value for value in ("scored", "evidence_not_ready", "same_evidence", "stride")
             if value in statuses),
            "skipped_committed",
        )
        event = {
            "observation": self._observation,
            "seconds": float(self.elapsed()),
            "phase": phase,
            "plan_start": int(plan_start),
            "plan_end": int(plan_end),
            "slots": slots,
            "status": status,
        }
        if "scored" in statuses:
            self._events.append(event)
        return event

    def close(self) -> None:
        self.scorer.close()

"""Diagnostic-only oracle-span scoring for Dual Vanilla LLaDA decoding.

The observer never locates an Agent field online.  Each existing full-sequence
warmup is compressed into per-position registry scores.  Once generation has
finished, natural JSON Agent spans from the final token canvas select the
positions to replay.  No token, decoder mask, NFE, prefetch state, or runtime
commit state is changed.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import JsonAgentPriorityConfig, JsonAgentSlotRuntime
from response_agent_timing import PassiveJsonAgentMonitor


class OracleLatentAgentObserver(PassiveJsonAgentMonitor):
    """Replay native final-layer Agent logits at final, oracle-known spans."""

    full_sequence_discovery_steps = 0

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        score_chunk_size: int = 32,
    ) -> None:
        if score_chunk_size < 1:
            raise ValueError("score_chunk_size must be positive")
        super().__init__(
            tokenizer=tokenizer,
            config=JsonAgentPriorityConfig(
                catalog=list(catalog),
                priority_slots=1,
                tracking_slots=1,
                probe_period=0,
            ),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
        )
        self.score_chunk_size = int(score_chunk_size)
        self._diagnostic_overhead_seconds = 0.0
        self._score_observations: List[Dict[str, object]] = []
        self._first_visible_seconds: List[Optional[float]] = []
        self._seen_visible = None
        self._detected_slots: List[Dict[str, object]] = []
        self._slot_metrics: List[Dict[str, object]] = []
        self._evaluation_agents: List[Optional[str]] = []
        self.final_plan_parse_success = False
        self._candidate_names = tuple(str(name) for name in catalog)
        self._candidate_ids = {
            name: tuple(self._encode(name + '"'))
            for name in self._candidate_names
        }
        if any(not ids for ids in self._candidate_ids.values()):
            raise ValueError("Every oracle-latent candidate must tokenize")
        self._candidate_tokenizations = self._build_tokenizer_diagnostics()

    def _build_tokenizer_diagnostics(self) -> List[Dict[str, object]]:
        canonical_anchors = ('agent":"', 'agent": "')
        rows = []
        for name in self._candidate_names:
            candidate_ids = self._candidate_ids[name]
            checks = []
            for anchor_text in canonical_anchors:
                anchor_ids = tuple(self._encode(anchor_text))
                combined_ids = tuple(self._encode(anchor_text + name + '"'))
                checks.append({
                    "anchor_text": anchor_text,
                    "anchor_token_ids": list(anchor_ids),
                    "combined_token_ids": list(combined_ids),
                    "separate_equals_combined": (
                        anchor_ids + candidate_ids == combined_ids
                    ),
                })
            rows.append({
                "agent_name": name,
                "token_ids": list(candidate_ids),
                "token_count": len(candidate_ids),
                "boundary_checks": checks,
            })
        return rows

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self._diagnostic_overhead_seconds = 0.0
        self._score_observations.clear()
        self._detected_slots.clear()
        self._slot_metrics.clear()
        self._evaluation_agents.clear()
        self.final_plan_parse_success = False
        available = max(
            0,
            min(self.gen_length, x.shape[1] - self.prompt_length),
        )
        self._first_visible_seconds = [None] * available
        self._seen_visible = torch.zeros(
            available, device=x.device, dtype=torch.bool
        )
        self._record_visibility(x, 0.0)

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(
            0.0,
            time.perf_counter()
            - self._started_at
            - self._diagnostic_overhead_seconds,
        )

    def _record_visibility(self, x: torch.Tensor, seconds: float) -> None:
        if self._seen_visible is None or not self._first_visible_seconds:
            return
        end = self.prompt_length + len(self._first_visible_seconds)
        visible = x[0, self.prompt_length:end] != self.mask_id
        newly_visible = visible & ~self._seen_visible
        self._seen_visible |= visible
        indices = torch.nonzero(newly_visible, as_tuple=False).flatten()
        for index in indices.detach().cpu().tolist():
            if self._first_visible_seconds[index] is None:
                self._first_visible_seconds[index] = float(seconds)

    @staticmethod
    def _synchronize(tensor: torch.Tensor) -> None:
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _compress_full_sequence_scores(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CPU [candidate, generation-position] score matrices."""
        available = len(self._first_visible_seconds)
        candidate_count = len(self._candidate_names)
        normalized = torch.full(
            (candidate_count, available),
            torch.nan,
            dtype=torch.float32,
        )
        raw_sums = torch.full_like(normalized, torch.nan)
        if available == 0:
            return normalized, raw_sums

        field_logits = logits[
            0,
            self.prompt_length:self.prompt_length + available,
        ]
        log_partitions = []
        for start in range(0, available, self.score_chunk_size):
            stop = min(available, start + self.score_chunk_size)
            log_partitions.append(
                torch.logsumexp(
                    field_logits[start:stop].float(), dim=-1
                )
            )
        log_partition = torch.cat(log_partitions, dim=0)

        for candidate_index, name in enumerate(self._candidate_names):
            token_ids = self._candidate_ids[name]
            valid_starts = available - len(token_ids) + 1
            if valid_starts <= 0:
                continue
            raw_score = torch.zeros(
                valid_starts,
                device=logits.device,
                dtype=torch.float32,
            )
            logprob_score = torch.zeros_like(raw_score)
            for offset, token_id in enumerate(token_ids):
                selected = field_logits[
                    offset:offset + valid_starts, int(token_id)
                ].float()
                raw_score += selected
                logprob_score += (
                    selected
                    - log_partition[offset:offset + valid_starts]
                )
            raw_sums[candidate_index, :valid_starts] = raw_score.detach().cpu()
            normalized[candidate_index, :valid_starts] = (
                logprob_score / float(len(token_ids))
            ).detach().cpu()
        return normalized, raw_sums

    def observe(
        self,
        logits,
        x,
        logits_start,
        global_step,
        is_last_agent_step=False,
    ) -> None:
        del is_last_agent_step
        self._observed_steps += 1
        full_sequence = (
            logits_start <= self.prompt_length
            and logits_start + logits.shape[1]
            >= self.prompt_length + len(self._first_visible_seconds)
        )
        if not full_sequence:
            return
        # Make the observation timestamp mean "full logits are available".
        # The synchronization wait belongs to the model forward, not observer
        # overhead, and is therefore completed before the diagnostic timer.
        self._synchronize(logits)
        started = time.perf_counter()
        seconds = self._elapsed()
        self._record_visibility(x, seconds)
        self._full_sequence_observations += 1
        value_scores, raw_sums = self._compress_full_sequence_scores(logits)
        end = self.prompt_length + len(self._first_visible_seconds)
        visibility = (
            x[0, self.prompt_length:end] != self.mask_id
        ).detach().to(device="cpu", dtype=torch.uint8)
        self._score_observations.append({
            "observation": self._full_sequence_observations,
            "iteration": (
                None if global_step is None else int(global_step)
            ),
            "wall_time": float(seconds),
            "visibility": visibility,
            "scores": value_scores,
            "raw_sums": raw_sums,
        })
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    def step_callback(self, nfe, num_block, block_step, x) -> None:
        del nfe, num_block, block_step
        # As above, wait for the normal token transfer before timestamping it;
        # only the following visibility scan is diagnostic overhead.
        self._synchronize(x)
        started = time.perf_counter()
        seconds = self._elapsed()
        self._record_visibility(x, seconds)
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    def decoder_mask(self, mask_index, mask_start=0):
        del mask_start
        return mask_index

    def has_unconfirmed_agents(self) -> bool:
        return False

    def probing_active(self) -> bool:
        return False

    def finalize(self, x: torch.Tensor) -> None:
        self._synchronize(x)
        started = time.perf_counter()
        self._record_visibility(x, self._elapsed())
        detected = []
        for anchor_start, pattern, _score, _ratio in sorted(
            self._materialized_anchor_candidates(x), key=lambda row: row[0]
        ):
            value_start = anchor_start + len(pattern)
            runtime = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=value_start,
            )
            agent = self._observed_catalog_value(x, runtime)
            if agent is None:
                continue
            token_ids = self._candidate_ids[str(agent)]
            value_end = value_start + len(token_ids)
            exact = bool(
                value_end <= x.shape[1]
                and torch.equal(
                    x[0, value_start:value_end],
                    torch.tensor(
                        token_ids, device=x.device, dtype=x.dtype
                    ),
                )
            )
            detected.append({
                "slot_id": len(detected),
                "span_agent": str(agent),
                "value_start": int(value_start),
                "value_end": int(value_end),
                "value_start_generation_offset": int(
                    value_start - self.prompt_length
                ),
                "exact_registry_value": exact,
                "fuzzy_matched_from": runtime.fuzzy_matched_from,
            })
        self._detected_slots = detected
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    @staticmethod
    def _parse_plan_agents(text: str) -> List[Optional[str]]:
        decoder = json.JSONDecoder()
        value = str(text).strip()
        for match in re.finditer(r"\[", value):
            try:
                plan, _ = decoder.raw_decode(value[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(plan, list) and all(
                isinstance(step, dict) for step in plan
            ):
                return [step.get("agent") for step in plan]
        return []

    def set_evaluation_plan_text(self, text: str) -> None:
        self._evaluation_agents = self._parse_plan_agents(text)
        self.final_plan_parse_success = bool(self._evaluation_agents)
        self._build_slot_metrics()

    @staticmethod
    def _first_stable_correct(
        events: Sequence[Dict[str, object]], *, strict: bool = False
    ) -> Optional[Dict[str, object]]:
        suffix_all_correct = True
        result = None
        for event in reversed(events):
            suffix_all_correct = suffix_all_correct and bool(event["correct"])
            if (
                suffix_all_correct
                and bool(event["correct"])
                and (not strict or bool(event["strict_pre_materialization"]))
            ):
                result = event
        return result

    @staticmethod
    def _lead(
        natural_time: Optional[float], event: Optional[Dict[str, object]]
    ) -> Optional[float]:
        if natural_time is None or event is None:
            return None
        return float(natural_time - float(event["wall_time"]))

    def _build_slot_metrics(self) -> None:
        final_agents = (
            self._evaluation_agents
            if self._evaluation_agents
            else [row["span_agent"] for row in self._detected_slots]
        )
        slot_count = min(len(final_agents), len(self._detected_slots))
        results = []
        for slot_id in range(slot_count):
            span = dict(self._detected_slots[slot_id])
            final_agent = final_agents[slot_id]
            offset = int(span["value_start_generation_offset"])
            final_ids = self._candidate_ids.get(str(final_agent))
            if final_ids is None or offset < 0:
                continue
            value_end_offset = offset + len(final_ids)
            first_char_time = (
                self._first_visible_seconds[offset]
                if offset < len(self._first_visible_seconds) else None
            )
            value_times = self._first_visible_seconds[offset:value_end_offset]
            natural_time = (
                max(float(value) for value in value_times)
                if len(value_times) == len(final_ids)
                and all(value is not None for value in value_times)
                else None
            )
            trajectory = []
            for observation in self._score_observations:
                candidate_scores = {}
                candidate_raw = {}
                ranked = []
                for candidate_index, name in enumerate(self._candidate_names):
                    score_tensor = observation["scores"]
                    raw_tensor = observation["raw_sums"]
                    valid = (
                        offset < score_tensor.shape[1]
                        and bool(torch.isfinite(
                            score_tensor[candidate_index, offset]
                        ))
                    )
                    score = (
                        float(score_tensor[candidate_index, offset])
                        if valid else None
                    )
                    raw = (
                        float(raw_tensor[candidate_index, offset])
                        if valid else None
                    )
                    candidate_scores[name] = score
                    candidate_raw[name] = raw
                    if score is not None:
                        ranked.append((name, score))
                if not ranked:
                    continue
                ranked.sort(key=lambda item: item[1], reverse=True)
                predicted_agent, top1_score = ranked[0]
                top2_score = ranked[1][1] if len(ranked) > 1 else None
                visible = observation["visibility"]
                visible_count = int(
                    visible[offset:min(value_end_offset, visible.shape[0])]
                    .sum().item()
                )
                wall_time = float(observation["wall_time"])
                strict = bool(
                    visible_count == 0
                    and first_char_time is not None
                    and wall_time < float(first_char_time)
                )
                trajectory.append({
                    "slot_id": slot_id,
                    "iteration": observation["iteration"],
                    "observation": observation["observation"],
                    "wall_time": wall_time,
                    "value_start": span["value_start"],
                    "value_end": int(span["value_start"] + len(final_ids)),
                    "agent_value_visible_token_count": visible_count,
                    "strict_pre_materialization": strict,
                    "candidate_scores": candidate_scores,
                    "candidate_raw_logit_sums": candidate_raw,
                    "invalid_candidates": [
                        name for name, score in candidate_scores.items()
                        if score is None
                    ],
                    "predicted_agent": predicted_agent,
                    "top1_score": float(top1_score),
                    "top2_score": (
                        None if top2_score is None else float(top2_score)
                    ),
                    "margin": (
                        None if top2_score is None
                        else float(top1_score - top2_score)
                    ),
                    "correct": predicted_agent == final_agent,
                })

            pre_natural = [
                event for event in trajectory
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            first = trajectory[0] if trajectory else None
            earliest_correct = next(
                (event for event in pre_natural if event["correct"]), None
            )
            stable_correct = self._first_stable_correct(pre_natural)
            strict_earliest = next(
                (
                    event for event in pre_natural
                    if event["correct"]
                    and event["strict_pre_materialization"]
                ),
                None,
            )
            strict_stable = self._first_stable_correct(
                pre_natural, strict=True
            )
            flip_count = sum(
                left["predicted_agent"] != right["predicted_agent"]
                for left, right in zip(pre_natural, pre_natural[1:])
            )
            first_stays_correct = bool(
                pre_natural
                and pre_natural[0]["correct"]
                and all(event["correct"] for event in pre_natural)
            )
            wrong_search_to_stable_calculation = None
            if final_agent == "calculation_agent" and stable_correct is not None:
                wrong_search = next(
                    (
                        event for event in pre_natural
                        if not event["correct"]
                        and event["predicted_agent"] == "search_agent"
                        and float(event["wall_time"])
                        < float(stable_correct["wall_time"])
                    ),
                    None,
                )
                if wrong_search is not None:
                    wrong_search_to_stable_calculation = float(
                        float(stable_correct["wall_time"])
                        - float(wrong_search["wall_time"])
                    )
            results.append({
                **span,
                "final_agent": final_agent,
                "final_agent_matches_span": final_agent == span["span_agent"],
                "agent_first_char_time": first_char_time,
                "natural_agent_time": natural_time,
                "first_latent_prediction_time": (
                    first["wall_time"] if first else None
                ),
                "first_latent_time": first["wall_time"] if first else None,
                "first_latent_prediction": (
                    first["predicted_agent"] if first else None
                ),
                "first_prediction_correct": (
                    first["correct"] if first else None
                ),
                "first_prediction_stays_correct": first_stays_correct,
                "earliest_correct_latent_time": (
                    earliest_correct["wall_time"] if earliest_correct else None
                ),
                "earliest_correct_time": (
                    earliest_correct["wall_time"] if earliest_correct else None
                ),
                "earliest_stable_correct_latent_time": (
                    stable_correct["wall_time"] if stable_correct else None
                ),
                "earliest_stable_correct_time": (
                    stable_correct["wall_time"] if stable_correct else None
                ),
                "strict_earliest_correct_latent_time": (
                    strict_earliest["wall_time"] if strict_earliest else None
                ),
                "strict_stable_correct_latent_time": (
                    strict_stable["wall_time"] if strict_stable else None
                ),
                "earliest_correct_lead": self._lead(
                    natural_time, earliest_correct
                ),
                "stable_correct_lead": self._lead(
                    natural_time, stable_correct
                ),
                "strict_earliest_correct_lead": self._lead(
                    natural_time, strict_earliest
                ),
                "strict_stable_correct_lead": self._lead(
                    natural_time, strict_stable
                ),
                "latent_prediction_count": len(trajectory),
                "prediction_count": len(trajectory),
                "pre_natural_prediction_count": len(pre_natural),
                "top1_flip_count": int(flip_count),
                "flip_count": int(flip_count),
                "wrong_search_to_stable_calculation_seconds": (
                    wrong_search_to_stable_calculation
                ),
                "trajectory": trajectory,
            })
        self._slot_metrics = results
        # Subclasses may derive additional diagnostic-only trajectories from
        # the same compressed full-sequence observations.  Keep this hook
        # before the large position-wide matrices are released.
        self._before_clear_score_observations()
        # Position-wide score matrices are no longer needed once oracle spans
        # have selected their scalar trajectories.
        self._score_observations.clear()

    def _before_clear_score_observations(self) -> None:
        """Extension hook for read-only diagnostics sharing oracle replay."""
        return None

    def metrics(self) -> Dict[str, object]:
        if not self._slot_metrics and self._detected_slots:
            self._build_slot_metrics()
        return {
            "policy": "oracle_latent",
            "timing_source": "oracle_span_full_sequence_logits",
            "diagnostic_only": True,
            "read_only": True,
            "writes_agent_tokens": False,
            "changes_decoder_mask": False,
            "triggers_prefetch": False,
            "extra_model_forwards": 0,
            "runtime_agent_registry": list(self._candidate_names),
            "candidate_tokenizations": self._candidate_tokenizations,
            "full_sequence_observation_count": self._full_sequence_observations,
            "observation_cadence": "existing_dual_block_warmups",
            "score_type": "mean_token_logprob",
            "raw_score_type": "raw_logit_sum",
            "score_chunk_size": self.score_chunk_size,
            "diagnostic_observer_overhead_seconds": float(
                self._diagnostic_overhead_seconds
            ),
            "final_plan_parse_success": self.final_plan_parse_success,
            "oracle_span_count": len(self._detected_slots),
            "agent_count": len(self._slot_metrics),
            "agent_slots": self._slot_metrics,
            "probe_forwards": 0,
        }

    def close(self) -> None:
        return None

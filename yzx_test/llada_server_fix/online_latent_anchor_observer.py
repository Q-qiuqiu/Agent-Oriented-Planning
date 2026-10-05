"""Read-only online latent JSON-Agent anchor diagnostic.

This observer shares the existing Dual Vanilla full-sequence warmups with the
oracle latent diagnostic.  At every warmup it locates potential ``agent`` JSON
anchors from logits, scores registry values at each inferred value position,
and later compares the online positions with final oracle spans.  It never
writes tokens, changes masks, commits an Agent, or triggers prefetch.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from oracle_latent_observer import OracleLatentAgentObserver


class OnlineLatentAnchorObserver(OracleLatentAgentObserver):
    """Compare online logits anchors with oracle Agent-value spans."""

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        score_chunk_size: int = 32,
        anchor_min_logit_margin: float = -6.0,
        anchor_position_tolerance: int = 4,
        anchor_stable_observations: int = 2,
        min_anchor_gap: int = 12,
    ) -> None:
        if anchor_stable_observations < 1:
            raise ValueError("anchor_stable_observations must be positive")
        if anchor_position_tolerance < 0:
            raise ValueError("anchor_position_tolerance cannot be negative")
        if min_anchor_gap < 1:
            raise ValueError("min_anchor_gap must be positive")
        super().__init__(
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            score_chunk_size=score_chunk_size,
        )
        self.anchor_min_logit_margin = float(anchor_min_logit_margin)
        self.anchor_position_tolerance = int(anchor_position_tolerance)
        self.anchor_stable_observations = int(anchor_stable_observations)
        self.min_anchor_gap = int(min_anchor_gap)
        self._online_anchor_variants = self._build_online_anchor_variants()
        self._anchor_candidate_tokenizations = (
            self._build_anchor_candidate_tokenizations()
        )
        self._online_slot_states: List[Dict[str, object]] = []
        self._online_anchor_observation_counts: List[int] = []

    def _build_online_anchor_variants(self) -> List[Dict[str, object]]:
        canonical = ('agent":"', 'agent": "')
        texts = list(canonical)
        for before_colon in (" ", "\t", "\n", "\n  "):
            for after_colon in ("", " ", "\t", "\n", "\n  "):
                texts.append(f'agent"{before_colon}:{after_colon}"')

        # Token-equivalent whitespace spellings are one detector pattern.  A
        # canonical spelling wins so that all-MASK speculative discovery uses
        # the same conservative variants as the legacy implementation.
        variants: Dict[Tuple[int, ...], Dict[str, object]] = {}
        for text in dict.fromkeys(texts):
            token_ids = tuple(self._encode(text))
            if not token_ids:
                raise ValueError("Tokenizer produced an empty online anchor")
            speculative = text in canonical
            previous = variants.get(token_ids)
            row = {
                "anchor_variant": text,
                "anchor_token_ids": token_ids,
                "speculative": speculative,
            }
            if previous is None or (
                speculative and not bool(previous["speculative"])
            ):
                variants[token_ids] = row
        return list(variants.values())

    def _build_anchor_candidate_tokenizations(self) -> List[Dict[str, object]]:
        rows = []
        for variant in self._online_anchor_variants:
            anchor_text = str(variant["anchor_variant"])
            anchor_ids = tuple(variant["anchor_token_ids"])
            for name in self._candidate_names:
                candidate_ids = self._candidate_ids[name]
                combined_ids = tuple(self._encode(anchor_text + name + '"'))
                rows.append({
                    "anchor_variant": anchor_text,
                    "anchor_token_ids": list(anchor_ids),
                    "candidate_agent": name,
                    "candidate_token_ids": list(candidate_ids),
                    "candidate_token_count": len(candidate_ids),
                    "combined_token_ids": list(combined_ids),
                    "separate_equals_combined": (
                        anchor_ids + candidate_ids == combined_ids
                    ),
                })
        return rows

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self._online_slot_states.clear()
        self._online_anchor_observation_counts.clear()

    def _online_anchor_candidates(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
    ) -> List[Dict[str, object]]:
        """Adapt legacy logits anchor discovery without a fixed slot count."""
        sequence_logits = logits[0]
        sequence_max = sequence_logits.amax(dim=-1)
        absolute_end = logits_start + sequence_logits.shape[0]
        generation_start = self.prompt_length
        generation_end = self.prompt_length + self.gen_length
        by_start: Dict[int, Dict[str, object]] = {}

        for variant in self._online_anchor_variants:
            pattern = tuple(variant["anchor_token_ids"])
            width = len(pattern)
            start = max(generation_start, logits_start)
            end = min(generation_end, absolute_end) - width + 1
            if end <= start:
                continue
            relative = start - logits_start
            count = end - start
            scores = torch.zeros(
                count, device=logits.device, dtype=torch.float32
            )
            compatible = torch.ones(
                count, device=logits.device, dtype=torch.bool
            )
            observed = torch.zeros_like(scores)
            base_positions = torch.arange(start, end, device=x.device)
            for offset, token_id in enumerate(pattern):
                row_start = relative + offset
                row_end = row_start + count
                target = sequence_logits[row_start:row_end, token_id].float()
                scores += target - sequence_max[row_start:row_end].float()
                current = x[0, base_positions + offset]
                compatible &= (current == self.mask_id) | (current == token_id)
                observed += (current == token_id).float()
            scores /= float(width)
            observed /= float(width)
            if bool(variant["speculative"]):
                eligible = compatible & (
                    (scores >= self.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
            else:
                # Preserve the legacy precision guard for unusual whitespace:
                # compact and normal-spaced anchors are speculative; tolerant
                # newline/tab spellings are accepted once naturally visible.
                eligible = compatible & (observed == 1.0)
            indices = torch.nonzero(eligible, as_tuple=False).flatten()
            for local_index in indices.detach().cpu().tolist():
                absolute_start = start + int(local_index)
                row = {
                    "anchor_start": absolute_start,
                    "anchor_end": absolute_start + width,
                    "value_start": absolute_start + width,
                    "anchor_variant": variant["anchor_variant"],
                    "anchor_token_ids": list(pattern),
                    "anchor_score": float(
                        scores[local_index].detach().cpu()
                    ),
                    "anchor_observed_ratio": float(
                        observed[local_index].detach().cpu()
                    ),
                    "speculative": bool(variant["speculative"]),
                }
                previous = by_start.get(absolute_start)
                if previous is None or (
                    row["anchor_observed_ratio"], row["anchor_score"]
                ) > (
                    previous["anchor_observed_ratio"],
                    previous["anchor_score"],
                ):
                    by_start[absolute_start] = row

        # Equivalent whitespace variants may begin a token or two apart.
        # Collapse local alternatives by evidence/confidence, then preserve
        # occurrence order exactly as the legacy controller does.
        cluster_radius = max(
            len(row["anchor_token_ids"])
            for row in self._online_anchor_variants
        )
        clusters: List[List[Dict[str, object]]] = []
        for row in sorted(by_start.values(), key=lambda item: item["anchor_start"]):
            if (
                not clusters
                or int(row["anchor_start"])
                - int(clusters[-1][-1]["anchor_start"])
                >= cluster_radius
            ):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        ranked = [
            max(
                cluster,
                key=lambda item: (
                    item["anchor_observed_ratio"], item["anchor_score"]
                ),
            )
            for cluster in clusters
        ]
        ranked.sort(key=lambda item: item["anchor_start"])

        selected: List[Dict[str, object]] = []
        for row in ranked:
            if any(
                abs(int(row["anchor_start"]) - int(old["anchor_start"]))
                < self.min_anchor_gap
                for old in selected
            ):
                continue
            selected.append(row)
        return selected

    def _track_online_anchors(
        self, anchors: List[Dict[str, object]], wall_time: float
    ) -> None:
        while len(self._online_slot_states) < len(anchors):
            self._online_slot_states.append({
                "current_anchor_start": None,
                "current_anchor_end": None,
                "current_value_start": None,
                "previous_anchor_start": None,
                "position_stable_count": 0,
                "first_anchor_seen_time": None,
                "first_anchor_stable_time": None,
            })
        for slot_id, anchor in enumerate(anchors):
            state = self._online_slot_states[slot_id]
            current = int(anchor["anchor_start"])
            previous = state["current_anchor_start"]
            delta = None if previous is None else current - int(previous)
            if previous is not None and abs(int(delta)) <= self.anchor_position_tolerance:
                stable_count = int(state["position_stable_count"]) + 1
            else:
                stable_count = 1
            if state["first_anchor_seen_time"] is None:
                state["first_anchor_seen_time"] = float(wall_time)
            if (
                stable_count >= self.anchor_stable_observations
                and state["first_anchor_stable_time"] is None
            ):
                state["first_anchor_stable_time"] = float(wall_time)
            state.update({
                "previous_anchor_start": previous,
                "current_anchor_start": current,
                "current_anchor_end": int(anchor["anchor_end"]),
                "current_value_start": int(anchor["value_start"]),
                "position_stable_count": stable_count,
            })
            anchor.update({
                "slot_id": slot_id,
                "previous_anchor_start": previous,
                "anchor_position_delta": delta,
                "position_stable_count": stable_count,
                "first_anchor_seen_time": state["first_anchor_seen_time"],
                "first_anchor_stable_time": state["first_anchor_stable_time"],
            })
        # A missing rank breaks position stability, but its first-seen history
        # remains available if that online slot reappears later.
        for state in self._online_slot_states[len(anchors):]:
            state["previous_anchor_start"] = state["current_anchor_start"]
            state["current_anchor_start"] = None
            state["current_anchor_end"] = None
            state["current_value_start"] = None
            state["position_stable_count"] = 0

    def observe(
        self,
        logits,
        x,
        logits_start,
        global_step,
        is_last_agent_step=False,
    ) -> None:
        before = len(self._score_observations)
        super().observe(
            logits,
            x,
            logits_start,
            global_step,
            is_last_agent_step,
        )
        if len(self._score_observations) == before:
            return
        started = time.perf_counter()
        observation = self._score_observations[-1]
        anchors = self._online_anchor_candidates(logits, x, logits_start)
        self._track_online_anchors(anchors, float(observation["wall_time"]))
        observation["online_anchors"] = anchors
        self._online_anchor_observation_counts.append(len(anchors))
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    @staticmethod
    def _first_stable_anchor(
        events: Sequence[Dict[str, object]], *, tolerance: int = 0
    ) -> Optional[Dict[str, object]]:
        suffix_correct = True
        result = None
        for event in reversed(events):
            error = event.get("position_error")
            correct = error is not None and abs(int(error)) <= tolerance
            suffix_correct = suffix_correct and correct
            if suffix_correct and correct:
                result = event
        return result

    @staticmethod
    def _count_flips(events: Sequence[Dict[str, object]]) -> int:
        predictions = [
            event["predicted_agent"]
            for event in events
            if event.get("predicted_agent") is not None
        ]
        return sum(left != right for left, right in zip(predictions, predictions[1:]))

    def _online_event(
        self,
        *,
        slot_id: int,
        final_agent: Optional[str],
        oracle_value_start: int,
        oracle_value_end: int,
        first_char_time: Optional[float],
        natural_time: Optional[float],
        observation: Dict[str, object],
        oracle_event: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        anchor = self._anchor_for_slot(
            observation=observation,
            slot_id=slot_id,
            oracle_value_start=oracle_value_start,
        )
        wall_time = float(observation["wall_time"])
        result: Dict[str, object] = {
            "slot_id": slot_id,
            "iteration": observation["iteration"],
            "observation": observation["observation"],
            "wall_time": wall_time,
            "online_anchor_found": anchor is not None,
            "online_anchor_start": None,
            "online_anchor_end": None,
            "online_value_start": None,
            "oracle_value_start": oracle_value_start,
            "position_error": None,
            "anchor_score": None,
            "anchor_observed_ratio": None,
            "anchor_variant": None,
            "anchor_token_ids": None,
            "previous_anchor_start": None,
            "anchor_position_delta": None,
            "position_stable_count": 0,
            "candidate_scores": {
                name: None for name in self._candidate_names
            },
            "candidate_raw_logit_sums": {
                name: None for name in self._candidate_names
            },
            "invalid_candidates": list(self._candidate_names),
            "predicted_agent": None,
            "top1_score": None,
            "top2_score": None,
            "margin": None,
            "correct": False,
            "strict_pre_materialization": bool(
                first_char_time is not None
                and wall_time < float(first_char_time)
            ),
            "agent_first_char_time": first_char_time,
            "natural_agent_time": natural_time,
            "oracle_predicted_agent": (
                oracle_event.get("predicted_agent") if oracle_event else None
            ),
            "oracle_correct": (
                oracle_event.get("correct") if oracle_event else None
            ),
        }
        if anchor is None:
            return result

        result.update({
            "online_anchor_start": int(anchor["anchor_start"]),
            "online_anchor_end": int(anchor["anchor_end"]),
            "online_value_start": int(anchor["value_start"]),
            "position_error": int(anchor["value_start"]) - oracle_value_start,
            "anchor_score": float(anchor["anchor_score"]),
            "anchor_observed_ratio": float(anchor["anchor_observed_ratio"]),
            "anchor_variant": anchor["anchor_variant"],
            "anchor_token_ids": list(anchor["anchor_token_ids"]),
            "previous_anchor_start": anchor["previous_anchor_start"],
            "anchor_position_delta": anchor["anchor_position_delta"],
            "position_stable_count": anchor["position_stable_count"],
        })
        offset = int(anchor["value_start"]) - self.prompt_length
        scores = observation["scores"]
        raw_sums = observation["raw_sums"]
        candidate_scores: Dict[str, Optional[float]] = {}
        candidate_raw: Dict[str, Optional[float]] = {}
        ranked = []
        for candidate_index, name in enumerate(self._candidate_names):
            valid = (
                offset >= 0
                and offset < scores.shape[1]
                and bool(torch.isfinite(scores[candidate_index, offset]))
            )
            score = float(scores[candidate_index, offset]) if valid else None
            raw = float(raw_sums[candidate_index, offset]) if valid else None
            candidate_scores[name] = score
            candidate_raw[name] = raw
            if score is not None:
                ranked.append((name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        predicted = ranked[0][0] if ranked else None
        top1 = ranked[0][1] if ranked else None
        top2 = ranked[1][1] if len(ranked) > 1 else None
        visible = observation["visibility"]
        oracle_offset = oracle_value_start - self.prompt_length
        oracle_visible_count = 0
        if 0 <= oracle_offset < visible.shape[0]:
            oracle_visible_count = int(
                visible[
                    oracle_offset:min(
                        oracle_value_end - self.prompt_length,
                        visible.shape[0],
                    )
                ].sum().item()
            )
        result.update({
            "candidate_scores": candidate_scores,
            "candidate_raw_logit_sums": candidate_raw,
            "invalid_candidates": [
                name for name, score in candidate_scores.items()
                if score is None
            ],
            "predicted_agent": predicted,
            "top1_score": top1,
            "top2_score": top2,
            "margin": (
                None if top1 is None or top2 is None else top1 - top2
            ),
            "correct": predicted == final_agent,
            "agent_value_visible_token_count": oracle_visible_count,
            "strict_pre_materialization": bool(
                oracle_visible_count == 0
                and first_char_time is not None
                and wall_time < float(first_char_time)
            ),
        })
        return result

    def _anchor_for_slot(
        self,
        *,
        observation: Dict[str, object],
        slot_id: int,
        oracle_value_start: int,
    ) -> Optional[Dict[str, object]]:
        """Return the rank-associated anchor for the coarse baseline."""
        del oracle_value_start
        anchors = observation.get("online_anchors") or []
        return anchors[slot_id] if slot_id < len(anchors) else None

    def _before_clear_score_observations(self) -> None:
        combined_slots = []
        for oracle_slot in self._slot_metrics:
            slot_id = int(oracle_slot["slot_id"])
            final_agent = oracle_slot.get("final_agent")
            oracle_value_start = int(oracle_slot["value_start"])
            oracle_value_end = int(oracle_slot["value_end"])
            first_char_time = oracle_slot.get("agent_first_char_time")
            natural_time = oracle_slot.get("natural_agent_time")
            oracle_by_observation = {
                event["observation"]: event
                for event in oracle_slot.get("trajectory") or []
            }
            trajectory = [
                self._online_event(
                    slot_id=slot_id,
                    final_agent=final_agent,
                    oracle_value_start=oracle_value_start,
                    oracle_value_end=oracle_value_end,
                    first_char_time=first_char_time,
                    natural_time=natural_time,
                    observation=observation,
                    oracle_event=oracle_by_observation.get(
                        observation["observation"]
                    ),
                )
                for observation in self._score_observations
            ]
            pre_natural = [
                event for event in trajectory
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            found = [
                event for event in pre_natural
                if event["online_anchor_found"]
            ]
            correct_anchors = [
                event for event in pre_natural
                if event.get("position_error") == 0
            ]
            first_correct_anchor = correct_anchors[0] if correct_anchors else None
            stable_anchor = self._first_stable_anchor(pre_natural)
            stable_anchor_tolerant = self._first_stable_anchor(
                pre_natural, tolerance=self.anchor_position_tolerance
            )
            predictions = [
                event for event in pre_natural
                if event.get("predicted_agent") is not None
            ]
            first_prediction = predictions[0] if predictions else None
            earliest_correct = next(
                (event for event in pre_natural if event["correct"]), None
            )
            stable_correct = self._first_stable_correct(pre_natural)
            strict_stable = self._first_stable_correct(
                pre_natural, strict=True
            )
            exact_anchor_predictions = [
                event for event in pre_natural
                if event.get("position_error") == 0
                and event.get("predicted_agent") is not None
            ]
            position_errors = [
                abs(int(event["position_error"]))
                for event in found
                if event.get("position_error") is not None
            ]
            oracle_stable_time = oracle_slot.get(
                "earliest_stable_correct_latent_time"
            )
            oracle_stable_lead = oracle_slot.get("stable_correct_lead")
            online_stable_lead = self._lead(natural_time, stable_correct)
            state = (
                self._online_slot_states[slot_id]
                if slot_id < len(self._online_slot_states) else {}
            )
            first_online_anchor_time = (
                found[0]["wall_time"] if found else None
            )
            anchor_lead = self._lead(natural_time, stable_anchor)
            oracle_first_time = oracle_slot.get("first_latent_time")
            first_stable_anchor_time = (
                stable_anchor["wall_time"] if stable_anchor else None
            )
            combined = dict(oracle_slot)
            combined.update({
                "oracle_trajectory": combined.pop("trajectory", []),
                "online_trajectory": trajectory,
                "first_online_anchor_time": first_online_anchor_time,
                "first_anchor_seen_time": first_online_anchor_time,
                "first_position_stable_time": state.get(
                    "first_anchor_stable_time"
                ),
                "first_correct_anchor_time": (
                    first_correct_anchor["wall_time"]
                    if first_correct_anchor else None
                ),
                "first_stable_correct_anchor_time": first_stable_anchor_time,
                "first_stable_tolerant_anchor_time": (
                    stable_anchor_tolerant["wall_time"]
                    if stable_anchor_tolerant else None
                ),
                "anchor_localization_lead": anchor_lead,
                "oracle_vs_online_anchor_delay": (
                    None
                    if first_stable_anchor_time is None
                    or oracle_first_time is None
                    else float(first_stable_anchor_time)
                    - float(oracle_first_time)
                ),
                "anchor_found": bool(found),
                "correct_anchor_found": bool(correct_anchors),
                "stable_correct_anchor": stable_anchor is not None,
                "stable_tolerant_anchor": stable_anchor_tolerant is not None,
                "anchor_found_observation_count": len(found),
                "correct_anchor_observation_count": len(correct_anchors),
                "mean_abs_position_error": (
                    sum(position_errors) / len(position_errors)
                    if position_errors else None
                ),
                "anchor_position_error": (
                    position_errors[-1] if position_errors else None
                ),
                "first_online_prediction_time": (
                    first_prediction["wall_time"]
                    if first_prediction else None
                ),
                "first_online_prediction": (
                    first_prediction["predicted_agent"]
                    if first_prediction else None
                ),
                "first_online_prediction_correct": (
                    first_prediction["correct"]
                    if first_prediction else None
                ),
                "earliest_correct_online_time": (
                    earliest_correct["wall_time"]
                    if earliest_correct else None
                ),
                "earliest_stable_correct_online_time": (
                    stable_correct["wall_time"]
                    if stable_correct else None
                ),
                "strict_stable_correct_online_time": (
                    strict_stable["wall_time"]
                    if strict_stable else None
                ),
                "online_earliest_correct_lead": self._lead(
                    natural_time, earliest_correct
                ),
                "online_stable_lead": online_stable_lead,
                "strict_online_stable_lead": self._lead(
                    natural_time, strict_stable
                ),
                "online_prediction_count": len(predictions),
                "online_flip_count": self._count_flips(pre_natural),
                "correct_anchor_flip_count": self._count_flips(
                    exact_anchor_predictions
                ),
                "wrong_anchor_observation_count": sum(
                    event.get("online_anchor_found", False)
                    and event.get("position_error") != 0
                    for event in pre_natural
                ),
                "wrong_prediction_wrong_anchor_count": sum(
                    not event["correct"]
                    and event.get("online_anchor_found", False)
                    and event.get("position_error") != 0
                    for event in pre_natural
                ),
                "wrong_prediction_correct_anchor_count": sum(
                    not event["correct"]
                    and event.get("position_error") == 0
                    for event in pre_natural
                ),
                "oracle_stable_correct_time": oracle_stable_time,
                "oracle_stable_lead": oracle_stable_lead,
                "lead_loss": (
                    None
                    if oracle_stable_lead is None
                    or online_stable_lead is None
                    else float(oracle_stable_lead)
                    - float(online_stable_lead)
                ),
            })
            combined_slots.append(combined)
        self._slot_metrics = combined_slots

    def metrics(self) -> Dict[str, object]:
        result = super().metrics()
        result.update({
            "policy": "online_latent_diagnostic",
            "timing_source": "online_anchor_and_oracle_full_sequence_logits",
            "anchor_detector": "legacy_json_agent_priority_logits",
            "anchor_min_logit_margin": self.anchor_min_logit_margin,
            "anchor_position_tolerance": self.anchor_position_tolerance,
            "anchor_stable_observations": self.anchor_stable_observations,
            "min_anchor_gap": self.min_anchor_gap,
            "anchor_candidate_tokenizations": (
                self._anchor_candidate_tokenizations
            ),
            "online_anchor_observation_counts": (
                self._online_anchor_observation_counts
            ),
            "online_anchor_slot_count": len(self._online_slot_states),
        })
        return result

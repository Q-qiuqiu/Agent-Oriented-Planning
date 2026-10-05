"""Read-only region-level latent Agent diagnostic.

The observer keeps the existing online anchor detector and persistent track
association, but treats a track position as the centre of a small region
rather than as an exact Agent-value boundary.  Candidate Agent identities are
scored across every alignment in that region.  Nothing in this module commits
an Agent, changes the token canvas/decoder mask, or triggers prefetch.
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from oracle_latent_observer import OracleLatentAgentObserver
from refined_online_latent_anchor_observer import (
    RefinedOnlineLatentAnchorObserver,
)


AGGREGATIONS = ("max", "top2_mean", "soft")
SCORE_VARIANTS = ("agent_only", "full_template")


class RegionLatentAgentObserver(RefinedOnlineLatentAnchorObserver):
    """Score Agent identity across persistent-track neighbourhoods."""

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
        region_radius: int = 4,
        region_top_k: int = 2,
        region_temperature: float = 1.0,
        main_aggregation: str = "top2_mean",
        anchor_position_tolerance: int = 4,
        anchor_stable_observations: int = 2,
        max_track_misses: int = 3,
        max_track_match_distance: int = 64,
        track_stable_observations: int = 2,
        track_score_weight: float = 0.05,
    ) -> None:
        if region_radius < 0:
            raise ValueError("region_radius cannot be negative")
        if region_top_k < 1:
            raise ValueError("region_top_k must be positive")
        if region_temperature <= 0:
            raise ValueError("region_temperature must be positive")
        if main_aggregation not in AGGREGATIONS:
            raise ValueError(
                f"main_aggregation must be one of {AGGREGATIONS}"
            )
        super().__init__(
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            persistent_tracking=True,
            score_chunk_size=score_chunk_size,
            anchor_min_logit_margin=anchor_min_logit_margin,
            # Preserve the existing coarse/refinement/tracking implementation.
            refinement_radius=4,
            refinement_anchor_weight=1.0,
            refinement_agent_weight=1.0,
            anchor_position_tolerance=anchor_position_tolerance,
            anchor_stable_observations=anchor_stable_observations,
            max_track_misses=max_track_misses,
            max_track_match_distance=max_track_match_distance,
            track_stable_observations=track_stable_observations,
            track_score_weight=track_score_weight,
        )
        self.region_radius = int(region_radius)
        self.region_top_k = int(region_top_k)
        self.region_temperature = float(region_temperature)
        self.main_aggregation = str(main_aggregation)
        self.main_score_variant = "agent_only"
        self._template_variants = self._build_template_variants()
        self._false_track_summary: Dict[str, object] = {}
        self._all_region_track_trajectory: List[Dict[str, object]] = []

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self._false_track_summary.clear()
        self._all_region_track_trajectory.clear()

    def _build_template_variants(self) -> List[Dict[str, object]]:
        rows = []
        for anchor_text in ('"agent":"', '"agent": "'):
            anchor_ids = tuple(self._encode(anchor_text))
            for name in self._candidate_names:
                agent_ids = self._candidate_ids[name]
                combined_ids = tuple(
                    self._encode(anchor_text + name + '"')
                )
                rows.append({
                    "anchor_text": anchor_text,
                    "anchor_token_ids": list(anchor_ids),
                    "candidate_agent": name,
                    "candidate_token_ids": list(agent_ids),
                    "candidate_token_count": len(agent_ids),
                    "template_token_ids": list(combined_ids),
                    "template_token_count": len(combined_ids),
                    "value_offset_tokens": len(anchor_ids),
                    "separate_equals_combined": (
                        anchor_ids + agent_ids == combined_ids
                    ),
                })
        return rows

    def _compress_template_scores(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return best canonical-template score indexed by value position."""
        available = len(self._first_visible_seconds)
        count = len(self._candidate_names)
        normalized = torch.full((count, available), torch.nan)
        raw_sums = torch.full_like(normalized, torch.nan)
        if available == 0:
            return normalized, raw_sums

        field_logits = logits[
            0, self.prompt_length:self.prompt_length + available
        ]
        partitions = []
        for start in range(0, available, self.score_chunk_size):
            stop = min(available, start + self.score_chunk_size)
            partitions.append(
                torch.logsumexp(field_logits[start:stop].float(), dim=-1)
            )
        log_partition = torch.cat(partitions, dim=0)
        name_to_index = {
            name: index for index, name in enumerate(self._candidate_names)
        }
        for row in self._template_variants:
            # A value-position index is meaningful only when tokenization has
            # a real boundary between the anchor and Agent value.  The failed
            # boundary checks remain in diagnostics rather than being silently
            # treated as aligned templates.
            if not row["separate_equals_combined"]:
                continue
            token_ids = tuple(row["template_token_ids"])
            value_offset = int(row["value_offset_tokens"])
            valid_starts = available - len(token_ids) + 1
            if valid_starts <= 0:
                continue
            raw = torch.zeros(
                valid_starts, device=logits.device, dtype=torch.float32
            )
            logprob = torch.zeros_like(raw)
            for offset, token_id in enumerate(token_ids):
                selected = field_logits[
                    offset:offset + valid_starts, int(token_id)
                ].float()
                raw += selected
                logprob += selected - log_partition[
                    offset:offset + valid_starts
                ]
            score = (logprob / float(len(token_ids))).detach().cpu()
            raw = raw.detach().cpu()
            candidate_index = name_to_index[str(row["candidate_agent"])]
            for template_start in range(valid_starts):
                value_index = template_start + value_offset
                if value_index >= available:
                    continue
                old = normalized[candidate_index, value_index]
                if not bool(torch.isfinite(old)) or score[template_start] > old:
                    normalized[candidate_index, value_index] = score[
                        template_start
                    ]
                    raw_sums[candidate_index, value_index] = raw[
                        template_start
                    ]
        return normalized, raw_sums

    def _aggregate_values(
        self, values: List[Tuple[int, float, Optional[float]]]
    ) -> Dict[str, object]:
        ranked = sorted(values, key=lambda row: row[1], reverse=True)
        if not ranked:
            return {
                "max_score": None,
                "top2_mean_score": None,
                "soft_score": None,
                "best_alignment_position": None,
                "best_alignment_score": None,
                "best_alignment_raw_logit_sum": None,
                "alignment_count": 0,
            }
        top = ranked[:self.region_top_k]
        tensor = torch.tensor([row[1] for row in ranked], dtype=torch.float64)
        soft = self.region_temperature * torch.logsumexp(
            tensor / self.region_temperature, dim=0
        )
        return {
            "max_score": float(ranked[0][1]),
            "top2_mean_score": float(
                sum(row[1] for row in top) / len(top)
            ),
            "soft_score": float(soft),
            "best_alignment_position": int(ranked[0][0]),
            "best_alignment_score": float(ranked[0][1]),
            "best_alignment_raw_logit_sum": ranked[0][2],
            "alignment_count": len(ranked),
        }

    def _region_bounds(
        self, snapshots: Dict[int, Dict[str, object]]
    ) -> Dict[int, Tuple[int, int]]:
        ordered = sorted(
            snapshots.items(),
            key=lambda item: int(item[1]["refined_value_start"]),
        )
        lower = self.prompt_length
        upper = self.prompt_length + len(self._first_visible_seconds) - 1
        result = {}
        for index, (track_id, snapshot) in enumerate(ordered):
            center = int(snapshot["refined_value_start"])
            left = max(lower, center - self.region_radius)
            right = min(upper, center + self.region_radius)
            if index:
                prior = int(ordered[index - 1][1]["refined_value_start"])
                left = max(left, (prior + center) // 2 + 1)
            if index + 1 < len(ordered):
                following = int(
                    ordered[index + 1][1]["refined_value_start"]
                )
                right = min(right, (center + following) // 2)
            result[int(track_id)] = (left, right)
        return result

    def _score_region(
        self,
        *,
        track_id: int,
        snapshot: Dict[str, object],
        left: int,
        right: int,
        observation: Dict[str, object],
    ) -> Dict[str, object]:
        matrices = {
            "agent_only": (
                observation["scores"], observation["raw_sums"]
            ),
            "full_template": (
                observation["template_scores"],
                observation["template_raw_sums"],
            ),
        }
        candidates: Dict[str, Dict[str, Dict[str, object]]] = {
            name: {} for name in self._candidate_names
        }
        predictions: Dict[str, Dict[str, Dict[str, object]]] = {}
        for score_variant, (score_tensor, raw_tensor) in matrices.items():
            for candidate_index, name in enumerate(self._candidate_names):
                values = []
                for absolute in range(left, right + 1):
                    offset = absolute - self.prompt_length
                    if offset < 0 or offset >= score_tensor.shape[1]:
                        continue
                    score = score_tensor[candidate_index, offset]
                    if not bool(torch.isfinite(score)):
                        continue
                    raw = raw_tensor[candidate_index, offset]
                    values.append((
                        absolute,
                        float(score),
                        float(raw) if bool(torch.isfinite(raw)) else None,
                    ))
                candidates[name][score_variant] = self._aggregate_values(
                    values
                )

            predictions[score_variant] = {}
            for aggregation in AGGREGATIONS:
                key = f"{aggregation}_score"
                ranked = [
                    (name, candidates[name][score_variant][key])
                    for name in self._candidate_names
                    if candidates[name][score_variant][key] is not None
                ]
                ranked.sort(key=lambda item: item[1], reverse=True)
                top1 = ranked[0] if ranked else (None, None)
                top2 = ranked[1] if len(ranked) > 1 else (None, None)
                predictions[score_variant][aggregation] = {
                    "predicted_agent": top1[0],
                    "top1_region_score": top1[1],
                    "top2_region_score": top2[1],
                    "region_margin": (
                        None
                        if top1[1] is None or top2[1] is None
                        else float(top1[1] - top2[1])
                    ),
                }

        main = predictions[self.main_score_variant][self.main_aggregation]
        predicted = main["predicted_agent"]
        support = (
            candidates[predicted][self.main_score_variant]
            if predicted is not None else {}
        )
        return {
            "track_id": int(track_id),
            "track_center": int(snapshot["refined_value_start"]),
            "region_left": int(left),
            "region_right": int(right),
            "region_radius": self.region_radius,
            "track_stable": bool(snapshot.get("track_stable")),
            "track_provisional": bool(snapshot.get("track_provisional")),
            "track_miss_count": int(snapshot.get("track_miss_count") or 0),
            "candidate_agent_scores": candidates,
            "predictions": predictions,
            "predicted_agent_max": predictions["agent_only"]["max"][
                "predicted_agent"
            ],
            "predicted_agent_top2": predictions["agent_only"][
                "top2_mean"
            ]["predicted_agent"],
            "predicted_agent_soft": predictions["agent_only"]["soft"][
                "predicted_agent"
            ],
            "predicted_agent": predicted,
            "top1_region_score": main["top1_region_score"],
            "top2_region_score": main["top2_region_score"],
            "region_margin": main["region_margin"],
            "best_alignment_position": support.get(
                "best_alignment_position"
            ),
            "best_alignment_score": support.get("best_alignment_score"),
        }

    def observe(
        self,
        logits,
        x,
        logits_start,
        global_step,
        is_last_agent_step=False,
    ) -> None:
        before = len(self._score_observations)
        # Deliberately bypass RefinedOnlineLatentAnchorObserver.observe().
        # Region centers come directly from coarse anchors; the failed
        # anchor_score+agent_score exact-offset argmax is not used here.
        OracleLatentAgentObserver.observe(
            self,
            logits, x, logits_start, global_step, is_last_agent_step
        )
        if len(self._score_observations) == before:
            return
        started = time.perf_counter()
        observation = self._score_observations[-1]
        coarse = self._online_anchor_candidates(logits, x, logits_start)
        track_candidates = []
        for row in coarse:
            candidate = dict(row)
            candidate.update({
                "coarse_anchor_start": int(row["anchor_start"]),
                "coarse_anchor_end": int(row["anchor_end"]),
                "coarse_value_start": int(row["value_start"]),
                "coarse_anchor_score": float(row["anchor_score"]),
                "refined_value_start": int(row["value_start"]),
                "refinement_offset": 0,
                "best_joint_score": float(row["anchor_score"]),
                "second_best_joint_score": None,
                "joint_margin": None,
                "refined_predicted_agent": None,
                "refinement_candidates": [],
            })
            track_candidates.append(candidate)
        observation["online_anchors"] = track_candidates
        observation["track_snapshots"] = self._track_candidates(
            track_candidates, observation
        )
        observation["coarse_anchor_count"] = len(coarse)
        observation["refined_anchor_count"] = 0
        self._online_anchor_observation_counts.append(len(coarse))
        template_scores, template_raw = self._compress_template_scores(logits)
        observation["template_scores"] = template_scores
        observation["template_raw_sums"] = template_raw
        snapshots = observation.get("track_snapshots") or {}
        bounds = self._region_bounds(snapshots)
        observation["region_track_events"] = {
            int(track_id): self._score_region(
                track_id=int(track_id),
                snapshot=snapshot,
                left=bounds[int(track_id)][0],
                right=bounds[int(track_id)][1],
                observation=observation,
            )
            for track_id, snapshot in snapshots.items()
        }
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    @staticmethod
    def _stable_usable_event(
        events: Sequence[Dict[str, object]],
        *,
        score_variant: str,
        aggregation: str,
        strict: bool = False,
    ) -> Optional[Dict[str, object]]:
        suffix_usable = True
        result = None
        for event in reversed(events):
            state = event["variant_results"][score_variant][aggregation]
            usable = bool(
                event.get("track_stable")
                and event.get("region_contains_oracle")
                and state.get("prediction_correct")
            )
            suffix_usable = suffix_usable and usable
            if suffix_usable and usable and (
                not strict or event.get("strict_pre_materialization")
            ):
                result = event
        return result

    @staticmethod
    def _prediction_flips(
        events: Sequence[Dict[str, object]],
        score_variant: str,
        aggregation: str,
        *,
        stable_only: bool = False,
    ) -> int:
        values = [
            event["variant_results"][score_variant][aggregation][
                "predicted_agent"
            ]
            for event in events
            if (not stable_only or event.get("track_stable"))
            and event["variant_results"][score_variant][aggregation].get(
                "predicted_agent"
            ) is not None
        ]
        return sum(left != right for left, right in zip(values, values[1:]))

    def _region_event_for_slot(
        self,
        *,
        slot: Dict[str, object],
        observation: Dict[str, object],
        track_id: Optional[int],
    ) -> Dict[str, object]:
        wall_time = float(observation["wall_time"])
        final_agent = slot.get("final_agent")
        oracle = int(slot["value_start"])
        source = (observation.get("region_track_events") or {}).get(track_id)
        base = {
            "slot_id": int(slot["slot_id"]),
            "mapped_slot_id": int(slot["slot_id"]),
            "track_id": track_id,
            "iteration": observation["iteration"],
            "observation": observation["observation"],
            "wall_time": wall_time,
            "track_center": None,
            "center_position_error": None,
            "region_left": None,
            "region_right": None,
            "region_radius": self.region_radius,
            "track_stable": False,
            "track_provisional": None,
            "oracle_value_start": oracle,
            "region_contains_oracle": False,
            "candidate_agent_scores": {},
            "variant_results": {
                variant: {
                    aggregation: {
                        "predicted_agent": None,
                        "top1_region_score": None,
                        "top2_region_score": None,
                        "region_margin": None,
                        "prediction_correct": False,
                        "usable_prediction_correct": False,
                        "out_of_region_accidental_correct": False,
                        "best_alignment_position": None,
                        "best_alignment_error": None,
                    }
                    for aggregation in AGGREGATIONS
                }
                for variant in SCORE_VARIANTS
            },
            "agent_first_char_time": slot.get("agent_first_char_time"),
            "natural_agent_time": slot.get("natural_agent_time"),
            "strict_pre_materialization": bool(
                slot.get("agent_first_char_time") is not None
                and wall_time < float(slot["agent_first_char_time"])
            ),
        }
        if source is None:
            return base

        contains = bool(
            int(source["region_left"]) <= oracle <= int(source["region_right"])
        )
        base.update({
            "track_center": int(source["track_center"]),
            "center_position_error": int(source["track_center"]) - oracle,
            "region_left": int(source["region_left"]),
            "region_right": int(source["region_right"]),
            "track_stable": bool(source["track_stable"]),
            "track_provisional": bool(source["track_provisional"]),
            "region_contains_oracle": contains,
            "candidate_agent_scores": source["candidate_agent_scores"],
        })
        for variant in SCORE_VARIANTS:
            for aggregation in AGGREGATIONS:
                prediction = source["predictions"][variant][aggregation]
                predicted = prediction["predicted_agent"]
                support = (
                    source["candidate_agent_scores"][predicted][variant]
                    if predicted is not None else {}
                )
                correct = predicted == final_agent
                best = support.get("best_alignment_position")
                base["variant_results"][variant][aggregation] = {
                    **prediction,
                    "prediction_correct": correct,
                    "usable_prediction_correct": bool(
                        source["track_stable"] and contains and correct
                    ),
                    "out_of_region_accidental_correct": bool(
                        not contains and correct
                    ),
                    "best_alignment_position": best,
                    "best_alignment_error": (
                        None if best is None else int(best) - oracle
                    ),
                }
        main = base["variant_results"][self.main_score_variant][
            self.main_aggregation
        ]
        base.update({
            "predicted_agent_max": base["variant_results"]["agent_only"][
                "max"
            ]["predicted_agent"],
            "predicted_agent_top2": base["variant_results"]["agent_only"][
                "top2_mean"
            ]["predicted_agent"],
            "predicted_agent_soft": base["variant_results"]["agent_only"][
                "soft"
            ]["predicted_agent"],
            "predicted_agent": main["predicted_agent"],
            "top1_region_score": main["top1_region_score"],
            "top2_region_score": main["top2_region_score"],
            "region_margin": main["region_margin"],
            "prediction_correct": main["prediction_correct"],
            "usable_prediction_correct": main[
                "usable_prediction_correct"
            ],
            "best_alignment_position": main["best_alignment_position"],
            "best_alignment_error": main["best_alignment_error"],
        })
        return base

    def _variant_slot_metrics(
        self,
        events: Sequence[Dict[str, object]],
        *,
        natural_time: Optional[float],
        score_variant: str,
        aggregation: str,
    ) -> Dict[str, object]:
        predictions = [
            event for event in events
            if event["variant_results"][score_variant][aggregation].get(
                "predicted_agent"
            ) is not None
        ]
        stable_predictions = [
            event for event in predictions if event.get("track_stable")
        ]
        first = predictions[0] if predictions else None
        first_stable = stable_predictions[0] if stable_predictions else None
        stable_correct = self._stable_usable_event(
            events,
            score_variant=score_variant,
            aggregation=aggregation,
        )
        strict_stable = self._stable_usable_event(
            events,
            score_variant=score_variant,
            aggregation=aggregation,
            strict=True,
        )
        def state(event):
            return event["variant_results"][score_variant][aggregation]
        first_stays = bool(
            first is not None
            and state(first)["prediction_correct"]
            and all(state(event)["prediction_correct"] for event in predictions)
        )
        return {
            "first_region_prediction_time": (
                first["wall_time"] if first else None
            ),
            "first_region_prediction": (
                state(first)["predicted_agent"] if first else None
            ),
            "first_region_prediction_correct": (
                state(first)["prediction_correct"] if first else None
            ),
            "first_stable_track_prediction_time": (
                first_stable["wall_time"] if first_stable else None
            ),
            "first_stable_track_prediction_correct": (
                state(first_stable)["prediction_correct"]
                if first_stable else None
            ),
            "first_prediction_stays_correct": first_stays,
            "region_stable_correct_time": (
                stable_correct["wall_time"] if stable_correct else None
            ),
            "strict_region_stable_correct_time": (
                strict_stable["wall_time"] if strict_stable else None
            ),
            "region_stable_lead": self._lead(
                natural_time, stable_correct
            ),
            "strict_region_stable_lead": self._lead(
                natural_time, strict_stable
            ),
            "region_prediction_count": len(predictions),
            "stable_track_prediction_count": len(stable_predictions),
            "region_flip_count": self._prediction_flips(
                events, score_variant, aggregation
            ),
            "stable_track_region_flip_count": self._prediction_flips(
                events, score_variant, aggregation, stable_only=True
            ),
            "out_of_region_accidental_correct_count": sum(
                state(event)["out_of_region_accidental_correct"]
                for event in predictions
            ),
            "best_alignment_position_at_commit": (
                state(stable_correct)["best_alignment_position"]
                if stable_correct else None
            ),
            "best_alignment_error_at_commit": (
                state(stable_correct)["best_alignment_error"]
                if stable_correct else None
            ),
        }

    def _build_false_track_summary(self) -> None:
        mapped_ids = set(self._evaluation_track_by_slot.values())
        slots = list(self._slot_metrics)
        oracle_positions = [int(slot["value_start"]) for slot in slots]
        assigned: Dict[int, List[int]] = {
            int(slot["slot_id"]): [] for slot in slots
        }
        false_ids = []
        stable_false_ids = []
        for track in self._tracks:
            track_id = int(track["track_id"])
            if track_id not in mapped_ids:
                false_ids.append(track_id)
                if track.get("stable"):
                    stable_false_ids.append(track_id)
            history = track.get("refined_position_history") or []
            if not history or not oracle_positions:
                continue
            distances = [
                min(abs(int(value) - oracle) for value in history)
                for oracle in oracle_positions
            ]
            nearest = min(range(len(distances)), key=distances.__getitem__)
            if distances[nearest] <= self.max_track_match_distance:
                assigned[int(slots[nearest]["slot_id"])].append(track_id)

        false_prediction_stats = {
            variant: {
                aggregation: {
                    "prediction_count": 0,
                    "accidental_correct_count": 0,
                    "stable_prediction_count": 0,
                    "stable_accidental_correct_count": 0,
                    "region_margins": [],
                    "stable_region_margins": [],
                }
                for aggregation in AGGREGATIONS
            }
            for variant in SCORE_VARIANTS
        }
        false_set = set(false_ids)
        for observation in self._score_observations:
            events = observation.get("region_track_events") or {}
            for track_id, event in events.items():
                if int(track_id) not in false_set or not slots:
                    continue
                center = int(event["track_center"])
                nearest_slot = min(
                    slots,
                    key=lambda slot: abs(int(slot["value_start"]) - center),
                )
                oracle = int(nearest_slot["value_start"])
                natural_time = nearest_slot.get("natural_agent_time")
                if (
                    natural_time is not None
                    and float(observation["wall_time"])
                    >= float(natural_time)
                ):
                    continue
                contains = bool(
                    int(event["region_left"])
                    <= oracle
                    <= int(event["region_right"])
                )
                for variant in SCORE_VARIANTS:
                    for aggregation in AGGREGATIONS:
                        prediction = event["predictions"][variant][aggregation]
                        predicted = prediction["predicted_agent"]
                        if predicted is None:
                            continue
                        stats = false_prediction_stats[variant][aggregation]
                        stats["prediction_count"] += 1
                        accidental = bool(
                            not contains
                            and predicted == nearest_slot.get("final_agent")
                        )
                        stats["accidental_correct_count"] += int(accidental)
                        margin = prediction.get("region_margin")
                        if margin is not None:
                            stats["region_margins"].append(float(margin))
                        if event.get("track_stable"):
                            stats["stable_prediction_count"] += 1
                            stats["stable_accidental_correct_count"] += int(
                                accidental
                            )
                            if margin is not None:
                                stats["stable_region_margins"].append(
                                    float(margin)
                                )
        for variant in SCORE_VARIANTS:
            for aggregation in AGGREGATIONS:
                stats = false_prediction_stats[variant][aggregation]
                total = stats["prediction_count"]
                stable_total = stats["stable_prediction_count"]
                stats["false_track_correct_rate"] = (
                    stats["accidental_correct_count"] / total
                    if total else None
                )
                stats["stable_false_track_correct_rate"] = (
                    stats["stable_accidental_correct_count"] / stable_total
                    if stable_total else None
                )
                stats["region_margin_mean"] = (
                    statistics.mean(stats["region_margins"])
                    if stats["region_margins"] else None
                )
                stats["region_margin_p50"] = (
                    statistics.median(stats["region_margins"])
                    if stats["region_margins"] else None
                )
                stats["stable_region_margin_mean"] = (
                    statistics.mean(stats["stable_region_margins"])
                    if stats["stable_region_margins"] else None
                )
                stats["stable_region_margin_p50"] = (
                    statistics.median(stats["stable_region_margins"])
                    if stats["stable_region_margins"] else None
                )
        self._false_track_summary = {
            "tracks_created": len(self._tracks),
            "tracks_mapped_to_real_slot": len(mapped_ids),
            "unmapped_false_tracks": len(false_ids),
            "stable_false_tracks": len(stable_false_ids),
            "duplicate_tracks_per_slot": {
                str(slot_id): max(0, len(ids) - 1)
                for slot_id, ids in assigned.items()
            },
            "nearby_tracks_per_slot": {
                str(slot_id): ids for slot_id, ids in assigned.items()
            },
            "false_track_prediction_stats": false_prediction_stats,
            "false_track_correct_definition": (
                "unmapped track prediction equals the nearest oracle slot's "
                "final Agent while its region does not contain that oracle"
            ),
        }

    def _build_all_track_trajectory(self) -> None:
        slot_by_id = {
            int(slot["slot_id"]): slot for slot in self._slot_metrics
        }
        slot_by_track = {
            int(track_id): slot_by_id[int(slot_id)]
            for slot_id, track_id in self._evaluation_track_by_slot.items()
            if int(slot_id) in slot_by_id
        }
        rows = []
        for observation in self._score_observations:
            for track_id, event in (
                observation.get("region_track_events") or {}
            ).items():
                track_id = int(track_id)
                slot = slot_by_track.get(track_id)
                oracle = int(slot["value_start"]) if slot else None
                contains = bool(
                    oracle is not None
                    and int(event["region_left"])
                    <= oracle
                    <= int(event["region_right"])
                )
                main = event["predictions"][self.main_score_variant][
                    self.main_aggregation
                ]
                predicted = main["predicted_agent"]
                support = (
                    event["candidate_agent_scores"][predicted][
                        self.main_score_variant
                    ]
                    if predicted is not None else {}
                )
                best = support.get("best_alignment_position")
                rows.append({
                    "track_id": track_id,
                    "mapped_slot_id": (
                        int(slot["slot_id"]) if slot else None
                    ),
                    "final_agent": slot.get("final_agent") if slot else None,
                    "iteration": observation["iteration"],
                    "observation": observation["observation"],
                    "wall_time": float(observation["wall_time"]),
                    **event,
                    "oracle_value_start": oracle,
                    "region_contains_oracle": contains,
                    "prediction_correct": (
                        predicted == slot.get("final_agent") if slot else None
                    ),
                    "best_alignment_error": (
                        None
                        if best is None or oracle is None
                        else int(best) - oracle
                    ),
                    "agent_first_char_time": (
                        slot.get("agent_first_char_time") if slot else None
                    ),
                    "natural_agent_time": (
                        slot.get("natural_agent_time") if slot else None
                    ),
                })
        self._all_region_track_trajectory = rows

    def _before_clear_score_observations(self) -> None:
        # Build the existing exact-position and oracle metrics first.  This also
        # performs diagnostic-only final-plan mapping from tracks to true slots.
        super()._before_clear_score_observations()
        for slot in self._slot_metrics:
            slot_id = int(slot["slot_id"])
            track_id = self._evaluation_track_by_slot.get(slot_id)
            natural_time = slot.get("natural_agent_time")
            trajectory = [
                self._region_event_for_slot(
                    slot=slot,
                    observation=observation,
                    track_id=track_id,
                )
                for observation in self._score_observations
            ]
            pre_natural = [
                event for event in trajectory
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            exact_online_stable = None
            exact_suffix = True
            online_pre_natural = [
                event for event in slot.get("online_trajectory") or []
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            for event in reversed(online_pre_natural):
                exact_correct = bool(
                    event.get("refined_position_error") == 0
                    and event.get("correct")
                )
                exact_suffix = exact_suffix and exact_correct
                if exact_suffix and exact_correct:
                    exact_online_stable = event
            first_stable_track = next(
                (event for event in pre_natural if event["track_stable"]),
                None,
            )
            first_contains = next(
                (
                    event for event in pre_natural
                    if event["track_stable"]
                    and event["region_contains_oracle"]
                ),
                None,
            )
            stable_contains = None
            suffix_contains = True
            for event in reversed(pre_natural):
                contains = bool(
                    event["track_stable"]
                    and event["region_contains_oracle"]
                )
                suffix_contains = suffix_contains and contains
                if suffix_contains and contains:
                    stable_contains = event
            errors = [
                abs(int(event["center_position_error"]))
                for event in pre_natural
                if event.get("center_position_error") is not None
            ]
            variant_metrics = {
                variant: {
                    aggregation: self._variant_slot_metrics(
                        pre_natural,
                        natural_time=natural_time,
                        score_variant=variant,
                        aggregation=aggregation,
                    )
                    for aggregation in AGGREGATIONS
                }
                for variant in SCORE_VARIANTS
            }
            main = variant_metrics[self.main_score_variant][
                self.main_aggregation
            ]
            oracle_lead = slot.get("oracle_stable_lead")
            region_lead = main["region_stable_lead"]
            slot.update({
                "region_trajectory": trajectory,
                "exact_online_stable_correct_time": (
                    exact_online_stable["wall_time"]
                    if exact_online_stable else None
                ),
                "exact_online_stable_lead": self._lead(
                    natural_time, exact_online_stable
                ),
                "region_score_variant": self.main_score_variant,
                "region_main_aggregation": self.main_aggregation,
                "region_variant_metrics": variant_metrics,
                "first_stable_track_time": (
                    first_stable_track["wall_time"]
                    if first_stable_track else None
                ),
                "first_region_contains_oracle_time": (
                    first_contains["wall_time"] if first_contains else None
                ),
                "first_stable_region_contains_oracle_time": (
                    stable_contains["wall_time"]
                    if stable_contains else None
                ),
                "region_contains_oracle_lead": self._lead(
                    natural_time, stable_contains
                ),
                "center_abs_error_p50": (
                    statistics.median(errors) if errors else None
                ),
                **main,
                "region_lead_retention": (
                    None
                    if oracle_lead is None
                    or float(oracle_lead) <= 0
                    or region_lead is None
                    else float(region_lead) / float(oracle_lead)
                ),
            })
        self._build_false_track_summary()
        self._build_all_track_trajectory()

    def metrics(self) -> Dict[str, object]:
        result = super().metrics()
        result.update({
            "policy": "online_latent_region",
            "timing_source": (
                "persistent_track_region_and_oracle_full_sequence_logits"
            ),
            "region_radius": self.region_radius,
            "region_top_k": self.region_top_k,
            "region_temperature": self.region_temperature,
            "region_main_aggregation": self.main_aggregation,
            "region_main_score_variant": self.main_score_variant,
            "region_aggregations": list(AGGREGATIONS),
            "region_score_variants": list(SCORE_VARIANTS),
            "full_template_tokenizations": self._template_variants,
            "region_center_source": "persistent_coarse_anchor_track",
            "exact_refinement_used_for_region_center": False,
            "track_false_positive_summary": self._false_track_summary,
            "region_track_trajectory": self._all_region_track_trajectory,
        })
        return result

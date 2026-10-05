"""Diagnostic-only raw-track merging and region hypothesis validation."""

from __future__ import annotations

import math
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

from region_latent_agent_observer import RegionLatentAgentObserver


class HypothesisRegionObserver(RegionLatentAgentObserver):
    """Merge noisy persistent tracks into validated local slot hypotheses."""

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
        hypothesis_merge_distance: int = 6,
        hypothesis_merge_gap: int = 2,
        hypothesis_min_seen: int = 2,
        hypothesis_min_support: float = 0.5,
        hypothesis_max_center_jump: int = 12,
        hypothesis_duplicate_observations: int = 2,
    ) -> None:
        if hypothesis_merge_distance < 0:
            raise ValueError("hypothesis_merge_distance cannot be negative")
        if hypothesis_merge_gap < 0:
            raise ValueError("hypothesis_merge_gap cannot be negative")
        if hypothesis_min_seen < 1:
            raise ValueError("hypothesis_min_seen must be positive")
        if not 0.0 <= hypothesis_min_support <= 1.0:
            raise ValueError("hypothesis_min_support must be in [0, 1]")
        if hypothesis_max_center_jump < 1:
            raise ValueError("hypothesis_max_center_jump must be positive")
        if hypothesis_duplicate_observations < 1:
            raise ValueError(
                "hypothesis_duplicate_observations must be positive"
            )
        super().__init__(
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            score_chunk_size=score_chunk_size,
            anchor_min_logit_margin=anchor_min_logit_margin,
            region_radius=region_radius,
            region_top_k=region_top_k,
            region_temperature=region_temperature,
            main_aggregation=main_aggregation,
            anchor_position_tolerance=anchor_position_tolerance,
            anchor_stable_observations=anchor_stable_observations,
            max_track_misses=max_track_misses,
            max_track_match_distance=max_track_match_distance,
            track_stable_observations=track_stable_observations,
            track_score_weight=track_score_weight,
        )
        self.hypothesis_merge_distance = int(hypothesis_merge_distance)
        self.hypothesis_merge_gap = int(hypothesis_merge_gap)
        self.hypothesis_min_seen = int(hypothesis_min_seen)
        self.hypothesis_min_support = float(hypothesis_min_support)
        self.hypothesis_max_center_jump = int(
            hypothesis_max_center_jump
        )
        self.hypothesis_duplicate_observations = int(
            hypothesis_duplicate_observations
        )
        self._hypotheses: List[Dict[str, object]] = []
        self._hypothesis_by_track: Dict[int, int] = {}
        self._hypothesis_pair_support: Dict[Tuple[int, int], int] = {}
        self._hypothesis_trajectory: List[Dict[str, object]] = []
        self._hypothesis_slot_mapping: Dict[int, int] = {}
        self._hypothesis_evaluation: Dict[str, object] = {}

    def initialize(self, x) -> None:
        super().initialize(x)
        self._hypotheses.clear()
        self._hypothesis_by_track.clear()
        self._hypothesis_pair_support.clear()
        self._hypothesis_trajectory.clear()
        self._hypothesis_slot_mapping.clear()
        self._hypothesis_evaluation.clear()

    @staticmethod
    def _representative_center(hypothesis: Dict[str, object]) -> float:
        history = hypothesis.get("center_history") or []
        return float(statistics.median(history[-6:])) if history else math.inf

    def _new_hypothesis(
        self,
        *,
        track_id: int,
        position: int,
        observation: Dict[str, object],
    ) -> int:
        hypothesis_id = len(self._hypotheses)
        number = int(observation["observation"])
        wall_time = float(observation["wall_time"])
        self._hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "member_track_ids": [track_id],
            "seen_count": 0,
            "first_seen_observation": number,
            "last_seen_observation": None,
            "first_seen_time": wall_time,
            "last_seen_time": None,
            "missing_count": 0,
            "total_missing_count": 0,
            "support_ratio": 0.0,
            "center_history": [],
            "center_time_history": [],
            "center_by_observation": {},
            "time_by_observation": {},
            "last_center": position,
            "large_jump_count": 0,
            "lifecycle_reset_count": 0,
            "archived_lifecycles": [],
            "validated": False,
            "first_validated_time": None,
            "duplicate": False,
            "suppressed_by": None,
            "active_root": True,
            "merged_into": None,
        })
        self._hypothesis_by_track[track_id] = hypothesis_id
        return hypothesis_id

    def _reset_hypothesis_lifecycle(
        self,
        hypothesis: Dict[str, object],
        *,
        position: int,
        observation: Dict[str, object],
    ) -> None:
        """Start a fresh local lifecycle without manufacturing a new track.

        A raw track can be reassociated to a distant anchor by the upstream
        tracker.  Keeping both locations in one validation window makes the
        support and center statistics meaningless, while creating a new
        hypothesis inflates the hypothesis count beyond the raw-track count.
        Archive the old lifecycle and reuse the same hypothesis identity.
        """
        centers = hypothesis.get("center_history") or []
        if centers:
            hypothesis["archived_lifecycles"].append({
                "first_seen_observation": hypothesis[
                    "first_seen_observation"
                ],
                "last_seen_observation": hypothesis[
                    "last_seen_observation"
                ],
                "first_seen_time": hypothesis["first_seen_time"],
                "last_seen_time": hypothesis["last_seen_time"],
                "seen_count": hypothesis["seen_count"],
                "support_ratio": hypothesis["support_ratio"],
                "center_history": list(centers),
                "validated": hypothesis["validated"],
            })
        number = int(observation["observation"])
        wall_time = float(observation["wall_time"])
        hypothesis.update({
            "seen_count": 0,
            "first_seen_observation": number,
            "last_seen_observation": None,
            "first_seen_time": wall_time,
            "last_seen_time": None,
            "missing_count": 0,
            "total_missing_count": 0,
            "support_ratio": 0.0,
            "center_history": [],
            "center_time_history": [],
            "center_by_observation": {},
            "time_by_observation": {},
            "last_center": int(position),
            "large_jump_count": 0,
            "validated": False,
            "first_validated_time": None,
            "duplicate": False,
            "suppressed_by": None,
        })
        hypothesis["lifecycle_reset_count"] = (
            int(hypothesis["lifecycle_reset_count"]) + 1
        )

    def _merge_target(
        self, position: int, observation_number: int
    ) -> Optional[int]:
        candidates = []
        for hypothesis in self._hypotheses:
            if (
                hypothesis.get("duplicate")
                or not hypothesis.get("active_root", True)
            ):
                continue
            last_seen = hypothesis.get("last_seen_observation")
            if last_seen is None:
                last_seen = hypothesis["first_seen_observation"]
            missing_between = observation_number - int(last_seen) - 1
            if missing_between > self.hypothesis_merge_gap:
                continue
            representative = self._representative_center(hypothesis)
            if not math.isfinite(representative):
                representative = float(hypothesis["last_center"])
            distance = abs(float(position) - representative)
            if distance <= self.hypothesis_merge_distance:
                candidates.append((
                    distance,
                    -float(hypothesis["support_ratio"]),
                    -int(hypothesis["seen_count"]),
                    int(hypothesis["hypothesis_id"]),
                ))
        return min(candidates)[-1] if candidates else None

    def _assign_active_tracks(
        self,
        snapshots: Dict[int, Dict[str, object]],
        observation: Dict[str, object],
    ) -> Dict[int, List[Tuple[int, int]]]:
        number = int(observation["observation"])
        active: Dict[int, List[Tuple[int, int]]] = {}
        for track_id, snapshot in sorted(
            snapshots.items(),
            key=lambda item: int(item[1]["refined_value_start"]),
        ):
            track_id = int(track_id)
            position = int(snapshot["refined_value_start"])
            hypothesis_id = self._hypothesis_by_track.get(track_id)
            if hypothesis_id is not None:
                hypothesis = self._hypotheses[hypothesis_id]
                if (
                    abs(position - int(hypothesis["last_center"]))
                    > self.hypothesis_max_center_jump
                ):
                    self._reset_hypothesis_lifecycle(
                        hypothesis,
                        position=position,
                        observation=observation,
                    )
            if hypothesis_id is None:
                hypothesis_id = self._merge_target(position, number)
                if hypothesis_id is None:
                    hypothesis_id = self._new_hypothesis(
                        track_id=track_id,
                        position=position,
                        observation=observation,
                    )
                else:
                    hypothesis = self._hypotheses[hypothesis_id]
                    hypothesis["member_track_ids"].append(track_id)
                    self._hypothesis_by_track[track_id] = hypothesis_id
            active.setdefault(hypothesis_id, []).append((track_id, position))
        return active

    def _recompute_hypothesis(
        self, hypothesis: Dict[str, object], current_observation: int
    ) -> None:
        centers = hypothesis["center_by_observation"]
        times = hypothesis["time_by_observation"]
        ordered = sorted(int(number) for number in centers)
        hypothesis["seen_count"] = len(ordered)
        hypothesis["first_seen_observation"] = ordered[0]
        hypothesis["last_seen_observation"] = ordered[-1]
        hypothesis["first_seen_time"] = float(times[ordered[0]])
        hypothesis["last_seen_time"] = float(times[ordered[-1]])
        hypothesis["center_history"] = [
            int(centers[number]) for number in ordered
        ]
        hypothesis["center_time_history"] = [
            float(times[number]) for number in ordered
        ]
        hypothesis["last_center"] = int(centers[ordered[-1]])
        span = current_observation - ordered[0] + 1
        missing = max(0, span - len(ordered))
        hypothesis["missing_count"] = missing
        hypothesis["total_missing_count"] = missing
        hypothesis["support_ratio"] = len(ordered) / float(span)
        history = hypothesis["center_history"]
        hypothesis["large_jump_count"] = sum(
            abs(right - left) > self.hypothesis_max_center_jump
            for left, right in zip(history, history[1:])
        )
        validated = bool(
            len(ordered) >= self.hypothesis_min_seen
            and float(hypothesis["support_ratio"])
            >= self.hypothesis_min_support
            and int(hypothesis["large_jump_count"]) == 0
        )
        hypothesis["validated"] = validated
        if validated and hypothesis["first_validated_time"] is None:
            hypothesis["first_validated_time"] = float(times[ordered[-1]])

    def _update_hypotheses(
        self,
        active: Dict[int, List[Tuple[int, int]]],
        observation: Dict[str, object],
    ) -> Dict[int, int]:
        number = int(observation["observation"])
        wall_time = float(observation["wall_time"])
        centers = {}
        for hypothesis in self._hypotheses:
            hypothesis_id = int(hypothesis["hypothesis_id"])
            if not hypothesis.get("active_root", True):
                continue
            positions = [value for _, value in active.get(hypothesis_id, [])]
            if positions:
                center = int(statistics.median(positions))
                hypothesis["center_by_observation"][number] = center
                hypothesis["time_by_observation"][number] = wall_time
                centers[hypothesis_id] = center
            self._recompute_hypothesis(hypothesis, number)
        return centers

    def _merge_close_hypotheses(
        self,
        centers: Dict[int, int],
        active: Dict[int, List[Tuple[int, int]]],
        observation: Dict[str, object],
    ) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, int]]]]:
        number = int(observation["observation"])
        changed = True
        while changed:
            changed = False
            roots = [
                hypothesis for hypothesis in self._hypotheses
                if hypothesis.get("active_root", True)
                and not hypothesis.get("duplicate")
            ]
            roots.sort(key=self._representative_center)
            for left_index, left_h in enumerate(roots):
                left = int(left_h["hypothesis_id"])
                left_last = int(left_h["last_seen_observation"])
                for right_h in roots[left_index + 1:]:
                    right = int(right_h["hypothesis_id"])
                    right_last = int(right_h["last_seen_observation"])
                    if left not in active and right not in active:
                        continue
                    if number - max(left_last, right_last) - 1 > (
                        self.hypothesis_merge_gap
                    ):
                        continue
                    current_left = centers.get(
                        left, int(left_h["last_center"])
                    )
                    current_right = centers.get(
                        right, int(right_h["last_center"])
                    )
                    current_distance = abs(current_left - current_right)
                    median_distance = abs(
                        self._representative_center(left_h)
                        - self._representative_center(right_h)
                    )
                    if (
                        current_distance > self.hypothesis_merge_distance
                        or median_distance > self.hypothesis_merge_distance
                    ):
                        continue
                    keep, drop = (left, right) if left < right else (right, left)
                    keep_h = self._hypotheses[keep]
                    drop_h = self._hypotheses[drop]
                    for observation_id, value in drop_h[
                        "center_by_observation"
                    ].items():
                        if observation_id in keep_h["center_by_observation"]:
                            value = int(statistics.median((
                                keep_h["center_by_observation"][observation_id],
                                value,
                            )))
                        keep_h["center_by_observation"][observation_id] = value
                        keep_h["time_by_observation"][observation_id] = max(
                            keep_h["time_by_observation"].get(
                                observation_id, 0.0
                            ),
                            drop_h["time_by_observation"][observation_id],
                        )
                    for track_id in drop_h["member_track_ids"]:
                        if track_id not in keep_h["member_track_ids"]:
                            keep_h["member_track_ids"].append(track_id)
                        self._hypothesis_by_track[int(track_id)] = keep
                    drop_h["active_root"] = False
                    drop_h["merged_into"] = keep
                    if drop in active:
                        active.setdefault(keep, []).extend(active.pop(drop))
                    if keep in active:
                        center = int(statistics.median(
                            value for _, value in active[keep]
                        ))
                        centers[keep] = center
                        keep_h["center_by_observation"][number] = center
                        keep_h["time_by_observation"][number] = float(
                            observation["wall_time"]
                        )
                    centers.pop(drop, None)
                    self._recompute_hypothesis(keep_h, number)
                    changed = True
                    break
                if changed:
                    break
        return centers, active

    def _suppress_duplicates(
        self,
        centers: Dict[int, int],
    ) -> None:
        close_pairs = set()
        candidates = [
            hypothesis_id for hypothesis_id in centers
            if self._hypotheses[hypothesis_id]["validated"]
            and not self._hypotheses[hypothesis_id]["duplicate"]
        ]
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1:]:
                pair = tuple(sorted((left, right)))
                distance = abs(centers[left] - centers[right])
                regions_overlap = distance <= 2 * self.region_radius
                if (
                    distance <= self.hypothesis_merge_distance
                    and regions_overlap
                ):
                    close_pairs.add(pair)
                    self._hypothesis_pair_support[pair] = (
                        self._hypothesis_pair_support.get(pair, 0) + 1
                    )
                    if (
                        self._hypothesis_pair_support[pair]
                        >= self.hypothesis_duplicate_observations
                    ):
                        left_h = self._hypotheses[left]
                        right_h = self._hypotheses[right]
                        strength_left = (
                            float(left_h["support_ratio"]),
                            int(left_h["seen_count"]),
                            -int(left_h["total_missing_count"]),
                            -left,
                        )
                        strength_right = (
                            float(right_h["support_ratio"]),
                            int(right_h["seen_count"]),
                            -int(right_h["total_missing_count"]),
                            -right,
                        )
                        keep, suppress = (
                            (left, right)
                            if strength_left >= strength_right
                            else (right, left)
                        )
                        self._hypotheses[suppress]["duplicate"] = True
                        self._hypotheses[suppress]["suppressed_by"] = keep
        for pair in list(self._hypothesis_pair_support):
            if pair not in close_pairs:
                self._hypothesis_pair_support[pair] = 0

    def _hypothesis_region_events(
        self,
        *,
        centers: Dict[int, int],
        active: Dict[int, List[Tuple[int, int]]],
        observation: Dict[str, object],
    ) -> Dict[int, Dict[str, object]]:
        lower = self.prompt_length
        upper = self.prompt_length + len(self._first_visible_seconds) - 1
        events = {}
        for hypothesis_id, center in centers.items():
            hypothesis = self._hypotheses[hypothesis_id]
            snapshot = {
                "refined_value_start": center,
                "track_stable": hypothesis["validated"],
                "track_provisional": not hypothesis["validated"],
                "track_miss_count": hypothesis["missing_count"],
            }
            event = self._score_region(
                track_id=hypothesis_id,
                snapshot=snapshot,
                left=max(lower, center - self.region_radius),
                right=min(upper, center + self.region_radius),
                observation=observation,
            )
            event.update({
                "hypothesis_id": hypothesis_id,
                "member_track_ids": list(hypothesis["member_track_ids"]),
                "active_member_track_ids": [
                    track_id for track_id, _ in active[hypothesis_id]
                ],
                "hypothesis_center": center,
                "seen_count": int(hypothesis["seen_count"]),
                "first_seen_time": hypothesis["first_seen_time"],
                "last_seen_time": hypothesis["last_seen_time"],
                "missing_count": int(hypothesis["missing_count"]),
                "support_ratio": float(hypothesis["support_ratio"]),
                "large_jump_count": int(hypothesis["large_jump_count"]),
                "lifecycle_reset_count": int(
                    hypothesis["lifecycle_reset_count"]
                ),
                "validated": bool(hypothesis["validated"]),
                "duplicate": bool(hypothesis["duplicate"]),
                "suppressed_by": hypothesis["suppressed_by"],
            })
            events[hypothesis_id] = event
        return events

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
            logits, x, logits_start, global_step, is_last_agent_step
        )
        if len(self._score_observations) == before:
            return
        started = time.perf_counter()
        observation = self._score_observations[-1]
        snapshots = observation.get("track_snapshots") or {}
        active = self._assign_active_tracks(snapshots, observation)
        centers = self._update_hypotheses(active, observation)
        centers, active = self._merge_close_hypotheses(
            centers, active, observation
        )
        self._suppress_duplicates(centers)
        observation["hypothesis_events"] = self._hypothesis_region_events(
            centers=centers,
            active=active,
            observation=observation,
        )
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    def _final_hypotheses(self) -> List[Dict[str, object]]:
        return [
            hypothesis for hypothesis in self._hypotheses
            if hypothesis.get("active_root", True)
            and hypothesis["validated"]
            and not hypothesis["duplicate"]
        ]

    def _map_hypotheses_to_slots(self) -> Dict[int, int]:
        hypotheses = sorted(
            self._final_hypotheses(), key=self._representative_center
        )
        slots = sorted(self._slot_metrics, key=lambda row: int(row["slot_id"]))
        m, n = len(slots), len(hypotheses)
        dp = [[(0, 0.0)] * (n + 1) for _ in range(m + 1)]
        action: List[List[Optional[str]]] = [
            [None] * (n + 1) for _ in range(m + 1)
        ]
        for i in range(1, m + 1):
            action[i][0] = "skip_slot"
        for j in range(1, n + 1):
            action[0][j] = "skip_hypothesis"
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                options = [
                    (dp[i - 1][j], "skip_slot"),
                    (dp[i][j - 1], "skip_hypothesis"),
                ]
                center = self._representative_center(hypotheses[j - 1])
                oracle = int(slots[i - 1]["value_start"])
                distance = abs(center - oracle)
                if distance <= self.region_radius:
                    prior = dp[i - 1][j - 1]
                    options.append((
                        (prior[0] + 1, prior[1] - float(distance)),
                        "match",
                    ))
                dp[i][j], action[i][j] = max(
                    options, key=lambda item: item[0]
                )
        mapping = {}
        i, j = m, n
        while i or j:
            if action[i][j] == "match":
                mapping[int(slots[i - 1]["slot_id"])] = int(
                    hypotheses[j - 1]["hypothesis_id"]
                )
                i -= 1
                j -= 1
            elif action[i][j] == "skip_slot":
                i -= 1
            else:
                j -= 1
        return mapping

    def _hypothesis_event_for_slot(
        self,
        *,
        slot: Dict[str, object],
        observation: Dict[str, object],
        hypothesis_id: Optional[int],
    ) -> Dict[str, object]:
        wall_time = float(observation["wall_time"])
        oracle = int(slot["value_start"])
        source = (observation.get("hypothesis_events") or {}).get(
            hypothesis_id
        )
        result = {
            "slot_id": int(slot["slot_id"]),
            "hypothesis_id": hypothesis_id,
            "iteration": observation["iteration"],
            "observation": observation["observation"],
            "wall_time": wall_time,
            "hypothesis_center": None,
            "region_left": None,
            "region_right": None,
            "oracle_value_start": oracle,
            "region_contains_oracle": False,
            "validated": False,
            "duplicate": False,
            "support_ratio": None,
            "predicted_agent": None,
            "prediction_correct": False,
            "usable_prediction_correct": False,
            "top1_region_score": None,
            "top2_region_score": None,
            "region_margin": None,
            "best_alignment_position": None,
            "best_alignment_error": None,
            "strict_pre_materialization": bool(
                slot.get("agent_first_char_time") is not None
                and wall_time < float(slot["agent_first_char_time"])
            ),
        }
        if source is None:
            return result
        predicted = source["predicted_agent"]
        best = source.get("best_alignment_position")
        contains = bool(
            int(source["region_left"]) <= oracle <= int(source["region_right"])
        )
        correct = predicted == slot.get("final_agent")
        result.update({
            "hypothesis_center": int(source["hypothesis_center"]),
            "region_left": int(source["region_left"]),
            "region_right": int(source["region_right"]),
            "region_contains_oracle": contains,
            "validated": bool(source["validated"]),
            "duplicate": bool(source["duplicate"]),
            "support_ratio": float(source["support_ratio"]),
            "seen_count": int(source["seen_count"]),
            "member_track_ids": list(source["member_track_ids"]),
            "predicted_agent": predicted,
            "prediction_correct": correct,
            "usable_prediction_correct": bool(
                source["validated"]
                and not source["duplicate"]
                and contains
                and correct
            ),
            "top1_region_score": source["top1_region_score"],
            "top2_region_score": source["top2_region_score"],
            "region_margin": source["region_margin"],
            "best_alignment_position": best,
            "best_alignment_error": (
                None if best is None else int(best) - oracle
            ),
        })
        return result

    @staticmethod
    def _first_stable_validated(
        events: Sequence[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        suffix = True
        result = None
        for event in reversed(events):
            usable = bool(event["usable_prediction_correct"])
            suffix = suffix and usable
            if suffix and usable:
                result = event
        return result

    def _build_hypothesis_metrics(self) -> None:
        self._hypothesis_slot_mapping = self._map_hypotheses_to_slots()
        mapped_ids = set(self._hypothesis_slot_mapping.values())
        for slot in self._slot_metrics:
            slot_id = int(slot["slot_id"])
            hypothesis_id = self._hypothesis_slot_mapping.get(slot_id)
            natural_time = slot.get("natural_agent_time")
            trajectory = [
                self._hypothesis_event_for_slot(
                    slot=slot,
                    observation=observation,
                    hypothesis_id=hypothesis_id,
                )
                for observation in self._score_observations
            ]
            pre_natural = [
                event for event in trajectory
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            stable = self._first_stable_validated(pre_natural)
            predictions = [
                event["predicted_agent"] for event in pre_natural
                if event.get("predicted_agent") is not None
            ]
            slot.update({
                "hypothesis_id": hypothesis_id,
                "hypothesis_trajectory": trajectory,
                "validated_region_stable_time": (
                    stable["wall_time"] if stable else None
                ),
                "validated_region_stable_lead": self._lead(
                    natural_time, stable
                ),
                "validated_region_flip_count": sum(
                    left != right
                    for left, right in zip(predictions, predictions[1:])
                ),
                "validated_region_center_at_stable": (
                    stable["hypothesis_center"] if stable else None
                ),
                "validated_region_margin_at_stable": (
                    stable["region_margin"] if stable else None
                ),
            })

        slot_by_hypothesis = {
            hypothesis_id: slot_id
            for slot_id, hypothesis_id in self._hypothesis_slot_mapping.items()
        }
        rows = []
        for observation in self._score_observations:
            for hypothesis_id, event in (
                observation.get("hypothesis_events") or {}
            ).items():
                rows.append({
                    "hypothesis_id": int(hypothesis_id),
                    "mapped_slot_id": slot_by_hypothesis.get(
                        int(hypothesis_id)
                    ),
                    "iteration": observation["iteration"],
                    "observation": observation["observation"],
                    "wall_time": float(observation["wall_time"]),
                    **event,
                })
        self._hypothesis_trajectory = rows
        final = self._final_hypotheses()
        duplicates = [h for h in self._hypotheses if h["duplicate"]]
        merged_roots = [
            h for h in self._hypotheses if h.get("active_root", True)
        ]
        self._hypothesis_evaluation = {
            "raw_tracks": len(self._tracks),
            "merged_hypotheses": len(merged_roots),
            "validated_hypotheses": len(final),
            "false_validated_hypotheses": len(final) - len(mapped_ids),
            "duplicate_hypotheses": len(duplicates),
            "true_slots": len(self._slot_metrics),
            "covered_slots": len(mapped_ids),
            "slot_precision": (
                len(mapped_ids) / len(final) if final else None
            ),
            "slot_recall": (
                len(mapped_ids) / len(self._slot_metrics)
                if self._slot_metrics else None
            ),
        }

    def _before_clear_score_observations(self) -> None:
        super()._before_clear_score_observations()
        self._build_hypothesis_metrics()

    def metrics(self) -> Dict[str, object]:
        result = super().metrics()
        result.update({
            "policy": "online_latent_hypothesis",
            "timing_source": (
                "merged_validated_hypothesis_region_and_oracle_logits"
            ),
            "hypothesis_merge_distance": self.hypothesis_merge_distance,
            "hypothesis_merge_gap": self.hypothesis_merge_gap,
            "hypothesis_min_seen": self.hypothesis_min_seen,
            "hypothesis_min_support": self.hypothesis_min_support,
            "hypothesis_max_center_jump": (
                self.hypothesis_max_center_jump
            ),
            "hypothesis_duplicate_observations": (
                self.hypothesis_duplicate_observations
            ),
            "hypothesis_evaluation": self._hypothesis_evaluation,
            "hypotheses": [dict(row) for row in self._hypotheses],
            "hypothesis_trajectory": self._hypothesis_trajectory,
        })
        return result

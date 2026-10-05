"""Local-refinement and persistent-tracking latent anchor diagnostics."""

from __future__ import annotations

import math
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from online_latent_anchor_observer import OnlineLatentAnchorObserver
from oracle_latent_observer import OracleLatentAgentObserver


class RefinedOnlineLatentAnchorObserver(OnlineLatentAnchorObserver):
    """Refine coarse anchors locally, optionally associating persistent tracks."""

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        persistent_tracking: bool,
        score_chunk_size: int = 32,
        anchor_min_logit_margin: float = -6.0,
        refinement_radius: int = 4,
        refinement_anchor_weight: float = 1.0,
        refinement_agent_weight: float = 1.0,
        anchor_position_tolerance: int = 4,
        anchor_stable_observations: int = 2,
        max_track_misses: int = 3,
        max_track_match_distance: int = 64,
        track_stable_observations: int = 2,
        track_score_weight: float = 0.05,
    ) -> None:
        if refinement_radius < 0:
            raise ValueError("refinement_radius cannot be negative")
        if refinement_anchor_weight < 0 or refinement_agent_weight < 0:
            raise ValueError("refinement weights cannot be negative")
        if refinement_anchor_weight + refinement_agent_weight == 0:
            raise ValueError("at least one refinement weight must be positive")
        if max_track_misses < 0:
            raise ValueError("max_track_misses cannot be negative")
        if max_track_match_distance < 1:
            raise ValueError("max_track_match_distance must be positive")
        if track_stable_observations < 1:
            raise ValueError("track_stable_observations must be positive")
        super().__init__(
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            score_chunk_size=score_chunk_size,
            anchor_min_logit_margin=anchor_min_logit_margin,
            anchor_position_tolerance=anchor_position_tolerance,
            anchor_stable_observations=anchor_stable_observations,
        )
        self.persistent_tracking = bool(persistent_tracking)
        self.refinement_radius = int(refinement_radius)
        self.refinement_anchor_weight = float(refinement_anchor_weight)
        self.refinement_agent_weight = float(refinement_agent_weight)
        self.max_track_misses = int(max_track_misses)
        self.max_track_match_distance = int(max_track_match_distance)
        self.track_stable_observations = int(track_stable_observations)
        self.track_score_weight = float(track_score_weight)
        self._tracks: List[Dict[str, object]] = []
        self._next_track_id = 0
        self._unmatched_candidate_count = 0
        self._track_missing_count = 0
        self._track_reassociation_count = 0
        self._slot_identity_shift_count = 0
        self._evaluation_track_by_slot: Dict[int, int] = {}

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self._tracks.clear()
        self._next_track_id = 0
        self._unmatched_candidate_count = 0
        self._track_missing_count = 0
        self._track_reassociation_count = 0
        self._slot_identity_shift_count = 0
        self._evaluation_track_by_slot.clear()

    def _structural_score_at(
        self,
        *,
        value_start: int,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        sequence_max: torch.Tensor,
    ) -> Optional[Dict[str, object]]:
        best = None
        generation_start = self.prompt_length
        generation_end = self.prompt_length + self.gen_length
        for variant in self._online_anchor_variants:
            pattern = tuple(variant["anchor_token_ids"])
            anchor_start = value_start - len(pattern)
            if anchor_start < generation_start or value_start > generation_end:
                continue
            relative = anchor_start - logits_start
            if relative < 0 or relative + len(pattern) > logits.shape[1]:
                continue
            score = torch.zeros((), device=logits.device, dtype=torch.float32)
            observed = 0
            compatible = True
            for offset, token_id in enumerate(pattern):
                absolute = anchor_start + offset
                current = int(x[0, absolute])
                if current not in (self.mask_id, token_id):
                    compatible = False
                    break
                row = relative + offset
                score += (
                    logits[0, row, token_id].float()
                    - sequence_max[row].float()
                )
                observed += int(current == token_id)
            if not compatible:
                continue
            score_value = float((score / float(len(pattern))).detach().cpu())
            row = {
                "anchor_start": anchor_start,
                "anchor_end": value_start,
                "anchor_variant": variant["anchor_variant"],
                "anchor_token_ids": list(pattern),
                "anchor_score": score_value,
                "anchor_observed_ratio": observed / float(len(pattern)),
            }
            if best is None or (
                row["anchor_observed_ratio"], row["anchor_score"]
            ) > (best["anchor_observed_ratio"], best["anchor_score"]):
                best = row
        return best

    def _refine_anchor(
        self,
        *,
        coarse: Dict[str, object],
        observation: Dict[str, object],
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        sequence_max: torch.Tensor,
    ) -> Dict[str, object]:
        coarse_value = int(coarse["value_start"])
        score_tensor = observation["scores"]
        raw_tensor = observation["raw_sums"]
        candidates = []
        for value_start in range(
            coarse_value - self.refinement_radius,
            coarse_value + self.refinement_radius + 1,
        ):
            offset = value_start - self.prompt_length
            if offset < 0 or offset >= score_tensor.shape[1]:
                continue
            structural = self._structural_score_at(
                value_start=value_start,
                logits=logits,
                x=x,
                logits_start=logits_start,
                sequence_max=sequence_max,
            )
            if structural is None:
                continue
            for candidate_index, name in enumerate(self._candidate_names):
                score = score_tensor[candidate_index, offset]
                if not bool(torch.isfinite(score)):
                    continue
                agent_score = float(score)
                raw_sum = float(raw_tensor[candidate_index, offset])
                joint = (
                    self.refinement_anchor_weight
                    * float(structural["anchor_score"])
                    + self.refinement_agent_weight * agent_score
                )
                candidates.append({
                    **structural,
                    "value_start": value_start,
                    "position_offset": value_start - coarse_value,
                    "predicted_agent": name,
                    "agent_sequence_score": agent_score,
                    "agent_raw_logit_sum": raw_sum,
                    "joint_score": joint,
                })
        candidates.sort(key=lambda row: row["joint_score"], reverse=True)
        if not candidates:
            result = dict(coarse)
            result.update({
                "coarse_anchor_start": int(coarse["anchor_start"]),
                "coarse_anchor_end": int(coarse["anchor_end"]),
                "coarse_value_start": coarse_value,
                "coarse_anchor_score": float(coarse["anchor_score"]),
                "refined_value_start": coarse_value,
                "refinement_offset": 0,
                "best_joint_score": None,
                "second_best_joint_score": None,
                "joint_margin": None,
                "refinement_candidates": [],
            })
            return result

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        result = dict(coarse)
        result.update({
            "coarse_anchor_start": int(coarse["anchor_start"]),
            "coarse_anchor_end": int(coarse["anchor_end"]),
            "coarse_value_start": coarse_value,
            "coarse_anchor_score": float(coarse["anchor_score"]),
            "anchor_start": int(best["anchor_start"]),
            "anchor_end": int(best["anchor_end"]),
            "value_start": int(best["value_start"]),
            "refined_value_start": int(best["value_start"]),
            "anchor_variant": best["anchor_variant"],
            "anchor_token_ids": best["anchor_token_ids"],
            "anchor_score": float(best["anchor_score"]),
            "anchor_observed_ratio": float(best["anchor_observed_ratio"]),
            "refinement_offset": int(best["position_offset"]),
            "refined_predicted_agent": best["predicted_agent"],
            "refined_agent_sequence_score": float(
                best["agent_sequence_score"]
            ),
            "best_joint_score": float(best["joint_score"]),
            "second_best_joint_score": (
                None if second is None else float(second["joint_score"])
            ),
            "joint_margin": (
                None if second is None
                else float(best["joint_score"] - second["joint_score"])
            ),
            "refinement_candidates": [
                {
                    "value_start": int(row["value_start"]),
                    "position_offset": int(row["position_offset"]),
                    "predicted_agent": row["predicted_agent"],
                    "anchor_variant": row["anchor_variant"],
                    "anchor_score": float(row["anchor_score"]),
                    "agent_sequence_score": float(row["agent_sequence_score"]),
                    "joint_score": float(row["joint_score"]),
                }
                for row in candidates
            ],
        })
        return result

    def _match_tracks(
        self, candidates: List[Dict[str, object]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        active = [
            index for index, track in enumerate(self._tracks)
            if int(track["miss_count"]) <= self.max_track_misses
        ]
        active.sort(key=lambda index: int(
            self._tracks[index]["last_refined_position"]
        ))
        candidate_order = sorted(
            range(len(candidates)),
            key=lambda index: int(candidates[index]["refined_value_start"]),
        )
        m, n = len(active), len(candidate_order)
        inf = float("inf")
        miss_penalty = self.max_track_match_distance / 2.0
        new_penalty = self.max_track_match_distance / 2.0
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        action: List[List[Optional[str]]] = [
            [None] * (n + 1) for _ in range(m + 1)
        ]
        dp[0][0] = 0.0
        for i in range(m + 1):
            for j in range(n + 1):
                current = dp[i][j]
                if not math.isfinite(current):
                    continue
                if i < m and current + miss_penalty < dp[i + 1][j]:
                    dp[i + 1][j] = current + miss_penalty
                    action[i + 1][j] = "miss"
                if j < n and current + new_penalty < dp[i][j + 1]:
                    dp[i][j + 1] = current + new_penalty
                    action[i][j + 1] = "new"
                if i < m and j < n:
                    track = self._tracks[active[i]]
                    candidate = candidates[candidate_order[j]]
                    delta = abs(
                        int(candidate["refined_value_start"])
                        - int(track["last_refined_position"])
                    )
                    if delta <= self.max_track_match_distance:
                        joint = candidate.get("best_joint_score")
                        score_cost = (
                            0.0 if joint is None
                            else -self.track_score_weight * float(joint)
                        )
                        match_cost = current + float(delta) + score_cost
                        if match_cost < dp[i + 1][j + 1]:
                            dp[i + 1][j + 1] = match_cost
                            action[i + 1][j + 1] = "match"

        matches = []
        missed = []
        new = []
        i, j = m, n
        while i or j:
            decision = action[i][j]
            if decision == "match":
                matches.append((active[i - 1], candidate_order[j - 1]))
                i -= 1
                j -= 1
            elif decision == "miss":
                missed.append(active[i - 1])
                i -= 1
            elif decision == "new":
                new.append(candidate_order[j - 1])
                j -= 1
            else:
                # Only reachable for an empty edge initialized at zero.
                if i:
                    missed.append(active[i - 1])
                    i -= 1
                elif j:
                    new.append(candidate_order[j - 1])
                    j -= 1
        matches.reverse()
        missed.reverse()
        new.reverse()
        return matches, missed, new

    def _new_track(
        self,
        candidate: Dict[str, object],
        *,
        observation: Dict[str, object],
        candidate_rank: int,
    ) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        wall_time = float(observation["wall_time"])
        track = {
            "track_id": track_id,
            "last_coarse_position": int(candidate["coarse_value_start"]),
            "last_refined_position": int(candidate["refined_value_start"]),
            "last_seen_iteration": observation["iteration"],
            "last_seen_observation": observation["observation"],
            "miss_count": 0,
            "total_missing_count": 0,
            "matched_observation_count": 1,
            "consecutive_match_count": 1,
            "stable": self.track_stable_observations <= 1,
            "provisional": self.track_stable_observations > 1,
            "first_seen_time": wall_time,
            "first_stable_time": (
                wall_time if self.track_stable_observations <= 1 else None
            ),
            "position_history": [int(candidate["coarse_value_start"])],
            "refined_position_history": [
                int(candidate["refined_value_start"])
            ],
            "time_history": [wall_time],
            "latest_anchor_score": candidate.get("anchor_score"),
            "latest_joint_score": candidate.get("best_joint_score"),
            "latest_predicted_agent": candidate.get(
                "refined_predicted_agent"
            ),
            "last_candidate_rank": candidate_rank,
            "reassociation_count": 0,
            "rank_shift_count": 0,
        }
        self._tracks.append(track)
        return len(self._tracks) - 1

    def _update_track(
        self,
        track_index: int,
        candidate: Dict[str, object],
        *,
        observation: Dict[str, object],
        candidate_rank: int,
    ) -> None:
        track = self._tracks[track_index]
        prior_refined = int(track["last_refined_position"])
        prior_coarse = int(track["last_coarse_position"])
        prior_misses = int(track["miss_count"])
        if prior_misses:
            track["reassociation_count"] = int(
                track["reassociation_count"]
            ) + 1
            self._track_reassociation_count += 1
        if candidate_rank != int(track["last_candidate_rank"]):
            track["rank_shift_count"] = int(track["rank_shift_count"]) + 1
            self._slot_identity_shift_count += 1
        consecutive = int(track["consecutive_match_count"]) + 1
        wall_time = float(observation["wall_time"])
        stable = bool(track["stable"]) or (
            consecutive >= self.track_stable_observations
        )
        if stable and track["first_stable_time"] is None:
            track["first_stable_time"] = wall_time
        track.update({
            "previous_coarse_position": prior_coarse,
            "previous_refined_position": prior_refined,
            "last_coarse_position": int(candidate["coarse_value_start"]),
            "last_refined_position": int(candidate["refined_value_start"]),
            "last_seen_iteration": observation["iteration"],
            "last_seen_observation": observation["observation"],
            "miss_count": 0,
            "matched_observation_count": int(
                track["matched_observation_count"]
            ) + 1,
            "consecutive_match_count": consecutive,
            "stable": stable,
            "provisional": not stable,
            "latest_anchor_score": candidate.get("anchor_score"),
            "latest_joint_score": candidate.get("best_joint_score"),
            "latest_predicted_agent": candidate.get(
                "refined_predicted_agent"
            ),
            "last_candidate_rank": candidate_rank,
        })
        track["position_history"].append(int(candidate["coarse_value_start"]))
        track["refined_position_history"].append(
            int(candidate["refined_value_start"])
        )
        track["time_history"].append(wall_time)

    def _track_candidates(
        self,
        candidates: List[Dict[str, object]],
        observation: Dict[str, object],
    ) -> Dict[int, Dict[str, object]]:
        matches, missed, new = self._match_tracks(candidates)
        matched_tracks = set()
        snapshots: Dict[int, Dict[str, object]] = {}
        candidate_ranks = {
            index: rank for rank, index in enumerate(sorted(
                range(len(candidates)),
                key=lambda item: int(candidates[item]["refined_value_start"]),
            ))
        }
        for track_index, candidate_index in matches:
            rank = candidate_ranks[candidate_index]
            self._update_track(
                track_index,
                candidates[candidate_index],
                observation=observation,
                candidate_rank=rank,
            )
            matched_tracks.add(track_index)
            track = self._tracks[track_index]
            snapshot = dict(candidates[candidate_index])
            snapshot.update({
                "track_id": track["track_id"],
                "track_stable": track["stable"],
                "track_provisional": track["provisional"],
                "track_miss_count": track["miss_count"],
                "previous_anchor_start": track.get(
                    "previous_refined_position"
                ),
                "anchor_position_delta": (
                    int(track["last_refined_position"])
                    - int(track.get(
                        "previous_refined_position",
                        track["last_refined_position"],
                    ))
                ),
                "position_stable_count": track[
                    "consecutive_match_count"
                ],
                "first_anchor_seen_time": track["first_seen_time"],
                "first_anchor_stable_time": track["first_stable_time"],
            })
            snapshots[int(track["track_id"])] = snapshot

        for track_index in missed:
            if track_index in matched_tracks:
                continue
            track = self._tracks[track_index]
            track["miss_count"] = int(track["miss_count"]) + 1
            track["total_missing_count"] = int(
                track["total_missing_count"]
            ) + 1
            track["consecutive_match_count"] = 0
            self._track_missing_count += 1

        for candidate_index in new:
            self._unmatched_candidate_count += 1
            rank = candidate_ranks[candidate_index]
            track_index = self._new_track(
                candidates[candidate_index],
                observation=observation,
                candidate_rank=rank,
            )
            track = self._tracks[track_index]
            snapshot = dict(candidates[candidate_index])
            snapshot.update({
                "track_id": track["track_id"],
                "track_stable": track["stable"],
                "track_provisional": track["provisional"],
                "track_miss_count": 0,
                "previous_anchor_start": None,
                "anchor_position_delta": None,
                "position_stable_count": 1,
                "first_anchor_seen_time": track["first_seen_time"],
                "first_anchor_stable_time": track["first_stable_time"],
            })
            snapshots[int(track["track_id"])] = snapshot

        # Tracks which expired before this observation are not returned by the
        # matcher, but their missing history remains explicit in the summary.
        return snapshots

    def observe(
        self,
        logits,
        x,
        logits_start,
        global_step,
        is_last_agent_step=False,
    ) -> None:
        before = len(self._score_observations)
        OracleLatentAgentObserver.observe(
            self,
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
        coarse = self._online_anchor_candidates(logits, x, logits_start)
        sequence_max = logits[0].amax(dim=-1)
        refined = [
            self._refine_anchor(
                coarse=row,
                observation=observation,
                logits=logits,
                x=x,
                logits_start=logits_start,
                sequence_max=sequence_max,
            )
            for row in coarse
        ]
        # Variant A deliberately keeps occurrence-rank association unchanged.
        if not self.persistent_tracking:
            self._track_online_anchors(
                refined, float(observation["wall_time"])
            )
            observation["online_anchors"] = refined
        else:
            observation["online_anchors"] = refined
            observation["track_snapshots"] = self._track_candidates(
                refined, observation
            )
        observation["coarse_anchor_count"] = len(coarse)
        observation["refined_anchor_count"] = len(refined)
        self._online_anchor_observation_counts.append(len(refined))
        self._diagnostic_overhead_seconds += time.perf_counter() - started

    @staticmethod
    def _track_representative(track: Dict[str, object]) -> float:
        history = track.get("refined_position_history") or []
        tail = history[-3:]
        return float(statistics.median(tail)) if tail else math.inf

    def _map_tracks_to_oracle_slots(
        self, oracle_slots: Sequence[Dict[str, object]]
    ) -> Dict[int, int]:
        tracks = sorted(
            self._tracks, key=self._track_representative
        )
        positions = [int(slot["value_start"]) for slot in oracle_slots]
        m, n = len(positions), len(tracks)
        inf = float("inf")
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        action: List[List[Optional[str]]] = [
            [None] * (n + 1) for _ in range(m + 1)
        ]
        for j in range(n + 1):
            dp[0][j] = 0.0
            if j:
                action[0][j] = "skip"
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if dp[i][j - 1] < dp[i][j]:
                    dp[i][j] = dp[i][j - 1]
                    action[i][j] = "skip"
                history = tracks[j - 1].get("refined_position_history") or []
                if history and math.isfinite(dp[i - 1][j - 1]):
                    error = min(abs(int(value) - positions[i - 1]) for value in history)
                    provisional_penalty = 0.0 if tracks[j - 1]["stable"] else 4.0
                    cost = dp[i - 1][j - 1] + error + provisional_penalty
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        action[i][j] = "match"
        mapping = {}
        i, j = m, n
        while i and j:
            if action[i][j] == "match":
                mapping[i - 1] = int(tracks[j - 1]["track_id"])
                i -= 1
                j -= 1
            else:
                j -= 1
        return mapping

    def _anchor_for_slot(
        self,
        *,
        observation: Dict[str, object],
        slot_id: int,
        oracle_value_start: int,
    ) -> Optional[Dict[str, object]]:
        del oracle_value_start
        if not self.persistent_tracking:
            return super()._anchor_for_slot(
                observation=observation,
                slot_id=slot_id,
                oracle_value_start=0,
            )
        track_id = self._evaluation_track_by_slot.get(slot_id)
        if track_id is None:
            return None
        return (observation.get("track_snapshots") or {}).get(track_id)

    def _online_event(self, **kwargs) -> Dict[str, object]:
        event = super()._online_event(**kwargs)
        observation = kwargs["observation"]
        slot_id = int(kwargs["slot_id"])
        oracle_value_start = int(kwargs["oracle_value_start"])
        anchor = self._anchor_for_slot(
            observation=observation,
            slot_id=slot_id,
            oracle_value_start=oracle_value_start,
        )
        if anchor is None:
            event.update({
                "coarse_anchor_start": None,
                "coarse_anchor_end": None,
                "coarse_value_start": None,
                "coarse_position_error": None,
                "refined_value_start": None,
                "refined_position_error": None,
                "refinement_offset": None,
                "best_joint_score": None,
                "second_best_joint_score": None,
                "joint_margin": None,
                "track_id": self._evaluation_track_by_slot.get(slot_id),
            })
            return event
        coarse_value = int(anchor["coarse_value_start"])
        refined_value = int(anchor["refined_value_start"])
        event.update({
            "coarse_anchor_start": int(anchor["coarse_anchor_start"]),
            "coarse_anchor_end": int(anchor["coarse_anchor_end"]),
            "coarse_value_start": coarse_value,
            "coarse_position_error": coarse_value - oracle_value_start,
            "refined_value_start": refined_value,
            "refined_position_error": refined_value - oracle_value_start,
            "refinement_offset": int(anchor["refinement_offset"]),
            "best_joint_score": anchor.get("best_joint_score"),
            "second_best_joint_score": anchor.get(
                "second_best_joint_score"
            ),
            "joint_margin": anchor.get("joint_margin"),
            "track_id": anchor.get("track_id"),
            "track_stable": anchor.get("track_stable"),
            "track_provisional": anchor.get("track_provisional"),
            "track_miss_count": anchor.get("track_miss_count"),
        })
        return event

    def _before_clear_score_observations(self) -> None:
        oracle_slots = list(self._slot_metrics)
        if self.persistent_tracking:
            self._evaluation_track_by_slot = self._map_tracks_to_oracle_slots(
                oracle_slots
            )
        super()._before_clear_score_observations()
        tracks_by_id = {
            int(track["track_id"]): track for track in self._tracks
        }
        for slot in self._slot_metrics:
            trajectory = slot.get("online_trajectory") or []
            natural_time = slot.get("natural_agent_time")
            pre_natural = [
                event for event in trajectory
                if natural_time is not None
                and float(event["wall_time"]) < float(natural_time)
            ]
            coarse_errors = [
                abs(int(event["coarse_position_error"]))
                for event in pre_natural
                if event.get("coarse_position_error") is not None
            ]
            refined_errors = [
                abs(int(event["refined_position_error"]))
                for event in pre_natural
                if event.get("refined_position_error") is not None
            ]
            first_coarse = next(
                (event for event in pre_natural
                 if event.get("coarse_value_start") is not None),
                None,
            )
            first_within4 = next(
                (event for event in pre_natural
                 if event.get("coarse_position_error") is not None
                 and abs(int(event["coarse_position_error"])) <= 4),
                None,
            )
            first_exact = next(
                (event for event in pre_natural
                 if event.get("refined_position_error") == 0),
                None,
            )
            track_id = (
                self._evaluation_track_by_slot.get(int(slot["slot_id"]))
                if self.persistent_tracking else None
            )
            track = tracks_by_id.get(track_id) if track_id is not None else None
            slot.update({
                "first_coarse_time": (
                    first_coarse["wall_time"] if first_coarse else None
                ),
                "first_within4_time": (
                    first_within4["wall_time"] if first_within4 else None
                ),
                "first_exact_refined_time": (
                    first_exact["wall_time"] if first_exact else None
                ),
                "first_stable_exact_refined_time": slot.get(
                    "first_stable_correct_anchor_time"
                ),
                "coarse_mean_position_error": (
                    statistics.mean(coarse_errors) if coarse_errors else None
                ),
                "refined_mean_position_error": (
                    statistics.mean(refined_errors)
                    if refined_errors else None
                ),
                "track_id": track_id,
                "track_reassociation_count": (
                    track.get("reassociation_count") if track else 0
                ),
                "track_rank_shift_count": (
                    track.get("rank_shift_count") if track else 0
                ),
                "track_position_history": (
                    track.get("position_history") if track else []
                ),
                "track_refined_position_history": (
                    track.get("refined_position_history") if track else []
                ),
            })

    def metrics(self) -> Dict[str, object]:
        result = super().metrics()
        policy = (
            "online_latent_refine_tracking"
            if self.persistent_tracking else "online_latent_refine"
        )
        stable_tracks = sum(bool(track["stable"]) for track in self._tracks)
        result.update({
            "policy": policy,
            "timing_source": (
                "local_refined_online_anchor_and_oracle_logits"
            ),
            "persistent_tracking": self.persistent_tracking,
            "refinement_radius": self.refinement_radius,
            "refinement_anchor_weight": self.refinement_anchor_weight,
            "refinement_agent_weight": self.refinement_agent_weight,
            "max_track_misses": self.max_track_misses,
            "max_track_match_distance": self.max_track_match_distance,
            "track_stable_observations": self.track_stable_observations,
            "track_score_weight": self.track_score_weight,
            "tracks_created": len(self._tracks),
            "online_anchor_slot_count": (
                len(self._tracks)
                if self.persistent_tracking
                else result.get("online_anchor_slot_count")
            ),
            "stable_tracks": stable_tracks,
            "provisional_tracks": len(self._tracks) - stable_tracks,
            "unmatched_candidate_count": self._unmatched_candidate_count,
            "track_missing_count": self._track_missing_count,
            "track_reassociation_count": self._track_reassociation_count,
            "slot_identity_shift_count": self._slot_identity_shift_count,
            "tracks": [dict(track) for track in self._tracks],
        })
        return result

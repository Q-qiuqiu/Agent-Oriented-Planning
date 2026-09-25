"""Compact read-only ordered Agent observer for existing Dual warmups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    JsonAgentSlotRuntime,
)


@dataclass
class _PositionTrack:
    track_id: int
    relative_pos: int
    pattern: Tuple[int, ...]
    anchor_score: float
    observed_ratio: float
    position_stable: int = 1
    agent: Optional[str] = None
    probability: float = 0.0
    margin: float = 0.0
    agent_stable: int = 0
    last_observation: int = 0


class OrderedPlanAgentObserver:
    """Track the first three ordered Agent fields without mutating decoding."""

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
        probability_threshold: float = 0.90,
        margin_threshold: float = 0.40,
        stable_observations: int = 2,
        position_drift: int = 4,
        candidate_limit: int = 16,
    ) -> None:
        self.elapsed = elapsed
        self.probability_threshold = float(probability_threshold)
        self.margin_threshold = float(margin_threshold)
        self.stable_observations = int(stable_observations)
        self.position_drift = int(position_drift)
        self.candidate_limit = int(candidate_limit)
        self.scorer = JsonAgentFieldController(
            tokenizer=tokenizer,
            config=JsonAgentPriorityConfig(
                catalog=list(catalog),
                priority_slots=candidate_limit,
                tracking_slots=candidate_limit,
                anchor_min_logit_margin=anchor_min_logit_margin,
                tentative_probability=probability_threshold,
                tentative_margin=margin_threshold,
                probe_period=0,
            ),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
        )
        self._tracks: List[_PositionTrack] = []
        self._next_track_id = 0
        self._observation = 0
        self._events: List[Dict[str, object]] = []

    @property
    def events(self) -> List[Dict[str, object]]:
        return self._events

    def initialize(self, x: torch.Tensor) -> None:
        self.scorer.initialize(x)

    @staticmethod
    def _rank_distribution(distribution: Dict[str, float]):
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        agent, probability = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return agent, float(probability), float(probability - second)

    def _anchor_candidates(self, logits, x, logits_start, plan_start, plan_end):
        sequence_logits = logits[0]
        sequence_max = sequence_logits.amax(dim=-1)
        logits_end = logits_start + sequence_logits.shape[0]
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}
        for pattern in self.scorer.anchor_variants:
            width = len(pattern)
            start = max(int(plan_start), int(logits_start))
            end = min(int(plan_end), int(logits_end)) - width + 1
            if end <= start:
                continue
            count = end - start
            relative = start - logits_start
            scores = torch.zeros(count, device=logits.device, dtype=torch.float32)
            compatible = torch.ones(count, device=logits.device, dtype=torch.bool)
            observed = torch.zeros(count, device=logits.device, dtype=torch.float32)
            for offset, token_id in enumerate(pattern):
                lo = relative + offset
                hi = lo + count
                scores += sequence_logits[lo:hi, token_id].float() - sequence_max[lo:hi].float()
                positions = torch.arange(start + offset, end + offset, device=x.device)
                current = x[0, positions]
                compatible &= (current == self.scorer.mask_id) | (current == token_id)
                observed += (current == token_id).float()
            scores /= float(width)
            observed /= float(width)
            if pattern in self.scorer.speculative_anchor_variants:
                eligible = compatible & (
                    (scores >= self.scorer.config.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
            else:
                eligible = compatible & (observed == 1.0)
            exact = torch.nonzero(eligible & (observed == 1.0), as_tuple=False).flatten()
            speculative = torch.nonzero(eligible & (observed != 1.0), as_tuple=False).flatten()
            indices = exact.detach().cpu().tolist()
            if speculative.numel():
                keep = min(int(speculative.numel()), self.candidate_limit * 4)
                best = torch.topk(scores[speculative], k=keep).indices
                indices.extend(speculative[best].detach().cpu().tolist())
            for local in indices:
                position = start + int(local)
                row = (
                    position,
                    tuple(pattern),
                    float(scores[local].detach().cpu()),
                    float(observed[local].detach().cpu()),
                )
                previous = candidates.get(position)
                if previous is None or (row[3], row[2]) > (previous[3], previous[2]):
                    candidates[position] = row

        radius = max(len(pattern) for pattern in self.scorer.anchor_variants)
        clusters: List[List[Tuple[int, Tuple[int, ...], float, float]]] = []
        for row in sorted(candidates.values(), key=lambda item: item[0]):
            if not clusters or row[0] - clusters[-1][0][0] >= radius:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        reduced = [max(cluster, key=lambda item: (item[3], item[2])) for cluster in clusters]
        reduced.sort(key=lambda item: item[0])
        ordered = []
        for row in reduced:
            if any(
                abs(row[0] - prior[0]) < self.scorer.config.min_anchor_gap
                for prior in ordered
            ):
                continue
            ordered.append(row)
        return ordered[: self.candidate_limit]

    def _match_track(self, relative_pos: int, used: set[int]) -> Optional[_PositionTrack]:
        choices = [
            track for track in self._tracks
            if track.track_id not in used
            and track.last_observation == self._observation - 1
            and abs(track.relative_pos - relative_pos) <= self.position_drift
        ]
        if not choices:
            return None
        return min(choices, key=lambda track: abs(track.relative_pos - relative_pos))

    def observe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        *,
        logits_start: int,
        global_step: int,
        plan_start: int,
        plan_end: int,
        phase: Optional[str],
    ) -> None:
        if logits_start > plan_start or logits_start + logits.shape[1] < plan_end:
            return
        self._observation += 1
        self.scorer.set_search_region(plan_start, plan_end)
        absolute = self._anchor_candidates(logits, x, logits_start, plan_start, plan_end)
        current = []
        for anchor_start, pattern, score, observed_ratio in absolute:
            runtime = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=anchor_start + len(pattern),
            )
            distribution = self.scorer._score_catalog(logits, logits_start, runtime)
            if distribution is None:
                continue
            agent, probability, margin = self._rank_distribution(distribution)
            entropy = -sum(
                float(value) * math.log(max(float(value), 1e-12))
                for value in distribution.values()
            )
            current.append({
                "relative_pos": int(anchor_start - plan_start),
                "pattern": tuple(pattern),
                "anchor_score": float(score),
                "observed_ratio": float(observed_ratio),
                "agent": agent,
                "probability": probability,
                "margin": margin,
                "entropy": entropy,
                "posterior": {str(name): float(value) for name, value in distribution.items()},
            })
        current.sort(key=lambda row: int(row["relative_pos"]))

        used: set[int] = set()
        active: List[Tuple[_PositionTrack, Dict[str, object]]] = []
        for row in current:
            relative_pos = int(row["relative_pos"])
            track = self._match_track(relative_pos, used)
            if track is None:
                track = _PositionTrack(
                    track_id=self._next_track_id,
                    relative_pos=relative_pos,
                    pattern=tuple(row["pattern"]),
                    anchor_score=float(row["anchor_score"]),
                    observed_ratio=float(row["observed_ratio"]),
                    last_observation=self._observation,
                )
                self._next_track_id += 1
                self._tracks.append(track)
            else:
                track.position_stable += 1
                track.relative_pos = relative_pos
                track.pattern = tuple(row["pattern"])
                track.anchor_score = float(row["anchor_score"])
                track.observed_ratio = float(row["observed_ratio"])
                track.last_observation = self._observation
            agent = str(row["agent"])
            track.agent_stable = track.agent_stable + 1 if track.agent == agent else 1
            track.agent = agent
            track.probability = float(row["probability"])
            track.margin = float(row["margin"])
            used.add(track.track_id)
            active.append((track, row))

        active.sort(key=lambda item: item[0].relative_pos)
        now = float(self.elapsed())
        slots = []
        for slot_index, (track, candidate) in enumerate(active[:3]):
            ready = (
                track.position_stable >= self.stable_observations
                and track.agent_stable >= self.stable_observations
                and track.probability >= self.probability_threshold
                and track.margin >= self.margin_threshold
            )
            slots.append({
                "slot": slot_index,
                "track_id": track.track_id,
                "cluster_id": track.track_id,
                "anchor_id": track.track_id,
                "relative_pos": track.relative_pos,
                "agent": track.agent,
                "probability": track.probability,
                "margin": track.margin,
                "anchor_score": track.anchor_score,
                "anchor_observed_ratio": track.observed_ratio,
                "position_stable_observations": track.position_stable,
                "agent_stable_observations": track.agent_stable,
                "ready": ready,
                "entropy": candidate["entropy"],
                "posterior": candidate["posterior"],
            })
        self._events.append({
            "observation": self._observation,
            "seconds": now,
            "step": int(global_step),
            "phase": phase,
            "plan_start": int(plan_start),
            "plan_end": int(plan_end),
            "slots": slots,
        })

    def close(self) -> None:
        self.scorer.close()

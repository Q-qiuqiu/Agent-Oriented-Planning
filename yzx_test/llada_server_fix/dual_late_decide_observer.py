"""Production latent-region/prefix/natural commits on Dual Vanilla logits."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from fusion_prefetch import FusionAgentPrefetchTracker, FusionGateConfig
from hypothesis_region_observer import HypothesisRegionObserver
from json_agent_priority import JsonAgentSlotRuntime


class DualVanillaLateDecideObserver(HypothesisRegionObserver):
    """Read-only stable-2 latent-region commit controller.

    Full-sequence logits are existing Dual Vanilla block warmups. This class
    never writes the canvas, changes a decoder mask, or requests a model
    forward. Final PLAN data is used only to evaluate already-fired online
    hypothesis triggers and never participates in trigger eligibility.
    """

    full_sequence_discovery_steps = 0
    region_radius = 4
    region_top_k = 2
    latent_stable_observations = 2

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        anchor_min_logit_margin: float = -6.0,
        score_chunk_size: int = 32,
        debug_trajectory: bool = False,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            score_chunk_size=score_chunk_size,
            anchor_min_logit_margin=anchor_min_logit_margin,
            region_radius=self.region_radius,
            region_top_k=self.region_top_k,
            region_temperature=1.0,
            main_aggregation="top2_mean",
            anchor_position_tolerance=4,
            anchor_stable_observations=2,
            max_track_misses=3,
            max_track_match_distance=64,
            track_stable_observations=2,
            track_score_weight=0.05,
            hypothesis_merge_distance=6,
            hypothesis_merge_gap=2,
            hypothesis_min_seen=2,
            hypothesis_min_support=0.5,
            hypothesis_max_center_jump=12,
            hypothesis_duplicate_observations=2,
        )
        self.catalog = tuple(str(agent) for agent in catalog)
        self.fusion = FusionAgentPrefetchTracker(
            FusionGateConfig(), catalog=self.catalog
        )
        self.debug_trajectory = bool(debug_trajectory)
        self._hypothesis_runtime: Dict[int, Dict[str, object]] = {}
        self._latent_triggers: List[Dict[str, object]] = []
        self._debug_events: List[Dict[str, object]] = []
        self._natural_value_starts: Dict[int, int] = {}
        self._evaluation_agents: List[Optional[str]] = []
        self.final_plan_parse_success = False

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self.fusion = FusionAgentPrefetchTracker(
            FusionGateConfig(), catalog=self.catalog
        )
        self._hypothesis_runtime.clear()
        self._latent_triggers.clear()
        self._debug_events.clear()
        self._natural_value_starts.clear()
        self._evaluation_agents.clear()
        self.final_plan_parse_success = False

    def _elapsed(self) -> float:
        """Formal latency clock: include online detection/scoring overhead."""
        if self._started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._started_at)

    def has_unconfirmed_agents(self) -> bool:
        return False

    def probing_active(self) -> bool:
        return False

    def decoder_mask(self, mask_index, mask_start=0):
        del mask_start
        return mask_index

    def _continuous_agent_prefix(
        self, x: torch.Tensor, runtime: JsonAgentSlotRuntime
    ) -> str:
        if runtime.name_start is None:
            return ""
        visible = []
        value_end = min(
            x.shape[1], runtime.name_start + self.value_width + 4
        )
        for token_id in x[
            0, runtime.name_start:value_end
        ].detach().cpu().tolist():
            if token_id == self.mask_id:
                break
            visible.append(token_id)
        if not visible:
            return ""
        text = self.tokenizer.decode(visible, skip_special_tokens=True)
        match = re.match(r"^[A-Za-z][A-Za-z0-9_]*", text)
        return match.group(0) if match else ""

    def _sync_natural_from_canvas(self, x: torch.Tensor) -> None:
        """Run high-frequency prefix/natural fallback without latent gates."""
        now = float(self._elapsed())
        rows = sorted(self._plan_agent_candidates(x), key=lambda row: row[0])
        for slot, (anchor_start, pattern, _score, _ratio) in enumerate(rows):
            runtime = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=anchor_start + len(pattern),
            )
            self._natural_value_starts[slot] = int(runtime.name_start)
            # Prefix deliberately runs before Natural in the same callback. A
            # complete value is itself a unique continuous prefix.
            prefix = self._continuous_agent_prefix(x, runtime)
            if prefix:
                self.fusion.observe_prefix(slot, prefix, seconds=now)
            agent = self._observed_catalog_value(x, runtime)
            if agent is not None:
                self.fusion.observe_natural(slot, str(agent), seconds=now)

    @staticmethod
    def _runtime_state() -> Dict[str, object]:
        return {
            "last_region_agent": None,
            "region_stable_count": 0,
            "trigger": None,
        }

    def _score_agent_region(
        self,
        logits: torch.Tensor,
        *,
        center: int,
    ) -> Optional[Dict[str, object]]:
        """Agent-only, normalized-logprob, top-2-mean region scoring."""
        sequence_end = min(
            logits.shape[1], self.prompt_length + self.gen_length
        )
        left = max(self.prompt_length, int(center) - self.region_radius)
        right = min(sequence_end - 1, int(center) + self.region_radius)
        max_tokens = max(len(ids) for ids in self._candidate_ids.values())
        row_end = min(sequence_end, right + max_tokens)
        if left >= row_end:
            return None
        local_logits = logits[0, left:row_end].float()
        partitions = torch.logsumexp(local_logits, dim=-1)
        agent_scores = []
        for agent in self._candidate_names:
            token_ids = self._candidate_ids[agent]
            valid_right = min(right, sequence_end - len(token_ids))
            if valid_right < left:
                continue
            count = valid_right - left + 1
            alignment = torch.zeros(
                count, device=logits.device, dtype=torch.float32
            )
            for offset, token_id in enumerate(token_ids):
                rows = local_logits[offset:offset + count]
                alignment += rows[:, int(token_id)] - partitions[
                    offset:offset + count
                ]
            alignment /= float(len(token_ids))
            top_count = min(self.region_top_k, count)
            top_values, top_indices = torch.topk(
                alignment, k=top_count, largest=True
            )
            agent_scores.append((
                agent,
                top_values.mean(),
                int(left + int(top_indices[0].detach().cpu())),
            ))
        if not agent_scores:
            return None
        stacked = torch.stack([row[1] for row in agent_scores])
        order = torch.argsort(stacked, descending=True).detach().cpu().tolist()
        best = agent_scores[int(order[0])]
        second = agent_scores[int(order[1])] if len(order) > 1 else None
        top1 = float(best[1].detach().cpu())
        top2 = float(second[1].detach().cpu()) if second else None
        return {
            "predicted_agent": str(best[0]),
            "top1_region_score": top1,
            "top2_region_score": top2,
            "region_margin": None if top2 is None else top1 - top2,
            "best_alignment_position": int(best[2]),
            "region_left": left,
            "region_right": right,
        }

    def _observe_latent_region(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        global_step: Optional[int],
    ) -> None:
        # Timestamp existing full-sequence logits before detector/scorer work.
        self._synchronize(logits)
        started = time.perf_counter()
        seconds = float(self._elapsed())
        self._full_sequence_observations += 1
        observation = {
            "observation": self._full_sequence_observations,
            "iteration": None if global_step is None else int(global_step),
            "wall_time": seconds,
        }
        coarse = self._online_anchor_candidates(logits, x, 0)
        candidates = []
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
            candidates.append(candidate)
        snapshots = self._track_candidates(candidates, observation)
        active = self._assign_active_tracks(snapshots, observation)
        centers = self._update_hypotheses(active, observation)
        centers, active = self._merge_close_hypotheses(
            centers, active, observation
        )
        self._suppress_duplicates(centers)

        active_ids = set(centers)
        for hypothesis in self._hypotheses:
            if not hypothesis.get("active_root", True):
                continue
            hypothesis_id = int(hypothesis["hypothesis_id"])
            state = self._hypothesis_runtime.setdefault(
                hypothesis_id, self._runtime_state()
            )
            valid = bool(
                hypothesis_id in active_ids
                and int(hypothesis["seen_count"]) >= 2
                and float(hypothesis["support_ratio"]) >= 0.5
            )
            if not valid:
                state["last_region_agent"] = None
                state["region_stable_count"] = 0
                continue
            # First-valid latent trigger wins for this persistent hypothesis.
            if state["trigger"] is not None:
                continue
            center = int(centers[hypothesis_id])
            prediction = self._score_agent_region(logits, center=center)
            if prediction is None:
                state["last_region_agent"] = None
                state["region_stable_count"] = 0
                continue
            agent = str(prediction["predicted_agent"])
            if state["last_region_agent"] == agent:
                state["region_stable_count"] = (
                    int(state["region_stable_count"]) + 1
                )
            else:
                state["last_region_agent"] = agent
                state["region_stable_count"] = 1
            if self.debug_trajectory:
                self._debug_events.append({
                    "hypothesis_id": hypothesis_id,
                    "observation": observation["observation"],
                    "wall_time": seconds,
                    "center": center,
                    "seen_count": int(hypothesis["seen_count"]),
                    "support_ratio": float(hypothesis["support_ratio"]),
                    "region_stable_count": int(
                        state["region_stable_count"]
                    ),
                    **prediction,
                })
            if (
                int(state["region_stable_count"])
                < self.latent_stable_observations
            ):
                continue
            # The candidate exists only after region scoring has completed.
            # Capture the event before mutating in-memory log/prefetch state.
            trigger_seconds = float(self._elapsed())
            prefetch = self.fusion.mark_prefetch(agent)
            trigger = {
                "hypothesis_id": hypothesis_id,
                "trigger_time": trigger_seconds,
                "observation": observation["observation"],
                "predicted_agent": agent,
                "hypothesis_center": center,
                "seen_count": int(hypothesis["seen_count"]),
                "support_ratio": float(hypothesis["support_ratio"]),
                "region_stable_count": int(state["region_stable_count"]),
                "prefetch_triggered": prefetch["prefetch_triggered"],
                "prefetch_reused": prefetch["prefetch_reused"],
                **prediction,
            }
            state["trigger"] = trigger
            self._latent_triggers.append(trigger)
        self._diagnostic_overhead_seconds += time.perf_counter() - started

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
        self._sync_natural_from_canvas(x)
        if logits_start == 0 and logits.shape[1] == x.shape[1]:
            self._observe_latent_region(logits, x, global_step)

    def step_callback(self, nfe, num_block, block_step, x) -> None:
        del nfe, num_block, block_step
        self._sync_natural_from_canvas(x)

    def finalize(self, x: torch.Tensor) -> None:
        self._sync_natural_from_canvas(x)

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

    @staticmethod
    def _percentile(values: Sequence[float], fraction: float):
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return None
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * fraction
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (
            ordered[upper] - ordered[lower]
        ) * (rank - lower)

    @classmethod
    def _stats(cls, values: Sequence[float]) -> Dict[str, object]:
        clean = [float(value) for value in values]
        return {
            "count": len(clean),
            "mean": statistics.mean(clean) if clean else None,
            "p50": cls._percentile(clean, 0.50),
            "p95": cls._percentile(clean, 0.95),
        }

    def _evaluation_mapping(self) -> Tuple[Dict[int, int], Dict[int, float]]:
        positions = {
            int(slot): int(position)
            for slot, position in self._natural_value_starts.items()
            if slot < len(self._evaluation_agents)
        }
        mapping = {}
        centers = {}
        for hypothesis in self._hypotheses:
            history = hypothesis.get("center_history") or []
            if not history or not positions:
                continue
            hypothesis_id = int(hypothesis["hypothesis_id"])
            center = float(statistics.median(history[-6:]))
            centers[hypothesis_id] = center
            slot = min(
                positions,
                key=lambda candidate: abs(center - positions[candidate]),
            )
            if abs(center - positions[slot]) <= self.region_radius:
                mapping[hypothesis_id] = int(slot)
        return mapping, centers

    def _evaluated_triggers(
        self,
    ) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, object]]]:
        mapping, centers = self._evaluation_mapping()
        rows = []
        primary = {}
        mapped_seen = set()
        for source in sorted(
            self._latent_triggers,
            key=lambda row: (
                float(row["trigger_time"]), int(row["hypothesis_id"])
            ),
        ):
            row = dict(source)
            hypothesis_id = int(row["hypothesis_id"])
            slot = mapping.get(hypothesis_id)
            duplicate = slot is not None and slot in mapped_seen
            final_agent = (
                self._evaluation_agents[slot]
                if slot is not None and slot < len(self._evaluation_agents)
                else None
            )
            if slot is None:
                category = "false_track"
            elif duplicate:
                category = "duplicate"
            else:
                mapped_seen.add(slot)
                category = (
                    "correct_real_slot"
                    if row["predicted_agent"] == final_agent
                    else "wrong_agent_real_slot"
                )
                primary[slot] = row
            row.update({
                "mapped_real_slot": slot,
                "mapped_final_agent": final_agent,
                "evaluation_center": centers.get(hypothesis_id),
                "false_track": slot is None,
                "duplicate": duplicate,
                "category": category,
            })
            rows.append(row)
        return rows, primary

    def _source_distribution(
        self, rows: Sequence[Dict[str, object]]
    ) -> Dict[str, object]:
        result = {}
        for agent in ("overall", *self.catalog):
            group = list(rows) if agent == "overall" else [
                row for row in rows if row.get("natural_agent") == agent
            ]
            by_source = {}
            for source in ("latent_region", "prefix", "natural"):
                source_rows = [
                    row for row in group
                    if row.get("commit_source") == source
                ]
                evaluated = [
                    row for row in source_rows
                    if row.get("correct") is not None
                ]
                correct = [row for row in evaluated if row["correct"]]
                by_source[source] = {
                    "count": len(source_rows),
                    "accuracy": (
                        len(correct) / len(evaluated) if evaluated else None
                    ),
                    "lead": self._stats([
                        row["lead_seconds"] for row in source_rows
                        if row.get("lead_seconds") is not None
                    ]),
                    "correct_lead": self._stats([
                        row["lead_seconds"] for row in correct
                        if row.get("lead_seconds") is not None
                    ]),
                }
            result[agent] = {"slots": len(group), "sources": by_source}
        return result

    def metrics(self) -> Dict[str, object]:
        final_agents = list(self._evaluation_agents)
        if not final_agents:
            final_agents = [
                (state.get("natural") or {}).get("agent")
                for state in self.fusion.slots
                if state.get("natural") is not None
            ]
        trigger_rows, latent_by_slot = self._evaluated_triggers()
        rows = []
        source_counts = {
            "latent_region": 0, "prefix": 0, "natural": 0
        }
        for slot, final_agent in enumerate(final_agents):
            self.fusion._ensure_slots(slot + 1)
            state = self.fusion.slots[slot]
            natural = state.get("natural") or {}
            fallback = state.get("commit") or {}
            latent = latent_by_slot.get(slot)
            candidates = []
            if latent is not None:
                candidates.append((
                    float(latent["trigger_time"]), 2,
                    "latent_region", latent,
                ))
            if fallback:
                priority = 0 if fallback.get("source") == "prefix" else 1
                candidates.append((
                    float(fallback["seconds"]), priority,
                    str(fallback["source"]), fallback,
                ))
            winner = min(candidates) if candidates else None
            source = winner[2] if winner else None
            commit = winner[3] if winner else {}
            if source in source_counts:
                source_counts[source] += 1
            committed_agent = (
                commit.get("predicted_agent")
                if source == "latent_region" else commit.get("agent")
            )
            natural_agent = natural.get("agent") or final_agent
            natural_seconds = natural.get("seconds")
            commit_seconds = winner[0] if winner else None
            correct = (
                committed_agent == natural_agent
                if committed_agent is not None and natural_agent is not None
                else None
            )
            lead = (
                float(natural_seconds) - float(commit_seconds)
                if natural_seconds is not None and commit_seconds is not None
                else None
            )
            rows.append({
                "slot_id": slot,
                "commit_source": source,
                "committed_agent": committed_agent,
                "natural_agent": natural_agent,
                # Retain the two legacy aliases consumed by older timing
                # clients while the canonical fields stay natural_agent/*.
                "natural_decoded_agent": natural_agent,
                "commit_wall_time": commit_seconds,
                "natural_agent_wall_time": natural_seconds,
                "T_natural_decode": natural_seconds,
                "prefix": commit.get("prefix"),
                "hypothesis_id": (
                    commit.get("hypothesis_id")
                    if source == "latent_region" else None
                ),
                "region_score": (
                    commit.get("top1_region_score")
                    if source == "latent_region" else None
                ),
                "region_margin": (
                    commit.get("region_margin")
                    if source == "latent_region" else None
                ),
                "stable_count": (
                    commit.get("region_stable_count")
                    if source == "latent_region" else None
                ),
                "semantic_confidence": None,
                "semantic_margin": None,
                "correct": correct,
                "strict_semantic_early": None,
                "lead_seconds": lead,
                "prefetch_triggered": commit.get("prefetch_triggered"),
                "prefetch_reused": commit.get("prefetch_reused"),
            })

        evaluated = [row for row in rows if row["correct"] is not None]
        correct = [row for row in evaluated if row["correct"]]
        latent_rows = [
            row for row in rows if row["commit_source"] == "latent_region"
        ]
        correct_latent = [row for row in latent_rows if row["correct"]]
        commit_times = [
            float(row["commit_wall_time"]) for row in rows
            if row["commit_wall_time"] is not None
        ]
        trigger_categories = {
            category: sum(row["category"] == category for row in trigger_rows)
            for category in (
                "correct_real_slot", "wrong_agent_real_slot",
                "false_track", "duplicate",
            )
        }
        trigger_precision = (
            trigger_categories["correct_real_slot"] / len(trigger_rows)
            if trigger_rows else None
        )
        return {
            "policy": "commit",
            "timing_source": "latent_region_prefix_natural_commit",
            "read_only": True,
            "writes_agent_tokens": False,
            "changes_decoder_mask": False,
            "triggers_real_prefetch": False,
            "extra_model_forwards": 0,
            "probe_forwards": 0,
            "runtime_agent_registry": list(self.catalog),
            "latent_region_radius": self.region_radius,
            "latent_region_aggregation": "agent_only_top2_mean",
            "latent_region_stable_observations": (
                self.latent_stable_observations
            ),
            "hypothesis_min_seen": 2,
            "hypothesis_min_support": 0.5,
            "full_sequence_observation_count": (
                self._full_sequence_observations
            ),
            "diagnostic_observer_overhead_seconds": float(
                self._diagnostic_overhead_seconds
            ),
            "final_plan_parse_success": self.final_plan_parse_success,
            "agent_count": len(rows),
            "latent_region_commit_count": source_counts["latent_region"],
            "semantic_commit_count": 0,
            "prefix_commit_count": source_counts["prefix"],
            "natural_commit_count": source_counts["natural"],
            "first_valid_commit_accuracy": (
                len(correct) / len(evaluated) if evaluated else None
            ),
            "latent_region_commit_accuracy": (
                len(correct_latent) / len(latent_rows)
                if latent_rows else None
            ),
            "correct_latent_coverage": (
                len(correct_latent) / len(rows) if rows else None
            ),
            "correct_latent_commit_count": len(correct_latent),
            "correct_latent_lead": self._stats([
                row["lead_seconds"] for row in correct_latent
                if row.get("lead_seconds") is not None
            ]),
            "source_distribution": self._source_distribution(rows),
            "latent_trigger_count": len(trigger_rows),
            "latent_trigger_precision": trigger_precision,
            "latent_trigger_categories": trigger_categories,
            "latent_trigger_events": trigger_rows,
            "first_agent_commit_time": (
                min(commit_times) if commit_times else None
            ),
            "all_agents_commit_time": (
                max(commit_times)
                if rows and len(commit_times) == len(rows) else None
            ),
            "agent_slots": rows,
            "natural_agent_sequence": [
                row["natural_agent"] for row in rows
                if row["natural_agent"] is not None
            ],
            "debug_trajectory": (
                list(self._debug_events) if self.debug_trajectory else []
            ),
        }

    def close(self) -> None:
        return None

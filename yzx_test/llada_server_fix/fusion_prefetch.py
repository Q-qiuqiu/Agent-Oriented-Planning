"""Read-only Global + Local + Natural Agent prefetch fusion.

The tracker consumes compact ordered-slot observations extracted from Dual
Vanilla's existing full-sequence warmups.  It never owns the generation
tensor, so it cannot modify decoding.  Natural materialization is delivered
separately after every normal token transfer and is always authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FusionGateConfig:
    global_stable: int = 2
    global_probability: float = 0.90
    global_margin: float = 0.40
    local_stable: int = 2
    local_probability: float = 0.75
    local_margin: float = 0.15
    position_drift: int = 4
    independent_min_position_gap: int = 12
    independent_structural_min_score: float = 1e-6


def _source_state() -> Dict[str, object]:
    return {
        "last_agent": None,
        "agent_stable": 0,
        "last_position": None,
        "position_stable": 0,
        "latest_ready": None,
        "first_prediction": None,
        "events": [],
    }


class FusionAgentPrefetchTracker:
    """Fuse three read-only sources independently for three ordered slots."""

    def __init__(
        self,
        config: FusionGateConfig,
        slot_count: int = 3,
        *,
        benchmark: Optional[str] = None,
        catalog: Sequence[str] = (),
    ) -> None:
        for value in (config.global_stable, config.local_stable):
            if value < 1:
                raise ValueError("Fusion stability counts must be positive")
        self.config = config
        self.slot_count = int(slot_count)
        self.benchmark = str(benchmark).lower() if benchmark else None
        self.catalog = tuple(str(agent) for agent in catalog)
        self._global_cluster_owner: Dict[object, int] = {}
        self._global_anchor_owner: Dict[object, int] = {}
        self.slots: List[Dict[str, object]] = [
            {
                "global": _source_state(),
                "local": _source_state(),
                "natural": None,
                "prefetch": None,
                "authoritative_agent": None,
                "current_prediction": None,
                "current_prediction_source": None,
                "switches": [],
                "correction": None,
            }
            for _ in range(self.slot_count)
        ]
        self.timeline: List[Dict[str, object]] = []

    def _thresholds(self, source: str):
        if source == "global":
            return (
                self.config.global_stable,
                self.config.global_probability,
                self.config.global_margin,
            )
        return (
            self.config.local_stable,
            self.config.local_probability,
            self.config.local_margin,
        )

    @staticmethod
    def _reset_evidence(state: Dict[str, object]) -> None:
        state["last_agent"] = None
        state["agent_stable"] = 0
        state["last_position"] = None
        state["position_stable"] = 0
        state["latest_ready"] = None

    def _update_source(
        self,
        slot: int,
        source: str,
        row: Optional[Dict[str, object]],
        *,
        seconds: float,
        step: int,
        independent: Optional[Dict[str, object]] = None,
    ) -> None:
        state = self.slots[slot][source]
        assert isinstance(state, dict)
        if row is None:
            self._reset_evidence(state)
            return
        agent = row.get("agent")
        position = row.get("relative_pos")
        probability = row.get("probability")
        margin = row.get("margin")
        if None in (agent, position, probability, margin):
            self._reset_evidence(state)
            return
        agent = str(agent)
        position = int(position)
        if state["last_agent"] == agent:
            state["agent_stable"] = int(state["agent_stable"]) + 1
        else:
            state["last_agent"] = agent
            state["agent_stable"] = 1
            state["latest_ready"] = None
        if (
            state["last_position"] is not None
            and abs(position - int(state["last_position"]))
            <= self.config.position_drift
        ):
            state["position_stable"] = int(state["position_stable"]) + 1
        else:
            state["position_stable"] = 1
            state["latest_ready"] = None
        state["last_position"] = position
        stable, probability_min, margin_min = self._thresholds(source)
        identity_ready = (
            int(state["agent_stable"]) >= stable
            and int(state["position_stable"]) >= stable
            and float(probability) >= probability_min
            and float(margin) >= margin_min
        )
        independent = independent or {}
        independent_slot_ready = bool(
            independent.get("independent_slot_ready", True)
        )
        ready = identity_ready and (
            source != "global" or independent_slot_ready
        )
        event = {
            "source": source,
            "seconds": float(seconds),
            "step": int(step),
            "agent": agent,
            "probability": float(probability),
            "margin": float(margin),
            "relative_pos": position,
            "anchor_observed_ratio": float(
                row.get("anchor_observed_ratio") or 0.0
            ),
            "agent_stable": int(state["agent_stable"]),
            "position_stable": int(state["position_stable"]),
            "candidate_agent": agent,
            "candidate_position": position,
            "cluster_id": independent.get(
                "cluster_id", row.get("track_id", position)
            ),
            "previous_slot_position": independent.get(
                "previous_slot_position"
            ),
            "position_gap": independent.get("position_gap"),
            "anchor_id": independent.get(
                "anchor_id", row.get("anchor_id", row.get("track_id", position))
            ),
            "anchor_reused": bool(independent.get("anchor_reused", False)),
            "cluster_reused": bool(independent.get("cluster_reused", False)),
            "structural_slot_score": float(
                independent.get("structural_slot_score", 0.0)
            ),
            "identity_ready": bool(identity_ready),
            "independent_slot_ready": bool(independent_slot_ready),
            "global_ready": bool(ready) if source == "global" else None,
            "ready": bool(ready),
        }
        state["events"].append(event)
        if ready:
            state["latest_ready"] = dict(event)
            if state["first_prediction"] is None:
                state["first_prediction"] = dict(event)

    @staticmethod
    def _row_identifier(row: Dict[str, object], name: str):
        value = row.get(name)
        if value is not None:
            return value
        # Old compact trajectories did not save ids.  Position is a safe,
        # conservative replay fallback because positional drift is already
        # handled by the ordered observer before rows reach this tracker.
        return ("position", int(row["relative_pos"]))

    def _independent_slot_diagnostics(
        self,
        rows: Sequence[Optional[Dict[str, object]]],
        slot: int,
    ) -> Dict[str, object]:
        """Require a distinct ordered field before Global may trigger.

        Agent identity is deliberately absent from this decision: repeated
        agents in two genuinely distinct JSON objects remain valid.
        """
        row = rows[slot]
        if row is None:
            return {"independent_slot_ready": False}
        position = int(row["relative_pos"])
        cluster_id = self._row_identifier(row, "track_id")
        anchor_id = self._row_identifier(row, "anchor_id")
        structural_score = float(
            row.get("structural_slot_score")
            or max(
                float(row.get("anchor_observed_ratio") or 0.0),
                float(row.get("object_structure_score") or 0.0),
            )
        )
        if slot == 0:
            return {
                "cluster_id": cluster_id,
                "anchor_id": anchor_id,
                "previous_slot_position": None,
                "position_gap": None,
                "anchor_reused": False,
                "cluster_reused": False,
                "structural_slot_score": structural_score,
                # Slot1's predictor is intentionally frozen.
                "independent_slot_ready": True,
            }

        previous = rows[slot - 1]
        previous_position = (
            int(previous["relative_pos"]) if previous is not None else None
        )
        position_gap = (
            position - previous_position
            if previous_position is not None else None
        )
        prior_rows = [candidate for candidate in rows[:slot] if candidate]
        prior_clusters = {
            self._row_identifier(candidate, "track_id")
            for candidate in prior_rows
        }
        prior_anchors = {
            self._row_identifier(candidate, "anchor_id")
            for candidate in prior_rows
        }
        cluster_reused = (
            cluster_id in prior_clusters
            or self._global_cluster_owner.get(cluster_id, slot) < slot
        )
        anchor_reused = (
            anchor_id in prior_anchors
            or self._global_anchor_owner.get(anchor_id, slot) < slot
        )
        ordered = bool(
            previous_position is not None and previous_position < position
        )
        gap_ok = bool(
            position_gap is not None
            and position_gap >= self.config.independent_min_position_gap
        )
        structure_ok = (
            structural_score >= self.config.independent_structural_min_score
        )
        reported_position_stable = row.get("position_stable_observations")
        if reported_position_stable is None:
            source_state = self.slots[slot]["global"]
            last_position = source_state.get("last_position")
            observed_position_stable = (
                int(source_state.get("position_stable") or 0) + 1
                if last_position is not None
                and abs(position - int(last_position)) <= self.config.position_drift
                else 1
            )
        else:
            observed_position_stable = int(reported_position_stable)
        position_stable = (
            observed_position_stable >= self.config.global_stable
        )
        return {
            "cluster_id": cluster_id,
            "anchor_id": anchor_id,
            "previous_slot_position": previous_position,
            "position_gap": position_gap,
            "anchor_reused": anchor_reused,
            "cluster_reused": cluster_reused,
            "structural_slot_score": structural_score,
            "position_ordered": ordered,
            "position_gap_ok": gap_ok,
            "new_structure_evidence": structure_ok,
            "independent_slot_ready": bool(
                position_stable
                and ordered
                and gap_ok
                and not anchor_reused
                and not cluster_reused
                and structure_ok
            ),
        }

    def _record_global_owners(
        self, rows: Sequence[Optional[Dict[str, object]]]
    ) -> None:
        for slot, row in enumerate(rows):
            if row is None:
                continue
            cluster_id = self._row_identifier(row, "track_id")
            anchor_id = self._row_identifier(row, "anchor_id")
            self._global_cluster_owner.setdefault(cluster_id, slot)
            self._global_anchor_owner.setdefault(anchor_id, slot)

    def _fixed_schema_slot3_completion(
        self, *, seconds: float, step: int
    ) -> None:
        """Resolve the third unique role only for fixed-role benchmarks."""
        if self.benchmark not in {"mmlu", "chronoqa"}:
            return
        if len(self.catalog) != 3:
            return
        first_two = []
        for slot in (0, 1):
            prefetch = self.slots[slot]["prefetch"]
            if prefetch is None:
                return
            first_two.append(str(prefetch["agent"]))
        if len(set(first_two)) != 2:
            return
        remaining = [agent for agent in self.catalog if agent not in first_two]
        if len(remaining) != 1:
            return
        state = self.slots[2]["global"]
        assert isinstance(state, dict)
        if state["first_prediction"] is not None:
            return
        event = {
            "source": "fixed_schema_completion",
            "seconds": float(seconds),
            "step": int(step),
            "agent": remaining[0],
            "candidate_agent": remaining[0],
            "probability": 1.0,
            "margin": 1.0,
            "relative_pos": None,
            "candidate_position": None,
            "cluster_id": None,
            "previous_slot_position": None,
            "position_gap": None,
            "anchor_id": None,
            "anchor_reused": False,
            "cluster_reused": False,
            "structural_slot_score": 1.0,
            "identity_ready": True,
            "independent_slot_ready": True,
            "global_ready": True,
            "ready": True,
            "resolution_source": "fixed_schema_completion",
        }
        state["latest_ready"] = dict(event)
        state["first_prediction"] = dict(event)
        state["events"].append(dict(event))

    def _set_current(
        self, slot: int, agent: Optional[str], source: Optional[str], seconds: float
    ) -> None:
        state = self.slots[slot]
        prior_agent = state["current_prediction"]
        prior_source = state["current_prediction_source"]
        if prior_agent == agent and prior_source == source:
            return
        if prior_agent is not None:
            state["switches"].append({
                "seconds": float(seconds),
                "from_agent": prior_agent,
                "from_source": prior_source,
                "to_agent": agent,
                "to_source": source,
            })
        state["current_prediction"] = agent
        state["current_prediction_source"] = source

    def _choose_first_prefetch(self, slot: int, seconds: float) -> None:
        state = self.slots[slot]
        if state["prefetch"] is not None:
            return
        natural = state["natural"]
        candidates = []
        if natural is not None:
            candidates.append(natural)
        for source in ("local", "global"):
            prediction = state[source]["first_prediction"]
            if prediction is not None:
                candidates.append(prediction)
        if not candidates:
            return
        priority = {
            "natural": 0,
            "local": 1,
            "fixed_schema_completion": 2,
            "global": 2,
        }
        winner = min(
            candidates,
            key=lambda row: (
                float(row["seconds"]), priority.get(str(row["source"]), 9)
            ),
        )
        state["prefetch"] = {
            "agent": winner["agent"],
            "source": winner["source"],
            "seconds": float(winner["seconds"]),
            "step": int(winner["step"]),
            "probability": winner.get("probability"),
            "margin": winner.get("margin"),
            "relative_pos": winner.get("relative_pos"),
        }
        self._set_current(
            slot, str(winner["agent"]), str(winner["source"]), seconds
        )

    def observe_predictions(
        self,
        rows: Sequence[Dict[str, object]],
        *,
        seconds: float,
        step: int,
    ) -> None:
        """Update Global and Local evidence from one existing warmup."""
        normalized: List[Optional[Dict[str, object]]] = [
            dict(rows[slot]) if slot < len(rows) else None
            for slot in range(self.slot_count)
        ]
        diagnostics = [
            self._independent_slot_diagnostics(normalized, slot)
            for slot in range(self.slot_count)
        ]
        for slot in range(self.slot_count):
            row = normalized[slot]
            self._update_source(
                slot, "global", row, seconds=seconds, step=step,
                independent=diagnostics[slot],
            )
            local_row = (
                row
                if row is not None
                and float(row.get("anchor_observed_ratio") or 0.0) >= 1.0
                else None
            )
            self._update_source(
                slot, "local", local_row, seconds=seconds, step=step
            )
        self._record_global_owners(normalized)
        # Slot1/2 remain exactly the old earliest-source fusion.  Their result
        # can then supply the explicitly allowed fixed-schema Slot3 rule.
        for slot in range(min(2, self.slot_count)):
            self._choose_first_prefetch(slot, seconds)
        if self.slot_count >= 3:
            self._fixed_schema_slot3_completion(seconds=seconds, step=step)
        for slot in range(2, self.slot_count):
            self._choose_first_prefetch(slot, seconds)
        for slot in range(self.slot_count):
            state = self.slots[slot]
            if state["natural"] is None:
                local_ready = state["local"]["latest_ready"]
                global_ready = state["global"]["latest_ready"]
                if local_ready is not None:
                    self._set_current(
                        slot, local_ready["agent"], "local", seconds
                    )
                elif global_ready is not None:
                    self._set_current(
                        slot, global_ready["agent"], "global", seconds
                    )
        self.timeline.append({
            "seconds": float(seconds),
            "step": int(step),
            "type": "prediction_observation",
            "slots": [
                {
                    "slot": slot,
                    "current_prediction": state["current_prediction"],
                    "current_prediction_source": state[
                        "current_prediction_source"
                    ],
                }
                for slot, state in enumerate(self.slots)
            ],
        })

    def observe_natural(
        self,
        slot: int,
        agent: str,
        *,
        seconds: float,
        step: int,
        relative_pos: Optional[int] = None,
    ) -> None:
        """Record the first fully materialized catalog Agent occurrence."""
        if not 0 <= slot < self.slot_count:
            return
        state = self.slots[slot]
        if state["natural"] is not None:
            return
        natural = {
            "source": "natural",
            "agent": str(agent),
            "seconds": float(seconds),
            "step": int(step),
            "relative_pos": relative_pos,
            "probability": 1.0,
            "margin": 1.0,
        }
        state["natural"] = natural
        state["authoritative_agent"] = str(agent)
        self._choose_first_prefetch(slot, seconds)
        prefetch = state["prefetch"]
        if prefetch is not None and prefetch["agent"] != agent:
            state["correction"] = {
                "wrong_prefetch": True,
                "wrong_prefetch_start": prefetch["seconds"],
                "corrected_agent": str(agent),
                "T_correction": float(seconds),
                "wrong_prefetch_duration": max(
                    0.0, float(seconds) - float(prefetch["seconds"])
                ),
            }
        self._set_current(slot, str(agent), "natural", seconds)
        self.timeline.append({
            "seconds": float(seconds),
            "step": int(step),
            "type": "natural_decode",
            "slot": int(slot),
            "agent": str(agent),
            "relative_pos": relative_pos,
        })

    @staticmethod
    def _prediction_metrics(prediction, natural):
        if prediction is None or natural is None:
            return {
                "coverage": prediction is not None,
                "correct": None,
                "lead": None,
            }
        return {
            "coverage": True,
            "correct": prediction["agent"] == natural["agent"],
            "lead": float(natural["seconds"]) - float(prediction["seconds"]),
        }

    def _global_error_classification(
        self,
        slot: int,
        prediction: Optional[Dict[str, object]],
        natural_agent: Optional[str],
    ) -> Optional[str]:
        if prediction is None or natural_agent is None:
            return None
        if prediction.get("agent") == natural_agent:
            return None
        if prediction.get("anchor_reused"):
            return "anchor_reuse"
        if prediction.get("cluster_reused"):
            return "cluster_reuse"
        earlier_natural = [
            state["natural"] for state in self.slots[:slot]
            if state.get("natural") is not None
        ]
        if any(
            row.get("agent") == prediction.get("agent")
            for row in earlier_natural
        ):
            return "slot1_identity_carryover"
        candidate_position = prediction.get("candidate_position")
        if candidate_position is not None and any(
            row.get("relative_pos") is not None
            and abs(
                int(candidate_position) - int(row["relative_pos"])
            ) <= self.config.position_drift
            for row in earlier_natural
        ):
            return "wrong_localization"
        return "true_identity_error"

    def metrics(self, final_agents: Sequence[str]) -> Dict[str, object]:
        final3 = list(final_agents[: self.slot_count])
        rows = []
        for slot, state in enumerate(self.slots):
            natural = state["natural"]
            prefetch = state["prefetch"]
            global_prediction = state["global"]["first_prediction"]
            local_prediction = state["local"]["first_prediction"]
            expected = final3[slot] if slot < len(final3) else None
            natural_agent = natural["agent"] if natural else expected
            prefetch_correct = (
                prefetch["agent"] == natural_agent
                if prefetch is not None and natural_agent is not None else None
            )
            lead = (
                float(natural["seconds"]) - float(prefetch["seconds"])
                if natural is not None and prefetch is not None else None
            )
            correction = state["correction"] or {}
            global_events = []
            for event in state["global"]["events"]:
                enriched = dict(event)
                enriched["natural_agent"] = natural_agent
                enriched["correct"] = (
                    enriched.get("agent") == natural_agent
                    if natural_agent is not None else None
                )
                global_events.append(enriched)
            global_error = self._global_error_classification(
                slot, global_prediction, natural_agent
            )
            rows.append({
                "slot": slot,
                "global_candidate": (
                    global_prediction.get("agent") if global_prediction else None
                ),
                "T_global": (
                    global_prediction.get("seconds") if global_prediction else None
                ),
                "S_global": (
                    global_prediction.get("step") if global_prediction else None
                ),
                "global_probability": (
                    global_prediction.get("probability")
                    if global_prediction else None
                ),
                "global_margin": (
                    global_prediction.get("margin") if global_prediction else None
                ),
                "global_resolution_source": (
                    global_prediction.get("resolution_source")
                    or global_prediction.get("source")
                    if global_prediction else None
                ),
                "global_error_classification": global_error,
                "global_candidate_position": (
                    global_prediction.get("candidate_position")
                    if global_prediction else None
                ),
                "global_cluster_id": (
                    global_prediction.get("cluster_id")
                    if global_prediction else None
                ),
                "global_anchor_id": (
                    global_prediction.get("anchor_id")
                    if global_prediction else None
                ),
                "global_correct": (
                    global_prediction.get("agent") == natural_agent
                    if global_prediction is not None and natural_agent is not None
                    else None
                ),
                "global_lead": (
                    float(natural["seconds"])
                    - float(global_prediction["seconds"])
                    if natural is not None and global_prediction is not None
                    else None
                ),
                "local_candidate": (
                    local_prediction.get("agent") if local_prediction else None
                ),
                "T_local": (
                    local_prediction.get("seconds") if local_prediction else None
                ),
                "S_local": (
                    local_prediction.get("step") if local_prediction else None
                ),
                "local_probability": (
                    local_prediction.get("probability") if local_prediction else None
                ),
                "local_margin": (
                    local_prediction.get("margin") if local_prediction else None
                ),
                "local_correct": (
                    local_prediction.get("agent") == natural_agent
                    if local_prediction is not None and natural_agent is not None
                    else None
                ),
                "local_lead": (
                    float(natural["seconds"])
                    - float(local_prediction["seconds"])
                    if natural is not None and local_prediction is not None
                    else None
                ),
                "prefetch_agent": prefetch.get("agent") if prefetch else None,
                "prefetch_source": prefetch.get("source") if prefetch else None,
                "T_prefetch": prefetch.get("seconds") if prefetch else None,
                "S_prefetch": prefetch.get("step") if prefetch else None,
                "prefetch_correct": prefetch_correct,
                "authoritative_agent": natural_agent,
                "natural_decoded_agent": natural.get("agent") if natural else None,
                "T_natural_decode": natural.get("seconds") if natural else None,
                "S_natural_decode": natural.get("step") if natural else None,
                "current_prediction": state["current_prediction"],
                "current_prediction_source": state[
                    "current_prediction_source"
                ],
                "fused_lead": lead,
                "lead_nonnegative": lead is None or lead >= -1e-9,
                "wrong_prefetch": prefetch_correct is False,
                "T_correction": correction.get("T_correction"),
                "corrected_agent": correction.get("corrected_agent"),
                "wrong_prefetch_duration": correction.get(
                    "wrong_prefetch_duration"
                ),
                "T_switch": (
                    state["switches"][0]["seconds"]
                    if state["switches"] else None
                ),
                "switches": list(state["switches"]),
                "global_events": global_events,
                "local_events": list(state["local"]["events"]),
            })

        complete = (
            len(final3) == self.slot_count
            and all(row["T_prefetch"] is not None for row in rows)
            and all(row["T_natural_decode"] is not None for row in rows)
        )
        first_prefetch_tuple = (
            [row["prefetch_agent"] for row in rows] if complete else None
        )
        natural_tuple = (
            [row["natural_decoded_agent"] for row in rows] if complete else final3
        )
        global_tuple = (
            [row["global_candidate"] for row in rows]
            if len(final3) == self.slot_count
            and all(row["global_candidate"] is not None for row in rows)
            else None
        )
        local_tuple = (
            [row["local_candidate"] for row in rows]
            if len(final3) == self.slot_count
            and all(row["local_candidate"] is not None for row in rows)
            else None
        )
        first3_fused = (
            max(float(row["T_prefetch"]) for row in rows) if complete else None
        )
        first3_natural = (
            max(float(row["T_natural_decode"]) for row in rows)
            if complete else None
        )
        first3_lead = (
            first3_natural - first3_fused
            if first3_natural is not None and first3_fused is not None else None
        )
        return {
            "read_only": True,
            "slot_count": self.slot_count,
            "config": {
                "global_stable": self.config.global_stable,
                "global_probability": self.config.global_probability,
                "global_margin": self.config.global_margin,
                "local_stable": self.config.local_stable,
                "local_probability": self.config.local_probability,
                "local_margin": self.config.local_margin,
                "position_drift": self.config.position_drift,
                "independent_min_position_gap": (
                    self.config.independent_min_position_gap
                ),
                "independent_structural_min_score": (
                    self.config.independent_structural_min_score
                ),
            },
            "agent_slots": rows,
            "first3_coverage": complete,
            "global_first3_tuple": global_tuple,
            "global_first3_exact": (
                global_tuple == natural_tuple if global_tuple is not None else None
            ),
            "local_first3_tuple": local_tuple,
            "local_first3_exact": (
                local_tuple == natural_tuple if local_tuple is not None else None
            ),
            "first_prefetch_tuple": first_prefetch_tuple,
            "first_prefetch_first3_exact": (
                first_prefetch_tuple == natural_tuple if complete else None
            ),
            "final_natural_first3_tuple": natural_tuple,
            "final_natural_first3_exact": (
                natural_tuple == final3 if complete else None
            ),
            "T_first3_fused": first3_fused,
            "T_first3_natural": first3_natural,
            "first3_lead": first3_lead,
            "first3_lead_nonnegative": (
                first3_lead is None or first3_lead >= -1e-9
            ),
            "wrong_prefetch_count": sum(
                row["wrong_prefetch"] is True for row in rows
            ),
            "source_winners": {
                source: sum(row["prefetch_source"] == source for row in rows)
                for source in (
                    "global", "fixed_schema_completion", "local", "natural"
                )
            },
            "timeline": list(self.timeline),
        }

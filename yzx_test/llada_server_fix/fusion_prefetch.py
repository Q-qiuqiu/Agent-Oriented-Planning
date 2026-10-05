"""Per-step semantic/prefix/natural Agent commit tracking.

The legacy module name is retained because the server imports it, but the
tracker now has one rule: the first valid commit for each ordered PLAN slot
wins.  It never mutates the decoding canvas and performs no log I/O.
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


def _slot_state() -> Dict[str, object]:
    return {
        "last_semantic_agent": None,
        "stable_count": 0,
        "last_position": None,
        "position_stable_count": 0,
        "latest_semantic": None,
        "commit": None,
        "natural": None,
    }


class FusionAgentPrefetchTracker:
    """Commit any number of ordered Agent slots independently."""

    def __init__(
        self,
        config: FusionGateConfig,
        slot_count: int = 0,
        *,
        catalog: Sequence[str] = (),
    ) -> None:
        for value in (config.global_stable, config.local_stable):
            if value < 1:
                raise ValueError("Fusion stability counts must be positive")
        self.config = config
        self.catalog = tuple(str(agent) for agent in catalog)
        if not self.catalog or len(self.catalog) != len(set(self.catalog)):
            raise ValueError("Agent catalog must be non-empty and unique")
        self.slots: List[Dict[str, object]] = []
        self._prefetched_agents: set[str] = set()
        self._ensure_slots(int(slot_count))

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def committed_slots(self) -> tuple[int, ...]:
        return tuple(
            slot for slot, state in enumerate(self.slots)
            if state["commit"] is not None
        )

    @property
    def natural_slots(self) -> tuple[int, ...]:
        """Slots whose complete natural Agent value has materialized."""
        return tuple(
            slot for slot, state in enumerate(self.slots)
            if state["natural"] is not None
        )

    def _ensure_slots(self, count: int) -> None:
        while len(self.slots) < count:
            self.slots.append(_slot_state())

    @staticmethod
    def _reset_semantic_state(state: Dict[str, object]) -> None:
        state["last_semantic_agent"] = None
        state["stable_count"] = 0
        state["last_position"] = None
        state["position_stable_count"] = 0
        state["latest_semantic"] = None

    def _semantic_thresholds(self, row: Dict[str, object]):
        # Preserve the existing two threshold sets. A naturally materialized
        # JSON key/anchor may use the local thresholds, but Agent value tokens
        # never participate in semantic scoring.
        if float(row.get("anchor_observed_ratio") or 0.0) >= 1.0:
            return (
                self.config.local_stable,
                self.config.local_probability,
                self.config.local_margin,
            )
        return (
            self.config.global_stable,
            self.config.global_probability,
            self.config.global_margin,
        )

    def mark_prefetch(self, agent: str) -> Dict[str, bool]:
        """Record the logical (not model-manager) prefetch state for Agent."""
        if agent not in self.catalog:
            return {
                "prefetch_triggered": False,
                "prefetch_reused": False,
            }
        triggered = agent not in self._prefetched_agents
        if triggered:
            self._prefetched_agents.add(agent)
        return {
            "prefetch_triggered": triggered,
            "prefetch_reused": not triggered,
        }

    def _commit(
        self,
        slot: int,
        agent: str,
        source: str,
        *,
        seconds: float,
        prefix: Optional[str] = None,
        probability: Optional[float] = None,
        margin: Optional[float] = None,
        stable_count: Optional[int] = None,
        strict_semantic_early: Optional[bool] = None,
    ) -> bool:
        self._ensure_slots(slot + 1)
        state = self.slots[slot]
        if state["commit"] is not None:
            return False
        if agent not in self.catalog:
            return False

        # This is the existing logical model-prefetch trigger. One PLAN step
        # commits once; repeated Agent/model names reuse the resident/loading
        # state instead of starting another model load.
        prefetch = self.mark_prefetch(agent)
        state["commit"] = {
            "source": source,
            "agent": agent,
            "seconds": float(seconds),
            "prefix": prefix,
            "probability": probability,
            "margin": margin,
            "stable_count": stable_count,
            "strict_semantic_early": strict_semantic_early,
            **prefetch,
        }
        return True

    def observe_predictions(
        self,
        rows: Sequence[Optional[Dict[str, object]]],
        *,
        seconds: float,
    ) -> None:
        """Apply the existing semantic gates independently to pending slots."""
        self._ensure_slots(len(rows))
        for slot, row in enumerate(rows):
            state = self.slots[slot]
            if row is None or state["commit"] is not None:
                continue
            evidence_ready = bool(row.get("semantic_evidence_ready", False))
            value_fully_masked = bool(
                row.get("agent_value_fully_masked", False)
            )
            if not (
                evidence_ready
                and value_fully_masked
                and bool(row.get("semantic_value_available", False))
            ):
                # Evidence from before this step matured cannot contribute to
                # this slot's later stability.
                self._reset_semantic_state(state)
                continue

            agent = row.get("agent")
            position = row.get("relative_pos")
            probability = row.get("probability")
            margin = row.get("margin")
            if None in (agent, position, probability, margin):
                continue
            agent = str(agent)
            position = int(position)
            if state["last_semantic_agent"] == agent:
                state["stable_count"] = int(state["stable_count"]) + 1
            else:
                state["last_semantic_agent"] = agent
                state["stable_count"] = 1
            if (
                state["last_position"] is not None
                and abs(position - int(state["last_position"]))
                <= self.config.position_drift
            ):
                state["position_stable_count"] = (
                    int(state["position_stable_count"]) + 1
                )
            else:
                state["position_stable_count"] = 1
            state["last_position"] = position
            semantic = {
                "agent": agent,
                "probability": float(probability),
                "margin": float(margin),
                "stable_count": int(state["stable_count"]),
                "position_stable_count": int(state["position_stable_count"]),
                "strict_semantic_early": value_fully_masked,
            }
            state["latest_semantic"] = semantic
            stable_min, probability_min, margin_min = self._semantic_thresholds(row)
            ready = (
                int(state["stable_count"]) >= stable_min
                and int(state["position_stable_count"]) >= stable_min
                and float(probability) >= probability_min
                and float(margin) >= margin_min
            )
            if ready:
                self._commit(
                    slot,
                    agent,
                    "semantic",
                    seconds=float(row.get("seconds", seconds)),
                    probability=float(probability),
                    margin=float(margin),
                    stable_count=int(state["stable_count"]),
                    strict_semantic_early=value_fully_masked,
                )

    def observe_prefix(
        self,
        slot: int,
        prefix: str,
        *,
        seconds: float,
    ) -> bool:
        """Commit a continuous value prefix only when it uniquely matches."""
        self._ensure_slots(slot + 1)
        state = self.slots[slot]
        if state["commit"] is not None:
            return False
        prefix = str(prefix)
        if not prefix:
            return False
        matches = [agent for agent in self.catalog if agent.startswith(prefix)]
        if len(matches) != 1:
            return False
        semantic = state.get("latest_semantic") or {}
        return self._commit(
            slot,
            matches[0],
            "prefix",
            seconds=seconds,
            prefix=prefix,
            probability=semantic.get("probability"),
            margin=semantic.get("margin"),
            stable_count=semantic.get("stable_count"),
        )

    def observe_natural(
        self,
        slot: int,
        agent: str,
        *,
        seconds: float,
    ) -> None:
        """Record full natural reveal and use it only if no earlier commit won."""
        self._ensure_slots(slot + 1)
        state = self.slots[slot]
        if state["natural"] is None:
            state["natural"] = {
                "agent": str(agent),
                "seconds": float(seconds),
            }
        semantic = state.get("latest_semantic") or {}
        self._commit(
            slot,
            str(agent),
            "natural",
            seconds=seconds,
            probability=semantic.get("probability"),
            margin=semantic.get("margin"),
            stable_count=semantic.get("stable_count"),
        )

    @staticmethod
    def _normalize_final_agents(final_steps: Sequence[object]):
        normalized = []
        for value in final_steps:
            if isinstance(value, dict):
                normalized.append(value.get("agent"))
            else:
                normalized.append(value)
        return normalized

    def metrics(self, final_agents: Sequence[object]) -> Dict[str, object]:
        normalized_agents = self._normalize_final_agents(final_agents)
        actual_count = len(normalized_agents)
        if not actual_count:
            actual_count = max(
                (
                    index + 1
                    for index, state in enumerate(self.slots)
                    if state["natural"] is not None or state["commit"] is not None
                ),
                default=0,
            )
            normalized_agents = [
                (self.slots[index]["natural"] or {}).get("agent")
                for index in range(actual_count)
            ]

        rows = []
        source_counts = {"semantic": 0, "prefix": 0, "natural": 0}
        for slot in range(actual_count):
            self._ensure_slots(slot + 1)
            state = self.slots[slot]
            commit = state["commit"] or {}
            natural = state["natural"] or {}
            semantic = state["latest_semantic"] or {}
            source = commit.get("source")
            if source in source_counts:
                source_counts[source] += 1
            committed_agent = commit.get("agent")
            natural_agent = natural.get("agent")
            correct = (
                committed_agent == natural_agent
                if committed_agent is not None and natural_agent is not None
                else None
            )
            natural_seconds = natural.get("seconds")
            commit_seconds = commit.get("seconds")
            lead = (
                float(natural_seconds) - float(commit_seconds)
                if natural_seconds is not None and commit_seconds is not None
                else None
            )
            rows.append(
                {
                    "slot_id": slot,
                    "commit_source": source,
                    "committed_agent": committed_agent,
                    "natural_agent": natural_agent,
                    "commit_wall_time": commit_seconds,
                    "natural_agent_wall_time": natural_seconds,
                    "prefix": commit.get("prefix"),
                    "semantic_confidence": (
                        commit.get("probability", semantic.get("probability"))
                    ),
                    "semantic_margin": (
                        commit.get("margin", semantic.get("margin"))
                    ),
                    "stable_count": (
                        commit.get("stable_count", semantic.get("stable_count"))
                    ),
                    "correct": correct,
                    "strict_semantic_early": (
                        commit.get("strict_semantic_early")
                        if source == "semantic" else None
                    ),
                    "lead_seconds": lead,
                    "prefetch_triggered": commit.get("prefetch_triggered"),
                    "prefetch_reused": commit.get("prefetch_reused"),
                }
            )

        commit_times = [
            float(row["commit_wall_time"])
            for row in rows
            if row["commit_wall_time"] is not None
        ]
        return {
            "read_only": True,
            "agent_count": actual_count,
            "semantic_commit_count": source_counts["semantic"],
            "prefix_commit_count": source_counts["prefix"],
            "natural_commit_count": source_counts["natural"],
            "first_agent_commit_time": min(commit_times) if commit_times else None,
            "all_agents_commit_time": (
                max(commit_times)
                if actual_count > 0 and len(commit_times) == actual_count
                else None
            ),
            "agent_slots": rows,
        }

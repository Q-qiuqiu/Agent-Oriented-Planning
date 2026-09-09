"""Passive Agent timing derived from an unchanged final planner response."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from json_agent_priority import JsonAgentFieldController


BENCHMARK_AGENT_REGISTRIES = {
    "huskyqa": frozenset(
        ("search_agent", "calculation_agent", "reasoning_agent")
    ),
    "iirc": frozenset(
        ("context_agent", "retrieval_agent", "reasoning_agent")
    ),
    "mmlu": frozenset(
        ("knowledge_agent", "reasoning_agent", "elimination_agent")
    ),
    "chronoqa": frozenset(
        ("evidence_agent", "temporal_agent", "verification_agent")
    ),
}


def infer_benchmark(agent_registry: Sequence[str]) -> Optional[str]:
    names = frozenset(str(name).strip().lower() for name in agent_registry)
    for benchmark, expected in BENCHMARK_AGENT_REGISTRIES.items():
        if names == expected:
            return benchmark
    return None


class PassiveJsonAgentMonitor(JsonAgentFieldController):
    """Observe naturally materialized Agent fields without changing decoding."""

    full_sequence_discovery_steps = 0

    def _record_materialized(self, x, global_step):
        candidates = self._materialized_anchor_candidates(x)
        self._assign_anchors(x, candidates)
        now = self._elapsed()
        for slot_index, runtime in enumerate(self.slots):
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is None:
                continue
            if runtime.first_observed_seconds is None:
                runtime.first_observed_seconds = now
                runtime.first_observed_step = global_step
            if runtime.recognized_seconds is None:
                runtime.recognized_seconds = now
                runtime.recognized_step = global_step
                self.logger.info(
                    "base_agent_observed %s",
                    json.dumps(
                        {
                            "slot": slot_index,
                            "agent": observed_name,
                            "seconds": now,
                            "step": global_step,
                            "fuzzy_matched_from": runtime.fuzzy_matched_from,
                        },
                        sort_keys=True,
                    ),
                )
            if runtime.confirmed_seconds is None:
                runtime.confirmed_seconds = now
                runtime.confirmed_step = global_step
            runtime.candidate = observed_name
            runtime.recognized_candidate = observed_name
            runtime.candidate_probability = 1.0
            runtime.candidate_margin = 1.0
            runtime.recognized_probability = 1.0
            runtime.recognized_margin = 1.0
            runtime.confirmed = True

    def observe(
        self,
        logits,
        x,
        logits_start,
        global_step,
        is_last_agent_step=False,
    ):
        del logits, logits_start, is_last_agent_step
        self._observed_steps += 1
        self._record_materialized(x, global_step)

    def step_callback(self, nfe, num_block, block_step, x):
        """Observe the canvas immediately after a normal decoder update."""
        del num_block, block_step
        self._record_materialized(x, int(nfe))

    def decoder_mask(self, mask_index, mask_start=0):
        del mask_start
        return mask_index

    def finalize(self, x):
        self._record_materialized(x, self._observed_steps)

    def metrics(self):
        result = super().metrics()
        result["policy"] = "base"
        result["timing_source"] = "passive_materialized_response"
        result["monitor_overhead_included"] = True
        for slot in result["agent_slots"]:
            slot["priority"] = False
        return result

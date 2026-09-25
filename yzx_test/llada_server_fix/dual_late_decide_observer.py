"""Read-only Global + Local + Natural fusion on Dual Vanilla warmups."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from fusion_prefetch import FusionAgentPrefetchTracker, FusionGateConfig
from json_agent_priority import JsonAgentPriorityConfig, JsonAgentSlotRuntime
from ordered_plan_observer import OrderedPlanAgentObserver
from response_agent_timing import PassiveJsonAgentMonitor


class DualVanillaLateDecideObserver(PassiveJsonAgentMonitor):
    """Observe existing Dual warmups and never mutate generation state."""

    full_sequence_discovery_steps = 0

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        priority_slots: int,
        tracking_slots: int,
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        anchor_min_logit_margin: float = -6.0,
        benchmark: Optional[str] = None,
        fusion_global_stable: int = 2,
        fusion_global_probability: float = 0.90,
        fusion_global_margin: float = 0.40,
        fusion_local_stable: int = 2,
        fusion_local_probability: float = 0.75,
        fusion_local_margin: float = 0.15,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            config=JsonAgentPriorityConfig(
                catalog=list(catalog),
                priority_slots=priority_slots,
                tracking_slots=tracking_slots,
                probe_period=0,
            ),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
        )
        self.fusion = FusionAgentPrefetchTracker(
            FusionGateConfig(
                global_stable=fusion_global_stable,
                global_probability=fusion_global_probability,
                global_margin=fusion_global_margin,
                local_stable=fusion_local_stable,
                local_probability=fusion_local_probability,
                local_margin=fusion_local_margin,
                position_drift=4,
            ),
            benchmark=benchmark,
            catalog=list(catalog),
        )
        self.ordered = OrderedPlanAgentObserver(
            tokenizer=tokenizer,
            catalog=list(catalog),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
            elapsed=self._elapsed,
            anchor_min_logit_margin=anchor_min_logit_margin,
        )

    def initialize(self, x: torch.Tensor) -> None:
        super().initialize(x)
        self.ordered.initialize(x)

    def has_unconfirmed_agents(self) -> bool:
        return False

    def decoder_mask(self, mask_index, mask_start=0):
        del mask_start
        return mask_index

    def _sync_natural_from_canvas(self, x: torch.Tensor, step: int) -> None:
        rows = []
        for anchor_start, pattern, _score, _ratio in (
            self._materialized_anchor_candidates(x)
        ):
            runtime = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=anchor_start + len(pattern),
            )
            agent = self._observed_catalog_value(x, runtime)
            if agent is not None:
                rows.append((int(anchor_start), str(agent)))
        now = float(self._elapsed())
        for slot, (anchor_start, agent) in enumerate(sorted(rows)[:3]):
            self.fusion.observe_natural(
                slot, agent, seconds=now, step=int(step),
                relative_pos=anchor_start - self.prompt_length,
            )

    def _sync_natural_slots(self) -> None:
        for slot, runtime in enumerate(self.slots[:3]):
            agent = runtime.materialized_candidate
            seconds = runtime.materialized_seconds
            step = runtime.materialized_step
            if agent is None and runtime.confirmed:
                agent = runtime.recognized_candidate or runtime.candidate
                seconds = runtime.confirmed_seconds
                step = runtime.confirmed_step
            if agent is None or seconds is None or step is None:
                continue
            self.fusion.observe_natural(
                slot, str(agent), seconds=float(seconds), step=int(step),
                relative_pos=(
                    int(runtime.anchor_start) - self.prompt_length
                    if runtime.anchor_start is not None else None
                ),
            )

    def observe(
        self, logits, x, logits_start, global_step, is_last_agent_step=False
    ):
        del is_last_agent_step
        self._observed_steps += 1
        self._record_materialized(x, global_step)
        self._sync_natural_from_canvas(x, global_step)
        self._sync_natural_slots()
        if logits_start != 0 or logits.shape[1] != x.shape[1]:
            return
        self._full_sequence_observations += 1
        self.ordered.observe(
            logits,
            x,
            logits_start=0,
            global_step=global_step,
            plan_start=self.prompt_length,
            plan_end=min(x.shape[1], self.prompt_length + self.gen_length),
            phase="dual_vanilla",
        )
        if self.ordered.events:
            event = self.ordered.events[-1]
            self.fusion.observe_predictions(
                event.get("slots") or [],
                seconds=float(event["seconds"]),
                step=int(event["step"]),
            )

    def step_callback(self, nfe, num_block, block_step, x):
        del num_block, block_step
        self._record_materialized(x, int(nfe))
        self._sync_natural_from_canvas(x, int(nfe))
        self._sync_natural_slots()

    def finalize(self, x: torch.Tensor) -> None:
        super().finalize(x)
        self._sync_natural_from_canvas(x, self._observed_steps)
        self._sync_natural_slots()

    def metrics(self):
        result = super().metrics()
        natural_slots = result.get("agent_slots") or []
        final_agents = [
            slot.get("materialized_candidate") or slot.get("agent")
            for slot in natural_slots
            if slot.get("materialized_candidate") or slot.get("agent")
        ]
        fusion = self.fusion.metrics(final_agents)
        fusion_slots = fusion.get("agent_slots") or []
        for index, source in enumerate(fusion_slots):
            if index >= len(natural_slots):
                break
            natural_slots[index].update({
                "prefetched_agent": source.get("prefetch_agent"),
                "prefetch_source": source.get("prefetch_source"),
                "T_slot_prefetch": source.get("T_prefetch"),
                "prefetch_correct": source.get("prefetch_correct"),
                "global_candidate": source.get("global_candidate"),
                "T_global": source.get("T_global"),
                "local_candidate": source.get("local_candidate"),
                "T_local": source.get("T_local"),
            })
        result.update({
            "policy": "commit",
            "timing_source": "dual_vanilla_global_local_natural_fusion",
            "read_only": True,
            "agent_slots": natural_slots,
            "final_agent_sequence": final_agents,
            "first3_tuple": final_agents[:3],
            "fusion_prefetch": fusion,
            "probe_forwards": 0,
        })
        return result

    def close(self) -> None:
        self.ordered.close()

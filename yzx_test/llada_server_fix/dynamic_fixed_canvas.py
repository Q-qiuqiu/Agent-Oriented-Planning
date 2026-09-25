"""Dynamic-END fixed canvas for reasoning/PLAN scheduling experiments.

Only the two opening markers are fixed.  END_PLANNING_REASONING and
END_PLAN_JSON must be produced by the model through the ordinary LLaDA
proposal/transfer path.  Once an END marker is materialized, positions after
it are removed from the active region and are physically compacted away after
all preceding holes have been filled.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import JsonAgentFieldController, JsonAgentPriorityConfig
from planner_json_repair import _plan_region, _strict_plan, _validate_plan
from response_agent_timing import PassiveJsonAgentMonitor


@dataclass
class DynamicFixedCanvasLayout:
    prompt_length: int
    gen_length: int
    reasoning_budget: int
    plan_budget: int
    prefix_ids: Tuple[int, ...]
    middle_ids: Tuple[int, ...]
    reasoning_actual_length: Optional[int] = None
    plan_actual_length: Optional[int] = None
    reasoning_compacted: bool = False
    plan_compacted: bool = False

    @property
    def generation_start(self) -> int:
        return self.prompt_length

    @property
    def prefix_start(self) -> int:
        return self.generation_start

    @property
    def reasoning_start(self) -> int:
        return self.prefix_start + len(self.prefix_ids)

    @property
    def reasoning_storage_length(self) -> int:
        if self.reasoning_compacted and self.reasoning_actual_length is not None:
            return self.reasoning_actual_length
        return self.reasoning_budget

    @property
    def reasoning_storage_end(self) -> int:
        return self.reasoning_start + self.reasoning_storage_length

    @property
    def reasoning_transfer_end(self) -> int:
        length = self.reasoning_actual_length or self.reasoning_budget
        return self.reasoning_start + length

    @property
    def middle_start(self) -> int:
        return self.reasoning_storage_end

    @property
    def plan_start(self) -> int:
        return self.middle_start + len(self.middle_ids)

    @property
    def plan_storage_length(self) -> int:
        if self.plan_compacted and self.plan_actual_length is not None:
            return self.plan_actual_length
        return self.plan_budget

    @property
    def plan_storage_end(self) -> int:
        return self.plan_start + self.plan_storage_length

    @property
    def plan_transfer_end(self) -> int:
        length = self.plan_actual_length or self.plan_budget
        return self.plan_start + length

    @property
    def generation_end(self) -> int:
        return self.plan_storage_end

    @property
    def initial_canvas_length(self) -> int:
        return (
            len(self.prefix_ids)
            + self.reasoning_budget
            + len(self.middle_ids)
            + self.plan_budget
        )

    @property
    def compacted(self) -> bool:
        return self.reasoning_compacted or self.plan_compacted

    def validate(self) -> None:
        if self.reasoning_budget <= 0 or self.plan_budget <= 0:
            raise ValueError("Reasoning and PLAN capacities must be positive.")
        if self.initial_canvas_length > self.gen_length:
            raise ValueError(
                "Fixed canvas exceeds gen_length: "
                f"{self.initial_canvas_length} > {self.gen_length}."
            )

    def initialize(self, x: torch.Tensor) -> None:
        self.validate()
        for start, ids in (
            (self.prefix_start, self.prefix_ids),
            (self.middle_start, self.middle_ids),
        ):
            values = torch.tensor(ids, dtype=x.dtype, device=x.device)
            x[:, start : start + len(ids)] = values

    def set_marker_boundary(self, phase: str, marker_end: int) -> None:
        if phase == "reasoning":
            length = marker_end - self.reasoning_start
            if not (0 < length <= self.reasoning_budget):
                raise ValueError(f"Invalid reasoning END boundary {marker_end}.")
            self.reasoning_actual_length = int(length)
        elif phase == "plan":
            length = marker_end - self.plan_start
            if not (0 < length <= self.plan_budget):
                raise ValueError(f"Invalid PLAN END boundary {marker_end}.")
            self.plan_actual_length = int(length)
        else:
            raise ValueError(f"Unknown phase {phase!r}.")

    def mark_compacted(self, phase: str) -> None:
        if phase == "reasoning":
            if self.reasoning_actual_length is None:
                raise ValueError("Cannot compact reasoning before its END marker.")
            self.reasoning_compacted = True
        elif phase == "plan":
            if self.plan_actual_length is None:
                raise ValueError("Cannot compact PLAN before its END marker.")
            self.plan_compacted = True
        else:
            raise ValueError(f"Unknown phase {phase!r}.")

    def bool_mask(self, x: torch.Tensor, region: str) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        if region == "reasoning":
            mask[:, self.reasoning_start : min(x.shape[1], self.reasoning_transfer_end)] = True
        elif region == "plan":
            mask[:, self.plan_start : min(x.shape[1], self.plan_transfer_end)] = True
        elif region == "fixed":
            mask[:, self.prefix_start : self.reasoning_start] = True
            mask[:, self.middle_start : self.plan_start] = True
        else:
            raise ValueError(f"Unknown region {region!r}.")
        return mask

    def fixed_ids(self) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for start, ids in (
            (self.prefix_start, self.prefix_ids),
            (self.middle_start, self.middle_ids),
        ):
            result.extend((start + offset, int(token)) for offset, token in enumerate(ids))
        return result


@dataclass(frozen=True)
class DynamicPlanCompletion:
    plan: List[Dict[str, object]]
    marker_start_offset: int
    marker_end_offset: int
    token_length: int
    tokens_after_json: int


class DynamicFixedCanvasMonitor(PassiveJsonAgentMonitor):
    """Materialization monitor and dynamic END-boundary controller."""

    def __init__(
        self,
        *,
        tokenizer,
        catalog: Sequence[str],
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        reasoning_budget: Optional[int] = None,
        plan_budget: Optional[int] = None,
        reasoning_ratio: float = 0.5,
        plan_ratio: float = 0.5,
        priority_slots: int = 3,
        tracking_slots: int = 16,
        structure_mode: str,
        agent_commit: bool = False,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            config=JsonAgentPriorityConfig(
                catalog=list(catalog),
                priority_slots=priority_slots,
                tracking_slots=max(priority_slots, tracking_slots),
                probe_period=0,
            ),
            prompt_length=prompt_length,
            gen_length=gen_length,
            mask_id=mask_id,
        )
        encode = lambda value: tuple(tokenizer.encode(value, add_special_tokens=False))
        prefix_ids = encode("PLANNING_REASONING\n")
        middle_ids = encode("\nPLAN_JSON\n")
        available = gen_length - len(prefix_ids) - len(middle_ids)
        if reasoning_budget is None and plan_budget is None:
            ratio_sum = reasoning_ratio + plan_ratio
            reasoning_budget = int(round(available * reasoning_ratio / ratio_sum))
            plan_budget = available - reasoning_budget
        elif reasoning_budget is None:
            plan_budget = int(plan_budget)
            reasoning_budget = available - plan_budget
        elif plan_budget is None:
            reasoning_budget = int(reasoning_budget)
            plan_budget = available - reasoning_budget
        else:
            reasoning_budget = int(reasoning_budget)
            plan_budget = int(plan_budget)
        if reasoning_budget + plan_budget > available:
            raise ValueError(
                f"Explicit capacities {reasoning_budget}+{plan_budget} exceed {available}."
            )
        self.layout = DynamicFixedCanvasLayout(
            prompt_length=prompt_length,
            gen_length=gen_length,
            reasoning_budget=reasoning_budget,
            plan_budget=plan_budget,
            prefix_ids=prefix_ids,
            middle_ids=middle_ids,
        )
        self.layout.validate()
        self.structure_mode = structure_mode
        self.agent_commit = bool(agent_commit)
        self.agent_observer = None
        if self.agent_commit:
            # ``all`` reuses normal PLAN warmups only and remains read-only.
            self.agent_observer = JsonAgentFieldController(
                tokenizer=tokenizer,
                config=JsonAgentPriorityConfig(
                    catalog=list(catalog),
                    priority_slots=priority_slots,
                    tracking_slots=max(priority_slots, tracking_slots),
                    tentative_probability=0.90,
                    tentative_margin=0.40,
                    allow_speculative_anchor_commit=True,
                    probe_period=0,
                ),
                prompt_length=prompt_length,
                gen_length=gen_length,
                mask_id=mask_id,
            )
        # Leading-newline token merges differ across tokenizers, so match both
        # forms while still requiring the complete materialized marker.
        self.reasoning_end_patterns = tuple(dict.fromkeys((
            encode("\nEND_PLANNING_REASONING"),
            encode(" END_PLANNING_REASONING"),
            encode("END_PLANNING_REASONING"),
        )))
        self.plan_end_patterns_dynamic = tuple(dict.fromkeys((
            encode("\nEND_PLAN_JSON"),
            encode(" END_PLAN_JSON"),
            encode("END_PLAN_JSON"),
        )))
        self.phase: Optional[str] = None
        self.phase_times: Dict[str, Optional[float]] = {
            "plan_phase_start": None,
            "reasoning_phase_start": None,
            "plan_region_complete": None,
            "reasoning_region_complete": None,
            "generation_complete": None,
        }
        self.marker_offsets: Dict[str, Optional[Tuple[int, int]]] = {
            "reasoning": None,
            "plan": None,
        }
        self.end_marker_seconds: Dict[str, Optional[float]] = {
            "reasoning": None,
            "plan": None,
        }
        self.capacity_exhausted = {"reasoning": False, "plan": False}
        self.phase_valid = {"reasoning": False, "plan": False}
        self.invalid_at_end = {"reasoning": False, "plan": False}
        self.plan_completion: Optional[DynamicPlanCompletion] = None
        self.plan_parseable_seconds: Optional[float] = None
        self.plan_json_complete_seconds: Optional[float] = None
        self.plan_json_start_materialized_seconds: Optional[float] = 0.0
        self._fixed_reference: List[Tuple[int, int]] = []
        self._snapshots: List[Dict[str, object]] = []
        self._final_occurrences: List[Tuple[int, str]] = []
        self._final_plan = None
        self._evaluation_plan = None
        self._last_step = 0
        self._nfe = 0

    def initialize(self, x: torch.Tensor) -> None:
        self.layout.initialize(x)
        super().initialize(x)
        if self.agent_observer is not None:
            self.agent_observer.initialize(x)
            self.agent_observer.set_search_region(
                self.layout.plan_start, self.layout.plan_storage_end
            )
        self.probe_forwards = 0
        self._fixed_reference = self.layout.fixed_ids()
        self._snapshot(x, step=0, nfe=0)

    def observe_plan_logits(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        *,
        logits_start: int,
        global_step: int,
    ) -> None:
        """Observe or commit confidence-gated Agent values inside PLAN."""

        if not self.agent_commit:
            return
        # Compaction can move the PLAN capacity, so refresh its exact bounds.
        # Localization stays inside the known PLAN region. In ``all`` the
        # nested controller is prediction-only; in fixed-PLAN ``commit`` it
        # writes and freezes values through its normal decoder-mask path.
        self.agent_observer.set_search_region(
            self.layout.plan_start, self.layout.plan_storage_end
        )
        self.agent_observer.observe(
            logits,
            x,
            logits_start=logits_start,
            global_step=global_step,
        )

    def decoder_mask(self, mask_index, mask_start=0):
        result = mask_index.clone()
        mask_end = mask_start + result.shape[1]
        for fixed_start, fixed_end in (
            (self.layout.prefix_start, self.layout.reasoning_start),
            (self.layout.middle_start, self.layout.plan_start),
        ):
            overlap_start = max(mask_start, fixed_start)
            overlap_end = min(mask_end, fixed_end)
            if overlap_start < overlap_end:
                result[:, overlap_start - mask_start : overlap_end - mask_start] = False
        if self.agent_observer is not None:
            result = self.agent_observer.decoder_mask(result, mask_start=mask_start)
        return result

    @property
    def observes_plan_warmups(self) -> bool:
        return self.agent_commit

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        key = f"{phase}_phase_start"
        if self.phase_times[key] is None:
            self.phase_times[key] = self._elapsed()

    def _phase_start(self, phase: str) -> int:
        return self.layout.reasoning_start if phase == "reasoning" else self.layout.plan_start

    def _phase_storage_end(self, phase: str) -> int:
        return (
            self.layout.reasoning_storage_end
            if phase == "reasoning"
            else self.layout.plan_storage_end
        )

    def _phase_marker_patterns(self, phase: str) -> Tuple[Tuple[int, ...], ...]:
        return (
            self.reasoning_end_patterns
            if phase == "reasoning"
            else self.plan_end_patterns_dynamic
        )

    def _find_marker(self, x: torch.Tensor, phase: str) -> Optional[Tuple[int, int]]:
        start = self._phase_start(phase)
        end = min(x.shape[1], self._phase_storage_end(phase))
        sequence = x[0, start:end]
        candidates = []
        for pattern in self._phase_marker_patterns(phase):
            if not pattern or sequence.shape[0] < len(pattern):
                continue
            target = x.new_tensor(pattern)
            matches = (sequence.unfold(0, len(pattern), 1) == target).all(dim=1)
            positions = matches.nonzero(as_tuple=False).flatten()
            if positions.numel():
                marker_start = start + int(positions[0].item())
                candidates.append((marker_start, marker_start + len(pattern)))
        return min(candidates, key=lambda item: (item[0], -item[1])) if candidates else None

    def observe_end_marker(self, x: torch.Tensor, phase: str) -> bool:
        if self.marker_offsets[phase] is not None:
            return True
        match = self._find_marker(x, phase)
        if match is None:
            return False
        region_start = self._phase_start(phase)
        marker_start, marker_end = match
        self.marker_offsets[phase] = (
            marker_start - region_start,
            marker_end - region_start,
        )
        self.layout.set_marker_boundary(phase, marker_end)
        self.end_marker_seconds[phase] = self._elapsed()
        return True

    def phase_remaining_masks(self, x: torch.Tensor, phase: str) -> int:
        return int(((x == self.mask_id) & self.layout.bool_mask(x, phase)).sum().item())

    def phase_ready(self, x: torch.Tensor, phase: str) -> bool:
        return self.marker_offsets[phase] is not None and self.phase_remaining_masks(x, phase) == 0

    @staticmethod
    def _raw_json_array(text: str, catalog: Optional[Sequence[str]] = None):
        # The complete materialized section, up to the natural END marker,
        # must be a single JSON array.  A valid prefix followed by generated
        # garbage is not a complete PLAN.
        try:
            decoder = json.JSONDecoder()
            prefix = len(text) - len(text.lstrip())
            value, end = decoder.raw_decode(text, idx=prefix)
            if text[end:].strip() or not isinstance(value, list):
                return None, None
            _validate_plan(value, catalog or ())
            return value, end
        except (json.JSONDecodeError, ValueError, TypeError):
            return None, None

    def _decode_ids(self, ids: torch.Tensor, *, skip_special_tokens=True) -> str:
        visible = ids[ids != self.mask_id].detach().cpu().tolist()
        return self.tokenizer.decode(visible, skip_special_tokens=skip_special_tokens)

    def _phase_payload(self, x: torch.Tensor, phase: str) -> Tuple[torch.Tensor, str]:
        marker = self.marker_offsets[phase]
        if marker is None:
            start, end = self._phase_start(phase), self._phase_storage_end(phase)
        else:
            start = self._phase_start(phase)
            end = start + marker[0]
        ids = x[0, start:end]
        return ids, self._decode_ids(ids, skip_special_tokens=False)

    def _compact_phase(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        marker = self.marker_offsets[phase]
        if marker is None:
            return x
        region_start = self._phase_start(phase)
        marker_end = region_start + marker[1]
        old_storage_end = self._phase_storage_end(phase)
        removed = old_storage_end - marker_end
        x = torch.cat([x[:, :marker_end], x[:, old_storage_end:]], dim=1)
        if (
            phase == "reasoning"
            and removed > 0
            and self.agent_observer is not None
        ):
            # PLAN tokens (including any speculative commits) move left with
            # the physical compaction. Keep observer/freeze coordinates aligned
            # without resetting their cross-warmup stability state.
            self.agent_observer.shift_positions(old_storage_end, -removed)
        self.layout.mark_compacted(phase)
        self._fixed_reference = self.layout.fixed_ids()
        return x

    def finish_phase(self, x: torch.Tensor, phase: str) -> Tuple[torch.Tensor, bool]:
        """Validate a naturally terminated region and compact its unused tail."""
        if not self.phase_ready(x, phase):
            return x, False
        payload_ids, payload_text = self._phase_payload(x, phase)
        now = self._elapsed()
        if phase == "reasoning":
            self.phase_valid[phase] = bool(payload_text.strip())
            self.invalid_at_end[phase] = not self.phase_valid[phase]
            if self.phase_valid[phase]:
                self.phase_times["reasoning_region_complete"] = now
        else:
            plan, char_end = self._raw_json_array(payload_text, self.config.catalog)
            self.phase_valid[phase] = plan is not None
            self.invalid_at_end[phase] = plan is None
            if plan is not None:
                valid_json_text = payload_text[:char_end]
                json_tokens = len(
                    self.tokenizer.encode(valid_json_text, add_special_tokens=False)
                )
                self.plan_completion = DynamicPlanCompletion(
                    plan=plan,
                    marker_start_offset=int(payload_ids.shape[0]),
                    marker_end_offset=int(self.marker_offsets[phase][1]),
                    token_length=int(payload_ids.shape[0]),
                    tokens_after_json=max(0, int(payload_ids.shape[0]) - json_tokens),
                )
                self.plan_json_complete_seconds = now
                self.plan_parseable_seconds = now
                self.phase_times["plan_region_complete"] = now
        return self._compact_phase(x, phase), True

    def mark_capacity_exhausted(self, phase: str) -> None:
        self.capacity_exhausted[phase] = True
        key = f"{phase}_region_complete"
        if self.phase_times[key] is None:
            self.phase_times[key] = self._elapsed()

    def _plan_json_bounds(self, x):
        del x
        start = self.layout.plan_start
        marker = self.marker_offsets["plan"]
        end = start + marker[0] if marker is not None else self.layout.plan_transfer_end
        return start, end

    def _current_occurrences(self, x: torch.Tensor) -> List[Tuple[int, str]]:
        candidates = self._plan_agent_candidates(x)
        self._assign_anchors(x, candidates)
        values = []
        for runtime in self.slots:
            if runtime.anchor_start is None:
                continue
            name = self._observed_catalog_value(x, runtime)
            if name is not None:
                values.append((int(runtime.anchor_start - self.layout.plan_start), str(name)))
        return sorted(set(values), key=lambda item: item[0])

    def _snapshot(self, x: torch.Tensor, *, step: int, nfe: int) -> None:
        occurrences = self._current_occurrences(x)
        self._snapshots.append({
            "step": int(step),
            "nfe": int(nfe),
            "seconds": float(self._elapsed()),
            "phase": self.phase,
            "plan_masks": self.phase_remaining_masks(x, "plan"),
            "reasoning_masks": self.phase_remaining_masks(x, "reasoning"),
            "reasoning_end_found": self.marker_offsets["reasoning"] is not None,
            "plan_end_found": self.marker_offsets["plan"] is not None,
            "materialized_agents": [
                {"plan_offset": offset, "agent": name} for offset, name in occurrences
            ],
        })
        self._last_step = int(step)
        self._nfe = int(nfe)

    def record_step(self, x, *, global_step, nfe, physical_block, local_step) -> None:
        del physical_block, local_step
        self._snapshot(x, step=global_step, nfe=nfe)

    @staticmethod
    def _first_cover_time(snapshots, targets):
        if not targets:
            return None, None
        target_set = set(targets)
        for row in snapshots:
            observed = {
                (int(item["plan_offset"]), str(item["agent"]))
                for item in row["materialized_agents"]
            }
            if target_set.issubset(observed):
                return row["seconds"], row["step"]
        return None, None

    def finalize(self, x: torch.Tensor) -> None:
        self._snapshot(x, step=self._last_step, nfe=self._nfe)
        self.phase_times["generation_complete"] = self._elapsed()
        self._final_occurrences = self._current_occurrences(x)
        plan_ids, plan_text = self._phase_payload(x, "plan")
        reasoning_ids, reasoning_text = self._phase_payload(x, "reasoning")
        if self.phase_remaining_masks(x, "plan") == 0:
            self._final_plan, self._json_end = self._raw_json_array(
                plan_text, self.config.catalog
            )
        else:
            self._final_plan, self._json_end = None, None
        self._plan_text = plan_text
        self._reasoning_text = reasoning_text
        self._plan_payload_ids = plan_ids.detach().cpu()
        self._reasoning_payload_ids = reasoning_ids.detach().cpu()
        self._final_ids = x[0, self.prompt_length :].detach().cpu()
        if self.agent_observer is not None:
            # Attribute speculative writes to anchors/values that actually
            # survived in the completed PLAN. This is evaluation-only and
            # happens after generation has ended.
            self.agent_observer.finalize(x)

    def set_evaluation_plan_text(self, content: str) -> None:
        """Set correctness GT from the final response returned to the client."""
        try:
            _start, _end, region = _plan_region(content)
            plan = _strict_plan(region)
            _validate_plan(plan, self.config.catalog)
        except (ValueError, TypeError, json.JSONDecodeError):
            self._evaluation_plan = None
            return
        self._evaluation_plan = plan

    def metrics(self) -> Dict[str, object]:
        final = self._final_occurrences
        first_time, first_step = self._first_cover_time(self._snapshots, final[:1])
        first3_time, first3_step = self._first_cover_time(
            self._snapshots, final[: min(3, len(final))]
        )
        all_time, all_step = self._first_cover_time(self._snapshots, final)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
        reasoning_tokens = getattr(self, "_reasoning_payload_ids", torch.tensor([])).tolist()
        plan_tokens = getattr(self, "_plan_payload_ids", torch.tensor([])).tolist()
        reasoning_effective = sum(
            int(token not in special_ids and token != self.mask_id)
            for token in reasoning_tokens
        )
        plan_effective = sum(
            int(token not in special_ids and token != self.mask_id)
            for token in plan_tokens
        )
        reasoning_used = self.layout.reasoning_actual_length or self.layout.reasoning_budget
        plan_used = self.layout.plan_actual_length or self.layout.plan_budget
        fixed_corruption = 0
        for position, token in self._fixed_reference:
            relative = position - self.prompt_length
            if relative < 0 or relative >= len(self._final_ids):
                fixed_corruption += 1
            else:
                fixed_corruption += int(int(self._final_ids[relative]) != token)
        sentences = [
            value.strip().lower()
            for value in re.split(r"(?<=[.!?。！？])\s+", self._reasoning_text.strip())
            if value.strip()
        ]
        repeated = len(sentences) - len(set(sentences))
        extra_plan_tokens = None
        if self._json_end is not None:
            extra_plan_tokens = len(
                self.tokenizer.encode(
                    self._plan_text[self._json_end :], add_special_tokens=False
                )
            )
        slots = []
        for slot, (offset, name) in enumerate(final):
            target_time, target_step = self._first_cover_time(
                self._snapshots, [(offset, name)]
            )
            slots.append({
                "slot": slot,
                "anchor_start": self.layout.plan_start + offset,
                "plan_offset": offset,
                "agent": name,
                "priority": False,
                "recognized_seconds": target_time,
                "recognized_step": target_step,
                "confirmed_seconds": target_time,
                "confirmed_step": target_step,
                "materialized_seconds": target_time,
                "materialized_step": target_step,
                "materialized_candidate": name,
                "confirmed": target_time is not None,
            })
        agents = [name for _, name in final]
        evaluation_plan = (
            self._evaluation_plan
            if self._evaluation_plan is not None else self._final_plan
        )
        parsed_agents = (
            [str(item["agent"]) for item in evaluation_plan]
            if evaluation_plan is not None else []
        )
        result = {
            "policy": self.structure_mode,
            "structure_mode": self.structure_mode,
            "timing_source": "dynamic_end_fixed_canvas_materialized_x",
            "catalog": list(self.config.catalog),
            "priority_slots": min(3, len(final)),
            "tracking_slots": len(final),
            "agent_slots": slots,
            "final_agent_sequence": agents,
            "first3_tuple": agents[:3],
            "first_agent_seconds": first_time,
            "first_agent_step": first_step,
            "first3_agent_seconds": first3_time,
            "first3_materialized_seconds": first3_time,
            "first3_agent_step": first3_step,
            "first3_materialized_step": first3_step,
            "all_final_agent_seconds": all_time,
            "all_final_agent_step": all_step,
            "reasoning_end_seconds": self.end_marker_seconds["reasoning"],
            "plan_json_start_seconds": self.plan_json_start_materialized_seconds,
            "plan_parseable_seconds": self.plan_parseable_seconds,
            "plan_json_complete_seconds": self.plan_json_complete_seconds,
            **self.phase_times,
            "layout": {
                "gen_length": self.layout.gen_length,
                "initial_canvas_length": self.layout.initial_canvas_length,
                "unused_generation_capacity": self.layout.gen_length - self.layout.initial_canvas_length,
                "reasoning_budget": self.layout.reasoning_budget,
                "plan_budget": self.layout.plan_budget,
                "prefix_tokens": len(self.layout.prefix_ids),
                "middle_tokens": len(self.layout.middle_ids),
                "suffix_tokens": 0,
                "reasoning_start": self.layout.reasoning_start,
                "reasoning_end": self.layout.reasoning_transfer_end,
                "plan_start": self.layout.plan_start,
                "plan_end": self.layout.plan_transfer_end,
                "compacted": self.layout.compacted,
            },
            "final_plan_parse_success": self._final_plan is not None,
            "final_plan": self._final_plan,
            "reasoning_nonempty": bool(self._reasoning_text.strip()),
            "reasoning_chars": len(self._reasoning_text.strip()),
            "reasoning_json_leak": any(
                marker in self._reasoning_text
                for marker in ('"agent"', "PLAN_JSON", "END_PLAN_JSON")
            ),
            "reasoning_effective_tokens": reasoning_effective,
            "reasoning_region_tokens_including_end": reasoning_used,
            "reasoning_capacity": self.layout.reasoning_budget,
            "unused_reasoning_capacity": max(0, self.layout.reasoning_budget - reasoning_used),
            "reasoning_capacity_utilization": reasoning_effective / self.layout.reasoning_budget,
            "reasoning_end_natural_success": self.marker_offsets["reasoning"] is not None,
            "reasoning_capacity_exhausted": self.capacity_exhausted["reasoning"],
            "reasoning_invalid_at_end": self.invalid_at_end["reasoning"],
            "reasoning_ends_cleanly": self._reasoning_text.rstrip().endswith((".", "!", "?", "。", "！", "？")),
            "reasoning_repeated_sentence_ratio": repeated / len(sentences) if sentences else None,
            "plan_capacity": self.layout.plan_budget,
            "plan_effective_tokens": plan_effective,
            "plan_region_tokens_including_end": plan_used,
            "unused_plan_capacity": max(0, self.layout.plan_budget - plan_used),
            "plan_capacity_utilization": plan_effective / self.layout.plan_budget,
            "plan_end_natural_success": self.marker_offsets["plan"] is not None,
            "plan_json_complete": self.plan_completion is not None,
            "plan_capacity_overflow": self.capacity_exhausted["plan"],
            "plan_capacity_exhausted": self.capacity_exhausted["plan"],
            "plan_invalid_at_end": self.invalid_at_end["plan"],
            "plan_invalid_at_capacity": self.capacity_exhausted["plan"] and self._final_plan is None,
            "plan_tokens_after_json_before_detection": (
                self.plan_completion.tokens_after_json if self.plan_completion else None
            ),
            "extra_plan_tail_tokens": extra_plan_tokens,
            "unresolved_plan_masks": self._snapshots[-1]["plan_masks"] if self._snapshots else None,
            "unresolved_reasoning_masks": self._snapshots[-1]["reasoning_masks"] if self._snapshots else None,
            "unresolved_mask_count": int((self._final_ids == self.mask_id).sum().item()),
            "fixed_token_corruption_count": fixed_corruption,
            "raw_generation_sha256": hashlib.sha256(self._final_ids.numpy().tobytes()).hexdigest(),
            "schedule": getattr(self, "schedule_log", []),
            "trajectory": self._snapshots,
        }
        if not self.agent_commit:
            result["first3_recognized_seconds"] = None
            result["first3_recognized_step"] = None
            result["first3_recognized_exact"] = None
            result["first3_commit_seconds"] = None
            result["first3_commit_step"] = None
            result["first3_commit_correct"] = None
            return result

        observed = self.agent_observer.metrics()
        observed_slots = observed.get("agent_slots") or []
        first_k = min(3, len(slots))
        shadow_times = []
        shadow_steps = []
        shadow_agents = []
        evaluated_shadows = []
        wrong_anchor_count = 0
        final_agent_match_count = 0

        for index, prediction in enumerate(observed_slots):
            shadow_agent = prediction.get("shadow_agent")
            shadow_seconds = prediction.get("shadow_seconds")
            shadow_step = prediction.get("shadow_step")
            if shadow_agent is None or shadow_seconds is None:
                continue
            expected_agent = agents[index] if index < len(agents) else None
            expected_offset = final[index][0] if index < len(final) else None
            shadow_offset = prediction.get("shadow_anchor_offset")
            wrong_anchor = (
                expected_offset is None
                or shadow_offset is None
                or int(shadow_offset) != int(expected_offset)
            )
            final_agent_match = (
                expected_agent is not None and shadow_agent == expected_agent
            )
            shadow_correct = (not wrong_anchor) and final_agent_match
            materialized_seconds = (
                slots[index].get("materialized_seconds")
                if index < len(slots) else None
            )
            shadow_lead = (
                float(materialized_seconds) - float(shadow_seconds)
                if materialized_seconds is not None else None
            )
            prediction.update({
                "shadow_wrong_anchor": wrong_anchor,
                "shadow_final_agent": expected_agent,
                "shadow_final_agent_match": final_agent_match,
                "shadow_correct": shadow_correct,
                "shadow_lead_vs_materialization": shadow_lead,
            })
            if index < len(slots):
                slots[index].update({
                    "predicted_agent": shadow_agent,
                    "predicted_seconds": shadow_seconds,
                    "predicted_step": shadow_step,
                    "recognized_seconds": shadow_seconds,
                    "recognized_step": shadow_step,
                    "probability": prediction.get("shadow_probability"),
                    "margin": prediction.get("shadow_margin"),
                    "shadow_agent": shadow_agent,
                    "shadow_seconds": shadow_seconds,
                    "shadow_step": shadow_step,
                    "shadow_anchor_offset": shadow_offset,
                    "shadow_anchor_observed_ratio": prediction.get(
                        "shadow_anchor_observed_ratio"
                    ),
                    "shadow_anchor_consistent_steps": prediction.get(
                        "shadow_anchor_consistent_steps"
                    ),
                    "shadow_probability": prediction.get("shadow_probability"),
                    "shadow_margin": prediction.get("shadow_margin"),
                    "shadow_wrong_anchor": wrong_anchor,
                    "shadow_final_agent": expected_agent,
                    "shadow_final_agent_match": final_agent_match,
                    "shadow_correct": shadow_correct,
                    "shadow_lead_vs_materialization": shadow_lead,
                })
            if index < first_k:
                shadow_agents.append(shadow_agent)
                shadow_times.append(shadow_seconds)
                if shadow_step is not None:
                    shadow_steps.append(shadow_step)
            evaluated_shadows.append(shadow_correct)
            wrong_anchor_count += int(wrong_anchor)
            final_agent_match_count += int(final_agent_match)

        complete_shadow = first_k > 0 and len(shadow_times) == first_k
        first3_shadow_seconds = max(shadow_times) if complete_shadow else None
        first3_shadow_step = (
            max(shadow_steps)
            if complete_shadow and len(shadow_steps) == first_k else None
        )
        first3_shadow_correct = (
            tuple(shadow_agents) == tuple(agents[:first_k])
            and all(
                observed_slots[index].get("shadow_correct") is True
                for index in range(first_k)
            )
            if complete_shadow else None
        )
        result.update({
            "policy": "all",
            "timing_source": "fixed_canvas_plan_first_plus_shadow_prefetch",
            "agent_slots": slots,
            "agent_observe": observed,
            "first3_shadow_seconds": first3_shadow_seconds,
            "first3_shadow_step": first3_shadow_step,
            "first3_shadow_correct": first3_shadow_correct,
            "first3_shadow_lead_vs_materialization": (
                first3_time - first3_shadow_seconds
                if first3_time is not None and first3_shadow_seconds is not None
                else None
            ),
            # Common recognition fields deliberately alias the deployable
            # shadow gate rather than an earlier, weaker recognition event.
            "first3_recognized_seconds": first3_shadow_seconds,
            "first3_recognized_step": first3_shadow_step,
            "first3_recognized_exact": first3_shadow_correct,
            "shadow_count": len(evaluated_shadows),
            "shadow_coverage": (
                len(shadow_times) / first_k if first_k else 0.0
            ),
            "shadow_correct_count": sum(evaluated_shadows),
            "shadow_accuracy": (
                sum(evaluated_shadows) / len(evaluated_shadows)
                if evaluated_shadows else None
            ),
            "shadow_wrong_count": (
                len(evaluated_shadows) - sum(evaluated_shadows)
            ),
            "shadow_wrong_rate": (
                1.0 - sum(evaluated_shadows) / len(evaluated_shadows)
                if evaluated_shadows else None
            ),
            "shadow_wrong_anchor_count": wrong_anchor_count,
            "shadow_wrong_anchor_rate": (
                wrong_anchor_count / len(evaluated_shadows)
                if evaluated_shadows else None
            ),
            "shadow_final_agent_match_count": final_agent_match_count,
            "shadow_final_agent_match_rate": (
                final_agent_match_count / len(evaluated_shadows)
                if evaluated_shadows else None
            ),
            # No token write occurs in ``all``.
            "first3_commit_seconds": None,
            "first3_commit_step": None,
            "first3_commit_correct": None,
            "commit_count": 0,
            "commit_coverage": 0.0,
            "commit_correct_count": 0,
            "wrong_commit_count": 0,
            "commit_accuracy": None,
            "wrong_commit_rate": None,
        })
        return result

    def close(self) -> None:
        if self.agent_observer is not None:
            self.agent_observer.close()
        return None

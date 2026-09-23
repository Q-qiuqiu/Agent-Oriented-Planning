"""Fixed structural canvas and passive timing for PLAN-first experiments.

Only the four section delimiters (and the newlines which separate them from
their regions) are written into the initial canvas.  Everything inside the
reasoning and PLAN regions is produced by the ordinary LLaDA proposal and
transfer policy.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from json_agent_priority import JsonAgentPriorityConfig
from response_agent_timing import PassiveJsonAgentMonitor


@dataclass
class FixedCanvasLayout:
    prompt_length: int
    gen_length: int
    reasoning_budget: int
    plan_budget: int
    prefix_ids: Tuple[int, ...]
    middle_ids: Tuple[int, ...]
    suffix_ids: Tuple[int, ...]
    plan_actual_end: Optional[int] = None

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
    def reasoning_end(self) -> int:
        return self.reasoning_start + self.reasoning_budget

    @property
    def middle_start(self) -> int:
        return self.reasoning_end

    @property
    def plan_start(self) -> int:
        return self.middle_start + len(self.middle_ids)

    @property
    def plan_capacity_end(self) -> int:
        return self.plan_start + self.plan_budget

    @property
    def plan_end(self) -> int:
        return self.plan_actual_end or self.plan_capacity_end

    @property
    def initial_generation_end(self) -> int:
        return self.plan_capacity_end + len(self.suffix_ids)

    @property
    def initial_canvas_length(self) -> int:
        return self.initial_generation_end - self.prompt_length

    @property
    def suffix_start(self) -> int:
        return self.plan_end

    @property
    def generation_end(self) -> int:
        return self.suffix_start + len(self.suffix_ids)

    @property
    def compacted(self) -> bool:
        return self.plan_actual_end is not None

    def validate(self) -> None:
        if self.reasoning_budget <= 0 or self.plan_budget <= 0:
            raise ValueError(
                "Fixed canvas requires positive reasoning and PLAN budgets; "
                f"got reasoning={self.reasoning_budget}, plan={self.plan_budget}."
            )
        if self.initial_canvas_length > self.gen_length:
            raise AssertionError("Fixed canvas exceeds configured gen_length.")
        if not (
            self.reasoning_start < self.reasoning_end
            < self.plan_start < self.plan_capacity_end
        ):
            raise AssertionError("Fixed canvas region ordering is invalid.")

    def bool_mask(self, x: torch.Tensor, region: str) -> torch.Tensor:
        mask = torch.zeros_like(x, dtype=torch.bool)
        if region == "reasoning":
            mask[:, self.reasoning_start : self.reasoning_end] = True
        elif region == "plan":
            mask[:, self.plan_start : self.plan_end] = True
        elif region == "fixed":
            mask[:, self.prefix_start : self.reasoning_start] = True
            mask[:, self.middle_start : self.plan_start] = True
            mask[:, self.suffix_start : self.generation_end] = True
        else:
            raise ValueError(f"Unknown fixed-canvas region {region!r}.")
        return mask

    def initialize(self, x: torch.Tensor) -> None:
        self.validate()
        for start, ids in (
            (self.prefix_start, self.prefix_ids),
            (self.middle_start, self.middle_ids),
            (self.plan_capacity_end, self.suffix_ids),
        ):
            values = torch.tensor(ids, dtype=x.dtype, device=x.device)
            x[:, start : start + len(ids)] = values

    def mark_plan_compacted(self, actual_end: int) -> None:
        if not (self.plan_start < actual_end <= self.plan_capacity_end):
            raise ValueError(
                f"Invalid compacted PLAN end {actual_end}; expected within "
                f"({self.plan_start}, {self.plan_capacity_end}]."
            )
        self.plan_actual_end = int(actual_end)

    def fixed_ids(self) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for start, ids in (
            (self.prefix_start, self.prefix_ids),
            (self.middle_start, self.middle_ids),
            (self.suffix_start, self.suffix_ids),
        ):
            result.extend((start + offset, int(token)) for offset, token in enumerate(ids))
        return result


@dataclass(frozen=True)
class PlanCompletion:
    plan: List[Dict[str, object]]
    char_end: int
    token_length: int
    actual_end: int
    continuous_prefix_tokens: int
    tokens_after_json: int
    alignment: str


class FixedCanvasMonitor(PassiveJsonAgentMonitor):
    """Read-only materialization monitor plus fixed-token protection."""

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
        encode = lambda value: tuple(
            tokenizer.encode(value, add_special_tokens=False)
        )
        prefix_ids = encode("PLANNING_REASONING\n")
        middle_ids = encode("\nEND_PLANNING_REASONING\n\nPLAN_JSON\n")
        suffix_ids = encode("\nEND_PLAN_JSON")
        available = gen_length - len(prefix_ids) - len(middle_ids) - len(suffix_ids)
        if available < 2:
            raise ValueError("Generation canvas is too short for both fixed regions.")
        if reasoning_budget is None and plan_budget is None:
            if reasoning_ratio <= 0 or plan_ratio <= 0:
                raise ValueError("reasoning_ratio and plan_ratio must be positive.")
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
                    "Explicit reasoning_budget + plan_budget exceeds the "
                    f"{available}-token non-delimiter canvas: got "
                    f"{reasoning_budget} + {plan_budget}."
                )
        self.layout = FixedCanvasLayout(
            prompt_length=prompt_length,
            gen_length=gen_length,
            reasoning_budget=reasoning_budget,
            plan_budget=plan_budget,
            prefix_ids=prefix_ids,
            middle_ids=middle_ids,
            suffix_ids=suffix_ids,
        )
        self.layout.validate()
        self.structure_mode = structure_mode
        self.phase = None
        self.phase_times: Dict[str, Optional[float]] = {
            "plan_phase_start": None,
            "reasoning_phase_start": None,
            "plan_region_complete": None,
            "reasoning_region_complete": None,
            "generation_complete": None,
        }
        self.plan_parseable_seconds: Optional[float] = None
        self.plan_json_complete_seconds: Optional[float] = None
        self.plan_json_start_materialized_seconds: Optional[float] = 0.0
        self.plan_completion: Optional[PlanCompletion] = None
        self.plan_capacity_overflow = False
        self.plan_alignment_error: Optional[str] = None
        self._last_plan_prefix_checked = -1
        self._snapshots: List[Dict[str, object]] = []
        self._fixed_reference: List[Tuple[int, int]] = []
        self._final_occurrences: List[Tuple[int, str]] = []
        self._final_plan = None
        self._last_step = 0
        self._nfe = 0

    def initialize(self, x: torch.Tensor) -> None:
        self.layout.initialize(x)
        super().initialize(x)
        self._last_plan_prefix_checked = -1
        self._fixed_reference = self.layout.fixed_ids()
        # The PLAN_JSON marker is present from initialization by construction.
        self.plan_json_start_materialized_seconds = 0.0
        self._snapshot(x, step=0, nfe=0)

    def _plan_json_bounds(self, x):
        del x
        return self.layout.plan_start, self.layout.plan_end

    def decoder_mask(self, mask_index, mask_start=0):
        """Protect delimiters even if a caller accidentally offers them."""
        fixed = self.layout.bool_mask(
            torch.empty(
                (mask_index.shape[0], self.prompt_length + self.gen_length),
                device=mask_index.device,
                dtype=torch.long,
            ),
            "fixed",
        )
        if mask_index.shape[1] == fixed.shape[1]:
            return mask_index & ~fixed
        stop = mask_start + mask_index.shape[1]
        return mask_index & ~fixed[:, mask_start:stop]

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        key = f"{phase}_phase_start"
        if key in self.phase_times and self.phase_times[key] is None:
            self.phase_times[key] = self._elapsed()

    def _decode_region(self, x: torch.Tensor, start: int, end: int) -> str:
        ids = x[0, start:end]
        ids = ids[ids != self.mask_id].detach().cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def _raw_json_array(text: str, catalog: Optional[Sequence[str]] = None):
        decoder = json.JSONDecoder()
        required = {"agent", "id", "task", "reason", "dep"}
        allowed_agents = set(catalog or ())
        for offset, character in enumerate(text):
            if character != "[":
                continue
            try:
                value, end = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(item, dict)
                    and required.issubset(item)
                    and bool(str(item.get("agent") or "").strip())
                    and (
                        not allowed_agents
                        or str(item.get("agent")).strip() in allowed_agents
                    )
                    and bool(str(item.get("task") or "").strip())
                    and bool(str(item.get("reason") or "").strip())
                    and isinstance(item.get("dep"), list)
                    for item in value
                )
            ):
                return value, offset + end
        return None, None

    def _continuous_plan_prefix_ids(self, x: torch.Tensor) -> List[int]:
        plan_ids = x[0, self.layout.plan_start : self.layout.plan_capacity_end]
        masks = torch.nonzero(plan_ids == self.mask_id, as_tuple=False).flatten()
        prefix_length = int(masks[0].item()) if masks.numel() else int(plan_ids.shape[0])
        return plan_ids[:prefix_length].detach().cpu().tolist()

    def _match_json_token_end(
        self, prefix_ids: Sequence[int], valid_json_text: str
    ) -> Tuple[Optional[int], Optional[str]]:
        encoded = list(
            self.tokenizer.encode(valid_json_text, add_special_tokens=False)
        )
        if list(prefix_ids[: len(encoded)]) == encoded:
            return len(encoded), "retokenized_exact"

        # Token merges at the closing bracket can make standalone re-tokenizing
        # differ. Locate an exact decoded boundary in the actual generated IDs;
        # retaining whitespace from the same closing token is also safe.
        start = max(1, len(encoded) - 4)
        stop = min(len(prefix_ids), len(encoded) + 4)
        candidates = list(range(start, stop + 1))
        if len(prefix_ids) not in candidates:
            candidates.append(len(prefix_ids))
        for length in candidates:
            decoded = self.tokenizer.decode(
                list(prefix_ids[:length]), skip_special_tokens=False
            )
            if decoded == valid_json_text:
                return length, "actual_prefix_exact"
            if decoded.startswith(valid_json_text) and decoded[len(valid_json_text) :].isspace():
                return length, "actual_prefix_trailing_whitespace"
        return None, None

    def detect_complete_plan_json(self, x: torch.Tensor) -> Optional[PlanCompletion]:
        """Parse only the contiguous, materialized prefix of PLAN capacity."""
        if self.plan_completion is not None:
            return self.plan_completion
        prefix_ids = self._continuous_plan_prefix_ids(x)
        if not prefix_ids:
            return None
        if len(prefix_ids) == self._last_plan_prefix_checked:
            return None
        self._last_plan_prefix_checked = len(prefix_ids)
        prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        plan, char_end = self._raw_json_array(prefix_text, self.config.catalog)
        if plan is None:
            return None
        valid_json_text = prefix_text[:char_end]
        token_length, alignment = self._match_json_token_end(
            prefix_ids, valid_json_text
        )
        if token_length is None:
            self.plan_alignment_error = (
                "A schema-valid JSON array was decoded, but its character end "
                "could not be aligned to the generated token prefix."
            )
            return None
        return PlanCompletion(
            plan=plan,
            char_end=int(char_end),
            token_length=int(token_length),
            actual_end=self.layout.plan_start + int(token_length),
            continuous_prefix_tokens=len(prefix_ids),
            tokens_after_json=max(0, len(prefix_ids) - int(token_length)),
            alignment=str(alignment),
        )

    def compact_plan(
        self, x: torch.Tensor, completion: PlanCompletion
    ) -> torch.Tensor:
        """Remove unused PLAN capacity and move END_PLAN_JSON next to JSON."""
        if self.layout.compacted:
            return x
        old_suffix_start = self.layout.plan_capacity_end
        old_suffix_end = old_suffix_start + len(self.layout.suffix_ids)
        suffix = x[:, old_suffix_start:old_suffix_end]
        expected = torch.tensor(
            self.layout.suffix_ids, dtype=x.dtype, device=x.device
        ).unsqueeze(0)
        if not torch.equal(suffix, expected):
            raise AssertionError("END_PLAN_JSON was corrupted before compaction.")
        x = torch.cat([x[:, : completion.actual_end], suffix], dim=1)
        self.layout.mark_plan_compacted(completion.actual_end)
        self.plan_completion = completion
        now = self._elapsed()
        self.plan_json_complete_seconds = now
        self.plan_parseable_seconds = now
        self.phase_times["plan_region_complete"] = now
        self._fixed_reference = self.layout.fixed_ids()
        return x

    def mark_plan_capacity_exhausted(self) -> None:
        if self.plan_completion is None:
            self.plan_capacity_overflow = True

    def _current_occurrences(self, x: torch.Tensor) -> List[Tuple[int, str]]:
        candidates = self._plan_agent_candidates(x)
        self._assign_anchors(x, candidates)
        values = []
        for runtime in self.slots:
            if runtime.anchor_start is None:
                continue
            name = self._observed_catalog_value(x, runtime)
            if name is not None:
                values.append((int(runtime.anchor_start), str(name)))
        return sorted(set(values), key=lambda item: item[0])

    def _snapshot(self, x: torch.Tensor, *, step: int, nfe: int) -> None:
        now = self._elapsed()
        occurrences = self._current_occurrences(x)
        plan_masks = int(
            ((x == self.mask_id) & self.layout.bool_mask(x, "plan")).sum().item()
        )
        reasoning_masks = int(
            ((x == self.mask_id) & self.layout.bool_mask(x, "reasoning")).sum().item()
        )
        if (
            plan_masks == 0
            and self.phase_times["plan_region_complete"] is None
            and self.plan_completion is not None
        ):
            self.phase_times["plan_region_complete"] = now
        if (
            reasoning_masks == 0
            and self.phase_times["reasoning_region_complete"] is None
        ):
            self.phase_times["reasoning_region_complete"] = now
        self._snapshots.append(
            {
                "step": int(step),
                "nfe": int(nfe),
                "seconds": float(now),
                "phase": self.phase,
                "plan_masks": plan_masks,
                "reasoning_masks": reasoning_masks,
                "materialized_agents": [
                    {"anchor_start": position, "agent": name}
                    for position, name in occurrences
                ],
            }
        )
        self._last_step = int(step)
        self._nfe = int(nfe)

    def record_step(
        self,
        x: torch.Tensor,
        *,
        global_step: int,
        nfe: int,
        physical_block: int,
        local_step: int,
    ) -> None:
        del physical_block, local_step
        self._snapshot(x, step=global_step, nfe=nfe)

    def _final_agent_occurrences(self, x: torch.Tensor) -> List[Tuple[int, str]]:
        # Rebuild slot assignment once from the fully materialized response.
        candidates = self._plan_agent_candidates(x)
        self._assign_anchors(x, candidates)
        values = []
        for runtime in self.slots:
            if runtime.anchor_start is None:
                continue
            value = self._observed_catalog_value(x, runtime)
            if value is not None:
                values.append((int(runtime.anchor_start), str(value)))
        return sorted(set(values), key=lambda item: item[0])

    @staticmethod
    def _first_cover_time(snapshots, targets):
        if not targets:
            return None, None
        target_set = set(targets)
        for row in snapshots:
            observed = {
                (int(item["anchor_start"]), str(item["agent"]))
                for item in row["materialized_agents"]
            }
            if target_set.issubset(observed):
                return row["seconds"], row["step"]
        return None, None

    def finalize(self, x: torch.Tensor) -> None:
        self._snapshot(x, step=self._last_step, nfe=self._nfe)
        self.phase_times["generation_complete"] = self._elapsed()
        self._final_occurrences = self._final_agent_occurrences(x)
        plan_text = self._decode_region(x, self.layout.plan_start, self.layout.plan_end)
        self._final_plan, json_end = self._raw_json_array(
            plan_text, self.config.catalog
        )
        self._json_end = json_end
        self._plan_text = plan_text
        self._reasoning_text = self._decode_region(
            x, self.layout.reasoning_start, self.layout.reasoning_end
        )
        self._final_ids = x[0, self.prompt_length : self.prompt_length + self.gen_length]

    def metrics(self) -> Dict[str, object]:
        final = self._final_occurrences
        first_time, first_step = self._first_cover_time(
            self._snapshots, final[:1]
        )
        first3_time, first3_step = self._first_cover_time(
            self._snapshots, final[: min(3, len(final))]
        )
        all_time, all_step = self._first_cover_time(self._snapshots, final)
        fixed_corruption = 0
        if hasattr(self, "_final_ids"):
            absolute = self._final_ids
            for position, token in self._fixed_reference:
                relative = position - self.prompt_length
                fixed_corruption += int(int(absolute[relative].item()) != token)
        plan_masks = self._snapshots[-1]["plan_masks"] if self._snapshots else None
        reasoning_masks = (
            self._snapshots[-1]["reasoning_masks"] if self._snapshots else None
        )
        extra_plan_tokens = None
        if getattr(self, "_json_end", None) is not None:
            extra_text = self._plan_text[self._json_end :]
            extra_plan_tokens = len(
                self.tokenizer.encode(extra_text, add_special_tokens=False)
            )
        special_ids = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
        region_ids = []
        if hasattr(self, "_final_ids"):
            rs = self.layout.reasoning_start - self.prompt_length
            reasoning_stop = self.layout.reasoning_end - self.prompt_length
            region_ids = self._final_ids[rs:reasoning_stop].detach().cpu().tolist()
        special_ratio = (
            sum(int(token in special_ids) for token in region_ids) / len(region_ids)
            if region_ids else None
        )
        reasoning_effective_tokens = sum(
            int(token not in special_ids and token != self.mask_id)
            for token in region_ids
        )
        reasoning_sentences = [
            sentence.strip().lower()
            for sentence in re.split(
                r"(?<=[.!?。！？])\s+", getattr(self, "_reasoning_text", "").strip()
            )
            if sentence.strip()
        ]
        repeated_sentences = len(reasoning_sentences) - len(set(reasoning_sentences))
        completion = self.plan_completion
        plan_effective_tokens = (
            completion.token_length
            if completion is not None
            else sum(
                int(token not in special_ids and token != self.mask_id)
                for token in (
                    self._final_ids[
                        self.layout.plan_start - self.prompt_length :
                        self.layout.plan_end - self.prompt_length
                    ].detach().cpu().tolist()
                    if hasattr(self, "_final_ids") else []
                )
            )
        )
        agents = [name for _, name in final]
        slots = []
        for slot, (position, name) in enumerate(final):
            target_time, target_step = self._first_cover_time(
                self._snapshots, [(position, name)]
            )
            slots.append(
                {
                    "slot": slot,
                    "anchor_start": position,
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
                }
            )
        return {
            "policy": self.structure_mode,
            "structure_mode": self.structure_mode,
            "timing_source": "fixed_canvas_materialized_x",
            "catalog": list(self.config.catalog),
            "priority_slots": min(3, len(final)),
            "tracking_slots": len(final),
            "agent_slots": slots,
            "final_agent_sequence": agents,
            "first3_tuple": agents[:3],
            "first_agent_seconds": first_time,
            "first_agent_step": first_step,
            "first3_agent_seconds": first3_time,
            "first3_agent_step": first3_step,
            "all_final_agent_seconds": all_time,
            "all_final_agent_step": all_step,
            "plan_parseable_seconds": self.plan_parseable_seconds,
            "plan_json_complete_seconds": self.plan_json_complete_seconds,
            **self.phase_times,
            "layout": {
                "gen_length": self.layout.gen_length,
                "initial_canvas_length": self.layout.initial_canvas_length,
                "unused_generation_capacity": (
                    self.layout.gen_length - self.layout.initial_canvas_length
                ),
                "reasoning_budget": self.layout.reasoning_budget,
                "plan_budget": self.layout.plan_budget,
                "prefix_tokens": len(self.layout.prefix_ids),
                "middle_tokens": len(self.layout.middle_ids),
                "suffix_tokens": len(self.layout.suffix_ids),
                "reasoning_start": self.layout.reasoning_start,
                "reasoning_end": self.layout.reasoning_end,
                "plan_start": self.layout.plan_start,
                "plan_end": self.layout.plan_end,
                "plan_capacity_end": self.layout.plan_capacity_end,
                "compacted": self.layout.compacted,
            },
            "final_plan_parse_success": self._final_plan is not None,
            "final_plan": self._final_plan,
            "reasoning_nonempty": bool(getattr(self, "_reasoning_text", "").strip()),
            "reasoning_chars": len(getattr(self, "_reasoning_text", "").strip()),
            "reasoning_json_leak": any(
                marker in getattr(self, "_reasoning_text", "")
                for marker in ('"agent"', "PLAN_JSON", "END_PLAN_JSON")
            ),
            "reasoning_special_token_ratio": special_ratio,
            "reasoning_effective_tokens": reasoning_effective_tokens,
            "reasoning_ends_cleanly": getattr(self, "_reasoning_text", "").rstrip().endswith(
                (".", "!", "?", "。", "！", "？")
            ),
            "reasoning_repeated_sentence_ratio": (
                repeated_sentences / len(reasoning_sentences)
                if reasoning_sentences else None
            ),
            "extra_plan_tail_tokens": extra_plan_tokens,
            "plan_capacity": self.layout.plan_budget,
            "plan_effective_tokens": plan_effective_tokens,
            "unused_plan_capacity": max(
                0, self.layout.plan_budget - plan_effective_tokens
            ),
            "plan_capacity_utilization": (
                plan_effective_tokens / self.layout.plan_budget
                if self.layout.plan_budget else None
            ),
            "plan_json_complete": completion is not None,
            "plan_capacity_overflow": self.plan_capacity_overflow,
            "plan_invalid_at_capacity": (
                self.plan_capacity_overflow and self._final_plan is None
            ),
            "plan_tokens_after_json_before_detection": (
                completion.tokens_after_json if completion is not None else None
            ),
            "plan_json_alignment": (
                completion.alignment if completion is not None else None
            ),
            "plan_json_alignment_error": self.plan_alignment_error,
            "unresolved_plan_masks": plan_masks,
            "unresolved_reasoning_masks": reasoning_masks,
            "unresolved_mask_count": (
                int(plan_masks) + int(reasoning_masks)
                if plan_masks is not None and reasoning_masks is not None
                else None
            ),
            "fixed_token_corruption_count": fixed_corruption,
            "raw_generation_sha256": (
                hashlib.sha256(
                    self._final_ids.detach().cpu().numpy().tobytes()
                ).hexdigest()
                if hasattr(self, "_final_ids") else None
            ),
            "schedule": getattr(self, "schedule_log", []),
            "trajectory": self._snapshots,
        }

    def close(self) -> None:
        return None


class NaturalReasoningPlanMonitor(PassiveJsonAgentMonitor):
    """Passive timing for the unchanged, non-fixed Dual Vanilla canvas."""

    def __init__(
        self, *, tokenizer, catalog, prompt_length, gen_length, mask_id,
        priority_slots=3, tracking_slots=16,
    ):
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
        self._reasoning_end_ids = tuple(self._encode("END_PLANNING_REASONING"))
        self.reasoning_end_seconds = None
        self.plan_json_start_seconds = None
        self.plan_parseable_seconds = None
        self.generation_complete_seconds = None
        self.plan_effective_tokens = None

    def _record_materialized(self, x, global_step):
        super()._record_materialized(x, global_step)
        now = self._elapsed()
        if (
            self.reasoning_end_seconds is None
            and self._materialized_pattern_starts(x, self._reasoning_end_ids)
        ):
            self.reasoning_end_seconds = now
        if self.plan_json_start_seconds is None and self._plan_json_bounds(x) is not None:
            self.plan_json_start_seconds = now
        if self.plan_parseable_seconds is None:
            bounds = self._plan_json_bounds(x)
            if bounds is not None:
                text = self.tokenizer.decode(
                    x[0, bounds[0] : bounds[1]][
                        x[0, bounds[0] : bounds[1]] != self.mask_id
                    ].detach().cpu().tolist(),
                    skip_special_tokens=True,
                )
                plan, _ = FixedCanvasMonitor._raw_json_array(text)
                if plan is not None:
                    self.plan_parseable_seconds = now

    def finalize(self, x):
        super().finalize(x)
        self.generation_complete_seconds = self._elapsed()
        bounds = self._plan_json_bounds(x)
        if bounds is not None:
            self.plan_effective_tokens = int(bounds[1] - bounds[0])

    def metrics(self):
        result = super().metrics()
        slots = [slot for slot in result.get("agent_slots", []) if slot.get("agent")]
        materialized = [
            slot.get("materialized_seconds") or slot.get("recognized_seconds")
            for slot in slots
        ]
        first3 = materialized[: min(3, len(materialized))]
        result.update(
            {
                "policy": "dual_vanilla",
                "structure_mode": "dual_vanilla",
                "timing_source": "passive_materialized_x",
                "final_agent_sequence": [slot.get("agent") for slot in slots],
                "first3_tuple": [slot.get("agent") for slot in slots[:3]],
                "first_agent_seconds": materialized[0] if materialized else None,
                "first3_agent_seconds": (
                    max(first3) if first3 and all(value is not None for value in first3) else None
                ),
                "all_final_agent_seconds": (
                    max(materialized)
                    if materialized and all(value is not None for value in materialized)
                    else None
                ),
                "reasoning_end_seconds": self.reasoning_end_seconds,
                "plan_json_start_seconds": self.plan_json_start_seconds,
                "plan_parseable_seconds": self.plan_parseable_seconds,
                "plan_effective_tokens": self.plan_effective_tokens,
                "plan_capacity": None,
                "plan_capacity_overflow": False,
                "generation_complete": self.generation_complete_seconds,
            }
        )
        return result

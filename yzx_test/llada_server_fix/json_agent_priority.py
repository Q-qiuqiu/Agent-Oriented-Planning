"""Priority decoding for Agent names embedded in a normal JSON response.

Unlike :mod:`agent_priority`, this controller does not reserve a routing region
or render a private output format.  It searches full-sequence logits for the
first Agent fields in a compact JSON plan and writes catalog-constrained Agent
names directly into those final response positions.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# Reuse the existing rotating Agent log configured by llada_server.
LOGGER = logging.getLogger("fastdllm.agent_priority")


def extract_agent_registry(
    messages: Sequence[Dict[str, str]], fallback: Sequence[str]
) -> List[str]:
    """Read ordered Agent definitions from an AOP system prompt.

    Both HuskyQA and IIRC define roles as ``- name_agent: description`` lines,
    but use different registries. Restrict extraction to system-message
    definition lines so an Agent-like string in the user query cannot expand
    the constrained catalog.
    """

    names = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = str(message.get("content") or "")
        for name in re.findall(
            r"(?m)^\s*-\s*([a-z][a-z0-9_]*_agent)\s*:", content
        ):
            if name not in names:
                names.append(name)
    return names or list(fallback)


@dataclass(frozen=True)
class JsonAgentPriorityConfig:
    catalog: Sequence[str]
    priority_slots: int = 3
    tracking_slots: Optional[int] = None
    anchor_min_logit_margin: float = -6.0
    anchor_stable_steps: int = 2
    tentative_probability: float = 0.45
    tentative_margin: float = 0.20
    confirm_probability: float = 0.72
    confirm_margin: float = 0.25
    confirm_stable_steps: int = 2
    discovery_steps: int = 4
    min_anchor_gap: int = 12
    # Permit the commit/shadow gate to use a stable logits-localized anchor
    # before every anchor token exists in ``x``. Keeping this opt-in preserves
    # the legacy observe/priority behavior.
    allow_speculative_anchor_commit: bool = False
    # Periodic cross-block probing experiment.  None keeps the legacy priority
    # behavior with speculative Agent-name writes.  Any explicit value
    # (including 0 = block-start observations only) switches the controller to
    # prediction-only mode: speculative writes are disabled so extra
    # observations can never change the generation trajectory.
    probe_period: Optional[int] = None

    def __post_init__(self) -> None:
        names = list(self.catalog)
        if not names or len(names) != len(set(names)):
            raise ValueError("Agent catalog must be non-empty and contain unique names.")
        if self.priority_slots < 1:
            raise ValueError("priority_slots must be positive.")
        if self.tracking_slots is not None and self.tracking_slots < self.priority_slots:
            raise ValueError("tracking_slots must be at least priority_slots.")
        if self.anchor_stable_steps < 1 or self.confirm_stable_steps < 1:
            raise ValueError("Stability step counts must be positive.")
        if self.discovery_steps < 1:
            raise ValueError("discovery_steps must be positive.")
        if self.probe_period is not None and self.probe_period < 0:
            raise ValueError("probe_period must be None or a non-negative integer.")


@dataclass
class JsonAgentSlotRuntime:
    anchor_start: Optional[int] = None
    anchor_token_ids: Tuple[int, ...] = ()
    name_start: Optional[int] = None
    anchor_score: float = -math.inf
    anchor_observed_ratio: float = 0.0
    anchor_consistent_steps: int = 0
    candidate: Optional[str] = None
    recognized_candidate: Optional[str] = None
    candidate_probability: float = 0.0
    candidate_margin: float = 0.0
    recognized_probability: Optional[float] = None
    recognized_margin: Optional[float] = None
    candidate_consistent_steps: int = 0
    confirmed: bool = False
    field_written: bool = False
    fuzzy_matched_from: Optional[str] = None
    first_observed_seconds: Optional[float] = None
    recognized_seconds: Optional[float] = None
    confirmed_seconds: Optional[float] = None
    first_observed_step: Optional[int] = None
    recognized_step: Optional[int] = None
    confirmed_step: Optional[int] = None
    # Probing-experiment bookkeeping.  predicted_* is the first logits-based
    # recognition (speculative, before the value exists in x); materialized_*
    # is the first time the Agent value appears naturally in the response.
    predicted_seconds: Optional[float] = None
    predicted_step: Optional[int] = None
    predicted_candidate: Optional[str] = None
    # ``shadow_*`` records the exact instant at which the writable speculative
    # gate would have fired, without requiring that a write actually occurs.
    shadow_seconds: Optional[float] = None
    shadow_step: Optional[int] = None
    shadow_candidate: Optional[str] = None
    shadow_anchor_offset: Optional[int] = None
    shadow_anchor_observed_ratio: Optional[float] = None
    shadow_anchor_consistent_steps: Optional[int] = None
    shadow_probability: Optional[float] = None
    shadow_margin: Optional[float] = None
    committed_seconds: Optional[float] = None
    committed_step: Optional[int] = None
    committed_candidate: Optional[str] = None
    committed_anchor_offset: Optional[int] = None
    commit_anchor_observed_ratio: Optional[float] = None
    commit_anchor_consistent_steps: Optional[int] = None
    commit_probability: Optional[float] = None
    commit_margin: Optional[float] = None
    commit_had_mask: Optional[bool] = None
    committed_token_count: Optional[int] = None
    final_anchor_match: Optional[bool] = None
    final_agent_candidate: Optional[str] = None
    final_agent_match: Optional[bool] = None
    materialized_seconds: Optional[float] = None
    materialized_step: Optional[int] = None
    materialized_candidate: Optional[str] = None
    last_distribution: Optional[Dict[str, float]] = field(default=None)


class JsonAgentFieldController:
    """Locate and prioritize the first Agent fields in ordinary plan JSON."""

    def __init__(
        self,
        tokenizer,
        config: JsonAgentPriorityConfig,
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.enabled = True
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_length = int(prompt_length)
        self.gen_length = int(gen_length)
        self.mask_id = int(mask_id)
        self.logger = logger
        # Dual Cache already performs one full-sequence forward per block.
        # Repeating the identical all-mask forward here neither reveals new
        # structure nor improves confidence, and speculative writes during that
        # phase can create self-fulfilling JSON anchors.  Reuse normal block
        # warm-ups for continuously improving anchor discovery instead.
        self.full_sequence_discovery_steps = 0
        self.tracking_slots = config.tracking_slots or config.priority_slots
        self.slots = [JsonAgentSlotRuntime() for _ in range(self.tracking_slots)]
        self._started_at: Optional[float] = None
        self._observed_steps = 0
        self._full_sequence_observations = 0

        # Periodic cross-block probing experiment state.
        self.probe_period = config.probe_period
        self.prediction_only = config.probe_period is not None
        self.probe_forwards = 0
        self.plan_complete_seconds: Optional[float] = None
        self.plan_complete_step: Optional[int] = None
        plan_end_variants = tuple(
            dict.fromkeys(("\nEND_PLAN_JSON", "END_PLAN_JSON"))
        )
        self.plan_end_patterns = tuple(
            tuple(self._encode(text)) for text in plan_end_variants
        )
        self._plan_end_tensors: Tuple[torch.Tensor, ...] = ()

        # Do not include the opening brace, indentation, or opening key quote.
        # The LLaDA tokenizer merges leading whitespace with ``{``/``\"`` into
        # context-dependent tokens (for example ``Ġ{`` and ``Ġ\"``), so a
        # standalone encoding of a pretty-printed object never matches the
        # sequence.  This key suffix is context-stable and remains specific to
        # an Agent JSON field.
        canonical_anchor_texts = ['agent":"', 'agent": "']
        # Materialized output may vary in JSON whitespace. Keep speculative
        # discovery on the two canonical forms so tolerant variants cannot add
        # all-mask false positives.
        anchor_texts = list(canonical_anchor_texts)
        for before_colon in (" ", "\t", "\n", "\n  "):
            for after_colon in ("", " ", "\t", "\n", "\n  "):
                anchor_texts.append(f'agent"{before_colon}:{after_colon}"')
        anchor_texts = list(dict.fromkeys(anchor_texts))
        self.anchor_variants = tuple(
            tuple(self._encode(text)) for text in anchor_texts
        )
        self.speculative_anchor_variants = frozenset(
            tuple(self._encode(text)) for text in canonical_anchor_texts
        )
        if any(not value for value in self.anchor_variants):
            raise ValueError("Tokenizer produced an empty JSON Agent anchor.")

        space_ids = self._encode(" ")
        if len(space_ids) != 1 or not self.tokenizer.decode(space_ids).isspace():
            raise ValueError("JSON Agent padding requires a single whitespace token.")
        self.space_token_id = int(space_ids[0])

        comma_ids = self._encode(",")
        if len(comma_ids) != 1 or self.tokenizer.decode(comma_ids) != ",":
            raise ValueError("JSON Agent layout requires a single comma token.")
        self.comma_token_id = int(comma_ids[0])

        self.catalog_value_ids = {
            name: tuple(self._encode(name + '"')) for name in config.catalog
        }
        if any(not ids for ids in self.catalog_value_ids.values()):
            raise ValueError("Every Agent name must tokenize to at least one token.")
        self.value_width = max(len(ids) for ids in self.catalog_value_ids.values())
        self.padded_catalog_ids = {
            name: ids + (self.space_token_id,) * (self.value_width - len(ids))
            for name, ids in self.catalog_value_ids.items()
        }
        self.catalog_names = tuple(self.padded_catalog_ids)
        self._catalog_target_ids: Optional[torch.Tensor] = None
        self._catalog_positions: Optional[torch.Tensor] = None
        self._anchor_target_tensors: Tuple[torch.Tensor, ...] = ()
        # Optional read-only localization bound.  Fixed-canvas PLAN-first uses
        # this to search only its known PLAN capacity; ordinary Dual Vanilla
        # leaves it unset and searches the complete generation canvas.
        self._search_start: Optional[int] = None
        self._search_end: Optional[int] = None

    def set_search_region(
        self, start: Optional[int], end: Optional[int]
    ) -> None:
        if (start is None) != (end is None):
            raise ValueError("Agent search region requires both start and end.")
        if start is not None and int(start) >= int(end):
            raise ValueError("Agent search region must be non-empty.")
        self._search_start = None if start is None else int(start)
        self._search_end = None if end is None else int(end)

    def shift_positions(self, start: int, delta: int) -> None:
        """Shift tracked absolute positions after a canvas compaction.

        Fixed-canvas reasoning compaction moves the entire PLAN region. Any
        speculative anchors and frozen Agent spans found before that movement
        must follow their tokens to the new absolute coordinates.
        """

        start = int(start)
        delta = int(delta)
        if delta == 0:
            return
        for runtime in self.slots:
            if runtime.anchor_start is not None and runtime.anchor_start >= start:
                runtime.anchor_start += delta
            if runtime.name_start is not None and runtime.name_start >= start:
                runtime.name_start += delta
        if self._search_start is not None and self._search_start >= start:
            self._search_start += delta
        if self._search_end is not None and self._search_end >= start:
            self._search_end += delta

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def initialize(self, x: torch.Tensor) -> None:
        if x.shape[0] != 1:
            raise ValueError("JSON Agent priority decoding requires batch size 1.")
        self._started_at = time.perf_counter()
        self._observed_steps = 0
        self._full_sequence_observations = 0
        self.probe_forwards = 0
        self.plan_complete_seconds = None
        self.plan_complete_step = None
        self._plan_end_tensors = tuple(
            torch.tensor(pattern, device=x.device, dtype=x.dtype)
            for pattern in self.plan_end_patterns
        )
        self._catalog_target_ids = torch.tensor(
            [self.padded_catalog_ids[name] for name in self.catalog_names],
            device=x.device,
            dtype=torch.long,
        )
        self._catalog_positions = torch.arange(
            self.value_width,
            device=x.device,
            dtype=torch.long,
        ).unsqueeze(0)
        self._anchor_target_tensors = tuple(
            torch.tensor(pattern, device=x.device, dtype=x.dtype)
            for pattern in self.anchor_variants
        )

    def close(self) -> None:
        return None

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.perf_counter() - self._started_at

    def _anchor_candidates(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
    ) -> List[Tuple[int, Tuple[int, ...], float, float]]:
        """Return ``(absolute_start, pattern, score, observed_ratio)`` candidates."""

        sequence_logits = logits[0]
        sequence_max = sequence_logits.amax(dim=-1)
        absolute_end = logits_start + sequence_logits.shape[0]
        generation_start = self.prompt_length
        generation_end = min(
            x.shape[1], self.prompt_length + self.gen_length
        )
        if self._search_start is not None:
            generation_start = max(generation_start, self._search_start)
            generation_end = min(generation_end, self._search_end)
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}

        for pattern in self.anchor_variants:
            width = len(pattern)
            start = max(generation_start, logits_start)
            end = min(generation_end, absolute_end) - width + 1
            if end <= start:
                continue
            relative = start - logits_start
            count = end - start
            scores = torch.zeros(count, device=logits.device, dtype=torch.float32)
            compatible = torch.ones(count, device=logits.device, dtype=torch.bool)
            observed = torch.zeros(count, device=logits.device, dtype=torch.float32)

            for offset, token_id in enumerate(pattern):
                row_start = relative + offset
                row_end = row_start + count
                target = sequence_logits[row_start:row_end, token_id].float()
                scores += target - sequence_max[row_start:row_end].float()
                absolute_positions = torch.arange(
                    start + offset,
                    end + offset,
                    device=x.device,
                )
                current = x[0, absolute_positions]
                compatible &= (current == self.mask_id) | (current == token_id)
                observed += (current == token_id).float()

            scores /= float(width)
            observed /= float(width)
            if pattern in self.speculative_anchor_variants:
                eligible = compatible & (
                    (scores >= self.config.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
            else:
                # Tolerant whitespace forms are evidence only after the normal
                # decoder has actually emitted them.
                eligible = compatible & (observed == 1.0)
            observed_indices = torch.nonzero(
                eligible & (observed == 1.0), as_tuple=False
            ).flatten()
            speculative_indices = torch.nonzero(
                eligible & (observed != 1.0), as_tuple=False
            ).flatten()
            if observed_indices.numel() == 0 and speculative_indices.numel() == 0:
                continue
            # A fully materialized anchor is direct evidence from the normal
            # response and must never be displaced by higher-logit speculative
            # positions. Limit only the speculative host transfers.
            selected_indices = observed_indices.detach().cpu().tolist()
            if speculative_indices.numel() > 0:
                top_count = min(
                    int(speculative_indices.numel()),
                    self.config.priority_slots * 8,
                )
                top = torch.topk(scores[speculative_indices], k=top_count).indices
                selected_indices.extend(
                    speculative_indices[top].detach().cpu().tolist()
                )
            for local_index in selected_indices:
                absolute_start = start + int(local_index)
                candidate = (
                    absolute_start,
                    pattern,
                    float(scores[local_index].detach().cpu()),
                    float(observed[local_index].detach().cpu()),
                )
                previous = candidates.get(absolute_start)
                if previous is None or (candidate[3], candidate[2]) > (
                    previous[3], previous[2]
                ):
                    candidates[absolute_start] = candidate

        # Different whitespace variants can describe the same field with starts
        # a token or two apart.  Collapse those local alternatives by confidence
        # before applying occurrence order; otherwise a weaker pretty-print
        # variant immediately before a compact anchor can steal its slot.
        clustered = []
        cluster_radius = max(len(pattern) for pattern in self.anchor_variants)
        for candidate in sorted(candidates.values(), key=lambda item: item[0]):
            if not clustered or candidate[0] - clustered[-1][-1][0] >= cluster_radius:
                clustered.append([candidate])
            else:
                clustered[-1].append(candidate)
        ranked = [
            max(cluster, key=lambda item: (item[3], item[2]))
            for cluster in clustered
        ]
        ranked.sort(key=lambda item: item[0])

        # The policy is explicitly occurrence based: priority slots follow
        # response order rather than confidence rank. Confidence gates
        # admission above; duplicate Agent names remain separate calls.
        selected = []
        for candidate in ranked:
            position = candidate[0]
            if any(
                abs(position - existing[0]) < self.config.min_anchor_gap
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == self.config.priority_slots:
                break

        # Timing-only slots observe every later field that has already been
        # materialized by the normal decoder. They never admit speculative
        # anchors and therefore cannot affect the priority prefetch policy.
        tracked = {candidate[0]: candidate for candidate in selected}
        for candidate in ranked:
            if candidate[3] == 1.0:
                tracked[candidate[0]] = candidate
        selected = []
        for candidate in sorted(tracked.values(), key=lambda item: item[0]):
            position = candidate[0]
            if any(
                abs(position - existing[0]) < self.config.min_anchor_gap
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == self.tracking_slots:
                break
        return selected

    def _assign_anchors(
        self,
        x: torch.Tensor,
        candidates: List[Tuple[int, Tuple[int, ...], float, float]],
    ) -> None:
        for slot_index, runtime in enumerate(self.slots):
            if slot_index >= len(candidates):
                # A later full-sequence observation can prove that an early
                # all-mask pass hallucinated more plan objects than the normal
                # response contains. Drop stale, unmaterialized slots instead
                # of reporting phantom Agent calls.
                if (
                    not runtime.confirmed
                    and not runtime.field_written
                    and runtime.shadow_seconds is None
                ):
                    self.slots[slot_index] = JsonAgentSlotRuntime()
                continue
            anchor_start, pattern, score, observed_ratio = candidates[slot_index]
            if runtime.anchor_start == anchor_start and runtime.anchor_token_ids == pattern:
                runtime.anchor_consistent_steps += 1
                runtime.anchor_score = score
                runtime.anchor_observed_ratio = observed_ratio
                continue
            # A field is written only after its anchor occurs naturally in x.
            # Once written, never erase or relocate that normal response text.
            if (
                runtime.confirmed
                or runtime.field_written
                or runtime.shadow_seconds is not None
            ):
                continue
            runtime.anchor_start = anchor_start
            runtime.anchor_token_ids = pattern
            runtime.name_start = anchor_start + len(pattern)
            runtime.anchor_score = score
            runtime.anchor_observed_ratio = observed_ratio
            runtime.anchor_consistent_steps = 1
            runtime.candidate = None
            runtime.recognized_candidate = None
            runtime.candidate_probability = 0.0
            runtime.candidate_margin = 0.0
            runtime.recognized_probability = None
            runtime.recognized_margin = None
            runtime.candidate_consistent_steps = 0
            runtime.last_distribution = None
            runtime.first_observed_seconds = None
            runtime.recognized_seconds = None
            runtime.confirmed_seconds = None
            runtime.first_observed_step = None
            runtime.recognized_step = None
            runtime.confirmed_step = None
            runtime.predicted_seconds = None
            runtime.predicted_step = None
            runtime.predicted_candidate = None
            runtime.shadow_seconds = None
            runtime.shadow_step = None
            runtime.shadow_candidate = None
            runtime.shadow_anchor_offset = None
            runtime.shadow_anchor_observed_ratio = None
            runtime.shadow_anchor_consistent_steps = None
            runtime.shadow_probability = None
            runtime.shadow_margin = None
            runtime.committed_seconds = None
            runtime.committed_step = None
            runtime.committed_candidate = None
            runtime.committed_anchor_offset = None
            runtime.commit_anchor_observed_ratio = None
            runtime.commit_anchor_consistent_steps = None
            runtime.commit_probability = None
            runtime.commit_margin = None
            runtime.commit_had_mask = None
            runtime.committed_token_count = None
            runtime.final_anchor_match = None
            runtime.final_agent_candidate = None
            runtime.final_agent_match = None
            runtime.materialized_seconds = None
            runtime.materialized_step = None
            runtime.materialized_candidate = None
            runtime.field_written = False
            runtime.fuzzy_matched_from = None

    def _score_catalog(
        self,
        logits: torch.Tensor,
        logits_start: int,
        runtime: JsonAgentSlotRuntime,
    ) -> Optional[Dict[str, float]]:
        if runtime.name_start is None:
            return None
        relative_start = runtime.name_start - logits_start
        relative_end = relative_start + self.value_width
        if relative_start < 0 or relative_end > logits.shape[1]:
            return None
        if self._catalog_target_ids is None or self._catalog_positions is None:
            raise RuntimeError(
                "Agent controller must be initialized before scoring."
            )
        field_logits = logits[0, relative_start:relative_end].float()
        # The per-position log-softmax normalizer is identical for every
        # equal-width candidate, so it cancels when candidate sequence scores
        # are normalized. Gathering candidate logits is exactly equivalent and
        # avoids a full-vocabulary log_softmax on every observation.
        candidate_logits = field_logits[
            self._catalog_positions,
            self._catalog_target_ids,
        ]
        probabilities = torch.softmax(candidate_logits.sum(dim=1), dim=0)
        probability_values = probabilities.detach().cpu().tolist()
        return dict(zip(self.catalog_names, probability_values))

    @staticmethod
    def _normalized_agent_name(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    def _observed_catalog_value(
        self,
        x: torch.Tensor,
        runtime: JsonAgentSlotRuntime,
    ) -> Optional[str]:
        """Return an Agent name already decoded in the normal response."""

        if runtime.name_start is None:
            return None
        for name, token_ids in self.catalog_value_ids.items():
            end = runtime.name_start + len(token_ids)
            if end > x.shape[1]:
                continue
            expected = torch.tensor(token_ids, device=x.device, dtype=x.dtype)
            if torch.equal(x[0, runtime.name_start:end], expected):
                runtime.fuzzy_matched_from = None
                return name

        # Map a naturally emitted near-match for prefetch reporting, without
        # rewriting the response. Separator differences normalize away; other
        # spelling perturbations are accepted only at unique edit distance one.
        window_end = min(x.shape[1], runtime.name_start + self.value_width + 5)
        token_ids = x[0, runtime.name_start:window_end].detach().cpu().tolist()
        visible = []
        for token_id in token_ids:
            if token_id == self.mask_id:
                break
            visible.append(token_id)
        if not visible:
            return None
        decoded = self.tokenizer.decode(visible, skip_special_tokens=True)
        match = re.match(r'^\s*([A-Za-z][A-Za-z0-9_ -]{2,31})\s*["\']', decoded)
        if match is None:
            return None
        raw_name = match.group(1).strip()
        normalized = self._normalized_agent_name(raw_name)
        matches = [
            name
            for name in self.catalog_value_ids
            if self._edit_distance(
                normalized, self._normalized_agent_name(name)
            ) <= 1
        ]
        if len(matches) == 1:
            runtime.fuzzy_matched_from = (
                raw_name
                if normalized != self._normalized_agent_name(matches[0])
                else None
            )
            return matches[0]
        return None

    def _write_candidate(
        self,
        x: torch.Tensor,
        runtime: JsonAgentSlotRuntime,
        candidate: str,
        global_step: int,
    ) -> bool:
        if (
            runtime.field_written
            or runtime.anchor_start is None
            or runtime.name_start is None
        ):
            return False
        value_ids = self.catalog_value_ids[candidate]
        name_end = runtime.name_start + len(value_ids)
        if name_end > x.shape[1]:
            return False
        # A commit is useful only while the target Agent value still contains
        # at least one MASK.  This also prevents a late observation from being
        # mislabeled as an early commit after natural materialization.
        had_mask = bool((x[:, runtime.name_start:name_end] == self.mask_id).any())
        if not had_mask:
            return False
        value = torch.tensor(value_ids, device=x.device, dtype=x.dtype)
        x[:, runtime.name_start:name_end] = value
        runtime.field_written = True
        now = self._elapsed()
        runtime.committed_seconds = now
        runtime.committed_step = global_step
        runtime.committed_candidate = candidate
        anchor_origin = (
            self._search_start
            if self._search_start is not None
            else self.prompt_length
        )
        runtime.committed_anchor_offset = runtime.anchor_start - anchor_origin
        runtime.commit_anchor_observed_ratio = runtime.anchor_observed_ratio
        runtime.commit_anchor_consistent_steps = runtime.anchor_consistent_steps
        runtime.commit_probability = runtime.candidate_probability
        runtime.commit_margin = runtime.candidate_margin
        runtime.commit_had_mask = had_mask
        runtime.committed_token_count = len(value_ids)
        # A successful commit is immediately materialized in the actual
        # generation canvas.  Freeze it through decoder_mask and retain this
        # exact timestamp rather than waiting for the next observation.
        runtime.materialized_seconds = now
        runtime.materialized_step = global_step
        runtime.materialized_candidate = candidate
        return True

    def _update_slot(
        self,
        x: torch.Tensor,
        slot_index: int,
        distribution: Dict[str, float],
        global_step: int,
        allow_write: bool = True,
        from_natural: bool = False,
    ) -> None:
        runtime = self.slots[slot_index]
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        candidate, probability = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = probability - second
        now = self._elapsed()
        if runtime.first_observed_seconds is None:
            runtime.first_observed_seconds = now
            runtime.first_observed_step = global_step
        if runtime.candidate == candidate:
            runtime.candidate_consistent_steps += 1
        else:
            runtime.candidate = candidate
            runtime.candidate_consistent_steps = 1
        runtime.candidate_probability = probability
        runtime.candidate_margin = margin
        runtime.last_distribution = dict(distribution)

        anchor_ready = (
            runtime.anchor_observed_ratio == 1.0
            or runtime.anchor_consistent_steps >= self.config.anchor_stable_steps
        )
        reliable = (
            anchor_ready
            and runtime.candidate_consistent_steps >= 2
            and probability >= self.config.tentative_probability
            and margin >= self.config.tentative_margin
        )
        if reliable:
            if runtime.recognized_candidate is None:
                runtime.recognized_candidate = candidate
                runtime.recognized_probability = probability
                runtime.recognized_margin = margin
                runtime.recognized_seconds = now
                runtime.recognized_step = global_step
            # A logits-based recognition that precedes natural materialization
            # is the prediction the probing experiment measures.  One-hot
            # distributions produced from an already-materialized value are
            # natural observations, not predictions.
            if not from_natural and runtime.predicted_seconds is None:
                runtime.predicted_seconds = now
                runtime.predicted_step = global_step
                runtime.predicted_candidate = candidate
            if self.config.allow_speculative_anchor_commit:
                commit_anchor_ready = (
                    runtime.anchor_consistent_steps
                    >= self.config.anchor_stable_steps
                )
            else:
                commit_anchor_ready = runtime.anchor_observed_ratio == 1.0
            candidate_width = len(self.catalog_value_ids[candidate])
            name_end = (
                runtime.name_start + candidate_width
                if runtime.name_start is not None else None
            )
            name_has_mask = bool(
                name_end is not None
                and name_end < x.shape[1]
                and (x[:, runtime.name_start:name_end] == self.mask_id).any()
            )
            would_commit = (
                slot_index < self.config.priority_slots
                and commit_anchor_ready
                and candidate == runtime.recognized_candidate
                and name_has_mask
            )
            if would_commit and runtime.shadow_seconds is None:
                anchor_origin = (
                    self._search_start
                    if self._search_start is not None
                    else self.prompt_length
                )
                runtime.shadow_seconds = now
                runtime.shadow_step = global_step
                runtime.shadow_candidate = candidate
                runtime.shadow_anchor_offset = runtime.anchor_start - anchor_origin
                runtime.shadow_anchor_observed_ratio = runtime.anchor_observed_ratio
                runtime.shadow_anchor_consistent_steps = (
                    runtime.anchor_consistent_steps
                )
                runtime.shadow_probability = probability
                runtime.shadow_margin = margin
            if (
                allow_write
                and would_commit
            ):
                self._write_candidate(
                    x,
                    runtime,
                    runtime.recognized_candidate,
                    global_step,
                )

        confirm = (
            reliable
            and candidate == runtime.recognized_candidate
            and runtime.anchor_observed_ratio == 1.0
            and probability >= self.config.confirm_probability
            and margin >= self.config.confirm_margin
            and runtime.candidate_consistent_steps >= self.config.confirm_stable_steps
        )
        if confirm and not runtime.confirmed:
            runtime.confirmed = True
            runtime.confirmed_seconds = now
            runtime.confirmed_step = global_step

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                "json_agent_step %s",
                json.dumps(
                    {
                        "step": global_step,
                        "slot": slot_index,
                        "anchor_start": runtime.anchor_start,
                        "anchor_score": runtime.anchor_score,
                        "anchor_consistent_steps": runtime.anchor_consistent_steps,
                        "candidate": candidate,
                        "probability": probability,
                        "margin": margin,
                        "candidate_consistent_steps": runtime.candidate_consistent_steps,
                        "recognized_seconds": runtime.recognized_seconds,
                        "confirmed_seconds": runtime.confirmed_seconds,
                        "confirmed": runtime.confirmed,
                        "fuzzy_matched_from": runtime.fuzzy_matched_from,
                    },
                    sort_keys=True,
                ),
            )

    def _priority_slots_confirmed(self) -> bool:
        priority = self.slots[:self.config.priority_slots]
        return len(priority) == self.config.priority_slots and all(
            runtime.confirmed for runtime in priority
        )

    def _scan_plan_complete(self, x: torch.Tensor, global_step: int) -> None:
        """Record when the END_PLAN_JSON marker first appears naturally."""

        if self.plan_complete_seconds is not None:
            return
        generation_start = self.prompt_length
        generation_end = min(x.shape[1], self.prompt_length + self.gen_length)
        sequence = x[0, generation_start:generation_end]
        for target in self._plan_end_tensors:
            width = target.shape[0]
            if sequence.shape[0] < width:
                continue
            windows = sequence.unfold(0, width, 1)
            if bool((windows == target).all(dim=1).any()):
                self.plan_complete_seconds = self._elapsed()
                self.plan_complete_step = global_step
                return

    def _observe_passive(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        global_step: int,
    ) -> None:
        """Shared read-only observation body used by observe() and probes."""

        # Anchor localization only needs logits that cover the region being
        # searched.  In ordinary Dual Vanilla that region is the complete
        # current generation canvas.  A fixed canvas can be shorter than the
        # configured maximum ``gen_length`` (and can shrink again after
        # compaction), so comparing against ``prompt_length + gen_length``
        # incorrectly rejects its existing full warmup logits.  Conversely,
        # block-local logits must not trigger a region-wide anchor rescan.
        search_start = (
            self.prompt_length
            if self._search_start is None
            else max(self.prompt_length, self._search_start)
        )
        search_end = min(
            x.shape[1],
            self.prompt_length + self.gen_length,
            self._search_end if self._search_end is not None else x.shape[1],
        )
        covers_search_region = (
            search_start < search_end
            and logits_start <= search_start
            and logits_start + logits.shape[1] >= search_end
        )
        if covers_search_region:
            self._full_sequence_observations += 1
            candidates = self._anchor_candidates(logits, x, logits_start)
            self._assign_anchors(x, candidates)
            self._scan_plan_complete(x, global_step)
        for slot_index, runtime in enumerate(self.slots):
            if runtime.confirmed:
                continue
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is not None and runtime.materialized_seconds is None:
                runtime.materialized_seconds = self._elapsed()
                runtime.materialized_step = global_step
                runtime.materialized_candidate = observed_name
            if observed_name is not None:
                distribution = {
                    name: 1.0 if name == observed_name else 0.0
                    for name in self.catalog_names
                }
            else:
                distribution = self._score_catalog(logits, logits_start, runtime)
            if distribution is not None:
                self._update_slot(
                    x,
                    slot_index,
                    distribution,
                    global_step,
                    allow_write=(
                        not self.prediction_only
                        and slot_index < self.config.priority_slots
                        and observed_name is None
                    ),
                    from_natural=observed_name is not None,
                )

    def probing_active(self) -> bool:
        """Whether periodic probes should continue.

        Probes stop once every priority Agent is stably recognized, so late
        blocks with no open question do not pay the extra full-sequence
        forward cost.
        """
        return self.prediction_only and not self._priority_slots_confirmed()

    def observe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        global_step: int,
        is_last_agent_step: bool = False,
    ) -> None:
        del is_last_agent_step
        self._observed_steps += 1
        if not self.prediction_only and self._priority_slots_confirmed():
            return
        self._observe_passive(logits, x, logits_start, global_step)

    def observe_probe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        global_step: int,
    ) -> None:
        """Consume one read-only full-sequence probe.

        The caller performs the extra forward; this method only advances the
        prediction state machine.  It never writes tokens, so the generation
        trajectory is independent of the probing frequency.  Available only in
        the experimental prediction-only mode.
        """
        if not self.prediction_only:
            return
        self.probe_forwards += 1
        self._observed_steps += 1
        self._observe_passive(logits, x, logits_start, global_step)

    def decoder_mask(
        self, mask_index: torch.Tensor, mask_start: int = 0
    ) -> torch.Tensor:
        result = mask_index.clone()
        mask_end = mask_start + result.shape[1]
        for runtime in self.slots:
            if runtime.name_start is None or not runtime.field_written:
                continue
            # A speculative anchor may still contain MASK tokens. Freeze only
            # the exact committed Agent value (including its closing quote).
            # Padding to catalog max width or writing the following comma can
            # overwrite a naturally shorter layout's next JSON key.
            start = runtime.name_start
            width = runtime.committed_token_count
            if width is None:
                width = len(self.catalog_value_ids[runtime.committed_candidate])
            end = runtime.name_start + width
            overlap_start = max(start, mask_start)
            overlap_end = min(end, mask_end)
            if overlap_start < overlap_end:
                result[:, overlap_start - mask_start:overlap_end - mask_start] = False
        return result

    def step_callback(self, nfe, num_block, block_step, x) -> None:
        """Timestamp natural materialization after an unchanged transfer.

        This callback reads ``x`` only.  It is used by the observe-only
        ablation so materialization timing is not delayed until the next block
        warmup, while logits-based recognition still comes exclusively from
        the warmup logits.
        """

        del num_block, block_step
        now = self._elapsed()
        for runtime in self.slots:
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is None or runtime.materialized_seconds is not None:
                continue
            runtime.materialized_seconds = now
            runtime.materialized_step = int(nfe)
            runtime.materialized_candidate = observed_name

    def has_unconfirmed_agents(self) -> bool:
        # Agent discovery must not extend a block's normal denoising loop.  The
        # next block warm-up supplies another full-sequence observation.
        return False

    def _materialized_anchor_candidates(
        self, x: torch.Tensor
    ) -> List[Tuple[int, Tuple[int, ...], float, float]]:
        """Find natural JSON Agent anchors without consulting model logits."""

        generation_start = self.prompt_length
        generation_end = min(
            x.shape[1], self.prompt_length + self.gen_length
        )
        if self._search_start is not None:
            generation_start = max(generation_start, self._search_start)
            generation_end = min(generation_end, self._search_end)
        sequence = x[0, generation_start:generation_end]
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}
        for pattern, target in zip(
            self.anchor_variants, self._anchor_target_tensors
        ):
            width = len(pattern)
            if sequence.shape[0] < width:
                continue
            windows = sequence.unfold(0, width, 1)
            matches = torch.nonzero(
                (windows == target).all(dim=1), as_tuple=False
            ).flatten()
            for relative_start in matches.detach().cpu().tolist():
                absolute_start = generation_start + int(relative_start)
                candidate = (absolute_start, pattern, 0.0, 1.0)
                previous = candidates.get(absolute_start)
                if previous is None or len(pattern) > len(previous[1]):
                    candidates[absolute_start] = candidate

        clustered = []
        cluster_radius = max(len(pattern) for pattern in self.anchor_variants)
        for candidate in sorted(candidates.values(), key=lambda item: item[0]):
            if not clustered or candidate[0] - clustered[-1][-1][0] >= cluster_radius:
                clustered.append([candidate])
            else:
                clustered[-1].append(candidate)
        selected = [
            max(cluster, key=lambda item: len(item[1]))
            for cluster in clustered
        ]
        return selected[:self.tracking_slots]

    def finalize(self, x: torch.Tensor) -> None:
        # Capture a field that materialized in the last generation block, where
        # no subsequent full-sequence warm-up exists. Exact natural JSON is
        # definitive evidence, so this records an end-of-generation upper bound
        # without writing or otherwise changing the response canvas.
        final_anchors = self._materialized_anchor_candidates(x)
        # Evaluate speculative writes against structure that really exists in
        # the final canvas.  Matching by value start handles equivalent anchor
        # whitespace variants while rejecting a hallucinated anchor position.
        final_values = {}
        for anchor_start, pattern, _score, _ratio in final_anchors:
            name_start = anchor_start + len(pattern)
            probe = JsonAgentSlotRuntime(
                anchor_start=anchor_start,
                anchor_token_ids=pattern,
                name_start=name_start,
            )
            final_values[name_start] = self._observed_catalog_value(x, probe)
        for runtime in self.slots:
            if runtime.committed_seconds is None:
                continue
            runtime.final_anchor_match = runtime.name_start in final_values
            runtime.final_agent_candidate = final_values.get(runtime.name_start)
            runtime.final_agent_match = bool(
                runtime.final_anchor_match
                and runtime.final_agent_candidate == runtime.committed_candidate
            )

        self._assign_anchors(x, final_anchors)
        self._scan_plan_complete(x, self._observed_steps)
        now = self._elapsed()
        final_step = self._observed_steps
        for runtime in self.slots:
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is None:
                continue
            if runtime.materialized_seconds is None:
                runtime.materialized_seconds = now
                runtime.materialized_step = final_step
                runtime.materialized_candidate = observed_name
            if runtime.first_observed_seconds is None:
                runtime.first_observed_seconds = now
                runtime.first_observed_step = final_step
            runtime.candidate = observed_name
            runtime.recognized_candidate = observed_name
            runtime.candidate_probability = 1.0
            runtime.candidate_margin = 1.0
            runtime.recognized_probability = 1.0
            runtime.recognized_margin = 1.0
            if runtime.recognized_seconds is None:
                runtime.recognized_seconds = now
                runtime.recognized_step = final_step
            if runtime.confirmed_seconds is None:
                runtime.confirmed_seconds = now
                runtime.confirmed_step = final_step
            runtime.confirmed = True

    def metrics(self) -> Dict[str, object]:
        slots = []
        for index, runtime in enumerate(self.slots):
            if (
                index >= self.config.priority_slots
                and runtime.anchor_start is None
                and runtime.recognized_candidate is None
            ):
                continue
            slots.append(
                {
                    "slot": index,
                    "priority": index < self.config.priority_slots,
                    "agent": runtime.recognized_candidate,
                    "anchor_start": runtime.anchor_start,
                    "anchor_score": (
                        runtime.anchor_score if math.isfinite(runtime.anchor_score) else None
                    ),
                    "anchor_observed_ratio": runtime.anchor_observed_ratio,
                    "first_observed_seconds": runtime.first_observed_seconds,
                    "recognized_seconds": runtime.recognized_seconds,
                    "confirmed_seconds": runtime.confirmed_seconds,
                    "first_observed_step": runtime.first_observed_step,
                    "recognized_step": runtime.recognized_step,
                    "confirmed_step": runtime.confirmed_step,
                    "predicted_seconds": runtime.predicted_seconds,
                    "predicted_step": runtime.predicted_step,
                    "predicted_agent": runtime.predicted_candidate,
                    "shadow_seconds": runtime.shadow_seconds,
                    "shadow_step": runtime.shadow_step,
                    "shadow_agent": runtime.shadow_candidate,
                    "shadow_anchor_offset": runtime.shadow_anchor_offset,
                    "shadow_anchor_observed_ratio": (
                        runtime.shadow_anchor_observed_ratio
                    ),
                    "shadow_anchor_consistent_steps": (
                        runtime.shadow_anchor_consistent_steps
                    ),
                    "shadow_probability": runtime.shadow_probability,
                    "shadow_margin": runtime.shadow_margin,
                    "committed_seconds": runtime.committed_seconds,
                    "committed_step": runtime.committed_step,
                    "committed_agent": runtime.committed_candidate,
                    "committed_anchor_offset": runtime.committed_anchor_offset,
                    "commit_anchor_observed_ratio": runtime.commit_anchor_observed_ratio,
                    "commit_anchor_consistent_steps": runtime.commit_anchor_consistent_steps,
                    "commit_probability": runtime.commit_probability,
                    "commit_margin": runtime.commit_margin,
                    "commit_had_mask": runtime.commit_had_mask,
                    "committed_token_count": runtime.committed_token_count,
                    "final_anchor_match": runtime.final_anchor_match,
                    "wrong_anchor": (
                        not runtime.final_anchor_match
                        if runtime.committed_seconds is not None
                        and runtime.final_anchor_match is not None
                        else None
                    ),
                    "final_agent": runtime.final_agent_candidate,
                    "final_agent_match": runtime.final_agent_match,
                    "commit_correct": (
                        runtime.final_agent_match
                        if runtime.committed_seconds is not None
                        and runtime.final_agent_match is not None
                        else (
                            runtime.committed_candidate
                            == runtime.materialized_candidate
                            if runtime.committed_seconds is not None
                            and runtime.materialized_candidate is not None
                            else None
                        )
                    ),
                    "wrong_commit": (
                        not runtime.final_agent_match
                        if runtime.committed_seconds is not None
                        and runtime.final_agent_match is not None
                        else None
                    ),
                    "commit_lead_vs_materialization": (
                        runtime.materialized_seconds - runtime.committed_seconds
                        if runtime.committed_seconds is not None
                        and runtime.materialized_seconds is not None
                        else None
                    ),
                    "materialized_seconds": runtime.materialized_seconds,
                    "materialized_step": runtime.materialized_step,
                    "materialized_candidate": runtime.materialized_candidate,
                    "probability": runtime.recognized_probability,
                    "margin": runtime.recognized_margin,
                    "confirmed": runtime.confirmed,
                    "fuzzy_matched_from": runtime.fuzzy_matched_from,
                }
            )
        recognized = [
            slot["recognized_seconds"]
            for slot in slots
            if slot["recognized_seconds"] is not None
        ]
        discovered_count = sum(slot["anchor_start"] is not None for slot in slots)
        recognized_count = len(recognized)
        all_recognized = (
            discovered_count > 0 and recognized_count == discovered_count
        )
        priority = slots[:self.config.priority_slots]
        priority_discovered = sum(
            slot["anchor_start"] is not None for slot in priority
        )
        priority_recognized = sum(
            slot["recognized_seconds"] is not None for slot in priority
        )
        priority_predicted = [
            slot["predicted_seconds"] for slot in priority
            if slot["predicted_seconds"] is not None
        ]
        priority_predicted_steps = [
            slot["predicted_step"] for slot in priority
            if slot["predicted_step"] is not None
        ]
        priority_materialized = [
            slot["materialized_seconds"] for slot in priority
            if slot["materialized_seconds"] is not None
        ]
        priority_materialized_steps = [
            slot["materialized_step"] for slot in priority
            if slot["materialized_step"] is not None
        ]
        priority_committed = [
            slot["committed_seconds"] for slot in priority
            if slot["committed_seconds"] is not None
        ]
        priority_committed_steps = [
            slot["committed_step"] for slot in priority
            if slot["committed_step"] is not None
        ]
        priority_shadow = [
            slot["shadow_seconds"] for slot in priority
            if slot["shadow_seconds"] is not None
        ]
        priority_shadow_steps = [
            slot["shadow_step"] for slot in priority
            if slot["shadow_step"] is not None
        ]
        committed_slots = [
            slot for slot in slots if slot.get("committed_seconds") is not None
        ]
        evaluated_commits = [
            slot for slot in committed_slots
            if slot.get("commit_correct") is not None
        ]
        correct_commits = sum(
            bool(slot.get("commit_correct"))
            for slot in evaluated_commits
        )
        wrong_anchors = sum(
            bool(slot.get("wrong_anchor")) for slot in evaluated_commits
        )
        final_agent_matches = sum(
            bool(slot.get("final_agent_match")) for slot in evaluated_commits
        )
        first3_recognized_seconds = (
            max(priority_predicted)
            if len(priority) == self.config.priority_slots
            and len(priority_predicted) == len(priority)
            else None
        )
        first3_materialized_seconds = (
            max(priority_materialized)
            if len(priority) == self.config.priority_slots
            and len(priority_materialized) == len(priority)
            else None
        )
        prediction_complete = (
            len(priority) == self.config.priority_slots
            and len(priority_predicted) == len(priority)
        )
        final_known = prediction_complete and all(
            slot.get("materialized_candidate") is not None for slot in priority
        )
        return {
            "priority_slots": self.config.priority_slots,
            "tracking_slots": self.tracking_slots,
            "catalog": list(self.config.catalog),
            "observed_steps": self._observed_steps,
            "full_sequence_observations": self._full_sequence_observations,
            "probe_period": self.probe_period,
            "prediction_only": self.prediction_only,
            "probe_forwards": self.probe_forwards,
            "plan_complete_seconds": self.plan_complete_seconds,
            "plan_complete_step": self.plan_complete_step,
            "discovered_agent_fields": discovered_count,
            "recognized_agent_fields": recognized_count,
            "all_priority_agents_recognized": (
                priority_discovered > 0
                and priority_recognized == priority_discovered
            ),
            "all_tracked_agents_recognized": all_recognized,
            "agent_slots": slots,
            # Do not label a partial result as "all recognized".  Keep the
            # partial timestamp separately so failed runs remain diagnosable.
            "all_recognized_seconds": (
                max(recognized) if all_recognized else None
            ),
            "first3_recognized_seconds": first3_recognized_seconds,
            "first3_recognized_step": (
                max(priority_predicted_steps)
                if prediction_complete
                and len(priority_predicted_steps) == len(priority)
                else None
            ),
            "first3_recognized_exact": (
                all(
                    slot.get("predicted_agent")
                    == slot.get("materialized_candidate")
                    for slot in priority
                )
                if final_known else None
            ),
            "first3_materialized_seconds": first3_materialized_seconds,
            "first3_materialized_step": (
                max(priority_materialized_steps)
                if len(priority) == self.config.priority_slots
                and len(priority_materialized_steps) == len(priority)
                else None
            ),
            "first3_agent_seconds": first3_materialized_seconds,
            "first3_shadow_seconds": (
                max(priority_shadow)
                if len(priority) == self.config.priority_slots
                and len(priority_shadow) == len(priority)
                else None
            ),
            "first3_shadow_step": (
                max(priority_shadow_steps)
                if len(priority) == self.config.priority_slots
                and len(priority_shadow_steps) == len(priority)
                else None
            ),
            "shadow_count": len([
                slot for slot in slots if slot.get("shadow_seconds") is not None
            ]),
            "shadow_coverage": (
                len(priority_shadow) / len(priority) if priority else 0.0
            ),
            "first3_commit_seconds": (
                max(priority_committed)
                if len(priority) == self.config.priority_slots
                and len(priority_committed) == len(priority)
                else None
            ),
            "first3_commit_step": (
                max(priority_committed_steps)
                if len(priority) == self.config.priority_slots
                and len(priority_committed_steps) == len(priority)
                else None
            ),
            "first3_commit_correct": (
                all(
                    bool(slot.get("commit_correct"))
                    for slot in priority
                )
                if len(priority) == self.config.priority_slots
                and len(priority_committed) == len(priority)
                and all(
                    slot.get("materialized_candidate") is not None
                    for slot in priority
                )
                else None
            ),
            "commit_count": len(committed_slots),
            "commit_coverage": (
                len(priority_committed) / len(priority) if priority else 0.0
            ),
            "commit_evaluated_count": len(evaluated_commits),
            "commit_correct_count": correct_commits,
            "wrong_commit_count": len(evaluated_commits) - correct_commits,
            "wrong_anchor_count": wrong_anchors,
            "wrong_anchor_rate": (
                wrong_anchors / len(evaluated_commits)
                if evaluated_commits else None
            ),
            "final_agent_match_count": final_agent_matches,
            "final_agent_match_rate": (
                final_agent_matches / len(evaluated_commits)
                if evaluated_commits else None
            ),
            "commit_accuracy": (
                correct_commits / len(evaluated_commits)
                if evaluated_commits else None
            ),
            "wrong_commit_rate": (
                1.0 - correct_commits / len(evaluated_commits)
                if evaluated_commits else None
            ),
            "latest_partial_recognized_seconds": (
                max(recognized) if recognized else None
            ),
        }

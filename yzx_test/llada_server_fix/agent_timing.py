"""Compact Agent prediction/materialization timing records."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


LOGGER = logging.getLogger("fastdllm.agent_timing")


class AgentTimingRecorder:
    """Keep one latest compact JSONL record per model/query/method."""

    schema_version = 7

    def __init__(self, log_path: str) -> None:
        if not log_path:
            raise ValueError("Agent timing log path cannot be empty.")
        self.log_path = Path(log_path).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8"):
            pass
        self.session_id = f"agent-session-{uuid.uuid4().hex}"
        self.started_unix = time.time()
        self._lock = threading.Lock()
        self._migration_backup_path: Optional[str] = None
        self._records = self._load_canonical_records()

    @staticmethod
    def _request_key(model: str, query: str, method: Optional[str]) -> str:
        value = f"{model}\0{query}\0method={method or 'base'}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _record_key(cls, record: Dict[str, Any]) -> str:
        query = record.get("query")
        if query is not None:
            return cls._request_key(
                str(record.get("model") or ""), str(query),
                record.get("method") or record.get("policy"),
            )
        query_hash = str(record.get("query_sha256") or "")
        if not query_hash:
            raise ValueError("Timing record has neither query nor query_sha256.")
        return hashlib.sha256(
            f"legacy\0{record.get('model')}\0{query_hash}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _slot_prediction(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = (
            (slot.get("prefetched_agent"), slot.get("T_slot_prefetch"),
             slot.get("prefetch_source")),
            (slot.get("predicted_agent"), slot.get("predicted_seconds"),
             "local"),
            (slot.get("shadow_agent"), slot.get("shadow_seconds"),
             "local"),
            (slot.get("committed_agent"), slot.get("committed_seconds"),
             "commit"),
        )
        for agent, seconds, source in candidates:
            if agent is None or seconds is None or source == "natural":
                continue
            return {
                "slot": slot.get("slot"),
                "agent": str(agent),
                "seconds": float(seconds),
                "source": source or "observer",
                "probability": slot.get("probability"),
                "margin": slot.get("margin"),
            }
        return None

    @staticmethod
    def _slot_natural(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        agent = (slot.get("materialized_candidate")
                 or slot.get("natural_decoded_agent") or slot.get("agent"))
        seconds = slot.get("materialized_seconds")
        if seconds is None:
            seconds = slot.get("T_natural_decode")
        if seconds is None:
            seconds = slot.get("confirmation_seconds")
        if agent is None or seconds is None:
            return None
        return {"slot": slot.get("slot"), "agent": str(agent),
                "seconds": float(seconds)}

    @staticmethod
    def _slot_fusion(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if (
            slot.get("T_fused_prefetch") is None
            and slot.get("T_natural_decode") is None
        ):
            return None
        fields = (
            "slot", "T_first_local_evidence", "T_local_prediction",
            "T_natural_decode",
            "T_fused_prefetch", "predicted_agent", "natural_agent",
            "prediction_correct", "source", "local_lead", "fused_lead",
            "local_observation_count_before_natural",
            "wrong_prefetch", "T_correction", "wrong_prefetch_duration",
            "corrected_agent",
        )
        return {field: slot.get(field) for field in fields}

    @staticmethod
    def _slot_commit(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fields = (
            "slot_id", "commit_source", "committed_agent", "natural_agent",
            "commit_wall_time",
            "natural_agent_wall_time", "prefix",
            "hypothesis_id", "region_score", "region_margin",
            "semantic_confidence", "semantic_margin", "stable_count",
            "correct", "strict_semantic_early", "lead_seconds",
            "prefetch_triggered", "prefetch_reused",
        )
        if not any(slot.get(field) is not None for field in fields):
            return None
        result = {field: slot.get(field) for field in fields}
        # Schema <=6 called the completed comparison target final_agent.
        if "natural_agent" not in slot:
            result["natural_agent"] = slot.get("final_agent")
        return result

    @classmethod
    def _compact_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        slots = record.get("agents") or []
        predicted = record.get("predicted_agents")
        if predicted is None:
            predicted = [cls._slot_prediction(slot) for slot in slots]
        natural = record.get("natural_agents")
        if natural is None:
            natural = [cls._slot_natural(slot) for slot in slots]
        predicted = [row for row in predicted if row is not None]
        natural = [row for row in natural if row is not None]
        method = record.get("method") or record.get("policy") or "base"
        query = str(record.get("query") or "")
        if method == "base" and isinstance(record.get("natural_agents"), list):
            return {
                "schema_version": cls.schema_version,
                "session_id": record.get("session_id"),
                "request_key": cls._request_key(
                    str(record.get("model") or ""), query, method
                ) if query else record.get("request_key"),
                "request_index": record.get("request_index"),
                "attempt_count": record.get("attempt_count", 1),
                "first_created_unix": record.get(
                    "first_created_unix", record.get("created_unix")
                ),
                "completion_id": record.get("completion_id"),
                "created_unix": record.get("created_unix"),
                "query": query,
                "query_sha256": record.get("query_sha256") or hashlib.sha256(
                    query.encode("utf-8")
                ).hexdigest(),
                "model": record.get("model"),
                "method": method,
                "status": record.get("status", "ok"),
                "error": record.get("error"),
                "natural_agents": natural,
                "agent_count": record.get("agent_count", len(natural)),
                "full_generation_time": record.get(
                    "full_generation_time", record.get("generation_seconds")
                ),
                "generation_seconds": record.get("generation_seconds"),
                "raw_output_sha256": record.get("raw_output_sha256"),
                "nfe": record.get("nfe"),
                "unresolved_mask_count": record.get("unresolved_mask_count"),
            }
        if method == "plan" and isinstance(record.get("natural_agents"), list):
            return {
                "schema_version": cls.schema_version,
                "session_id": record.get("session_id"),
                "request_key": cls._request_key(
                    str(record.get("model") or ""), query, method
                ) if query else record.get("request_key"),
                "request_index": record.get("request_index"),
                "attempt_count": record.get("attempt_count", 1),
                "first_created_unix": record.get(
                    "first_created_unix", record.get("created_unix")
                ),
                "completion_id": record.get("completion_id"),
                "created_unix": record.get("created_unix"),
                "query": query,
                "query_sha256": record.get("query_sha256") or hashlib.sha256(
                    query.encode("utf-8")
                ).hexdigest(),
                "model": record.get("model"),
                "method": method,
                "status": record.get("status", "ok"),
                "error": record.get("error"),
                "natural_agents": natural,
                "agent_count": record.get("agent_count", len(natural)),
                "generation_seconds": record.get("generation_seconds"),
                "nfe": record.get("nfe"),
                "raw_output_sha256": record.get("raw_output_sha256"),
                "final_plan_parse_success": record.get(
                    "final_plan_parse_success"
                ),
                "plan_end_natural_success": record.get(
                    "plan_end_natural_success"
                ),
                "reasoning_end_natural_success": record.get(
                    "reasoning_end_natural_success"
                ),
                "plan_capacity_exhausted": record.get(
                    "plan_capacity_exhausted"
                ),
                "reasoning_capacity_exhausted": record.get(
                    "reasoning_capacity_exhausted"
                ),
                "unresolved_mask_count": record.get("unresolved_mask_count"),
            }
        if method == "commit" and isinstance(record.get("agent_commits"), list):
            commits = [
                cls._slot_commit(slot) for slot in record["agent_commits"]
            ]
            commits = [row for row in commits if row is not None]
            return {
                "schema_version": cls.schema_version,
                "session_id": record.get("session_id"),
                "request_key": cls._request_key(
                    str(record.get("model") or ""), query, method
                ) if query else record.get("request_key"),
                "request_index": record.get("request_index"),
                "attempt_count": record.get("attempt_count", 1),
                "first_created_unix": record.get(
                    "first_created_unix", record.get("created_unix")
                ),
                "completion_id": record.get("completion_id"),
                "created_unix": record.get("created_unix"),
                "query": query,
                "query_sha256": record.get("query_sha256") or hashlib.sha256(
                    query.encode("utf-8")
                ).hexdigest(),
                "model": record.get("model"),
                "method": method,
                "status": record.get("status", "ok"),
                "error": record.get("error"),
                "agent_commits": commits,
                "agent_count": record.get("agent_count", len(commits)),
                "latent_region_commit_count": record.get(
                    "latent_region_commit_count", 0
                ),
                "semantic_commit_count": record.get(
                    "semantic_commit_count", 0
                ),
                "prefix_commit_count": record.get("prefix_commit_count", 0),
                "natural_commit_count": record.get(
                    "natural_commit_count", 0
                ),
                "semantic_observation_stride": record.get(
                    "semantic_observation_stride"
                ),
                "semantic_scorer_call_count": record.get(
                    "semantic_scorer_call_count", 0
                ),
                "semantic_scorer_total_time": record.get(
                    "semantic_scorer_total_time", 0.0
                ),
                "semantic_scorer_mean_time": record.get(
                    "semantic_scorer_mean_time", 0.0
                ),
                "semantic_skipped_same_evidence_count": record.get(
                    "semantic_skipped_same_evidence_count", 0
                ),
                "semantic_skipped_evidence_not_ready_count": record.get(
                    "semantic_skipped_evidence_not_ready_count", 0
                ),
                "semantic_skipped_stride_count": record.get(
                    "semantic_skipped_stride_count", 0
                ),
                "first_valid_commit_accuracy": record.get(
                    "first_valid_commit_accuracy"
                ),
                "latent_region_commit_accuracy": record.get(
                    "latent_region_commit_accuracy"
                ),
                "correct_latent_coverage": record.get(
                    "correct_latent_coverage"
                ),
                "correct_latent_commit_count": record.get(
                    "correct_latent_commit_count", 0
                ),
                "correct_latent_lead": record.get("correct_latent_lead"),
                "source_distribution": record.get("source_distribution"),
                "latent_trigger_count": record.get("latent_trigger_count", 0),
                "latent_trigger_precision": record.get(
                    "latent_trigger_precision"
                ),
                "latent_trigger_categories": record.get(
                    "latent_trigger_categories"
                ),
                "latent_trigger_events": record.get("latent_trigger_events"),
                "runtime_agent_registry": record.get(
                    "runtime_agent_registry"
                ),
                "latent_region_radius": record.get("latent_region_radius"),
                "latent_region_aggregation": record.get(
                    "latent_region_aggregation"
                ),
                "latent_region_stable_observations": record.get(
                    "latent_region_stable_observations"
                ),
                "hypothesis_min_seen": record.get("hypothesis_min_seen"),
                "hypothesis_min_support": record.get(
                    "hypothesis_min_support"
                ),
                "full_sequence_observation_count": record.get(
                    "full_sequence_observation_count"
                ),
                "extra_model_forwards": record.get(
                    "extra_model_forwards", 0
                ),
                "read_only": record.get("read_only"),
                "writes_agent_tokens": record.get("writes_agent_tokens"),
                "changes_decoder_mask": record.get("changes_decoder_mask"),
                "triggers_real_prefetch": record.get(
                    "triggers_real_prefetch"
                ),
                "diagnostic_observer_overhead_seconds": record.get(
                    "diagnostic_observer_overhead_seconds"
                ),
                "generation_seconds_excluding_diagnostic": record.get(
                    "generation_seconds_excluding_diagnostic"
                ),
                "first_agent_commit_time": record.get(
                    "first_agent_commit_time"
                ),
                "all_agents_commit_time": record.get(
                    "all_agents_commit_time"
                ),
                "full_generation_time": record.get(
                    "full_generation_time", record.get("generation_seconds")
                ),
                "generation_seconds": record.get("generation_seconds"),
                "raw_output_sha256": record.get("raw_output_sha256"),
                "nfe": record.get("nfe"),
                "final_plan_parse_success": record.get(
                    "final_plan_parse_success"
                ),
                "unresolved_mask_count": record.get("unresolved_mask_count"),
            }
        return {
            "schema_version": cls.schema_version,
            "session_id": record.get("session_id"),
            "request_key": cls._request_key(
                str(record.get("model") or ""), query, method
            ) if query else record.get("request_key"),
            "request_index": record.get("request_index"),
            "attempt_count": record.get("attempt_count", 1),
            "first_created_unix": record.get(
                "first_created_unix", record.get("created_unix")),
            "completion_id": record.get("completion_id"),
            "created_unix": record.get("created_unix"),
            "query": query,
            "query_sha256": record.get("query_sha256") or hashlib.sha256(
                query.encode("utf-8")).hexdigest(),
            "model": record.get("model"),
            "method": method,
            "status": record.get("status", "ok"),
            "error": record.get("error"),
            "predicted_agents": predicted,
            "natural_agents": natural,
            "agent_fusion": record.get("agent_fusion") or [],
            "first3_prediction_seconds": max(
                (row["seconds"] for row in predicted[:3]), default=None
            ) if len(predicted) >= 3 else None,
            "first3_natural_seconds": max(
                (row["seconds"] for row in natural[:3]), default=None
            ) if len(natural) >= 3 else None,
            "all_natural_seconds": max(
                (row["seconds"] for row in natural), default=None),
            "agent_count": record.get("agent_count", len(natural)),
            "first_agent_seconds": record.get("first_agent_seconds"),
            "generation_seconds": record.get("generation_seconds"),
            "nfe": record.get("nfe"),
            "final_plan_parse_success": record.get(
                "final_plan_parse_success"
            ),
            "plan_end_natural_success": record.get(
                "plan_end_natural_success"
            ),
            "reasoning_end_natural_success": record.get(
                "reasoning_end_natural_success"
            ),
            "plan_capacity_exhausted": record.get(
                "plan_capacity_exhausted"
            ),
            "reasoning_capacity_exhausted": record.get(
                "reasoning_capacity_exhausted"
            ),
            "unresolved_mask_count": record.get("unresolved_mask_count"),
            "local_observation_stride": record.get(
                "local_observation_stride"
            ),
            "local_agent_probability": record.get(
                "local_agent_probability"
            ),
            "local_observer_warmup_calls": record.get(
                "local_observer_warmup_calls"
            ),
            "local_observer_refinement_calls": record.get(
                "local_observer_refinement_calls"
            ),
            "local_observer_wall_time": record.get(
                "local_observer_wall_time"
            ),
            "T_first3_plan": record.get("T_first3_plan"),
            "T_first3_all": record.get("T_first3_all"),
            "all_first3_speculative_exact": record.get(
                "all_first3_speculative_exact"
            ),
            "local_first3_coverage": record.get("local_first3_coverage"),
            "local_first3_exact": record.get("local_first3_exact"),
            "correct_first3_lead": record.get("correct_first3_lead"),
            "prediction_incremental_lead": record.get(
                "prediction_incremental_lead"
            ),
            "wrong_speculative_rate": record.get("wrong_speculative_rate"),
            "natural_fallback_rate": record.get("natural_fallback_rate"),
            "all_not_later_than_plan": record.get("all_not_later_than_plan"),
        }

    def _read_records_from_disk(self) -> List[Dict[str, Any]]:
        records = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid timing JSONL at {self.log_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError("Timing JSONL records must be JSON objects.")
                records.append(value)
        return records

    def _atomic_write_log(self, records: List[Dict[str, Any]]) -> None:
        temporary = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, self.log_path)

    def _backup_legacy_log(self) -> Path:
        suffix = self.session_id.rsplit("-", 1)[-1][:12]
        backup = self.log_path.with_name(
            f"{self.log_path.name}.precanonical.{suffix}.bak")
        shutil.copy2(self.log_path, backup)
        self._migration_backup_path = str(backup)
        return backup

    def _load_canonical_records(self) -> Dict[str, Dict[str, Any]]:
        raw = self._read_records_from_disk()
        canonical: Dict[str, Dict[str, Any]] = {}
        migration = any(row.get("schema_version") != self.schema_version for row in raw)
        for source in raw:
            record = (dict(source) if source.get("schema_version") == self.schema_version
                      else self._compact_record(source))
            key = self._record_key(record)
            previous = canonical.get(key)
            if previous is not None:
                migration = True
                record["attempt_count"] = max(
                    int(record.get("attempt_count") or 1),
                    int(previous.get("attempt_count") or 1) + 1)
                record["first_created_unix"] = previous.get(
                    "first_created_unix", previous.get("created_unix"))
                record["request_index"] = previous.get("request_index")
            elif record.get("request_index") is None:
                migration = True
                record["request_index"] = len(canonical) + 1
            canonical[key] = record
        if migration and raw:
            self._backup_legacy_log()
            self._atomic_write_log(list(canonical.values()))
        return canonical

    def record(self, *, completion_id: str, created_unix: int, query: str,
               model: str, temperature: float,
               requested_max_tokens: Optional[int],
               metrics: Optional[Dict[str, Any]] = None,
               error: Optional[str] = None) -> Dict[str, Any]:
        del temperature, requested_max_tokens
        metrics = metrics or {}
        priority = metrics.get("agent_priority") or {}
        slots = priority.get("agent_slots") or []
        method = str(metrics.get("method") or priority.get("policy") or "base")
        commits = []
        if method == "commit":
            commits = [self._slot_commit(slot) for slot in slots]
            commits = [row for row in commits if row is not None]
            # The commit schema supersedes the legacy predicted/natural/fusion
            # summaries. Avoid computing metrics which will not be persisted.
            predicted = []
            natural = []
            fusion = []
        elif method == "base":
            # Base observes only complete Agent values naturally present in x.
            # Do not derive prediction, fusion, or commit summaries for it.
            predicted = []
            natural = [self._slot_natural(slot) for slot in slots]
            natural = [row for row in natural if row is not None]
            fusion = []
        elif method in {
            "oracle_latent", "online_latent_diagnostic",
            "online_latent_refine", "online_latent_refine_tracking",
            "online_latent_region",
            "online_latent_hypothesis",
        }:
            # Oracle trajectories are already finalized by the diagnostic
            # observer. They are persisted verbatim in a dedicated branch
            # below and must not be interpreted as runtime predictions.
            predicted = []
            natural = []
            fusion = []
        else:
            predicted = [self._slot_prediction(slot) for slot in slots]
            natural = [self._slot_natural(slot) for slot in slots]
            fusion = [self._slot_fusion(slot) for slot in slots]
            predicted = [row for row in predicted if row is not None]
            natural = [row for row in natural if row is not None]
            fusion = [row for row in fusion if row is not None]
            natural_by_slot = {row["slot"]: row for row in natural}
            for row in predicted:
                expected = natural_by_slot.get(row["slot"])
                row["correct"] = (
                    row["agent"] == expected["agent"] if expected else None
                )
        request_key = self._request_key(model, query, method)
        record = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "request_key": request_key,
            "request_index": None,
            "attempt_count": None,
            "first_created_unix": None,
            "completion_id": completion_id,
            "created_unix": int(created_unix),
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "model": model,
            "method": method,
            "status": "error" if error else "ok",
            "error": error,
            "predicted_agents": predicted,
            "natural_agents": natural,
            "agent_fusion": fusion,
            "agent_commits": commits,
            "first3_prediction_seconds": max(
                (row["seconds"] for row in predicted[:3]), default=None
            ) if len(predicted) >= 3 else None,
            "first3_natural_seconds": max(
                (row["seconds"] for row in natural[:3]), default=None
            ) if len(natural) >= 3 else None,
            "all_natural_seconds": max(
                (row["seconds"] for row in natural), default=None),
            "agent_count": len(natural),
            "first_agent_seconds": priority.get("first_agent_seconds"),
            "generation_seconds": metrics.get("generation_seconds"),
            "full_generation_time": metrics.get("generation_seconds"),
            "raw_output_sha256": metrics.get("raw_output_sha256"),
            "nfe": metrics.get("nfe"),
            "final_plan_parse_success": priority.get(
                "final_plan_parse_success"
            ),
            "plan_end_natural_success": priority.get(
                "plan_end_natural_success"
            ),
            "reasoning_end_natural_success": priority.get(
                "reasoning_end_natural_success"
            ),
            "plan_capacity_exhausted": priority.get(
                "plan_capacity_exhausted"
            ),
            "reasoning_capacity_exhausted": priority.get(
                "reasoning_capacity_exhausted"
            ),
            "unresolved_mask_count": priority.get(
                "unresolved_mask_count", metrics.get("unresolved_mask_count")
            ),
            "local_observation_stride": priority.get(
                "local_observation_stride"
            ),
            "local_agent_probability": priority.get(
                "local_agent_probability"
            ),
            "local_observer_warmup_calls": priority.get(
                "local_observer_warmup_calls"
            ),
            "local_observer_refinement_calls": priority.get(
                "local_observer_refinement_calls"
            ),
            "local_observer_wall_time": priority.get(
                "local_observer_wall_time"
            ),
            "T_first3_plan": priority.get("T_first3_plan"),
            "T_first3_all": priority.get("T_first3_all"),
            "all_first3_speculative_exact": priority.get(
                "all_first3_speculative_exact"
            ),
            "local_first3_coverage": priority.get("local_first3_coverage"),
            "local_first3_exact": priority.get("local_first3_exact"),
            "correct_first3_lead": priority.get("correct_first3_lead"),
            "prediction_incremental_lead": priority.get(
                "prediction_incremental_lead"
            ),
            "wrong_speculative_rate": priority.get("wrong_speculative_rate"),
            "natural_fallback_rate": priority.get("natural_fallback_rate"),
            "all_not_later_than_plan": priority.get("all_not_later_than_plan"),
            "semantic_observation_stride": priority.get(
                "semantic_observation_stride"
            ),
            "semantic_scorer_call_count": priority.get(
                "semantic_scorer_call_count"
            ),
            "semantic_scorer_total_time": priority.get(
                "semantic_scorer_total_time"
            ),
            "semantic_scorer_mean_time": priority.get(
                "semantic_scorer_mean_time"
            ),
            "semantic_skipped_same_evidence_count": priority.get(
                "semantic_skipped_same_evidence_count"
            ),
            "semantic_skipped_evidence_not_ready_count": priority.get(
                "semantic_skipped_evidence_not_ready_count"
            ),
            "semantic_skipped_stride_count": priority.get(
                "semantic_skipped_stride_count"
            ),
            "semantic_min_task_visible_tokens": priority.get(
                "semantic_min_task_visible_tokens"
            ),
            "semantic_min_rationale_visible_tokens": priority.get(
                "semantic_min_rationale_visible_tokens"
            ),
            "semantic_require_materialized_agent_key": priority.get(
                "semantic_require_materialized_agent_key"
            ),
        }
        if method == "base":
            # Keep the baseline log focused on natural Agent materialization.
            base_fields = (
                "schema_version", "session_id", "request_key",
                "request_index", "attempt_count", "first_created_unix",
                "completion_id", "created_unix", "query", "query_sha256",
                "model", "method", "status", "error", "natural_agents",
                "agent_count", "full_generation_time", "generation_seconds",
                "raw_output_sha256", "nfe", "unresolved_mask_count",
            )
            record["agent_count"] = len(natural)
            record = {field: record.get(field) for field in base_fields}
        elif method == "plan":
            # Large-scale PLAN runs deliberately use a narrow schema.  Do not
            # serialize empty prediction/fusion fields or ablation-only data.
            plan_fields = (
                "schema_version", "session_id", "request_key",
                "request_index", "attempt_count", "first_created_unix",
                "completion_id", "created_unix", "query", "query_sha256",
                "model", "method", "status", "error", "natural_agents",
                "agent_count", "generation_seconds", "nfe",
                "raw_output_sha256",
                "final_plan_parse_success", "plan_end_natural_success",
                "reasoning_end_natural_success", "plan_capacity_exhausted",
                "reasoning_capacity_exhausted", "unresolved_mask_count",
            )
            record = {field: record.get(field) for field in plan_fields}
        elif method == "commit":
            # Event timestamps were captured inside generation before any
            # serialization or file I/O. Persist only the requested per-slot
            # commit data and request-level aggregates after generation ends.
            commit_fields = (
                "schema_version", "session_id", "request_key",
                "request_index", "attempt_count", "first_created_unix",
                "completion_id", "created_unix", "query", "query_sha256",
                "model", "method", "status", "error", "agent_commits",
                "agent_count", "latent_region_commit_count",
                "semantic_commit_count",
                "prefix_commit_count", "natural_commit_count",
                "semantic_observation_stride",
                "semantic_scorer_call_count",
                "semantic_scorer_total_time",
                "semantic_scorer_mean_time",
                "semantic_skipped_same_evidence_count",
                "semantic_skipped_evidence_not_ready_count",
                "semantic_skipped_stride_count",
                "first_valid_commit_accuracy",
                "latent_region_commit_accuracy", "correct_latent_coverage",
                "correct_latent_commit_count", "correct_latent_lead",
                "source_distribution", "latent_trigger_count",
                "latent_trigger_precision", "latent_trigger_categories",
                "latent_trigger_events", "runtime_agent_registry",
                "latent_region_radius", "latent_region_aggregation",
                "latent_region_stable_observations",
                "hypothesis_min_seen", "hypothesis_min_support",
                "full_sequence_observation_count", "extra_model_forwards",
                "read_only", "writes_agent_tokens",
                "changes_decoder_mask", "triggers_real_prefetch",
                "diagnostic_observer_overhead_seconds",
                "generation_seconds_excluding_diagnostic",
                "first_agent_commit_time", "all_agents_commit_time",
                "full_generation_time", "generation_seconds",
                "raw_output_sha256", "nfe", "final_plan_parse_success",
                "unresolved_mask_count",
            )
            record.update({
                "agent_count": priority.get("agent_count", len(commits)),
                "latent_region_commit_count": priority.get(
                    "latent_region_commit_count", 0
                ),
                "semantic_commit_count": priority.get(
                    "semantic_commit_count", 0
                ),
                "prefix_commit_count": priority.get("prefix_commit_count", 0),
                "natural_commit_count": priority.get(
                    "natural_commit_count", 0
                ),
                "first_agent_commit_time": priority.get(
                    "first_agent_commit_time"
                ),
                "all_agents_commit_time": priority.get(
                    "all_agents_commit_time"
                ),
            })
            for field in (
                "first_valid_commit_accuracy",
                "latent_region_commit_accuracy", "correct_latent_coverage",
                "correct_latent_commit_count", "correct_latent_lead",
                "source_distribution", "latent_trigger_count",
                "latent_trigger_precision", "latent_trigger_categories",
                "latent_trigger_events", "runtime_agent_registry",
                "latent_region_radius", "latent_region_aggregation",
                "latent_region_stable_observations",
                "hypothesis_min_seen", "hypothesis_min_support",
                "full_sequence_observation_count", "extra_model_forwards",
                "read_only", "writes_agent_tokens",
                "changes_decoder_mask", "triggers_real_prefetch",
                "diagnostic_observer_overhead_seconds",
            ):
                record[field] = priority.get(field)
            for field in (
                "semantic_observation_stride",
                "semantic_scorer_call_count",
                "semantic_scorer_total_time",
                "semantic_scorer_mean_time",
                "semantic_skipped_same_evidence_count",
                "semantic_skipped_evidence_not_ready_count",
                "semantic_skipped_stride_count",
            ):
                record[field] = priority.get(field)
            overhead = priority.get("diagnostic_observer_overhead_seconds")
            generation = metrics.get("generation_seconds")
            record["generation_seconds_excluding_diagnostic"] = (
                max(0.0, float(generation) - float(overhead))
                if isinstance(generation, (int, float))
                and isinstance(overhead, (int, float))
                else None
            )
            record = {field: record.get(field) for field in commit_fields}
        elif method in {
            "oracle_latent", "online_latent_diagnostic",
            "online_latent_refine", "online_latent_refine_tracking",
            "online_latent_region",
            "online_latent_hypothesis",
        }:
            oracle_slots = []
            for source_slot in slots:
                slot = dict(source_slot)
                trajectory_fields = (
                    ("trajectory",)
                    if method == "oracle_latent"
                    else (
                        "oracle_trajectory",
                        "online_trajectory",
                        *(("region_trajectory",)
                          if method in {
                              "online_latent_region",
                              "online_latent_hypothesis",
                          } else ()),
                        *(("hypothesis_trajectory",)
                          if method == "online_latent_hypothesis" else ()),
                    )
                )
                for trajectory_field in trajectory_fields:
                    slot[trajectory_field] = [
                        {
                            "request_id": completion_id,
                            "final_agent": slot.get("final_agent"),
                            "agent_first_char_time": slot.get(
                                "agent_first_char_time"
                            ),
                            "natural_agent_time": slot.get(
                                "natural_agent_time"
                            ),
                            **dict(event),
                        }
                        for event in source_slot.get(trajectory_field) or []
                    ]
                oracle_slots.append(slot)
            slot_field = (
                "oracle_latent_slots"
                if method == "oracle_latent"
                else "online_latent_slots"
            )
            record = {
                "schema_version": self.schema_version,
                "session_id": self.session_id,
                "request_key": request_key,
                "request_index": None,
                "attempt_count": None,
                "first_created_unix": None,
                "completion_id": completion_id,
                "request_id": completion_id,
                "created_unix": int(created_unix),
                "query": query,
                "query_sha256": hashlib.sha256(
                    query.encode("utf-8")
                ).hexdigest(),
                "model": model,
                "method": method,
                "status": "error" if error else "ok",
                "error": error,
                "runtime_agent_registry": priority.get(
                    "runtime_agent_registry"
                ),
                "candidate_tokenizations": priority.get(
                    "candidate_tokenizations"
                ),
                "anchor_candidate_tokenizations": priority.get(
                    "anchor_candidate_tokenizations"
                ),
                "full_template_tokenizations": priority.get(
                    "full_template_tokenizations"
                ),
                "score_type": priority.get("score_type"),
                "raw_score_type": priority.get("raw_score_type"),
                "score_chunk_size": priority.get("score_chunk_size"),
                "observation_cadence": priority.get(
                    "observation_cadence"
                ),
                "full_sequence_observation_count": priority.get(
                    "full_sequence_observation_count"
                ),
                "oracle_span_count": priority.get("oracle_span_count"),
                slot_field: oracle_slots,
                "agent_count": len(oracle_slots),
                "diagnostic_only": True,
                "read_only": priority.get("read_only"),
                "extra_model_forwards": priority.get(
                    "extra_model_forwards", 0
                ),
                "diagnostic_observer_overhead_seconds": priority.get(
                    "diagnostic_observer_overhead_seconds"
                ),
                "anchor_detector": priority.get("anchor_detector"),
                "anchor_min_logit_margin": priority.get(
                    "anchor_min_logit_margin"
                ),
                "anchor_position_tolerance": priority.get(
                    "anchor_position_tolerance"
                ),
                "anchor_stable_observations": priority.get(
                    "anchor_stable_observations"
                ),
                "min_anchor_gap": priority.get("min_anchor_gap"),
                "online_anchor_observation_counts": priority.get(
                    "online_anchor_observation_counts"
                ),
                "online_anchor_slot_count": priority.get(
                    "online_anchor_slot_count"
                ),
                "persistent_tracking": priority.get(
                    "persistent_tracking"
                ),
                "refinement_radius": priority.get("refinement_radius"),
                "refinement_anchor_weight": priority.get(
                    "refinement_anchor_weight"
                ),
                "refinement_agent_weight": priority.get(
                    "refinement_agent_weight"
                ),
                "max_track_misses": priority.get("max_track_misses"),
                "max_track_match_distance": priority.get(
                    "max_track_match_distance"
                ),
                "track_stable_observations": priority.get(
                    "track_stable_observations"
                ),
                "track_score_weight": priority.get("track_score_weight"),
                "tracks_created": priority.get("tracks_created"),
                "stable_tracks": priority.get("stable_tracks"),
                "provisional_tracks": priority.get("provisional_tracks"),
                "unmatched_candidate_count": priority.get(
                    "unmatched_candidate_count"
                ),
                "track_missing_count": priority.get(
                    "track_missing_count"
                ),
                "track_reassociation_count": priority.get(
                    "track_reassociation_count"
                ),
                "slot_identity_shift_count": priority.get(
                    "slot_identity_shift_count"
                ),
                "tracks": priority.get("tracks"),
                "region_radius": priority.get("region_radius"),
                "region_top_k": priority.get("region_top_k"),
                "region_temperature": priority.get(
                    "region_temperature"
                ),
                "region_main_aggregation": priority.get(
                    "region_main_aggregation"
                ),
                "region_main_score_variant": priority.get(
                    "region_main_score_variant"
                ),
                "region_center_source": priority.get(
                    "region_center_source"
                ),
                "exact_refinement_used_for_region_center": priority.get(
                    "exact_refinement_used_for_region_center"
                ),
                "region_aggregations": priority.get(
                    "region_aggregations"
                ),
                "region_score_variants": priority.get(
                    "region_score_variants"
                ),
                "track_false_positive_summary": priority.get(
                    "track_false_positive_summary"
                ),
                "region_track_trajectory": [
                    {"request_id": completion_id, **dict(event)}
                    for event in priority.get("region_track_trajectory") or []
                ],
                "hypothesis_merge_distance": priority.get(
                    "hypothesis_merge_distance"
                ),
                "hypothesis_merge_gap": priority.get(
                    "hypothesis_merge_gap"
                ),
                "hypothesis_min_seen": priority.get(
                    "hypothesis_min_seen"
                ),
                "hypothesis_min_support": priority.get(
                    "hypothesis_min_support"
                ),
                "hypothesis_max_center_jump": priority.get(
                    "hypothesis_max_center_jump"
                ),
                "hypothesis_duplicate_observations": priority.get(
                    "hypothesis_duplicate_observations"
                ),
                "hypothesis_evaluation": priority.get(
                    "hypothesis_evaluation"
                ),
                "hypotheses": priority.get("hypotheses"),
                "hypothesis_trajectory": [
                    {"request_id": completion_id, **dict(event)}
                    for event in priority.get("hypothesis_trajectory") or []
                ],
                "generation_seconds": metrics.get("generation_seconds"),
                "generation_seconds_excluding_diagnostic": metrics.get(
                    "generation_seconds_excluding_diagnostic"
                ),
                "nfe": metrics.get("nfe"),
                "final_plan_parse_success": priority.get(
                    "final_plan_parse_success"
                ),
                "unresolved_mask_count": metrics.get(
                    "unresolved_mask_count"
                ),
            }
        with self._lock:
            previous = self._records.get(request_key)
            if previous is None:
                record["request_index"] = len(self._records) + 1
                record["attempt_count"] = 1
                record["first_created_unix"] = int(created_unix)
            else:
                record["request_index"] = previous["request_index"]
                record["attempt_count"] = int(previous.get("attempt_count") or 1) + 1
                record["first_created_unix"] = previous.get(
                    "first_created_unix", previous.get("created_unix"))
            self._records[request_key] = record
            self._atomic_write_log(list(self._records.values()))
        LOGGER.info(
            "agent_timing_saved request=%s method=%s commits=%d natural=%d",
            record["request_index"], method, len(commits), len(natural))
        return record

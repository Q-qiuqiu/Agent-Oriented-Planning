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

    schema_version = 3

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
             "plan_region_observer"),
            (slot.get("shadow_agent"), slot.get("shadow_seconds"),
             "plan_region_observer"),
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

    @classmethod
    def _compact_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        slots = record.get("agents") or []
        predicted = [cls._slot_prediction(slot) for slot in slots]
        natural = [cls._slot_natural(slot) for slot in slots]
        predicted = [row for row in predicted if row is not None]
        natural = [row for row in natural if row is not None]
        method = record.get("method") or record.get("policy") or "base"
        query = str(record.get("query") or "")
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
            "first3_prediction_seconds": max(
                (row["seconds"] for row in predicted[:3]), default=None
            ) if len(predicted) >= 3 else None,
            "first3_natural_seconds": max(
                (row["seconds"] for row in natural[:3]), default=None
            ) if len(natural) >= 3 else None,
            "all_natural_seconds": max(
                (row["seconds"] for row in natural), default=None),
            "generation_seconds": record.get("generation_seconds"),
            "nfe": record.get("nfe"),
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
        predicted = [self._slot_prediction(slot) for slot in slots]
        natural = [self._slot_natural(slot) for slot in slots]
        predicted = [row for row in predicted if row is not None]
        natural = [row for row in natural if row is not None]
        natural_by_slot = {row["slot"]: row for row in natural}
        for row in predicted:
            expected = natural_by_slot.get(row["slot"])
            row["correct"] = row["agent"] == expected["agent"] if expected else None
        method = str(metrics.get("method") or priority.get("policy") or "base")
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
            "first3_prediction_seconds": max(
                (row["seconds"] for row in predicted[:3]), default=None
            ) if len(predicted) >= 3 else None,
            "first3_natural_seconds": max(
                (row["seconds"] for row in natural[:3]), default=None
            ) if len(natural) >= 3 else None,
            "all_natural_seconds": max(
                (row["seconds"] for row in natural), default=None),
            "generation_seconds": metrics.get("generation_seconds"),
            "nfe": metrics.get("nfe"),
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
            "agent_timing_saved request=%s method=%s predictions=%d natural=%d",
            record["request_index"], method, len(predicted), len(natural))
        return record

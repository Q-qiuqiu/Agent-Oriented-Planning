"""OpenAI-compatible API server for MASK_SLOT-accelerated LLaDA planning."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from agent_timing import AgentTimingRecorder
from generate import generate_with_dual_cache, generate_with_fixed_canvas_dual_cache
from dynamic_fixed_canvas import DynamicFixedCanvasMonitor as FixedCanvasMonitor
from dual_late_decide_observer import DualVanillaLateDecideObserver
from fixed_canvas import NaturalReasoningPlanMonitor
from json_agent_priority import extract_agent_registry
from model.modeling_llada import LLaDAModelLM
from oracle_latent_observer import OracleLatentAgentObserver
from online_latent_anchor_observer import OnlineLatentAnchorObserver
from refined_online_latent_anchor_observer import (
    RefinedOnlineLatentAnchorObserver,
)
from region_latent_agent_observer import RegionLatentAgentObserver
from hypothesis_region_observer import HypothesisRegionObserver
from planner_json_repair import repair_plan_json_response
from response_agent_timing import infer_benchmark


KNOWN_AGENT_NAMES = [
    "search_agent",
    "calculation_agent",
    "reasoning_agent",
    "context_agent",
    "retrieval_agent",
    "knowledge_agent",
    "elimination_agent",
    "evidence_agent",
    "temporal_agent",
    "verification_agent",
    "code_agent",
    "math_agent",
    "commonsense_agent",
]

DEFAULT_AGENT_TIMING_LOG_DIR = str(
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "fastdllm_log"
    / "llada_server_fix"
)


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stop: Optional[Union[str, List[str]]] = None


@dataclass
class ServerConfig:
    model_path: str
    served_model_name: str
    device: str
    block_size: int
    max_gen_length: int
    steps_per_block: int
    agent_slots: int
    agent_timing_slots: int
    agent_names: List[str]
    cache_mode: str
    threshold: float
    agent_anchor_margin: float
    agent_timing_log_dir: str
    plan_json_repair: bool
    api_key: Optional[str]
    structure_mode: str
    reasoning_budget: Optional[int]
    plan_budget: Optional[int]
    reasoning_ratio: float
    plan_ratio: float
    method: str
    local_observation_stride: int
    all_local_probability: float
    oracle_score_chunk_size: int
    online_anchor_position_tolerance: int
    online_anchor_stable_observations: int
    online_refinement_radius: int
    online_refinement_anchor_weight: float
    online_refinement_agent_weight: float
    online_max_track_misses: int
    online_max_track_match_distance: int
    online_track_stable_observations: int
    online_track_score_weight: float
    online_region_radius: int
    online_region_top_k: int
    online_region_temperature: float
    online_region_main_aggregation: str
    hypothesis_merge_distance: int
    hypothesis_merge_gap: int
    hypothesis_min_seen: int
    hypothesis_min_support: float
    hypothesis_max_center_jump: int
    hypothesis_duplicate_observations: int


class LLaDAPlannerRuntime:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.lock = None
        self.timing_recorders: Dict[str, AgentTimingRecorder] = {}

    def _timing_recorder(self, benchmark: str) -> AgentTimingRecorder:
        recorder = self.timing_recorders.get(benchmark)
        if recorder is None:
            log_path = (
                Path(self.config.agent_timing_log_dir)
                / f"{benchmark}_full_timings.jsonl"
            )
            recorder = AgentTimingRecorder(str(log_path))
            self.timing_recorders[benchmark] = recorder
        return recorder

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        self.model = LLaDAModelLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        self.lock = asyncio.Lock()

    @staticmethod
    def message_content_to_text(message: ChatMessage) -> str:
        if message.content is None:
            return ""
        if isinstance(message.content, str):
            return message.content
        text_parts = []
        for part in message.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            else:
                raise ValueError(
                    f"Unsupported message content part {part.type!r}; only text is supported."
                )
        return "\n".join(text_parts)

    def prepare_messages(self, messages: List[ChatMessage]):
        normalized = []
        has_user_message = False
        for message in messages:
            if message.role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported role {message.role!r}; expected system, user, or assistant."
                )
            normalized.append(
                {
                    "role": message.role,
                    "content": self.message_content_to_text(message),
                }
            )
            if message.role == "user":
                has_user_message = True

        if not has_user_message:
            raise ValueError("At least one user message is required.")
        return normalized

    def record_agent_timing(
        self,
        *,
        completion_id: str,
        created: int,
        request: ChatCompletionRequest,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            messages = self.prepare_messages(request.messages)
            registry = extract_agent_registry(messages, self.config.agent_names)
            benchmark = infer_benchmark(registry)
            if benchmark is None:
                logging.getLogger("fastdllm.agent_timing").warning(
                    "Skipping Agent timing: unrecognized Agent registry %s",
                    registry,
                )
                return
            query = next(
                (
                    str(message.get("content") or "")
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            requested_tokens = (
                request.max_completion_tokens or request.max_tokens
            )
            self._timing_recorder(benchmark).record(
                completion_id=completion_id,
                created_unix=created,
                query=query,
                model=request.model,
                temperature=request.temperature,
                requested_max_tokens=requested_tokens,
                metrics=metrics,
                error=error,
            )
        except Exception:
            # A late filesystem failure must be visible in the server log, but
            # must not discard an otherwise successful benchmark response.
            logging.getLogger("fastdllm.agent_timing").exception(
                "Failed to persist Agent timing for %s", completion_id
            )

    def effective_lengths(self, requested_tokens: Optional[int]):
        visible_tokens = requested_tokens or self.config.max_gen_length
        if visible_tokens > self.config.max_gen_length:
            raise ValueError(
                f"Requested max_tokens={visible_tokens} exceeds server limit "
                f"{self.config.max_gen_length}."
            )
        gen_length = max(
            self.config.block_size,
            math.ceil(visible_tokens / self.config.block_size) * self.config.block_size,
        )
        num_blocks = gen_length // self.config.block_size
        steps = num_blocks * self.config.steps_per_block
        if self.config.cache_mode == "dual" and self.config.steps_per_block < self.config.block_size:
            raise ValueError(
                "Dual Cache requires steps_per_block >= block_size so an unfinished "
                "block cannot be skipped."
            )
        return visible_tokens, gen_length, steps

    @staticmethod
    def apply_stop(text: str, stop: Optional[Union[str, List[str]]]):
        if not stop:
            return text
        stops = [stop] if isinstance(stop, str) else stop
        indices = [text.find(value) for value in stops if value and value in text]
        return text[:min(indices)] if indices else text

    def generate(self, request: ChatCompletionRequest):
        if request.model != self.config.served_model_name:
            raise ValueError(
                f"Model {request.model!r} is not served; use "
                f"{self.config.served_model_name!r}."
            )
        if request.n != 1:
            raise ValueError("Only n=1 is supported.")
        if request.top_p != 1.0:
            raise ValueError("top_p sampling is not supported; use top_p=1.")

        requested_tokens = request.max_completion_tokens or request.max_tokens
        visible_tokens, gen_length, steps = self.effective_lengths(requested_tokens)
        messages = self.prepare_messages(request.messages)
        request_agent_names = extract_agent_registry(
            messages, self.config.agent_names
        )
        if not request_agent_names:
            raise ValueError(
                "No Agent registry was found in the system prompt. Define roles "
                "as '- name_agent: description' lines or start the server with "
                "--agent_names."
            )
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)
        mask_id = self.tokenizer.mask_token_id or 126336
        controller = None
        if self.config.method in {"plan", "all"}:
            fixed_structure_mode = "fixed_canvas_plan_first"
            controller = FixedCanvasMonitor(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                priority_slots=3,
                tracking_slots=max(16, self.config.agent_timing_slots),
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                reasoning_budget=self.config.reasoning_budget,
                plan_budget=self.config.plan_budget,
                reasoning_ratio=self.config.reasoning_ratio,
                plan_ratio=self.config.plan_ratio,
                structure_mode=fixed_structure_mode,
                local_agent_observe=self.config.method == "all",
                local_observation_stride=self.config.local_observation_stride,
                local_agent_probability=self.config.all_local_probability,
                lightweight_tracking=self.config.method == "plan",
            )
        elif self.config.method == "commit":
            controller = DualVanillaLateDecideObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                anchor_min_logit_margin=self.config.agent_anchor_margin,
            )
        elif self.config.method == "base":
            # Passive materialization timing only.  It never writes a catalog
            # token or changes decoder masks, so this is true Dual Vanilla.
            controller = NaturalReasoningPlanMonitor(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
            )
        elif self.config.method == "oracle_latent":
            # Diagnostic-only upper bound. It compresses existing full-sequence
            # warmup logits for offline oracle-span replay and never participates
            # in commit, token transfer, decoder masking, or prefetch.
            controller = OracleLatentAgentObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                score_chunk_size=self.config.oracle_score_chunk_size,
            )
        elif self.config.method == "online_latent_diagnostic":
            # Read-only online anchor localization plus oracle-span control.
            # Both paths reuse the same existing full-sequence logits.
            controller = OnlineLatentAnchorObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                score_chunk_size=self.config.oracle_score_chunk_size,
                anchor_min_logit_margin=self.config.agent_anchor_margin,
                anchor_position_tolerance=(
                    self.config.online_anchor_position_tolerance
                ),
                anchor_stable_observations=(
                    self.config.online_anchor_stable_observations
                ),
            )
        elif self.config.method in {
            "online_latent_refine", "online_latent_refine_tracking"
        }:
            controller = RefinedOnlineLatentAnchorObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                persistent_tracking=(
                    self.config.method == "online_latent_refine_tracking"
                ),
                score_chunk_size=self.config.oracle_score_chunk_size,
                anchor_min_logit_margin=self.config.agent_anchor_margin,
                refinement_radius=self.config.online_refinement_radius,
                refinement_anchor_weight=(
                    self.config.online_refinement_anchor_weight
                ),
                refinement_agent_weight=(
                    self.config.online_refinement_agent_weight
                ),
                anchor_position_tolerance=(
                    self.config.online_anchor_position_tolerance
                ),
                anchor_stable_observations=(
                    self.config.online_anchor_stable_observations
                ),
                max_track_misses=self.config.online_max_track_misses,
                max_track_match_distance=(
                    self.config.online_max_track_match_distance
                ),
                track_stable_observations=(
                    self.config.online_track_stable_observations
                ),
                track_score_weight=self.config.online_track_score_weight,
            )
        elif self.config.method == "online_latent_region":
            # Read-only persistent-track region scoring. Exact Agent value
            # alignment is metadata only and never gates decoding or prefetch.
            controller = RegionLatentAgentObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                score_chunk_size=self.config.oracle_score_chunk_size,
                anchor_min_logit_margin=self.config.agent_anchor_margin,
                region_radius=self.config.online_region_radius,
                region_top_k=self.config.online_region_top_k,
                region_temperature=self.config.online_region_temperature,
                main_aggregation=(
                    self.config.online_region_main_aggregation
                ),
                anchor_position_tolerance=(
                    self.config.online_anchor_position_tolerance
                ),
                anchor_stable_observations=(
                    self.config.online_anchor_stable_observations
                ),
                max_track_misses=self.config.online_max_track_misses,
                max_track_match_distance=(
                    self.config.online_max_track_match_distance
                ),
                track_stable_observations=(
                    self.config.online_track_stable_observations
                ),
                track_score_weight=self.config.online_track_score_weight,
            )
        elif self.config.method == "online_latent_hypothesis":
            controller = HypothesisRegionObserver(
                tokenizer=self.tokenizer,
                catalog=request_agent_names,
                prompt_length=input_ids.shape[1],
                gen_length=gen_length,
                mask_id=mask_id,
                score_chunk_size=self.config.oracle_score_chunk_size,
                anchor_min_logit_margin=self.config.agent_anchor_margin,
                region_radius=self.config.online_region_radius,
                region_top_k=self.config.online_region_top_k,
                region_temperature=self.config.online_region_temperature,
                main_aggregation=(
                    self.config.online_region_main_aggregation
                ),
                anchor_position_tolerance=(
                    self.config.online_anchor_position_tolerance
                ),
                anchor_stable_observations=(
                    self.config.online_anchor_stable_observations
                ),
                max_track_misses=self.config.online_max_track_misses,
                max_track_match_distance=(
                    self.config.online_max_track_match_distance
                ),
                track_stable_observations=(
                    self.config.online_track_stable_observations
                ),
                track_score_weight=self.config.online_track_score_weight,
                hypothesis_merge_distance=(
                    self.config.hypothesis_merge_distance
                ),
                hypothesis_merge_gap=self.config.hypothesis_merge_gap,
                hypothesis_min_seen=self.config.hypothesis_min_seen,
                hypothesis_min_support=self.config.hypothesis_min_support,
                hypothesis_max_center_jump=(
                    self.config.hypothesis_max_center_jump
                ),
                hypothesis_duplicate_observations=(
                    self.config.hypothesis_duplicate_observations
                ),
            )

        generation_kwargs = {
            "model": self.model,
            "prompt": input_ids,
            "steps": steps,
            "gen_length": gen_length,
            "block_length": self.config.block_size,
            "temperature": request.temperature,
            "remasking": "low_confidence",
            "threshold": self.config.threshold,
            "mask_id": mask_id,
            "agent_controller": controller,
        }
        uses_dual_vanilla = self.config.method in {
            "base", "commit", "oracle_latent", "online_latent_diagnostic",
            "online_latent_refine", "online_latent_refine_tracking",
            "online_latent_region",
            "online_latent_hypothesis",
        }
        if (
            uses_dual_vanilla
            and controller is not None
            and hasattr(controller, "step_callback")
        ):
            generation_kwargs["step_callback"] = controller.step_callback
        if uses_dual_vanilla:
            generate_fn = generate_with_dual_cache
        else:
            generate_fn = generate_with_fixed_canvas_dual_cache
            generation_kwargs["structure_mode"] = controller.structure_mode

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids, nfe = generate_fn(**generation_kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started_at
        suffix_ids = output_ids[:, input_ids.shape[1]:]
        raw_output_token_ids = suffix_ids[0, :visible_tokens].detach().cpu().tolist()
        raw_output_sha256 = hashlib.sha256(
            suffix_ids[0, :visible_tokens].detach().cpu().numpy().tobytes()
        ).hexdigest()
        content = self.tokenizer.decode(
            suffix_ids[0, :visible_tokens], skip_special_tokens=True
        )
        content = self.apply_stop(content, request.stop).strip()
        model_completion_tokens = len(
            self.tokenizer(content, add_special_tokens=False).input_ids
        )
        repair_report = {
            "applied": False,
            "method": "disabled",
            "operations": [],
        }
        if self.config.plan_json_repair:
            content, repair_report = repair_plan_json_response(
                content,
                request_agent_names,
                repair_agents=KNOWN_AGENT_NAMES,
            )
        if controller is not None and hasattr(
            controller, "set_evaluation_plan_text"
        ):
            # Observer correctness is evaluated against the final parsed PLAN
            # returned to the benchmark, not against speculative anchors.
            controller.set_evaluation_plan_text(content)
        if controller is not None:
            controller.close()
        # OpenAI usage describes the text actually returned to the caller.  A
        # diffusion LM always allocates a fixed output canvas, so counting the
        # canvas (often all 1024 positions) as completion tokens substantially
        # overstates useful throughput when special/padding tokens are skipped.
        completion_tokens = len(
            self.tokenizer(content, add_special_tokens=False).input_ids
        )
        prompt_tokens = int(input_ids.shape[1])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_tokens),
            "total_tokens": prompt_tokens + int(completion_tokens),
        }
        metrics = {
            "nfe": int(nfe),
            "normal_nfe": int(nfe),
            "generated_tokens": int(model_completion_tokens),
            "returned_tokens": int(completion_tokens),
            "generation_seconds": generation_seconds,
            "tps": (
                model_completion_tokens / generation_seconds
                if generation_seconds > 0 else 0.0
            ),
            "plan_json_repair": repair_report,
            "structure_mode": self.config.structure_mode,
            "method": self.config.method,
            "raw_output_sha256": raw_output_sha256,
            "raw_output_token_ids": raw_output_token_ids,
            "unresolved_mask_count": int((suffix_ids == mask_id).sum().item()),
        }
        if controller is not None:
            metrics["agent_priority"] = controller.metrics()
            metrics["agent_priority"]["policy"] = (
                self.config.method or self.config.structure_mode
            )
            if not uses_dual_vanilla:
                metrics["fixed_canvas"] = metrics["agent_priority"]
            probe_forwards = int(getattr(controller, "probe_forwards", 0) or 0)
            probe_wall_time = float(
                getattr(controller, "probe_wall_time", 0.0) or 0.0
            )
            metrics["probe_forwards"] = probe_forwards
            metrics["total_forwards"] = int(nfe) + probe_forwards
            metrics["probe_nfe"] = probe_forwards
            metrics["total_nfe"] = int(nfe) + probe_forwards
            metrics["probe_wall_time"] = probe_wall_time
            diagnostic_overhead = float(
                metrics["agent_priority"].get(
                    "diagnostic_observer_overhead_seconds", 0.0
                ) or 0.0
            )
            if diagnostic_overhead:
                metrics["diagnostic_observer_overhead_seconds"] = (
                    diagnostic_overhead
                )
                metrics["generation_seconds_excluding_diagnostic"] = max(
                    0.0, generation_seconds - diagnostic_overhead
                )
        return content, usage, metrics


runtime: Optional[LLaDAPlannerRuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if runtime is None:
        raise RuntimeError("Server runtime was not configured.")
    runtime.load()
    yield


app = FastAPI(title="Fast-dLLM LLaDA OpenAI API", lifespan=lifespan)


def check_authorization(request: Request):
    if runtime.config.api_key is None:
        return
    if request.headers.get("authorization") != f"Bearer {runtime.config.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.exception_handler(HTTPException)
async def openai_http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def openai_validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": runtime.model is not None}


@app.get("/v1/models")
async def list_models(request: Request):
    check_authorization(request)
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.config.served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fast-dllm",
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    check_authorization(request)
    if model_id != runtime.config.served_model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} was not found.")
    return {
        "id": runtime.config.served_model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "fast-dllm",
    }


def completion_chunk(completion_id, created, model, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    check_authorization(request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        async with runtime.lock:
            content, usage, metrics = await asyncio.to_thread(runtime.generate, payload)
    except ValueError as error:
        runtime.record_agent_timing(
            completion_id=completion_id,
            created=created,
            request=payload,
            error=str(error),
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        runtime.record_agent_timing(
            completion_id=completion_id,
            created=created,
            request=payload,
            error=str(error),
        )
        raise HTTPException(status_code=500, detail=str(error)) from error

    runtime.record_agent_timing(
        completion_id=completion_id,
        created=created,
        request=payload,
        metrics=metrics,
    )

    if not payload.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "fastdllm": metrics,
        }

    async def stream_response():
        chunks = [
            completion_chunk(
                completion_id, created, payload.model, {"role": "assistant"}
            ),
            completion_chunk(
                completion_id, created, payload.model, {"content": content}
            ),
            completion_chunk(
                completion_id, created, payload.model, {}, finish_reason="stop"
            ),
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        if payload.stream_options and payload.stream_options.get("include_usage"):
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [],
                "usage": usage,
                "fastdllm": metrics,
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve LLaDA with an OpenAI API.")
    parser.add_argument(
        "--method",
        choices=(
            "base", "commit", "plan", "all", "oracle_latent",
            "online_latent_diagnostic",
            "online_latent_refine", "online_latent_refine_tracking",
            "online_latent_region",
            "online_latent_hypothesis",
        ),
        default="base",
        help=(
            "base=Dual Vanilla with natural Agent timing only; "
            "commit=Dual Vanilla with Semantic/Prefix/Natural per-step commit; "
            "plan=Fixed Canvas PLAN-first with natural timing only; "
            "all=PLAN-first plus read-only PLAN-region prediction; "
            "oracle_latent=diagnostic-only oracle-span replay of existing "
            "full-sequence logits; online_latent_diagnostic=read-only online "
            "latent anchor localization compared against oracle spans; "
            "online_latent_refine adds +/-4 template refinement; "
            "online_latent_refine_tracking also adds persistent monotonic "
            "slot tracks; online_latent_region scores Agent identity across "
            "each persistent track's local region."
            " online_latent_hypothesis merges and validates noisy tracks "
            "before applying the same region scorer."
        ),
    )
    parser.add_argument("--model_path", default="/data/labshare/Param/llada")
    parser.add_argument("--served_model_name", default="/data/labshare/Param/llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7004)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--max_gen_length", type=int, default=1024)
    parser.add_argument("--steps_per_block", type=int, default=32)
    parser.add_argument(
        "--reasoning_budget", "--reasoning-budget",
        type=int,
        default=None,
        help=(
            "Optional explicit reasoning capacity. If plan_budget is omitted, "
            "PLAN receives the remaining non-delimiter canvas."
        ),
    )
    parser.add_argument(
        "--plan_budget", "--plan-budget",
        type=int,
        default=None,
        help=(
            "Optional maximum PLAN capacity, not a target output length. If "
            "reasoning_budget is omitted, reasoning receives the remainder."
        ),
    )
    parser.add_argument(
        "--reasoning_ratio", "--reasoning-ratio",
        type=float,
        default=0.5,
        help="Default share of non-delimiter canvas assigned to reasoning.",
    )
    parser.add_argument(
        "--plan_ratio", "--plan-ratio",
        type=float,
        default=0.5,
        help="Default share of non-delimiter canvas assigned to PLAN capacity.",
    )
    parser.add_argument("--agent_slots", type=int, default=3)
    parser.add_argument(
        "--agent_timing_slots",
        type=int,
        default=3,
        help=(
            "Maximum normal-response Agent fields to time for plan/all. Base "
            "and commit discover every PLAN Agent field dynamically."
        ),
    )
    parser.add_argument(
        "--agent_names",
        default="",
        help=(
            "Optional comma-separated fallback Agent registry. By default the "
            "registry must be extracted from '- name_agent: description' lines "
            "in the request system prompt."
        ),
    )
    parser.add_argument("--cache_mode", choices=("dual",), default="dual")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--agent_anchor_margin",
        type=float,
        default=-6.0,
        help="Minimum mean target-vs-top logit margin for a speculative JSON Agent anchor.",
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--log_level", default="info")
    parser.add_argument(
        "--agent_timing_log_dir",
        "--agent-timing-log-dir",
        default=DEFAULT_AGENT_TIMING_LOG_DIR,
        help=(
            "Directory for {benchmark}_full_timings.jsonl files. The benchmark "
            "is inferred from the request Agent registry."
        ),
    )
    parser.add_argument(
        "--local-observation-stride",
        type=int,
        default=1,
        help=(
            "For --method all, score existing PLAN block-local logits only "
            "every N refinement steps. Full block warmup observations remain "
            "enabled; this never adds a model forward."
        ),
    )
    parser.add_argument(
        "--all-local-prob",
        type=float,
        default=0.90,
        help=(
            "Probability threshold for the read-only PLAN-local predictor "
            "used only by --method all. The margin remains 0.40 and stable "
            "count remains 2."
        ),
    )
    parser.add_argument(
        "--oracle-score-chunk-size",
        type=int,
        default=32,
        help=(
            "Vocabulary logsumexp position chunk size used only by "
            "--method oracle_latent. It changes diagnostic memory/overhead, "
            "not model forwards or decoding."
        ),
    )
    parser.add_argument(
        "--online-anchor-position-tolerance",
        type=int,
        default=4,
        help=(
            "Maximum token-position drift counted as stable by the read-only "
            "online latent anchor diagnostic."
        ),
    )
    parser.add_argument(
        "--online-anchor-stable-observations",
        type=int,
        default=2,
        help=(
            "Consecutive nearby anchor observations used only as a diagnostic "
            "position-stability field; it never commits."
        ),
    )
    parser.add_argument(
        "--online-refinement-radius",
        type=int,
        default=4,
        help="Local value-start search radius for refined anchor diagnostics.",
    )
    parser.add_argument(
        "--online-refinement-anchor-weight",
        type=float,
        default=1.0,
        help="Anchor structural-score weight in local joint scoring.",
    )
    parser.add_argument(
        "--online-refinement-agent-weight",
        type=float,
        default=1.0,
        help="Agent sequence-score weight in local joint scoring.",
    )
    parser.add_argument(
        "--online-max-track-misses",
        type=int,
        default=3,
        help="Observations a persistent diagnostic track may remain missing.",
    )
    parser.add_argument(
        "--online-max-track-match-distance",
        type=int,
        default=64,
        help="Maximum token distance for persistent track reassociation.",
    )
    parser.add_argument(
        "--online-track-stable-observations",
        type=int,
        default=2,
        help="Consecutive matches needed to label a diagnostic track stable.",
    )
    parser.add_argument(
        "--online-track-score-weight",
        type=float,
        default=0.05,
        help="Joint-score tie-break weight in monotonic track matching.",
    )
    parser.add_argument(
        "--online-region-radius",
        type=int,
        default=4,
        help="Token radius around each persistent track for region scoring.",
    )
    parser.add_argument(
        "--online-region-top-k",
        type=int,
        default=2,
        help="Number of best alignments averaged by top-k region scoring.",
    )
    parser.add_argument(
        "--online-region-temperature",
        type=float,
        default=1.0,
        help="Temperature for soft position aggregation in region scoring.",
    )
    parser.add_argument(
        "--online-region-main-aggregation",
        choices=("max", "top2_mean", "soft"),
        default="top2_mean",
        help="Primary region aggregation used by summary/timeline fields.",
    )
    parser.add_argument(
        "--hypothesis-merge-distance",
        type=int,
        default=6,
        help="Maximum median-position distance for merging raw tracks.",
    )
    parser.add_argument(
        "--hypothesis-merge-gap",
        type=int,
        default=2,
        help="Missing observations allowed when joining adjacent raw tracks.",
    )
    parser.add_argument(
        "--hypothesis-min-seen",
        type=int,
        default=2,
        help="Minimum observations required to validate a hypothesis.",
    )
    parser.add_argument(
        "--hypothesis-min-support",
        type=float,
        default=0.5,
        help="Minimum seen/span ratio required to validate a hypothesis.",
    )
    parser.add_argument(
        "--hypothesis-max-center-jump",
        type=int,
        default=12,
        help="Center jump above which a hypothesis is marked unstable.",
    )
    parser.add_argument(
        "--hypothesis-duplicate-observations",
        type=int,
        default=2,
        help="Nearby overlapping observations required for suppression.",
    )
    parser.add_argument(
        "--plan_json_repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Conservatively repair and validate the fixed planner JSON schema "
            "before returning a response. Use --no-plan_json_repair to disable."
        ),
    )
    return parser.parse_args()


def main():
    global runtime
    args = parse_args()
    if args.cache_mode != "dual":
        raise ValueError("--method base|commit|plan|all requires --cache_mode dual.")
    args.structure_mode = (
        "fixed_canvas_plan_first"
        if args.method in {"plan", "all"} else "dual_vanilla"
    )
    if args.local_observation_stride < 1:
        raise ValueError("--local-observation-stride must be at least 1.")
    if not 0.0 <= args.all_local_prob <= 1.0:
        raise ValueError("--all-local-prob must be within [0,1].")
    if args.oracle_score_chunk_size < 1:
        raise ValueError("--oracle-score-chunk-size must be at least 1.")
    if args.online_anchor_position_tolerance < 0:
        raise ValueError("--online-anchor-position-tolerance cannot be negative.")
    if args.online_anchor_stable_observations < 1:
        raise ValueError("--online-anchor-stable-observations must be at least 1.")
    if args.online_refinement_radius < 0:
        raise ValueError("--online-refinement-radius cannot be negative.")
    if (
        args.online_refinement_anchor_weight < 0
        or args.online_refinement_agent_weight < 0
        or args.online_refinement_anchor_weight
        + args.online_refinement_agent_weight == 0
    ):
        raise ValueError("online refinement weights must be non-negative and non-zero.")
    if args.online_max_track_misses < 0:
        raise ValueError("--online-max-track-misses cannot be negative.")
    if args.online_max_track_match_distance < 1:
        raise ValueError("--online-max-track-match-distance must be positive.")
    if args.online_track_stable_observations < 1:
        raise ValueError("--online-track-stable-observations must be positive.")
    if args.online_region_radius < 0:
        raise ValueError("--online-region-radius cannot be negative.")
    if args.online_region_top_k < 1:
        raise ValueError("--online-region-top-k must be positive.")
    if args.online_region_temperature <= 0:
        raise ValueError("--online-region-temperature must be positive.")
    if args.hypothesis_merge_distance < 0:
        raise ValueError("--hypothesis-merge-distance cannot be negative.")
    if args.hypothesis_merge_gap < 0:
        raise ValueError("--hypothesis-merge-gap cannot be negative.")
    if args.hypothesis_min_seen < 1:
        raise ValueError("--hypothesis-min-seen must be positive.")
    if not 0.0 <= args.hypothesis_min_support <= 1.0:
        raise ValueError("--hypothesis-min-support must be within [0,1].")
    if args.hypothesis_max_center_jump < 1:
        raise ValueError("--hypothesis-max-center-jump must be positive.")
    if args.hypothesis_duplicate_observations < 1:
        raise ValueError(
            "--hypothesis-duplicate-observations must be positive."
        )
    agent_names = [name.strip() for name in args.agent_names.split(",") if name.strip()]
    if args.max_gen_length % args.block_size != 0:
        raise ValueError("max_gen_length must be divisible by block_size.")
    if args.method in {"plan", "all"} and args.agent_timing_slots < args.agent_slots:
        raise ValueError("agent_timing_slots must be at least agent_slots.")
    if args.reasoning_budget is not None and args.reasoning_budget <= 0:
        raise ValueError("reasoning_budget must be positive when provided.")
    if args.plan_budget is not None and args.plan_budget <= 0:
        raise ValueError("plan_budget must be positive when provided.")
    if args.reasoning_ratio <= 0 or args.plan_ratio <= 0:
        raise ValueError("reasoning_ratio and plan_ratio must be positive.")
    if not str(args.agent_timing_log_dir).strip():
        raise ValueError("agent_timing_log_dir cannot be empty.")

    runtime = LLaDAPlannerRuntime(
        ServerConfig(
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            device=args.device,
            block_size=args.block_size,
            max_gen_length=args.max_gen_length,
            steps_per_block=args.steps_per_block,
            agent_slots=args.agent_slots,
            agent_timing_slots=args.agent_timing_slots,
            agent_names=agent_names,
            cache_mode=args.cache_mode,
            threshold=args.threshold,
            agent_anchor_margin=args.agent_anchor_margin,
            agent_timing_log_dir=args.agent_timing_log_dir,
            plan_json_repair=args.plan_json_repair,
            api_key=args.api_key,
            structure_mode=args.structure_mode,
            reasoning_budget=args.reasoning_budget,
            plan_budget=args.plan_budget,
            reasoning_ratio=args.reasoning_ratio,
            plan_ratio=args.plan_ratio,
            method=args.method,
            local_observation_stride=args.local_observation_stride,
            all_local_probability=args.all_local_prob,
            oracle_score_chunk_size=args.oracle_score_chunk_size,
            online_anchor_position_tolerance=(
                args.online_anchor_position_tolerance
            ),
            online_anchor_stable_observations=(
                args.online_anchor_stable_observations
            ),
            online_refinement_radius=args.online_refinement_radius,
            online_refinement_anchor_weight=(
                args.online_refinement_anchor_weight
            ),
            online_refinement_agent_weight=(
                args.online_refinement_agent_weight
            ),
            online_max_track_misses=args.online_max_track_misses,
            online_max_track_match_distance=(
                args.online_max_track_match_distance
            ),
            online_track_stable_observations=(
                args.online_track_stable_observations
            ),
            online_track_score_weight=args.online_track_score_weight,
            online_region_radius=args.online_region_radius,
            online_region_top_k=args.online_region_top_k,
            online_region_temperature=args.online_region_temperature,
            online_region_main_aggregation=(
                args.online_region_main_aggregation
            ),
            hypothesis_merge_distance=args.hypothesis_merge_distance,
            hypothesis_merge_gap=args.hypothesis_merge_gap,
            hypothesis_min_seen=args.hypothesis_min_seen,
            hypothesis_min_support=args.hypothesis_min_support,
            hypothesis_max_center_jump=args.hypothesis_max_center_jump,
            hypothesis_duplicate_observations=(
                args.hypothesis_duplicate_observations
            ),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

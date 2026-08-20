# import time
# import uuid
# import threading
# from typing import List, Optional, Dict, Any, Union

# import torch
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # =========================
# # 配置区
# # =========================
# MODEL_PATH = "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct"
# SERVED_MODEL_NAME = "llama3-8b-instruct"

# HOST = "0.0.0.0"
# PORT = 7002

# DEFAULT_MAX_TOKENS = 2048
# DEFAULT_TEMPERATURE = 0.1
# DEFAULT_TOP_P = 0.9

# # 基础 Transformers generate 不适合多个请求同时抢 GPU。
# # 这里用锁保证一次只跑一个请求，稳定优先。
# GENERATE_LOCK = threading.Lock()


# # =========================
# # OpenAI-compatible 请求格式
# # =========================
# class ChatMessage(BaseModel):
#     role: str
#     content: str


# class ChatCompletionRequest(BaseModel):
#     model: Optional[str] = SERVED_MODEL_NAME
#     messages: List[ChatMessage]

#     max_tokens: Optional[int] = Field(default=DEFAULT_MAX_TOKENS)
#     temperature: Optional[float] = Field(default=DEFAULT_TEMPERATURE)
#     top_p: Optional[float] = Field(default=DEFAULT_TOP_P)

#     stream: Optional[bool] = False
#     stop: Optional[Union[str, List[str]]] = None


# # =========================
# # 加载模型，只执行一次
# # =========================
# print(f"Loading tokenizer from: {MODEL_PATH}")
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     local_files_only=True,
# )

# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# if torch.cuda.is_available():
#     if torch.cuda.is_bf16_supported():
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float16
# else:
#     dtype = torch.float32

# print(f"Loading model from: {MODEL_PATH}")
# print(f"Using dtype: {dtype}")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=dtype,
#     device_map="auto",
#     local_files_only=True,
# )

# model.eval()


# def get_input_device():
#     """
#     device_map='auto' 时，模型可能被切到多张卡。
#     输入一般放到 embedding 所在设备即可。
#     """
#     if hasattr(model, "hf_device_map"):
#         for key in ["model.embed_tokens", "model.tok_embeddings", "transformer.wte"]:
#             if key in model.hf_device_map:
#                 dev = model.hf_device_map[key]
#                 if isinstance(dev, int):
#                     return torch.device(f"cuda:{dev}")
#                 return torch.device(dev)

#         for _, dev in model.hf_device_map.items():
#             if dev not in ["cpu", "disk"]:
#                 if isinstance(dev, int):
#                     return torch.device(f"cuda:{dev}")
#                 return torch.device(dev)

#     return next(model.parameters()).device


# INPUT_DEVICE = get_input_device()
# print(f"Input device: {INPUT_DEVICE}")


# def get_eos_token_ids():
#     """
#     Llama 3 Instruct 常用 <|eot_id|> 作为一轮对话结束标记。
#     只用 tokenizer.eos_token_id 有时不会及时停。
#     """
#     eos_ids = []

#     if tokenizer.eos_token_id is not None:
#         eos_ids.append(tokenizer.eos_token_id)

#     eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
#         eos_ids.append(eot_id)

#     return eos_ids


# EOS_TOKEN_IDS = get_eos_token_ids()
# print(f"EOS token ids: {EOS_TOKEN_IDS}")


# # =========================
# # FastAPI
# # =========================
# app = FastAPI()


# @app.get("/v1/models")
# def list_models():
#     return {
#         "object": "list",
#         "data": [
#             {
#                 "id": SERVED_MODEL_NAME,
#                 "object": "model",
#                 "created": int(time.time()),
#                 "owned_by": "local",
#             }
#         ],
#     }


# @app.post("/v1/chat/completions")
# def chat_completions(req: ChatCompletionRequest):
#     if req.stream:
#         raise HTTPException(
#             status_code=400,
#             detail="This minimal server does not support stream=True yet.",
#         )

#     if not req.messages:
#         raise HTTPException(status_code=400, detail="messages cannot be empty")

#     messages = [m.model_dump() for m in req.messages]

#     try:
#         input_ids = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_tensors="pt",
#         ).to(INPUT_DEVICE)

#         prompt_tokens = input_ids.shape[-1]

#         max_new_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
#         temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
#         top_p = DEFAULT_TOP_P if req.top_p is None else req.top_p

#         do_sample = temperature > 0

#         generation_kwargs = {
#             "input_ids": input_ids,
#             "max_new_tokens": max_new_tokens,
#             "do_sample": do_sample,
#             "top_p": top_p,
#             "eos_token_id": EOS_TOKEN_IDS,
#             "pad_token_id": tokenizer.pad_token_id,
#         }

#         if do_sample:
#             generation_kwargs["temperature"] = temperature

#         with GENERATE_LOCK:
#             with torch.no_grad():
#                 outputs = model.generate(**generation_kwargs)

#         generated_ids = outputs[0][prompt_tokens:]
#         completion_text = tokenizer.decode(
#             generated_ids,
#             skip_special_tokens=True,
#         )

#         # 简单支持 stop 字符串
#         if req.stop is not None:
#             stops = [req.stop] if isinstance(req.stop, str) else req.stop
#             for s in stops:
#                 idx = completion_text.find(s)
#                 if idx != -1:
#                     completion_text = completion_text[:idx]
#                     break

#         completion_text = completion_text.strip()

#         completion_tokens = len(generated_ids)
#         total_tokens = prompt_tokens + completion_tokens

#         return {
#             "id": f"chatcmpl-{uuid.uuid4().hex}",
#             "object": "chat.completion",
#             "created": int(time.time()),
#             "model": req.model or SERVED_MODEL_NAME,
#             "choices": [
#                 {
#                     "index": 0,
#                     "message": {
#                         "role": "assistant",
#                         "content": completion_text,
#                     },
#                     "finish_reason": "stop",
#                 }
#             ],
#             "usage": {
#                 "prompt_tokens": prompt_tokens,
#                 "completion_tokens": completion_tokens,
#                 "total_tokens": total_tokens,
#             },
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host=HOST, port=PORT)

from __future__ import annotations

import os
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 配置
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
)

SERVED_MODEL_NAME = os.getenv(
    "SERVED_MODEL_NAME",
    "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7002"))

# 固定使用单张 GPU
CUDA_DEVICE = int(os.getenv("CUDA_DEVICE", "0"))
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}")

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

# 普通 Transformers 推理不适合多个请求同时操作同一个模型和 GPU。
# 使用锁将请求串行化，确保测试稳定。
GENERATE_LOCK = threading.Lock()


# ============================================================
# OpenAI-compatible 请求格式
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = SERVED_MODEL_NAME
    messages: List[ChatMessage]

    max_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P,
        gt=0.0,
        le=1.0,
    )

    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# ============================================================
# 环境检查
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This server requires an NVIDIA GPU.")

if CUDA_DEVICE >= torch.cuda.device_count():
    raise RuntimeError(
        f"CUDA device {CUDA_DEVICE} does not exist. "
        f"Available GPU count: {torch.cuda.device_count()}"
    )

torch.cuda.set_device(CUDA_DEVICE)

gpu_name = torch.cuda.get_device_name(CUDA_DEVICE)
gpu_memory_gb = (
    torch.cuda.get_device_properties(CUDA_DEVICE).total_memory
    / 1024**3
)

print("=" * 70)
print("PyTorch Transformers baseline server")
print(f"Model path:       {MODEL_PATH}")
print(f"Served name:      {SERVED_MODEL_NAME}")
print(f"CUDA device:      {DEVICE}")
print(f"GPU:              {gpu_name}")
print(f"GPU memory:       {gpu_memory_gb:.2f} GiB")
print(f"PyTorch version:  {torch.__version__}")
print("=" * 70)


# ============================================================
# 加载 Tokenizer
# ============================================================

print(f"Loading tokenizer from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ============================================================
# 加载模型
# ============================================================

print(f"Loading model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,

    # 明确使用 Transformers eager attention。
    # 不自动使用 PyTorch SDPA 或 FlashAttention。
    attn_implementation="eager",

    local_files_only=True,
)

# 明确将整个模型放到单张 GPU
model = model.to(DEVICE)
model.eval()

# 明确启用标准 KV Cache
model.config.use_cache = True

print(f"Model dtype:      {model.dtype}")
print(f"Model device:     {next(model.parameters()).device}")
print(
    "Attention impl:  "
    f"{getattr(model.config, '_attn_implementation', 'unknown')}"
)

allocated_gb = torch.cuda.memory_allocated(CUDA_DEVICE) / 1024**3
reserved_gb = torch.cuda.memory_reserved(CUDA_DEVICE) / 1024**3

print(f"GPU allocated:    {allocated_gb:.2f} GiB")
print(f"GPU reserved:     {reserved_gb:.2f} GiB")
print("=" * 70)


# ============================================================
# EOS Token
# ============================================================

def get_eos_token_ids() -> List[int]:
    eos_ids: List[int] = []

    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # Llama 3 Instruct 通常用 <|eot_id|> 结束一轮对话
    vocab = tokenizer.get_vocab()
    eot_id = vocab.get("<|eot_id|>")

    if eot_id is not None and eot_id not in eos_ids:
        eos_ids.append(eot_id)

    return eos_ids


EOS_TOKEN_IDS = get_eos_token_ids()
EOS_TOKEN_ID_SET = set(EOS_TOKEN_IDS)

print(f"EOS token ids: {EOS_TOKEN_IDS}")


# ============================================================
# 工具函数
# ============================================================

def message_to_dict(message: ChatMessage) -> Dict[str, str]:
    """同时兼容 Pydantic v1 和 v2。"""

    if hasattr(message, "model_dump"):
        return message.model_dump()

    return message.dict()


def synchronize_cuda() -> None:
    """等待当前 GPU 上的 CUDA 操作执行完成。"""

    torch.cuda.synchronize(DEVICE)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """
    根据最后一个位置的 logits 选择下一个 token。

    temperature <= 0:
        使用 greedy decoding。

    temperature > 0:
        使用 temperature + top-p sampling。
    """

    if temperature <= 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)

    if top_p >= 1.0:
        return torch.multinomial(
            probabilities,
            num_samples=1,
        )

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # 保留使累积概率达到 top_p 的 token。
    # 减去当前概率，确保至少保留第一个 token。
    remove_mask = (
        cumulative_probabilities - sorted_probabilities
    ) >= top_p

    sorted_probabilities = sorted_probabilities.masked_fill(
        remove_mask,
        0.0,
    )

    probability_sum = sorted_probabilities.sum(
        dim=-1,
        keepdim=True,
    )

    sorted_probabilities = sorted_probabilities / probability_sum

    sampled_sorted_index = torch.multinomial(
        sorted_probabilities,
        num_samples=1,
    )

    return torch.gather(
        sorted_indices,
        dim=-1,
        index=sampled_sorted_index,
    )


def apply_stop_strings(
    text: str,
    stop: Optional[Union[str, List[str]]],
) -> tuple[str, bool]:
    """
    对生成后的文本应用 OpenAI 风格 stop 字符串。

    返回：
        截断后的文本；
        是否命中了 stop 字符串。
    """

    if stop is None:
        return text, False

    stop_strings = [stop] if isinstance(stop, str) else stop

    earliest_index: Optional[int] = None

    for stop_string in stop_strings:
        if not stop_string:
            continue

        index = text.find(stop_string)

        if index != -1:
            if earliest_index is None or index < earliest_index:
                earliest_index = index

    if earliest_index is None:
        return text, False

    return text[:earliest_index], True


def validate_sequence_length(
    prompt_tokens: int,
    max_new_tokens: int,
) -> None:
    """
    检查 prompt + output 是否超过模型最大上下文长度。
    """

    max_position_embeddings = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if max_position_embeddings is None:
        return

    total_requested_tokens = prompt_tokens + max_new_tokens

    if total_requested_tokens > max_position_embeddings:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested sequence is too long: "
                f"prompt_tokens={prompt_tokens}, "
                f"max_tokens={max_new_tokens}, "
                f"total={total_requested_tokens}, "
                f"model_limit={max_position_embeddings}"
            ),
        )


# ============================================================
# 手写 PyTorch 自回归生成
# ============================================================

@torch.inference_mode()
def generate_tokens(
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """
    使用最基础的 PyTorch + Transformers forward 实现生成。

    包含两个阶段：

    1. Prefill
       一次处理完整 prompt，并生成第一个 token。

    2. Decode
       使用 KV Cache，每轮输入一个 token。
    """

    batch_size = input_ids.shape[0]

    if batch_size != 1:
        raise ValueError("This baseline server only supports batch size 1.")

    prompt_tokens = input_ids.shape[-1]

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    generated_token_list: List[torch.Tensor] = []

    stopped_by_eos = False

    # 清除此前可能排队的 GPU 操作
    synchronize_cuda()

    generation_start = time.perf_counter()

    # --------------------------------------------------------
    # Prefill
    # --------------------------------------------------------

    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    next_token = sample_next_token(
        logits=outputs.logits[:, -1, :],
        temperature=temperature,
        top_p=top_p,
    )

    past_key_values = outputs.past_key_values

    # sample_next_token 中可能包含 CUDA 操作，必须同步后再停止计时
    synchronize_cuda()

    prefill_seconds = time.perf_counter() - prefill_start
    ttft_seconds = time.perf_counter() - generation_start

    generated_token_list.append(next_token)

    next_token_id = int(next_token.item())

    if next_token_id in EOS_TOKEN_ID_SET:
        stopped_by_eos = True

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decode_start = time.perf_counter()
    decoded_tokens = 0

    while (
        len(generated_token_list) < max_new_tokens
        and not stopped_by_eos
    ):
        # attention mask 长度需要覆盖 prompt + 已生成 token
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=-1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values

        next_token = sample_next_token(
            logits=outputs.logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
        )

        generated_token_list.append(next_token)
        decoded_tokens += 1

        next_token_id = int(next_token.item())

        if next_token_id in EOS_TOKEN_ID_SET:
            stopped_by_eos = True

    synchronize_cuda()

    decode_seconds = time.perf_counter() - decode_start
    generation_seconds = time.perf_counter() - generation_start

    generated_ids = torch.cat(
        generated_token_list,
        dim=-1,
    )

    completion_tokens = generated_ids.shape[-1]

    # Prefill throughput 表示处理输入 prompt 的速度
    prefill_tps = (
        prompt_tokens / prefill_seconds
        if prefill_seconds > 0
        else 0.0
    )

    # Decode TPS 不包含 Prefill 阶段生成的第一个 token
    decode_tps = (
        decoded_tokens / decode_seconds
        if decode_seconds > 0 and decoded_tokens > 0
        else 0.0
    )

    # 完整生成速度：输出 token / (prefill + decode)
    end_to_end_tps = (
        completion_tokens / generation_seconds
        if generation_seconds > 0
        else 0.0
    )

    # 每个 decode token 的平均耗时
    tpot_ms = (
        decode_seconds / decoded_tokens * 1000
        if decoded_tokens > 0
        else 0.0
    )

    return {
        "generated_ids": generated_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stopped_by_eos": stopped_by_eos,
        "metrics": {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generation_seconds": generation_seconds,
            "ttft_seconds": ttft_seconds,
            "prefill_tokens_per_second": prefill_tps,
            "decode_tokens_per_second": decode_tps,
            "end_to_end_tokens_per_second": end_to_end_tps,
            "time_per_output_token_ms": tpot_ms,
        },
    }


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="PyTorch Transformers Baseline Server",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": SERVED_MODEL_NAME,
        "device": str(DEVICE),
        "gpu": gpu_name,
        "dtype": str(model.dtype),
        "attention_implementation": getattr(
            model.config,
            "_attn_implementation",
            "unknown",
        ),
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
) -> Dict[str, Any]:
    request_start = time.perf_counter()

    if req.stream:
        raise HTTPException(
            status_code=400,
            detail=(
                "This PyTorch baseline server does not support "
                "stream=True."
            ),
        )

    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )

    if req.model not in [None, SERVED_MODEL_NAME]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model: {req.model}. "
                f"Available model: {SERVED_MODEL_NAME}"
            ),
        )

    max_new_tokens = (
        req.max_tokens
        if req.max_tokens is not None
        else DEFAULT_MAX_TOKENS
    )

    temperature = (
        req.temperature
        if req.temperature is not None
        else DEFAULT_TEMPERATURE
    )

    top_p = (
        req.top_p
        if req.top_p is not None
        else DEFAULT_TOP_P
    )

    messages = [
        message_to_dict(message)
        for message in req.messages
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(DEVICE)

        prompt_tokens = input_ids.shape[-1]

        validate_sequence_length(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )

        queue_start = time.perf_counter()

        with GENERATE_LOCK:
            queue_seconds = time.perf_counter() - queue_start

            result = generate_tokens(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        generated_ids = result["generated_ids"]

        completion_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        completion_text, stopped_by_string = apply_stop_strings(
            completion_text,
            req.stop,
        )

        completion_text = completion_text.strip()

        completion_tokens = result["completion_tokens"]
        total_tokens = prompt_tokens + completion_tokens

        if result["stopped_by_eos"] or stopped_by_string:
            finish_reason = "stop"
        else:
            finish_reason = "length"

        request_seconds = time.perf_counter() - request_start

        metrics = result["metrics"]
        metrics["queue_seconds"] = queue_seconds
        metrics["request_seconds"] = request_seconds

        print(
            f"[generation] "
            f"prompt={prompt_tokens}, "
            f"completion={completion_tokens}, "
            f"prefill={metrics['prefill_seconds']:.4f}s, "
            f"decode={metrics['decode_seconds']:.4f}s, "
            f"TTFT={metrics['ttft_seconds']:.4f}s, "
            f"decode_TPS="
            f"{metrics['decode_tokens_per_second']:.2f}, "
            f"E2E_TPS="
            f"{metrics['end_to_end_tokens_per_second']:.2f}"
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },

            # OpenAI 标准响应之外的额外性能字段
            "metrics": {
                "prefill_seconds": round(
                    metrics["prefill_seconds"],
                    6,
                ),
                "decode_seconds": round(
                    metrics["decode_seconds"],
                    6,
                ),
                "generation_seconds": round(
                    metrics["generation_seconds"],
                    6,
                ),
                "queue_seconds": round(
                    metrics["queue_seconds"],
                    6,
                ),
                "request_seconds": round(
                    metrics["request_seconds"],
                    6,
                ),
                "ttft_seconds": round(
                    metrics["ttft_seconds"],
                    6,
                ),
                "prefill_tokens_per_second": round(
                    metrics["prefill_tokens_per_second"],
                    2,
                ),
                "decode_tokens_per_second": round(
                    metrics["decode_tokens_per_second"],
                    2,
                ),
                "end_to_end_tokens_per_second": round(
                    metrics["end_to_end_tokens_per_second"],
                    2,
                ),
                "time_per_output_token_ms": round(
                    metrics["time_per_output_token_ms"],
                    2,
                ),
            },
        }

    except HTTPException:
        raise

    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=f"CUDA out of memory: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    # 必须保持 workers=1。
    # 多 worker 会在显存中加载多个模型副本。
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
    )
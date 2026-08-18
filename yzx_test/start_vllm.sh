#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Unified model-server launcher with per-model Conda environments
#
# Usage:
#   bash start_vllm.sh list
#   bash start_vllm.sh <model_name> <cuda_visible_devices>
#   bash start_vllm.sh <model_name> <cuda_visible_devices> --background
#   bash start_vllm.sh <model_name> <cuda_visible_devices> -- <extra vLLM args>
#
# Examples:
#   bash start_vllm.sh phi4-4b 6
#   bash start_vllm.sh llama3-8b 0
#   bash start_vllm.sh llada 4 --background
#   bash start_vllm.sh qwen3-4b 2 --background
#   bash start_vllm.sh llama3.2-3b 6 -- --disable-log-requests
#   bash start_vllm.sh qwen3-30b 0,1,2,3 -- --tensor-parallel-size 4
#
# Environment selection:
#   - hunyuan-1.8b uses Conda environment: vllm012
#   - minicpm3-4b uses Conda environment: minicpm3
#   - LLaDA/Qwen2.5/InternLM2.5/SmolLM2 models below use Conda environment: base
#   - Other models use the environment active when this script is invoked.
#
# Explicit Python override:
#   PYTHON_BIN=/path/to/python bash start_vllm.sh llama3-1b 5
#   PYTHON_BIN takes priority over the per-model Conda mapping.
# ============================================================

PYTHON_BIN_OVERRIDE="${PYTHON_BIN:-}"
VLLM_MODULE="${VLLM_MODULE:-vllm.entrypoints.openai.api_server}"
LOG_DIR="${VLLM_LOG_DIR:-./vllm_logs}"

declare -A MODEL_PATH
declare -A MODEL_PORT
declare -A MODEL_TP
declare -A MODEL_MAX_LEN
declare -A MODEL_GPU_UTIL
declare -A MODEL_TOOL_PARSER
declare -A MODEL_AUTO_TOOL
declare -A MODEL_ENFORCE_EAGER
declare -A MODEL_EXTRA_ARGS
declare -A MODEL_CONDA_ENV
declare -A MODEL_RUNNER

register_model() {
    local name="$1"
    local path="$2"
    local port="$3"
    local tp="$4"
    local max_len="$5"
    local gpu_util="$6"
    local tool_parser="${7:-}"
    local auto_tool="${8:-false}"
    local enforce_eager="${9:-false}"
    local extra_args="${10:-}"

    MODEL_PATH["$name"]="$path"
    MODEL_PORT["$name"]="$port"
    MODEL_TP["$name"]="$tp"
    MODEL_MAX_LEN["$name"]="$max_len"
    MODEL_GPU_UTIL["$name"]="$gpu_util"
    MODEL_TOOL_PARSER["$name"]="$tool_parser"
    MODEL_AUTO_TOOL["$name"]="$auto_tool"
    MODEL_ENFORCE_EAGER["$name"]="$enforce_eager"
    MODEL_EXTRA_ARGS["$name"]="$extra_args"
    MODEL_RUNNER["$name"]="vllm"
}

register_fastdllm_server() {
    local name="$1"
    local server_path="$2"
    local port="$3"
    local max_len="$4"
    local extra_args="$5"

    MODEL_PATH["$name"]="$server_path"
    MODEL_PORT["$name"]="$port"
    MODEL_TP["$name"]="N/A"
    MODEL_MAX_LEN["$name"]="$max_len"
    MODEL_GPU_UTIL["$name"]="N/A"
    MODEL_TOOL_PARSER["$name"]=""
    MODEL_AUTO_TOOL["$name"]="false"
    MODEL_ENFORCE_EAGER["$name"]="false"
    MODEL_EXTRA_ARGS["$name"]="$extra_args"
    MODEL_RUNNER["$name"]="fastdllm"
}

set_model_conda_env() {
    local model_name="$1"
    local conda_env="$2"
    MODEL_CONDA_ENV["$model_name"]="$conda_env"
}

# ============================================================
# Model configuration
#
# register_model arguments:
#   alias model_path port tensor_parallel_size max_model_len
#   gpu_memory_utilization tool_call_parser enable_auto_tool_choice
#   enforce_eager "extra arguments"
# ============================================================

register_model \
    "qwen3-30b" \
    "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507" \
    "7001" "4" "32768" "0.8" \
    "hermes" "true" "false" ""
register_model \
    "llama3-8b" \
    "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct" \
    "7002" "1" "8192" "0.8" \
    "hermes" "true" "false" ""
register_fastdllm_server \
    "llada" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7003" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada4" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7004" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada5" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7005" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada6" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7006" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada7" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7007" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada8" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7008" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"
register_fastdllm_server \
    "llada9" \
    "/home/yzx/Fast-dLLM/v1/llada/fastdllm_server.py" \
    "7009" "1024" \
    "--gen-length 1024 --block-size 32 --cache-mode dual --threshold 0.9 --steps 1024"

register_model \
    "llama3-3b" \
    "/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct" \
    "7011" "1" "8192" "0.8" \
    "hermes" "true" "false" ""
register_model \
    "gemma3-4b" \
    "/data/labshare/Param/gemma-3-4b-it" \
    "7012" "1" "8192" "0.8" \
    "hermes" "true" "false" ""
register_model \
    "qwen3-4b" \
    "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507" \
    "7013" "1" "8192" "0.8" \
    "hermes" "true" "false" ""
register_model \
    "phi4-4b" \
    "/data/labshare/Param/Phi-4-mini-instruct" \
    "7014" "1" "8192" "0.8" \
    "hermes" "true" "false" ""
register_model \
    "minicpm3-4b" \
    "/data/labshare/Param/MiniCPM3-4B" \
    "7015" "1" "8192" "0.8" \
    "" "false" "false" \
    "--trust-remote-code"


register_model \
    "llama3-1b" \
    "/data/labshare/Param/llama/llama3/Llama-3.2-1B-Instruct" \
    "7021" "1" "8192" "0.4" \
    "hermes" "true" "false" ""
register_model \
    "gemma3-1b" \
    "/data/labshare/Param/gemma-3-1b-it" \
    "7022" "1" "8192" "0.4" \
    "hermes" "true" "false" ""
register_model \
    "qwen3-1.7b" \
    "/data/labshare/Param/Qwen/Qwen3-1.7B" \
    "7023" "1" "8192" "0.4" \
    "hermes" "true" "false" \
    "--chat-template /data/labshare/Param/Qwen/Qwen3-1.7B/qwen3_nonthinking.jinja"
register_model \
    "hunyuan-1.8b" \
    "/data/labshare/Param/Hunyuan-1.8B-Instruct" \
    "7024" "1" "8192" "0.4" \
    "" "false" "false" \
    "--chat-template /data/labshare/Param/Hunyuan-1.8B-Instruct/hunyuan_nonthinking.jinja"
register_model \
    "lfm2.5-1.2b" \
    "/data/labshare/Param/LFM2.5-1.2B-Instruct" \
    "7025" "1" "8192" "0.4" \
    "hermes" "true" "false" ""
register_model \
    "minicpm5-1b" \
    "/data/labshare/Param/MiniCPM5-1B" \
    "7026" "1" "8192" "0.4" \
    "" "false" "false" \
    "--chat-template /data/labshare/Param/MiniCPM5-1B/minicpm5_nonthinking.jinja"
register_model \
    "deepseekr1-1.5b" \
    "/data/labshare/Param/DeepSeek-R1-Distill-Qwen-1.5B" \
    "7027" "1" "8192" "0.4" \
    "" "false" "false" \
    "--chat-template /data/labshare/Param/DeepSeek-R1-Distill-Qwen-1.5B/deepseek_nonthinking.jinja"
register_model \
    "qwen2.5-math-1.5b" \
    "/data/labshare/Param/Qwen/Qwen2.5-Math-1.5B-Instruct" \
    "7028" "1" "4096" "0.4" \
    "" "false" "false" ""
register_model \
    "qwen2.5-coder-1.5b" \
    "/data/labshare/Param/Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    "7029" "1" "8192" "0.4" \
    "" "false" "false" ""
register_model \
    "internlm2.5-1.8b" \
    "/data/labshare/Param/internlm2_5-1_8b-chat" \
    "7030" "1" "8192" "0.4" \
    "" "false" "false" \
    "--trust-remote-code"
register_model \
    "smollm2-1.7b" \
    "/data/labshare/Param/SmolLM2-1.7B-Instruct" \
    "7031" "1" "8192" "0.4" \
    "" "false" "false" ""
# ============================================================
# Per-model Conda environment mapping
#
# Models not listed here use the currently active environment.
# Left: model alias; right: Conda environment name.
# ============================================================

set_model_conda_env "hunyuan-1.8b" "vllm012"
set_model_conda_env "minicpm3-4b" "minicpm3"
set_model_conda_env "llada" "base"
set_model_conda_env "qwen2.5-math-1.5b" "base"
set_model_conda_env "qwen2.5-coder-1.5b" "base"
set_model_conda_env "internlm2.5-1.8b" "base"
set_model_conda_env "smollm2-1.7b" "base"

usage() {
    cat <<'USAGE'
Usage:
  bash start_vllm.sh list
  bash start_vllm.sh <model_name> <cuda_visible_devices> [--background] [-- extra_vllm_args...]

Examples:
  bash start_vllm.sh phi4-4b 6
  bash start_vllm.sh llama3-8b 0
  bash start_vllm.sh llada 4 --background
  bash start_vllm.sh qwen3-4b 2 --background
  bash start_vllm.sh llama3.2-3b 6 -- --disable-log-requests
  bash start_vllm.sh qwen3-30b 0,1,2,3 -- --tensor-parallel-size 4

Environment variables:
  PYTHON_BIN       Explicit Python executable; overrides model Conda mapping.
  VLLM_MODULE      vLLM API server module.
  VLLM_LOG_DIR     Background log directory. Default: ./vllm_logs
USAGE
}

list_models() {
    printf "%-20s %-6s %-5s %-9s %-14s %s\n" \
        "MODEL" "PORT" "TP" "MAX_LEN" "CONDA_ENV" "PATH"
    printf "%-20s %-6s %-5s %-9s %-14s %s\n" \
        "--------------------" "------" "-----" "---------" "--------------" "----"

    local name
    local env_name

    while IFS= read -r name; do
        env_name="${MODEL_CONDA_ENV[$name]-current}"

        printf "%-20s %-6s %-5s %-9s %-14s %s\n" \
            "$name" \
            "${MODEL_PORT[$name]}" \
            "${MODEL_TP[$name]}" \
            "${MODEL_MAX_LEN[$name]}" \
            "$env_name" \
            "${MODEL_PATH[$name]}"
    done < <(
        for name in "${!MODEL_PATH[@]}"; do
            printf '%s\t%s\n' "${MODEL_PORT[$name]}" "$name"
        done | sort -n -k1,1 -k2,2 | cut -f2-
    )
}

shell_split_append() {
    # Split a trusted, locally configured argument string into an array.
    # Do not put untrusted input in MODEL_EXTRA_ARGS.
    local input="$1"
    local -n output_array="$2"

    if [[ -n "$input" ]]; then
        local -a parsed=()
        read -r -a parsed <<< "$input"
        output_array+=("${parsed[@]}")
    fi
}

activate_conda_env() {
    local target_env="$1"

    if [[ -z "$target_env" ]]; then
        return 0
    fi

    if [[ "${CONDA_DEFAULT_ENV:-}" == "$target_env" ]]; then
        return 0
    fi

    local conda_command
    conda_command="$(command -v conda || true)"

    if [[ -z "$conda_command" ]]; then
        echo "Error: conda command was not found." >&2
        echo "Activate Conda or add its condabin directory to PATH." >&2
        exit 1
    fi

    local conda_base
    conda_base="$("$conda_command" info --base)"

    local conda_init_script="${conda_base}/etc/profile.d/conda.sh"

    if [[ ! -f "$conda_init_script" ]]; then
        echo "Error: Conda initialization script was not found:" >&2
        echo "  $conda_init_script" >&2
        exit 1
    fi

    echo "Activating Conda environment: $target_env"

    # Conda initialization scripts may reference unset variables.
    set +u
    # shellcheck source=/dev/null
    source "$conda_init_script"

    if ! conda activate "$target_env"; then
        set -u
        echo "Error: failed to activate Conda environment: $target_env" >&2
        exit 1
    fi

    set -u
}

resolve_python_executable() {
    local target_conda_env="${1:-}"

    if [[ -n "$PYTHON_BIN_OVERRIDE" ]]; then
        if [[ "$PYTHON_BIN_OVERRIDE" == */* ]]; then
            if [[ ! -x "$PYTHON_BIN_OVERRIDE" ]]; then
                echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN_OVERRIDE" >&2
                exit 1
            fi
            printf '%s\n' "$PYTHON_BIN_OVERRIDE"
            return 0
        fi

        local overridden_python
        overridden_python="$(command -v "$PYTHON_BIN_OVERRIDE" || true)"

        if [[ -z "$overridden_python" ]]; then
            echo "Error: PYTHON_BIN command was not found: $PYTHON_BIN_OVERRIDE" >&2
            exit 1
        fi

        printf '%s\n' "$overridden_python"
        return 0
    fi

    local selected_python

    # A virtual environment can remain at the front of PATH even when Conda
    # reports the requested environment as active. For explicitly mapped
    # models, use that Conda environment's Python directly.
    if [[ -n "$target_conda_env" ]]; then
        selected_python="${CONDA_PREFIX:-}/bin/python"

        if [[ -z "${CONDA_PREFIX:-}" || ! -x "$selected_python" ]]; then
            echo "Error: no usable Python executable was found for Conda environment: $target_conda_env" >&2
            exit 1
        fi

        printf '%s\n' "$selected_python"
        return 0
    fi

    selected_python="$(command -v python || true)"

    if [[ -z "$selected_python" || ! -x "$selected_python" ]]; then
        echo "Error: no usable Python executable was found." >&2
        exit 1
    fi

    printf '%s\n' "$selected_python"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

if [[ "$1" == "list" || "$1" == "--list" ]]; then
    list_models
    exit 0
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    echo "Error: model_name and cuda_visible_devices are required." >&2
    echo >&2
    usage >&2
    exit 1
fi

MODEL_NAME="$1"
CUDA_DEVICES="$2"
shift 2

if [[ -z "${MODEL_PATH[$MODEL_NAME]+x}" ]]; then
    echo "Error: unknown model name: $MODEL_NAME" >&2
    echo >&2
    list_models >&2
    exit 1
fi

BACKGROUND=false
USER_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --)
            shift
            USER_EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            USER_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

MODEL="${MODEL_PATH[$MODEL_NAME]}"
PORT="${MODEL_PORT[$MODEL_NAME]}"
TP="${MODEL_TP[$MODEL_NAME]}"
MAX_LEN="${MODEL_MAX_LEN[$MODEL_NAME]}"
GPU_UTIL="${MODEL_GPU_UTIL[$MODEL_NAME]}"
TOOL_PARSER="${MODEL_TOOL_PARSER[$MODEL_NAME]}"
AUTO_TOOL="${MODEL_AUTO_TOOL[$MODEL_NAME]}"
ENFORCE_EAGER="${MODEL_ENFORCE_EAGER[$MODEL_NAME]}"
CONFIG_EXTRA_ARGS="${MODEL_EXTRA_ARGS[$MODEL_NAME]}"
TARGET_CONDA_ENV="${MODEL_CONDA_ENV[$MODEL_NAME]-}"
RUNNER="${MODEL_RUNNER[$MODEL_NAME]}"

if [[ ! -e "$MODEL" ]]; then
    echo "Warning: model path does not exist: $MODEL" >&2
fi

# PYTHON_BIN explicitly set: use it directly and do not switch environments.
# Otherwise, activate the environment configured for this model.
if [[ -z "$PYTHON_BIN_OVERRIDE" && -n "$TARGET_CONDA_ENV" ]]; then
    activate_conda_env "$TARGET_CONDA_ENV"
fi

PYTHON_EXECUTABLE="$(resolve_python_executable "$TARGET_CONDA_ENV")"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

if [[ "$RUNNER" == "fastdllm" ]]; then
    CMD=(
        env
        "FASTDLLM_PORT=$PORT"
        "$PYTHON_EXECUTABLE"
        "$MODEL"
    )
else
    CMD=(
        "$PYTHON_EXECUTABLE"
        -m "$VLLM_MODULE"
        --model "$MODEL"
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_LEN"
        --gpu-memory-utilization "$GPU_UTIL"
        --port "$PORT"
    )

    if [[ -n "$TOOL_PARSER" ]]; then
        CMD+=(--tool-call-parser "$TOOL_PARSER")

        if [[ "$AUTO_TOOL" == "true" ]]; then
            CMD+=(--enable-auto-tool-choice)
        fi
    elif [[ "$AUTO_TOOL" == "true" ]]; then
        echo "Error: auto tool choice is enabled for '$MODEL_NAME'," >&2
        echo "but no tool-call parser is configured." >&2
        exit 1
    fi

    if [[ "$ENFORCE_EAGER" == "true" ]]; then
        CMD+=(--enforce-eager)
    fi
fi

shell_split_append "$CONFIG_EXTRA_ARGS" CMD
CMD+=("${USER_EXTRA_ARGS[@]}")

ACTIVE_CONDA_ENV="${CONDA_DEFAULT_ENV:-none}"
ACTIVE_CONDA_PREFIX="${CONDA_PREFIX:-none}"
PYTHON_VERSION="$("$PYTHON_EXECUTABLE" -V 2>&1)"

echo "============================================================"
echo "Model name:           $MODEL_NAME"
echo "Model path:           $MODEL"
echo "Runner:               $RUNNER"
echo "Requested Conda env:  ${TARGET_CONDA_ENV:-current}"
echo "Active Conda env:     $ACTIVE_CONDA_ENV"
echo "Conda prefix:         $ACTIVE_CONDA_PREFIX"
echo "Python executable:    $PYTHON_EXECUTABLE"
echo "Python version:       $PYTHON_VERSION"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Port:                 $PORT"
echo "Tensor parallel:      $TP"
echo "Max model length:     $MAX_LEN"
echo "GPU utilization:      $GPU_UTIL"
echo "Background:           $BACKGROUND"
printf 'Command:              '
printf '%q ' "${CMD[@]}"
printf '\n'
echo "============================================================"

if [[ "$BACKGROUND" == "true" ]]; then
    mkdir -p "$LOG_DIR"

    SAFE_CUDA_DEVICES="${CUDA_DEVICES//,/-}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_gpu-${SAFE_CUDA_DEVICES}_port-${PORT}.log"
    PID_FILE="${LOG_DIR}/${MODEL_NAME}_gpu-${SAFE_CUDA_DEVICES}_port-${PORT}.pid"

    nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" >"$PID_FILE"

    echo "Started in background."
    echo "PID:      $PID"
    echo "PID file: $PID_FILE"
    echo "Log file: $LOG_FILE"
    echo "Follow:   tail -f '$LOG_FILE'"
else
    exec "${CMD[@]}"
fi

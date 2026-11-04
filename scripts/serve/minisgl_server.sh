#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_minisgl_server.sh --model PATH [options]

Options:
  --env NAME                     Expected active conda env name. Optional validation only.
  --model PATH                   Model path or HF repo. Required.
  --gpus CSV                     CUDA_VISIBLE_DEVICES. Default: 0,1,2,3
  --host HOST                    Default: 127.0.0.1
  --port PORT                    Default: 19193
  --tp-size N                    Tensor parallel size. Default: number of GPUs in --gpus
  --max-running-requests N       Default: 1
  --max-seq-len-override N       Optional
  --cuda-graph-max-bs N          Default: 1
  --num-pages N                  Default: 256
  --attention-backend NAME       Default: trtllm
  --moe-backend NAME             Optional
  --memory-ratio FLOAT           Optional
  --launch-backend NAME          Default: ray
  --ray-address ADDR             Default: auto
  --ray-log-dir DIR              Optional
  --ray-nsys                     Enable Nsight Systems profiling for each Ray scheduler actor
  --ray-nsys-output-prefix PATH  Optional Nsight Systems output prefix
  --disable-pynccl               Pass through
  --dummy-weight                 Pass through
  --extra-args "..."             Extra args appended as shell words
EOF
}

count_csv_items() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys
items = [x for x in sys.argv[1].split(",") if x]
print(len(items))
PY
}

ENV_NAME=""
MODEL=""
GPUS="0,1,2,3"
HOST="127.0.0.1"
PORT="19193"
TP_SIZE=""
MAX_RUNNING_REQUESTS="1"
MAX_SEQ_LEN_OVERRIDE=""
CUDA_GRAPH_MAX_BS="1"
NUM_PAGES="256"
ATTENTION_BACKEND="trtllm"
MOE_BACKEND=""
MEMORY_RATIO=""
LAUNCH_BACKEND="ray"
RAY_ADDRESS="auto"
RAY_LOG_DIR=""
RAY_NSYS=0
RAY_NSYS_OUTPUT_PREFIX=""
DISABLE_PYNCCL=0
DUMMY_WEIGHT=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --max-seq-len-override) MAX_SEQ_LEN_OVERRIDE="$2"; shift 2 ;;
    --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --num-pages) NUM_PAGES="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
    --memory-ratio) MEMORY_RATIO="$2"; shift 2 ;;
    --launch-backend) LAUNCH_BACKEND="$2"; shift 2 ;;
    --ray-address) RAY_ADDRESS="$2"; shift 2 ;;
    --ray-log-dir) RAY_LOG_DIR="$2"; shift 2 ;;
    --ray-nsys) RAY_NSYS=1; shift ;;
    --ray-nsys-output-prefix) RAY_NSYS_OUTPUT_PREFIX="$2"; shift 2 ;;
    --disable-pynccl) DISABLE_PYNCCL=1; shift ;;
    --dummy-weight) DUMMY_WEIGHT=1; shift ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  usage >&2
  exit 1
fi

if [[ -n "$ENV_NAME" ]]; then
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Expected active conda env '$ENV_NAME', but no conda env is activated." >&2
    exit 1
  fi
  if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Expected active conda env '$ENV_NAME', got '$CONDA_DEFAULT_ENV'." >&2
    exit 1
  fi
fi

if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_csv_items "$GPUS")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"

cmd=(
  python -m minisgl
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  --num-pages "$NUM_PAGES"
  --attention-backend "$ATTENTION_BACKEND"
  --launch-backend "$LAUNCH_BACKEND"
  --ray-address "$RAY_ADDRESS"
)

if [[ -n "$RAY_LOG_DIR" ]]; then
  cmd+=(--ray-log-dir "$RAY_LOG_DIR")
fi
if [[ "$RAY_NSYS" -eq 1 ]]; then
  cmd+=(--ray-nsys)
fi
if [[ -n "$RAY_NSYS_OUTPUT_PREFIX" ]]; then
  cmd+=(--ray-nsys-output-prefix "$RAY_NSYS_OUTPUT_PREFIX")
fi

if [[ -n "$MAX_SEQ_LEN_OVERRIDE" ]]; then
  cmd+=(--max-seq-len-override "$MAX_SEQ_LEN_OVERRIDE")
fi
if [[ -n "$MOE_BACKEND" ]]; then
  cmd+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$MEMORY_RATIO" ]]; then
  cmd+=(--memory-ratio "$MEMORY_RATIO")
fi
if [[ "$DISABLE_PYNCCL" -eq 1 ]]; then
  cmd+=(--disable-pynccl)
fi
if [[ "$DUMMY_WEIGHT" -eq 1 ]]; then
  cmd+=(--dummy-weight)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra <<< "$EXTRA_ARGS"
  cmd+=("${extra[@]}")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

# Triton on Blackwell may require an explicit ptxas-blackwell path during
# first-time kernel compilation. Mirror the vLLM startup defaults so local mp
# runs do not depend on the caller shell pre-exporting toolchain variables.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CUDA_TARGET_ROOT="${CONDA_PREFIX}/targets/sbsa-linux"
  if [[ -f "${CUDA_TARGET_ROOT}/include/cuda_bf16.h" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDA_PATH="${CONDA_PREFIX}"
    export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${CONDA_PREFIX}/bin/nvcc}"
    export PATH="${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${CUDA_TARGET_ROOT}/bin:${PATH}"
    if [[ -n "${FLASHINFER_WORKSPACE_BASE:-}" && \
          "${FLASHINFER_WORKSPACE_BASE##*/}" != "cuda-target-sbsa-linux" ]]; then
      export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE%/}/cuda-target-sbsa-linux"
    fi
  elif [[ -z "${CUDA_HOME:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  fi
  if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
  fi
  if [[ -z "${CUDA_NVCC_EXECUTABLE:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_NVCC_EXECUTABLE="${CONDA_PREFIX}/bin/nvcc"
  fi
fi

if [[ -z "${TRITON_PTXAS_BLACKWELL_PATH:-}" ]]; then
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/ptxas" ]]; then
    export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME}/bin/ptxas"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/ptxas" ]]; then
    export TRITON_PTXAS_BLACKWELL_PATH="${CONDA_PREFIX}/bin/ptxas"
  fi
fi

cd "$CALIB_DIR"
exec "${cmd[@]}"

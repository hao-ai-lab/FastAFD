#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_vllm_server.sh --model PATH [options]

Options:
  --env NAME                Expected active conda env name. Default: minisgl-cuda130
  --model PATH              Model path or HF repo. Required.
  --gpus CSV                CUDA_VISIBLE_DEVICES. Default: 0,1,2,3
  --port PORT               Default: 8000
  --host HOST               Default: 127.0.0.1
  --tp-size N               Tensor parallel size. Default: number of GPUs in --gpus
  --max-model-len N         Optional
  --served-model-name NAME  Optional
  --extra-args "..."        Extra args appended as shell words
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

ENV_NAME="minisgl-cuda130"
MODEL=""
GPUS="0,1,2,3"
PORT="8000"
HOST="127.0.0.1"
TP_SIZE=""
MAX_MODEL_LEN=""
SERVED_MODEL_NAME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
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

if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$(count_csv_items "$GPUS")"
fi

cmd=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
  cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra <<< "$EXTRA_ARGS"
  cmd+=("${extra[@]}")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "Expected active conda env '$ENV_NAME', but no conda env is activated." >&2
  exit 1
fi
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  echo "Expected active conda env '$ENV_NAME', got '$CONDA_DEFAULT_ENV'." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

# vLLM/Triton on Blackwell looks for an explicit ptxas-blackwell path.
# Mirror the mini-sgl clean-shell defaults so ad-hoc runs don't depend on
# the caller having pre-exported the toolchain env.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CUDA_TARGET_ROOT="${CONDA_PREFIX}/targets/sbsa-linux"
  if [[ -f "${CUDA_TARGET_ROOT}/include/cuda_bf16.h" ]]; then
    # CUDA 13 Conda packages keep headers/binaries at the prefix and expose
    # target libraries through $CONDA_PREFIX/lib64. Pointing CUDA_HOME at the
    # sbsa target directly makes FlashInfer link against a non-existent
    # targets/sbsa-linux/lib64 directory during a cold JIT build.
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

exec "${cmd[@]}"

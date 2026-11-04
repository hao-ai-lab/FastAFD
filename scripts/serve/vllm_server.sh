#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_vllm_server.sh --model PATH [options]

Options:
  --env NAME|PATH           Expected active conda env name, or the path of a
                            Python env (contains '/'; a directory with
                            bin/python and bin/vllm, e.g. a venv) whose own
                            bin/vllm is run without conda. Default: minisgl-cuda130
  --model PATH              Model path or HF repo. Required.
  --gpus CSV                CUDA_VISIBLE_DEVICES. Default: 0,1,2,3
  --port PORT               Default: 8000
  --host HOST               Default: 127.0.0.1
  --tp-size N               Tensor parallel size. Default: number of GPUs in --gpus
  --max-model-len N         Optional
  --served-model-name NAME  Optional
  --extra-args "..."        Extra args appended as shell words
  --identity-json FILE      Before serving, record the reference identity (vLLM
                            source and extensions, torch, FlashInfer) to FILE

Environment:
  VLLM_ENV_SETUP            Path-type --env only: a bash file sourced before
                            vLLM starts (e.g. to name the CUDA toolkit for a
                            venv that has none; a conda env uses its own
                            activation scripts instead)
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
IDENTITY_JSON=""

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
    --identity-json) IDENTITY_JSON="$2"; shift 2 ;;
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../lib/common.sh"
ENV_PREFIX="$(fastafd_python_env_prefix "$ENV_NAME")" || exit 1

cmd=(
  "${ENV_PREFIX:+$ENV_PREFIX/bin/}vllm" serve "$MODEL"
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
if [[ -n "$ENV_PREFIX" ]]; then
  # A path-type env is used as is: no conda activation, its own bin/ first.
  if [[ ! -x "$ENV_PREFIX/bin/vllm" ]]; then
    echo "Python env '$ENV_PREFIX' has no executable bin/vllm." >&2
    exit 1
  fi
  if [[ -n "${VLLM_ENV_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$VLLM_ENV_SETUP"
    unset VLLM_ENV_SETUP  # vLLM warns about unknown VLLM_* variables
  fi
  export PATH="$ENV_PREFIX/bin:$PATH"
else
  if [[ -n "${VLLM_ENV_SETUP:-}" ]]; then
    echo "VLLM_ENV_SETUP applies only to a path-type --env; conda env '$ENV_NAME' uses its own activation scripts." >&2
    exit 1
  fi
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Expected active conda env '$ENV_NAME', but no conda env is activated." >&2
    exit 1
  fi
  if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Expected active conda env '$ENV_NAME', got '$CONDA_DEFAULT_ENV'." >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

# vLLM/Triton on Blackwell looks for an explicit ptxas-blackwell path.
# Mirror the mini-sgl clean-shell defaults so ad-hoc runs don't depend on
# the caller having pre-exported the toolchain env. A path-type env is not a
# conda env: it runs with the caller's toolchain or the one VLLM_ENV_SETUP names.
if [[ -z "$ENV_PREFIX" && -n "${CONDA_PREFIX:-}" ]]; then
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
    # FlashInfer's JIT compiles host code with -isystem $CUDA_HOME/include and
    # links with -L$CUDA_HOME/lib64{,/stubs}. Some CUDA 13 Conda envs keep the
    # headers and the libcuda stub only in the target tree, so name those
    # directories instead of relying on the caller's CPATH or LIBRARY_PATH.
    if [[ ! -f "${CONDA_PREFIX}/include/cuda_fp8.h" ]]; then
      export FLASHINFER_EXTRA_CFLAGS="${FLASHINFER_EXTRA_CFLAGS:+${FLASHINFER_EXTRA_CFLAGS} }-isystem ${CUDA_TARGET_ROOT}/include -isystem ${CUDA_TARGET_ROOT}/include/cccl"
    fi
    if [[ ! -e "${CONDA_PREFIX}/lib64/stubs/libcuda.so" && -f "${CUDA_TARGET_ROOT}/lib/stubs/libcuda.so" ]]; then
      export FLASHINFER_EXTRA_LDFLAGS="${FLASHINFER_EXTRA_LDFLAGS:+${FLASHINFER_EXTRA_LDFLAGS} }-L${CUDA_TARGET_ROOT}/lib -L${CUDA_TARGET_ROOT}/lib/stubs"
    fi
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

if [[ -n "$IDENTITY_JSON" ]]; then
  # Record what this env will actually import (a version string is not an
  # identity). A failed probe does not block serving: the report then lacks
  # reference_identity and check_alignment_report.py warns.
  if ! PYTHONDONTWRITEBYTECODE=1 "${ENV_PREFIX:+$ENV_PREFIX/bin/}python" -B \
      "$SCRIPT_DIR/../validate/reference_identity.py" --output "$IDENTITY_JSON"; then
    echo "WARNING: reference identity probe failed; $IDENTITY_JSON was not written." >&2
  fi
fi

exec "${cmd[@]}"

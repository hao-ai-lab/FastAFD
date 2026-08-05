#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fastafd_server.sh --model PATH [options]

Options:
  --env NAME                     Expected active conda env name. Optional validation only.
  --model PATH                   Model path or HF repo. Required.
  --host HOST                    Default: 127.0.0.1
  --port PORT                    Default: 19193
  --launch-backend NAME          Default: ray
  --ray-address ADDR             Default: auto
  --ray-log-dir DIR              Optional directory for Ray-side AFD logs
  --ray-nsys                     Enable Nsight Systems profiling for Ray-side AFD workers
  --ray-nsys-output-prefix PATH  Nsight Systems output prefix
  --afd-report-dir DIR           Optional directory for AFD worker/coordinator logs
  --afd-attn-dp-size N           Default: 1
  --afd-mlp-dp-size N            Default: 1
  --afd-attn-tp-size N           Default: 4
  --afd-mlp-tp-size N            Default: 4
  --afd-mlp-ep-size N            Default: 1
  --afd-moe-a2a-backend NAME     Default: none
  --afd-moe-runner-backend NAME  Default: auto
  --afd-batch-size N             Default: 8
  --afd-max-batched-tokens N     Default: 8192
  --afd-max-running-requests N   Default: auto
  --afd-max-new-tokens N         Default: 1
  --cache-type NAME              Default: naive
  --cuda-graph-max-bs N          Default: 8
  --afd-decode-graph-bs CSV      Optional static decode graph buckets
  --afd-num-mb N                 Default: 1
  --afd-async-pending-steps N    Default: 3
  --afd-force-multi-mb-graph-overlap
                                  Diagnostic opt-in for cross-step MB graph overlap
  --afd-disable-overlap          Disable cross-step overlap
  --afd-device-comm-num-sms N    Default: 1
  --page-size N                  Default: 1
  --max-seq-len-override N       Optional
  --extra-args "..."             Extra args appended as shell words
EOF
}

ENV_NAME=""
MODEL=""
HOST="127.0.0.1"
PORT="19193"
LAUNCH_BACKEND="ray"
RAY_ADDRESS="auto"
RAY_LOG_DIR=""
RAY_NSYS=0
RAY_NSYS_OUTPUT_PREFIX=""
AFD_REPORT_DIR=""
AFD_ATTN_DP_SIZE="1"
AFD_MLP_DP_SIZE="1"
AFD_ATTN_TP_SIZE="4"
AFD_MLP_TP_SIZE="4"
AFD_MLP_EP_SIZE="1"
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-none}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-auto}"
AFD_BATCH_SIZE="8"
AFD_MAX_BATCHED_TOKENS="8192"
AFD_MAX_RUNNING_REQUESTS="0"
AFD_MAX_NEW_TOKENS="1"
CACHE_TYPE="naive"
CUDA_GRAPH_MAX_BS="8"
AFD_DECODE_GRAPH_BS=""
AFD_NUM_MB="1"
AFD_ASYNC_PENDING_STEPS="3"
AFD_FORCE_MULTI_MB_GRAPH_OVERLAP=0
AFD_DISABLE_OVERLAP=0
AFD_DEVICE_COMM_NUM_SMS="1"
PAGE_SIZE="1"
MAX_SEQ_LEN_OVERRIDE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --launch-backend) LAUNCH_BACKEND="$2"; shift 2 ;;
    --ray-address) RAY_ADDRESS="$2"; shift 2 ;;
    --ray-log-dir) RAY_LOG_DIR="$2"; shift 2 ;;
    --ray-nsys) RAY_NSYS=1; shift ;;
    --ray-nsys-output-prefix) RAY_NSYS_OUTPUT_PREFIX="$2"; shift 2 ;;
    --afd-report-dir) AFD_REPORT_DIR="$2"; shift 2 ;;
    --afd-attn-dp-size) AFD_ATTN_DP_SIZE="$2"; shift 2 ;;
    --afd-mlp-dp-size) AFD_MLP_DP_SIZE="$2"; shift 2 ;;
    --afd-attn-tp-size) AFD_ATTN_TP_SIZE="$2"; shift 2 ;;
    --afd-mlp-tp-size) AFD_MLP_TP_SIZE="$2"; shift 2 ;;
    --afd-mlp-ep-size) AFD_MLP_EP_SIZE="$2"; shift 2 ;;
    --afd-moe-a2a-backend) AFD_MOE_A2A_BACKEND="$2"; shift 2 ;;
    --afd-moe-runner-backend) AFD_MOE_RUNNER_BACKEND="$2"; shift 2 ;;
    --afd-batch-size) AFD_BATCH_SIZE="$2"; shift 2 ;;
    --afd-max-batched-tokens) AFD_MAX_BATCHED_TOKENS="$2"; shift 2 ;;
    --afd-max-running-requests) AFD_MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --afd-max-new-tokens) AFD_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --cache-type) CACHE_TYPE="$2"; shift 2 ;;
    --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --afd-decode-graph-bs) AFD_DECODE_GRAPH_BS="$2"; shift 2 ;;
    --afd-num-mb) AFD_NUM_MB="$2"; shift 2 ;;
    --afd-async-pending-steps) AFD_ASYNC_PENDING_STEPS="$2"; shift 2 ;;
    --afd-force-multi-mb-graph-overlap) AFD_FORCE_MULTI_MB_GRAPH_OVERLAP=1; shift ;;
    --afd-disable-overlap) AFD_DISABLE_OVERLAP=1; shift ;;
    --afd-device-comm-num-sms) AFD_DEVICE_COMM_NUM_SMS="$2"; shift 2 ;;
    --page-size) PAGE_SIZE="$2"; shift 2 ;;
    --max-seq-len-override) MAX_SEQ_LEN_OVERRIDE="$2"; shift 2 ;;
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
  --mode afd-serve
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --launch-backend "$LAUNCH_BACKEND"
  --ray-address "$RAY_ADDRESS"
  --afd-attn-dp-size "$AFD_ATTN_DP_SIZE"
  --afd-mlp-dp-size "$AFD_MLP_DP_SIZE"
  --afd-attn-tp-size "$AFD_ATTN_TP_SIZE"
  --afd-mlp-tp-size "$AFD_MLP_TP_SIZE"
  --afd-mlp-ep-size "$AFD_MLP_EP_SIZE"
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND"
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND"
  --afd-batch-size "$AFD_BATCH_SIZE"
  --afd-max-batched-tokens "$AFD_MAX_BATCHED_TOKENS"
  --afd-max-running-requests "$AFD_MAX_RUNNING_REQUESTS"
  --afd-max-new-tokens "$AFD_MAX_NEW_TOKENS"
  --cache-type "$CACHE_TYPE"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  --afd-num-mb "$AFD_NUM_MB"
  --afd-async-pending-steps "$AFD_ASYNC_PENDING_STEPS"
  --afd-device-comm-num-sms "$AFD_DEVICE_COMM_NUM_SMS"
  --page-size "$PAGE_SIZE"
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
if [[ -n "$AFD_REPORT_DIR" ]]; then
  cmd+=(--afd-report-dir "$AFD_REPORT_DIR")
fi
if [[ -n "$AFD_DECODE_GRAPH_BS" ]]; then
  cmd+=(--afd-decode-graph-bs "$AFD_DECODE_GRAPH_BS")
fi
if [[ "$AFD_FORCE_MULTI_MB_GRAPH_OVERLAP" -eq 1 ]]; then
  cmd+=(--afd-force-multi-mb-graph-overlap)
fi
if [[ "$AFD_DISABLE_OVERLAP" -eq 1 ]]; then
  cmd+=(--afd-disable-overlap)
fi
if [[ -n "$MAX_SEQ_LEN_OVERRIDE" ]]; then
  cmd+=(--max-seq-len-override "$MAX_SEQ_LEN_OVERRIDE")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra <<< "$EXTRA_ARGS"
  cmd+=("${extra[@]}")
fi

export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CUDA_TARGET_ROOT="${CONDA_PREFIX}/targets/sbsa-linux"
  if [[ -f "${CUDA_TARGET_ROOT}/include/cuda_bf16.h" ]]; then
    # General MiniSGL extensions (pynccl, Triton/TVM-FFI AOT) resolve CUDA
    # from the Conda prefix. The NVFP4 module retargets FlashInfer in-process.
    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDA_PATH="${CONDA_PREFIX}"
    export CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE:-${CONDA_PREFIX}/bin/nvcc}"
    export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${CONDA_PREFIX}/bin/nvcc}"
    export PATH="${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${CUDA_TARGET_ROOT}/bin:${PATH}"
    if [[ -n "${FLASHINFER_WORKSPACE_BASE:-}" && \
          "${FLASHINFER_WORKSPACE_BASE##*/}" != "cuda-target-sbsa-linux" ]]; then
      export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE%/}/cuda-target-sbsa-linux"
    fi
  fi
fi

cd "$CALIB_DIR"
exec "${cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# AFD AG/EG, Qwen3-30B-A3B-FP8 workload:
#   512 prompts x 8192 tokens, generate 16 tokens
#   attention: dp=${AFD_ATTN_DP_SIZE:-2}, tp=4
#   MLP: dp=4, tp=1, ep=4
#   mb=2, CUDA graph replay, NSYS capture over decode, then vLLM TP=1 alignment.

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-8192}"
export NUM_PROMPTS="${NUM_PROMPTS:-512}"
export PROMPT_FILE="${PROMPT_FILE:-$CALIB_DIR/reports/prompts_${NUM_PROMPTS}x${PROMPT_LEN}_seed20260527.txt}"
if [[ -f "$PROMPT_FILE" ]]; then
  export PROMPT_KIND="${PROMPT_KIND:-existing}"
fi
export MAX_TOKENS="${MAX_TOKENS:-16}"
export AFD_MAX_NEW_TOKENS="${AFD_MAX_NEW_TOKENS:-$MAX_TOKENS}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
export BATCH_SUBMIT="${BATCH_SUBMIT:-1}"
export IGNORE_EOS="${IGNORE_EOS:-1}"

export AFD_ATTN_DP_SIZE="${AFD_ATTN_DP_SIZE:-${ATTN_DP_SIZE:-2}}"
export ATTN_DP_SIZE="$AFD_ATTN_DP_SIZE"
export AFD_ATTN_TP_SIZE="${AFD_ATTN_TP_SIZE:-${ATTN_TP_SIZE:-4}}"
export ATTN_TP_SIZE="$AFD_ATTN_TP_SIZE"
export MLP_DP_SIZE="${MLP_DP_SIZE:-4}"
export MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
export AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
export AFD_NUM_MB="${AFD_NUM_MB:-2}"
export AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
export MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-8256}"
export MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-256}"
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,24,32,40,48,56,64,128,256}"
export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-32768}"

export AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend trtllm --memory-ratio 0.65}"

export NSYS="${NSYS:-1}"
export NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
_PREFILL_REQS_PER_DP_STEP=$(( (AFD_MAX_BATCHED_TOKENS + PROMPT_LEN - 1) / PROMPT_LEN ))
if (( _PREFILL_REQS_PER_DP_STEP < 1 )); then
  _PREFILL_REQS_PER_DP_STEP=1
fi
_PREFILL_GLOBAL_STEPS=$(( (NUM_PROMPTS + ATTN_DP_SIZE * _PREFILL_REQS_PER_DP_STEP - 1) / (ATTN_DP_SIZE * _PREFILL_REQS_PER_DP_STEP) ))
_DECODE_GRAPH_STEPS=$(( MAX_TOKENS > 0 ? MAX_TOKENS - 1 : 0 ))
export NSYS_START_STEP="${NSYS_START_STEP:-$_PREFILL_GLOBAL_STEPS}"
export NSYS_STOP_STEP="${NSYS_STOP_STEP:-$((_PREFILL_GLOBAL_STEPS + _DECODE_GRAPH_STEPS))}"
if [[ -z "${MINISGL_RAY_NSYS_BIN:-}" ]]; then
  if [[ -n "${CUDA_13_HOME:-}" && -x "$CUDA_13_HOME/bin/nsys" ]]; then
    export MINISGL_RAY_NSYS_BIN="$CUDA_13_HOME/bin/nsys"
  elif [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nsys" ]]; then
    export MINISGL_RAY_NSYS_BIN="$CUDA_HOME/bin/nsys"
  else
    export MINISGL_RAY_NSYS_BIN="$(command -v nsys || true)"
  fi
fi

export RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-1}"
export VLLM_GPUS="${VLLM_GPUS:-0}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
export VLLM_PROMPT_LOGPROBS="${VLLM_PROMPT_LOGPROBS:-0}"
export VLLM_SCORE_CONCURRENCY="${VLLM_SCORE_CONCURRENCY:-16}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export ALIGN_MIN_TOP10_RATE="${ALIGN_MIN_TOP10_RATE:-1.0}"

export CLEAN_FIRST="${CLEAN_FIRST:-0}"
export CLEAN_AFTER="${CLEAN_AFTER:-0}"
export POST_SHUTDOWN_SLEEP="${POST_SHUTDOWN_SLEEP:-20}"
export AFD_STOP_TIMEOUT_S="${AFD_STOP_TIMEOUT_S:-60}"

export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen3_30b_a3b_fp8_attn${ATTN_DP_SIZE}dp_tp${ATTN_TP_SIZE}_mlp${MLP_DP_SIZE}dp_ep${AFD_MLP_EP_SIZE}_mb${AFD_NUM_MB}_nsys_align_${TIMESTAMP}}"

exec "$SCRIPT_DIR/run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh"

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

ENV_NAME="${ENV_NAME:-minisgl-cuda130}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPT_LEN="${PROMPT_LEN:-8192}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
PROMPT_KIND="${PROMPT_KIND:-realistic}"
PROMPT_SEED="${PROMPT_SEED:-20260527}"
MAX_TOKENS="${MAX_TOKENS:-16}"
AFD_MAX_NEW_TOKENS="${AFD_MAX_NEW_TOKENS:-$MAX_TOKENS}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
BATCH_SUBMIT="${BATCH_SUBMIT:-1}"
IGNORE_EOS="${IGNORE_EOS:-1}"
WARMUP_REPLAYS="${WARMUP_REPLAYS:-0}"

ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-4}"
MLP_DP_SIZE="${MLP_DP_SIZE:-4}"
MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-deepep}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"

AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-256}"
AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,24,32,40,48,56,64,128,256}"
AFD_NUM_MB="${AFD_NUM_MB:-2}"
AFD_ASYNC_PENDING_STEPS="${AFD_ASYNC_PENDING_STEPS:-3}"
AFD_FORCE_MULTI_MB_GRAPH_OVERLAP="${AFD_FORCE_MULTI_MB_GRAPH_OVERLAP:-0}"
AFD_DISABLE_OVERLAP="${AFD_DISABLE_OVERLAP:-0}"
AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-32768}"
MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-8256}"
MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-64}"
AFD_DEVICE_COMM_NUM_SMS="${AFD_DEVICE_COMM_NUM_SMS:-4}"
AFD_CACHE_TYPE="${AFD_CACHE_TYPE:-naive}"
AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend trtllm}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MINISGL_PORT="${MINISGL_PORT:-}"
MINISGL_RAY_ADDRESS="${MINISGL_RAY_ADDRESS:-auto}"
RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-0}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_PORT="${VLLM_PORT:-}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
VLLM_PROMPT_LOGPROBS="${VLLM_PROMPT_LOGPROBS:-0}"
VLLM_SCORE_CONCURRENCY="${VLLM_SCORE_CONCURRENCY:-16}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-3600}"
VLLM_WAIT_READY_TIMEOUT="${VLLM_WAIT_READY_TIMEOUT:-1800}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
NSYS="${NSYS:-0}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-${MINISGL_RAY_NSYS_CUDA_GRAPH_TRACE:-graph}}"
NSYS_START_STEP="${NSYS_START_STEP:-${MINISGL_RAY_NSYS_START_STEP:-0}}"
NSYS_STOP_STEP="${NSYS_STOP_STEP:-${MINISGL_RAY_NSYS_STOP_STEP:-0}}"
# Opt-in: kill_afd_ray_tasks.sh SIGKILLs matching processes on every live Ray
# node, other people's jobs included. Set 1 only on a cluster you have to yourself.
CLEAN_FIRST="${CLEAN_FIRST:-0}"
CLEAN_AFTER="${CLEAN_AFTER:-0}"
WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-1800}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-3600}"
POST_SHUTDOWN_SLEEP="${POST_SHUTDOWN_SLEEP:-20}"
# Grace (seconds) for the AFD supervisor tree to shut down on SIGTERM before
# SIGKILL. The per-rank workers flush their CPU NVTX trace + finalize their
# nsys-rep during this window; with many workers / multi-node it needs to be
# larger than the old hardcoded 10s, or the workers get killed mid-flush and
# their minisgl_rank*.nsys-rep / *_nvtx_cpu.json never get written.
AFD_STOP_TIMEOUT_S="${AFD_STOP_TIMEOUT_S:-10}"

export MINISGL_LOCAL_CACHE_BASE="${MINISGL_LOCAL_CACHE_BASE:-$CALIB_DIR/cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$MINISGL_LOCAL_CACHE_BASE/flashinfer}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/tvm_ffi}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_jit}"
export N2M_M2N_GIN_BUILD_DIR="${N2M_M2N_GIN_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/gin_comm}"
export MINISGL_DEEPEP_BUILD_DIR="${MINISGL_DEEPEP_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_moe}"
export MINISGL_DEEPGEMM_BUILD_DIR="${MINISGL_DEEPGEMM_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepgemm}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_fp8_long8k_dp4_ep4_attn1_tp4_${TIMESTAMP}}"
if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$CALIB_DIR/$RUN_DIR"
fi
PROMPT_FILE="${PROMPT_FILE:-$RUN_DIR/prompts_${NUM_PROMPTS}x${PROMPT_LEN}.txt}"
SAMPLE_JSON="$RUN_DIR/sample.json"
WARMUP_JSON="$RUN_DIR/warmup.json"
ALIGNMENT_JSON="$RUN_DIR/alignment.json"
AFD_LOG="$RUN_DIR/afd.log"
SAMPLE_LOG="$RUN_DIR/sample.log"
WARMUP_LOG="$RUN_DIR/warmup.log"
VLLM_LOG="$RUN_DIR/vllm.log"
SCORE_LOG="$RUN_DIR/score.log"
RAY_LOG_DIR="$RUN_DIR/ray_logs"
NSYS_OUTPUT_PREFIX="${NSYS_OUTPUT_PREFIX:-$RAY_LOG_DIR/minisgl}"

mkdir -p "$RUN_DIR" "$RAY_LOG_DIR"

if [[ -n "$ENV_NAME" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "Expected active conda env '$ENV_NAME', got '${CONDA_DEFAULT_ENV:-<none>}'." >&2
    echo "Run: source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_NAME" >&2
    exit 1
  fi
fi

CUDA_13_HOME="${CUDA_13_HOME:-/usr/local/cuda}"
# Fall back to the active conda env's CUDA when the default system path has no
# nvcc (e.g. a conda-provided cuda-toolkit install with nvcc under $CONDA_PREFIX).
if [[ ! -x "$CUDA_13_HOME/bin/nvcc" && -x "${CONDA_PREFIX:-}/bin/nvcc" ]]; then
  CUDA_13_HOME="$CONDA_PREFIX"
fi
if [[ -x "$CUDA_13_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$CUDA_13_HOME"
  export CUDA_PATH="$CUDA_13_HOME"
  export CUDA_NVCC_EXECUTABLE="$CUDA_13_HOME/bin/nvcc"
  export MINISGL_RAY_NSYS_BIN="${MINISGL_RAY_NSYS_BIN:-$CUDA_13_HOME/bin/nsys}"
  export PATH="$CUDA_13_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_13_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$CUDA_13_HOME/lib64:${LIBRARY_PATH:-}"
else
  echo "CUDA 13.0 nvcc not found at $CUDA_13_HOME/bin/nvcc" >&2
  exit 1
fi
if [[ "$NSYS" == "1" && ! -x "$MINISGL_RAY_NSYS_BIN" ]]; then
  echo "Nsight Systems nsys not found at $MINISGL_RAY_NSYS_BIN" >&2
  exit 1
fi

pick_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [[ -z "$MINISGL_PORT" ]]; then
  MINISGL_PORT="$(pick_free_port)"
fi
if [[ "$RUN_VLLM_ALIGNMENT" == "1" && -z "$VLLM_PORT" ]]; then
  VLLM_PORT="$(pick_free_port)"
fi

process_is_running() {
  local pid="$1"
  local stat
  stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR==1 {print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

kill_tree() {
  local pid="$1"
  local signal_name="${2:-TERM}"
  local children
  children="$(pgrep -P "$pid" || true)"
  for child in $children; do
    kill_tree "$child" "$signal_name"
  done
  kill "-${signal_name}" "$pid" 2>/dev/null || true
}

stop_process_tree() {
  local pid="$1"
  local timeout_s="${2:-10}"
  local waited=0
  if ! process_is_running "$pid"; then
    return 0
  fi
  kill_tree "$pid" TERM
  while process_is_running "$pid"; do
    if (( waited >= timeout_s * 10 )); then
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  if process_is_running "$pid"; then
    kill_tree "$pid" KILL
  fi
}

if [[ "$CLEAN_FIRST" == "1" ]]; then
  "$SCRIPT_ROOT/kill_afd_ray_tasks.sh" >/dev/null 2>&1 || true
fi

echo "Generating prompt file: $PROMPT_FILE"
if [[ "$PROMPT_KIND" == "realistic" ]]; then
  python "$SCRIPT_ROOT/data_gen/generate_realistic_long_prompts.py" \
    --model "$MODEL_PATH" \
    --target-tokens "$PROMPT_LEN" \
    --count "$NUM_PROMPTS" \
    --seed "$PROMPT_SEED" \
    --output "$PROMPT_FILE"
elif [[ "$PROMPT_KIND" == "existing" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Existing prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  echo "Using existing prompt file: $PROMPT_FILE"
elif [[ "$PROMPT_KIND" == "hello" ]]; then
  python - "$MODEL_PATH" "$PROMPT_LEN" "$NUM_PROMPTS" "$PROMPT_FILE" <<'PY'
from pathlib import Path
import sys

from minisgl.hf_support import load_tokenizer

model, target_raw, count_raw, output_raw = sys.argv[1:5]
target = int(target_raw)
count = int(count_raw)
output = Path(output_raw)
tok = load_tokenizer(model)

unit = " hello"
lo, hi = 1, max(target * 4, 1024)
while len(tok.encode((unit * hi).strip(), add_special_tokens=True)) <= target:
    hi *= 2
while lo < hi:
    mid = (lo + hi + 1) // 2
    n = len(tok.encode((unit * mid).strip(), add_special_tokens=True))
    if n <= target:
        lo = mid
    else:
        hi = mid - 1

prompt = (unit * lo).strip()
prompt_tokens = len(tok.encode(prompt, add_special_tokens=True))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join([prompt] * count) + "\n", encoding="utf-8")
print(f"prompt_tokens_add_special={prompt_tokens}")
print(f"num_prompts={count}")
print(f"chars_per_prompt={len(prompt)}")
PY
else
  echo "Unsupported PROMPT_KIND=$PROMPT_KIND; use realistic, existing, or hello." >&2
  exit 1
fi

echo "Running AFD FP8 long-prefill workload"
echo "  model: $MODEL_PATH"
echo "  run_dir: $RUN_DIR"
echo "  prompt_file: $PROMPT_FILE"
echo "  prompt_kind: $PROMPT_KIND"
echo "  prompt_seed: $PROMPT_SEED"
echo "  prompt_len: $PROMPT_LEN"
echo "  num_prompts: $NUM_PROMPTS"
echo "  max_tokens: $MAX_TOKENS"
echo "  afd_max_new_tokens: $AFD_MAX_NEW_TOKENS"
echo "  warmup_replays: $WARMUP_REPLAYS"
echo "  sample_concurrency: $SAMPLE_CONCURRENCY"
echo "  batch_submit: $BATCH_SUBMIT"
echo "  ignore_eos: $IGNORE_EOS"
echo "  attn_dp/tp: $ATTN_DP_SIZE/$ATTN_TP_SIZE"
echo "  mlp_dp/tp/ep: $MLP_DP_SIZE/$MLP_TP_SIZE/$AFD_MLP_EP_SIZE"
echo "  graph_bs: $AFD_DECODE_GRAPH_BS"
echo "  afd_num_mb: $AFD_NUM_MB"
echo "  afd_async_pending_steps: $AFD_ASYNC_PENDING_STEPS"
echo "  afd_force_multi_mb_graph_overlap: $AFD_FORCE_MULTI_MB_GRAPH_OVERLAP"
echo "  afd_disable_overlap: $AFD_DISABLE_OVERLAP"
echo "  max_batched_tokens: $AFD_MAX_BATCHED_TOKENS"
echo "  max_comm_tokens: ${MINISGL_AFD_MAX_COMM_TOKENS:-<auto>}"
echo "  deepep_cpu_sync_disabled: ${MINISGL_AFD_DISABLE_DEEPEP_CPU_SYNC:-0}"
echo "  max_seq_len: $MINISGL_MAX_SEQ_LEN"
echo "  page_size: $MINISGL_PAGE_SIZE"
echo "  nsys: $NSYS"
echo "  nsys_cuda_graph_trace: $NSYS_CUDA_GRAPH_TRACE"
echo "  nsys_start_step: $NSYS_START_STEP"
echo "  nsys_stop_step: $NSYS_STOP_STEP"
echo "  flashinfer_workspace_base: ${FLASHINFER_WORKSPACE_BASE:-<default>}"
echo "  tvm_ffi_cache_dir: ${TVM_FFI_CACHE_DIR:-<default>}"
echo "  ep_jit_cache_dir: ${EP_JIT_CACHE_DIR:-<default>}"
echo "  gin_build_dir: ${N2M_M2N_GIN_BUILD_DIR:-<default>}"
echo "  deepep_build_dir: ${MINISGL_DEEPEP_BUILD_DIR:-<default>}"
echo "  deepgemm_build_dir: ${MINISGL_DEEPGEMM_BUILD_DIR:-<default>}"
echo "  cuda_home: ${CUDA_HOME:-<unset>}"
echo "  cuda_nvcc: ${CUDA_NVCC_EXECUTABLE:-<unset>}"
echo "  nsys_bin: ${MINISGL_RAY_NSYS_BIN:-<unset>}"
echo "  run_vllm_alignment: $RUN_VLLM_ALIGNMENT"
if [[ "$RUN_VLLM_ALIGNMENT" == "1" ]]; then
  echo "  vllm_gpus: $VLLM_GPUS"
  echo "  vllm_port: $VLLM_PORT"
  echo "  vllm_tp_size: $VLLM_TP_SIZE"
  echo "  vllm_max_model_len: $VLLM_MAX_MODEL_LEN"
  echo "  vllm_extra_args: $VLLM_EXTRA_ARGS"
  echo "  vllm_prompt_logprobs: $VLLM_PROMPT_LOGPROBS"
  echo "  vllm_score_concurrency: $VLLM_SCORE_CONCURRENCY"
  echo "  vllm_use_flashinfer_moe_fp8: ${VLLM_USE_FLASHINFER_MOE_FP8:-<unset>}"
  echo "  vllm_use_deep_gemm: ${VLLM_USE_DEEP_GEMM:-<unset>}"
  echo "  vllm_moe_use_deep_gemm: ${VLLM_MOE_USE_DEEP_GEMM:-<unset>}"
fi
echo "  clean_first: $CLEAN_FIRST"
echo "  clean_after: $CLEAN_AFTER"
echo "  post_shutdown_sleep: $POST_SHUTDOWN_SLEEP"

AFD_PID=""
VLLM_PID=""
cleanup() {
  if [[ -n "${AFD_PID:-}" ]]; then
    stop_process_tree "$AFD_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    stop_process_tree "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_cmd=(
  "$SCRIPT_ROOT/serve/fastafd_server.sh"
  --env "$ENV_NAME"
  --model "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$MINISGL_PORT"
  --launch-backend ray
  --ray-address "$MINISGL_RAY_ADDRESS"
  --ray-log-dir "$RAY_LOG_DIR"
  --afd-report-dir "$RAY_LOG_DIR"
  --afd-attn-dp-size "$ATTN_DP_SIZE"
  --afd-mlp-dp-size "$MLP_DP_SIZE"
  --afd-attn-tp-size "$ATTN_TP_SIZE"
  --afd-mlp-tp-size "$MLP_TP_SIZE"
  --afd-mlp-ep-size "$AFD_MLP_EP_SIZE"
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND"
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND"
  --afd-batch-size "$AFD_BATCH_SIZE"
  --afd-max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS"
  --afd-max-new-tokens "$AFD_MAX_NEW_TOKENS"
  --afd-max-batched-tokens "$AFD_MAX_BATCHED_TOKENS"
  --cache-type "$AFD_CACHE_TYPE"
  --page-size "$MINISGL_PAGE_SIZE"
  --cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS"
  --afd-decode-graph-bs "$AFD_DECODE_GRAPH_BS"
  --afd-num-mb "$AFD_NUM_MB"
  --afd-async-pending-steps "$AFD_ASYNC_PENDING_STEPS"
  --afd-device-comm-num-sms "$AFD_DEVICE_COMM_NUM_SMS"
  --max-seq-len-override "$MINISGL_MAX_SEQ_LEN"
  --extra-args "$AFD_SERVER_EXTRA_ARGS"
)
if [[ "$AFD_FORCE_MULTI_MB_GRAPH_OVERLAP" == "1" ]]; then
  start_cmd+=(--afd-force-multi-mb-graph-overlap)
fi
if [[ "$AFD_DISABLE_OVERLAP" == "1" ]]; then
  start_cmd+=(--afd-disable-overlap)
fi
if [[ "$NSYS" == "1" ]]; then
  export MINISGL_RAY_NSYS_CUDA_GRAPH_TRACE="$NSYS_CUDA_GRAPH_TRACE"
  export MINISGL_RAY_NSYS_START_STEP="$NSYS_START_STEP"
  export MINISGL_RAY_NSYS_STOP_STEP="$NSYS_STOP_STEP"
  start_cmd+=(--ray-nsys --ray-nsys-output-prefix "$NSYS_OUTPUT_PREFIX")
fi

RAY_DEDUP_LOGS=0 "${start_cmd[@]}" >"$AFD_LOG" 2>&1 &
AFD_PID=$!
echo "  afd_pid: $AFD_PID"
echo "  afd_log: $AFD_LOG"

deadline=$((SECONDS + WAIT_READY_TIMEOUT))
until curl -fsS "http://${SERVER_HOST}:${MINISGL_PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$AFD_PID" 2>/dev/null; then
    echo "AFD exited before ready." >&2
    tail -n 200 "$AFD_LOG" >&2 || true
    exit 2
  fi
  if (( SECONDS >= deadline )); then
    echo "AFD failed to become ready." >&2
    tail -n 200 "$AFD_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
echo "AFD ready: http://${SERVER_HOST}:${MINISGL_PORT}"

if (( WARMUP_REPLAYS < 0 || WARMUP_REPLAYS + 1 > AFD_MAX_NEW_TOKENS )); then
  echo "WARMUP_REPLAYS=$WARMUP_REPLAYS must be >= 0 and warmup tokens must fit AFD_MAX_NEW_TOKENS=$AFD_MAX_NEW_TOKENS" >&2
  exit 1
fi
if (( WARMUP_REPLAYS > 0 )); then
  warmup_cmd=(
    python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" sample
    --model "$MODEL_PATH"
    --minisgl-url "http://${SERVER_HOST}:${MINISGL_PORT}"
    --prompt-file "$PROMPT_FILE"
    --prompt-repeat 1
    --max-tokens "$((WARMUP_REPLAYS + 1))"
    --temperature 0
    --top-p 1
    --top-k 1
    --sample-concurrency "$SAMPLE_CONCURRENCY"
    --timeout "$SAMPLE_TIMEOUT"
    --sample-json "$WARMUP_JSON"
  )
  if [[ "$BATCH_SUBMIT" == "1" ]]; then
    warmup_cmd+=(--batch-submit)
  fi
  if [[ "$IGNORE_EOS" == "1" ]]; then
    warmup_cmd+=(--ignore-eos)
  fi
  "${warmup_cmd[@]}" >"$WARMUP_LOG" 2>&1
fi

sample_cmd=(
  python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" sample
  --model "$MODEL_PATH"
  --minisgl-url "http://${SERVER_HOST}:${MINISGL_PORT}"
  --prompt-file "$PROMPT_FILE"
  --prompt-repeat 1
  --max-tokens "$MAX_TOKENS"
  --temperature 0
  --top-p 1
  --top-k 1
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --timeout "$SAMPLE_TIMEOUT"
  --sample-json "$SAMPLE_JSON"
)
if [[ "$BATCH_SUBMIT" == "1" ]]; then
  sample_cmd+=(--batch-submit)
fi
if [[ "$IGNORE_EOS" == "1" ]]; then
  sample_cmd+=(--ignore-eos)
fi
"${sample_cmd[@]}" >"$SAMPLE_LOG" 2>&1

stop_process_tree "$AFD_PID" "$AFD_STOP_TIMEOUT_S" || true
wait "$AFD_PID" >/dev/null 2>&1 || true
AFD_PID=""
sleep "$POST_SHUTDOWN_SLEEP"
# kill_afd_ray_tasks.sh SIGKILLs matching processes of every user on every Ray node;
# on the shared cluster run it only when explicitly asked (CLEAN_AFTER=1).
if [[ "$CLEAN_AFTER" == "1" ]]; then
  "$SCRIPT_ROOT/kill_afd_ray_tasks.sh" >/dev/null 2>&1 || true
fi

if [[ "$RUN_VLLM_ALIGNMENT" == "1" ]]; then
  if [[ "${VLLM_SHARDED_SCORE:-0}" == "1" ]]; then
    echo "Starting sharded vLLM alignment baseline"
    echo "  shard_nodes: ${VLLM_SHARDED_NODE_LIST:-<unset>}"
    echo "  shard_log_dir: $RUN_DIR/vllm_shards"
    python "$SCRIPT_ROOT/experiments/vllm/score_vllm_sharded_alignment.py" \
      --sample-json "$SAMPLE_JSON" \
      --model "$MODEL_PATH" \
      --node-list "$VLLM_SHARDED_NODE_LIST" \
      --port "$VLLM_PORT" \
      --tp-size "$VLLM_TP_SIZE" \
      --extra-args "$VLLM_EXTRA_ARGS --max-model-len $VLLM_MAX_MODEL_LEN" \
      --prompt-logprobs "$VLLM_PROMPT_LOGPROBS" \
      --score-concurrency "$VLLM_SCORE_CONCURRENCY" \
      --timeout "$VLLM_TIMEOUT" \
      --ready-timeout "$VLLM_WAIT_READY_TIMEOUT" \
      --gpus-per-server "${VLLM_SHARDED_GPUS_PER_SERVER:-4}" \
      --conda-sh "${VLLM_CONDA_SH:-${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}}" \
      --env-name "$ENV_NAME" \
      --cuda-home "$CUDA_13_HOME" \
      --output-json "$ALIGNMENT_JSON" \
      --log-dir "$RUN_DIR/vllm_shards" \
      >"$SCORE_LOG" 2>&1
  else
    echo "Starting vLLM alignment baseline"
    echo "  vllm_log: $VLLM_LOG"
    env \
      -u RUN_VLLM_ALIGNMENT \
      -u VLLM_GPUS \
      -u VLLM_PORT \
      -u VLLM_TP_SIZE \
      -u VLLM_MAX_MODEL_LEN \
      -u VLLM_PROMPT_LOGPROBS \
      -u VLLM_SCORE_CONCURRENCY \
      -u VLLM_TIMEOUT \
      -u VLLM_WAIT_READY_TIMEOUT \
      -u VLLM_EXTRA_ARGS \
      "$SCRIPT_ROOT/serve/vllm_server.sh" \
      --env "$ENV_NAME" \
      --model "$MODEL_PATH" \
      --gpus "$VLLM_GPUS" \
      --port "$VLLM_PORT" \
      --host "$SERVER_HOST" \
      --tp-size "$VLLM_TP_SIZE" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --extra-args "$VLLM_EXTRA_ARGS" \
      >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    deadline=$((SECONDS + VLLM_WAIT_READY_TIMEOUT))
    until curl -fsS "http://${SERVER_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
      if ! process_is_running "$VLLM_PID"; then
        echo "vLLM exited before ready." >&2
        tail -n 200 "$VLLM_LOG" >&2 || true
        exit 2
      fi
      if (( SECONDS >= deadline )); then
        echo "vLLM failed to become ready." >&2
        tail -n 200 "$VLLM_LOG" >&2 || true
        exit 1
      fi
      sleep 2
    done
    echo "vLLM ready: http://${SERVER_HOST}:${VLLM_PORT}"

    python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" score \
      --sample-json "$SAMPLE_JSON" \
      --model "$MODEL_PATH" \
      --vllm-url "http://${SERVER_HOST}:${VLLM_PORT}" \
      --prompt-logprobs "$VLLM_PROMPT_LOGPROBS" \
      --score-concurrency "$VLLM_SCORE_CONCURRENCY" \
      --timeout "$VLLM_TIMEOUT" \
      --output-json "$ALIGNMENT_JSON" \
      >"$SCORE_LOG" 2>&1
  fi

  python - "$ALIGNMENT_JSON" "${ALIGN_MIN_TOP1_RATE:-}" "${ALIGN_MIN_TOP10_RATE:-}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_top1_raw = sys.argv[2]
min_top10_raw = sys.argv[3]
report = json.loads(path.read_text(encoding="utf-8"))
summary = report.get("summary", {})
payload = {
    "overall_top1_agreement_rate": summary.get("overall_top1_agreement_rate"),
    "overall_top10_hit_rate": summary.get("overall_top10_hit_rate"),
    "overall_top100_hit_rate": summary.get("overall_top100_hit_rate"),
    "overall_avg_rank_in_vllm": summary.get("overall_avg_rank_in_vllm"),
    "overall_max_rank_in_vllm": summary.get("overall_max_rank_in_vllm"),
}
print("Alignment Summary")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if min_top1_raw:
    min_top1 = float(min_top1_raw)
    if (summary.get("overall_top1_agreement_rate") or 0.0) < min_top1:
        raise SystemExit(
            f"overall_top1_agreement_rate={summary.get('overall_top1_agreement_rate')} < {min_top1}"
        )
if min_top10_raw:
    min_top10 = float(min_top10_raw)
    if (summary.get("overall_top10_hit_rate") or 0.0) < min_top10:
        raise SystemExit(
            f"overall_top10_hit_rate={summary.get('overall_top10_hit_rate')} < {min_top10}"
        )
PY

  if [[ -n "${VLLM_PID:-}" ]]; then
    stop_process_tree "$VLLM_PID" || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
    VLLM_PID=""
  fi
fi

python - "$RUN_DIR" "$NUM_PROMPTS" "$MAX_TOKENS" <<'PY'
from pathlib import Path
import json
import re
import sys

run = Path(sys.argv[1])
expected_prompts = int(sys.argv[2])
expected_tokens = int(sys.argv[3])
logdir = run / "ray_logs"

print(f"RUN_DIR={run}")
sample_path = run / "sample.json"
if sample_path.exists():
    data = json.loads(sample_path.read_text())
    samples = data.get("samples", [])
    token_counts = [len(sample.get("generated_token_ids", [])) for sample in samples]
    print(f"num_samples={len(samples)} expected={expected_prompts}")
    print(f"generated_token_counts={sorted(set(token_counts))} expected={expected_tokens}")
    print(f"total_generated={sum(token_counts)}")
else:
    print("sample_json_missing")

alignment_path = run / "alignment.json"
if alignment_path.exists():
    report = json.loads(alignment_path.read_text())
    print("alignment_summary=" + json.dumps(report.get("summary", {}), ensure_ascii=False))
else:
    print("alignment_json_missing")

coord_paths = [
    path
    for path in (logdir / "afd_coordinator.log", logdir / "afd-coordinator.log")
    if path.exists()
]
if coord_paths:
    txt = "\n".join(path.read_text(errors="ignore") for path in coord_paths)
    printed_lines = set()
    for line in txt.splitlines():
        if (
            "centralized_scheduler ready" in line
            or "ready max_seq_len" in line
            or "decode_graph_capabilities" in line
        ):
            if line in printed_lines:
                continue
            printed_lines.add(line)
            print(line)
    phases = []
    for line in txt.splitlines():
        if "schedule_global_step" in line:
            m = re.search(r"phase=(\w+).*sizes=([^ ]+).*token_offsets=([^\n]+)", line)
            phases.append((m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?"))
            continue
        if "[afd_step]" in line:
            m = re.search(r"\[afd_step\] step=(\d+)(?: dp=(\d+))? phase=(\w+) bs=(\d+)", line)
            if m:
                step, dp, phase, bs = m.groups()
                phases.append((phase, f"step={step},dp={dp if dp is not None else 0},bs={bs}", ""))
    print(f"num_schedule_steps={len(phases)}")
    print("phase_counts=" + str({p: sum(1 for x, _, __ in phases if x == p) for p in sorted({x for x, _, __ in phases})}))
    print(f"first_12_phases={phases[:12]}")
    print(f"last_12_phases={phases[-12:]}")

for label, patterns, legacy_replay, afd_replay in (
    (
        "attention",
        ("attention_rank*.log", "attention_dp*_rank*.log"),
        "attention_decode_graph:done",
        "afd_ag_decode_graph:replay",
    ),
    (
        "mlp",
        ("mlp_dp*_rank*.log",),
        "model_decode_graph:done",
        "afd_eg_decode_graph:replay",
    ),
):
    files = sorted({path for pattern in patterns for path in logdir.glob(pattern)})
    print(f"{label}_files={len(files)}")
    for path in files:
        txt = path.read_text(errors="ignore")
        replay_count = txt.count(legacy_replay) + txt.count(afd_replay)
        graph_done = (
            txt.count("attention_decode_graph:done")
            if label == "attention"
            else txt.count("model_decode_graph:done")
        )
        graph_done += txt.count("afd_decode_graphs:captured")
        print(
            f"{path.name}: target_replay={replay_count} "
            f"all_graph_done={graph_done} warmups={txt.count('warmup:done')}"
        )

print("sample_log_tail:")
sample_log = run / "sample.log"
print(sample_log.read_text(errors="ignore")[-4000:] if sample_log.exists() else "")
PY

echo "Done."

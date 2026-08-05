#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_afd_online_alignment_report.sh --model PATH --prompt-file FILE [options]

This runs a sequential online AFD-vs-vLLM alignment pipeline:
1. Start the online AFD server.
2. Sample deterministic outputs through the OpenAI-compatible API.
3. Stop AFD.
4. Start vLLM and score sampled tokens with prompt_logprobs.
5. Fail if the configured alignment thresholds are not met.

Options:
  --env NAME                  Expected active conda env. Default: minisgl-cuda130
  --vllm-env NAME             Conda env used only for vLLM. Defaults to --env.
  --model PATH                Model path or HF repo. Required.
  --vllm-model PATH           Optional vLLM-only model path. Defaults to --model.
  --prompt-file FILE          UTF-8 prompt file. Required.
  --prompt-repeat N           Repeat prompt set in memory. Default: 1
  --batch-size N              AFD online batch size. Default: 8
  --max-new-tokens N          Sampling max new tokens. Default: 1
  --cuda-graph-max-bs N       AFD decode graph cap. Default: batch size
  --decode-graph-bs CSV       Optional static AFD decode graph buckets
  --afd-num-mb N              AFD microbatches per step. Default: 1
  --max-batched-tokens N      Static eager prefill token budget / comm buffer sizing. Default: 8192
  --cache-type NAME           KV cache manager strategy for AFD. Default: naive
  --gpus CSV                  CUDA_VISIBLE_DEVICES for vLLM. Default: 0,1,2,3
  --afd-attn-dp-size N        AFD attention-side DP size. Default: 1
  --afd-mlp-dp-size N         AFD model/mlp-side DP size. Default: 1
  --attn-tp-size N            AFD attention-side TP size. Default: 4
  --mlp-tp-size N             AFD model/mlp-side TP size. Default: 4
  --afd-mlp-ep-size N         AFD model/mlp-side EP size. Default: 1
  --vllm-tp-size N            vLLM TP size. Default: number of GPUs in --gpus
  --ray-address ADDR          Ray address for AFD phase. Default: auto
  --ray-nsys                 Enable Nsight Systems profiling for AFD Ray workers
  --ray-nsys-output-prefix PATH
                              Optional Nsight Systems output prefix. Defaults to
                              <artifacts>/ray_logs/minisgl when --ray-nsys is set.
  --host HOST                 AFD/vLLM host. Default: 127.0.0.1
  --afd-port PORT             Default: 19193
  --vllm-port PORT            Default: 18000
  --prompt-logprobs N         Default: 5
  --sample-concurrency N      Concurrent AFD sampling requests. Default: 32
  --batch-submit              Submit all prompts to AFD in one HTTP batch. Default: on
  --no-batch-submit           Use one HTTP request per prompt.
  --ignore-eos                Ignore EOS during AFD sampling.
  --afd-max-running-requests N
                              Static maximum live AFD requests. Default: auto
  --page-size N               AFD page size. Default: 1
  --max-seq-len-override N    Optional
  --vllm-max-model-len N      Optional
  --output-json FILE          Optional final report JSON path
  --keep-artifacts            Keep temp artifacts and print paths
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

pick_free_port() {
  local width="${1:-1}"
  python - "$width" <<'PY'
import random
import socket
import sys

width = int(sys.argv[1])
for _ in range(2000):
    base = random.randrange(20000, 60000 - width)
    sockets = []
    try:
        for port in range(base, base + width):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        print(base)
        break
    except OSError:
        for sock in sockets:
            sock.close()
else:
    raise RuntimeError(f"failed to find free port block width={width}")
PY
}

process_is_running() {
  local pid="$1"
  local stat
  stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR==1 {print $1}')"
  [[ -n "$stat" && "$stat" != Z* ]]
}

try_http_ready() {
  local url="$1"
  python - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as response:
    response.read()
PY
}

wait_for_ready() {
  local url="$1"
  local name="$2"
  local log_file="$3"
  local pid="$4"
  local timeout_s="${5:-900}"
  local deadline=$((SECONDS + timeout_s))
  local last_error=""

  while (( SECONDS < deadline )); do
    if ! process_is_running "$pid"; then
      return 2
    fi
    if last_error="$(try_http_ready "$url" 2>&1)"; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 2
  done

  echo "$name failed to become ready: $url" >&2
  echo "See log: $log_file" >&2
  if [[ -n "$last_error" ]]; then
    echo "Last error: $last_error" >&2
  fi
  return 1
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
    for _ in $(seq 1 50); do
      if ! process_is_running "$pid"; then
        break
      fi
      sleep 0.1
    done
  fi
}

print_log_tail() {
  local log_file="$1"
  local lines="${2:-120}"
  if [[ -f "$log_file" ]]; then
    echo "--- tail -n $lines $log_file ---" >&2
    tail -n "$lines" "$log_file" >&2 || true
    echo "--- end log tail ---" >&2
  fi
}

fail_with_logs() {
  local message="$1"
  local exit_code="$2"
  shift 2

  echo "$message" >&2
  for log_file in "$@"; do
    echo "See log: $log_file" >&2
    print_log_tail "$log_file"
  done
  exit "$exit_code"
}

ensure_active_env() {
  local expected="$1"
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Expected active conda env '$expected', but no conda env is activated." >&2
    exit 1
  fi
  if [[ "$CONDA_DEFAULT_ENV" != "$expected" ]]; then
    echo "Expected active conda env '$expected', got '$CONDA_DEFAULT_ENV'." >&2
    exit 1
  fi
}

log_step() {
  local message="$1"
  printf '\n==> %s\n' "$message"
}

ENV_NAME="minisgl-cuda130"
VLLM_ENV_NAME=""
MODEL=""
VLLM_MODEL=""
PROMPT_FILE=""
PROMPT_REPEAT="1"
AFD_BATCH_SIZE="8"
MAX_NEW_TOKENS="1"
AFD_CUDA_GRAPH_MAX_BS=""
AFD_DECODE_GRAPH_BS=""
AFD_NUM_MB="1"
AFD_MAX_BATCHED_TOKENS="8192"
AFD_MAX_RUNNING_REQUESTS="0"
CACHE_TYPE="naive"
GPUS="0,1,2,3"
ATTN_DP_SIZE="1"
MLP_DP_SIZE="1"
ATTN_TP_SIZE="4"
MLP_TP_SIZE="4"
AFD_MLP_EP_SIZE="1"
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-none}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-auto}"
VLLM_TP_SIZE=""
RAY_ADDRESS="auto"
RAY_NSYS=0
RAY_NSYS_OUTPUT_PREFIX=""
SERVER_HOST="127.0.0.1"
AFD_PORT=""
VLLM_PORT=""
PROMPT_LOGPROBS="5"
SAMPLE_CONCURRENCY="32"
BATCH_SUBMIT=1
IGNORE_EOS=0
MAX_SEQ_LEN_OVERRIDE=""
VLLM_MAX_MODEL_LEN=""
OUTPUT_JSON=""
KEEP_ARTIFACTS=0
AFD_DEVICE_COMM_NUM_SMS="1"
PAGE_SIZE="1"
VLLM_EXTRA_ARGS_USER_SET=1
if [[ ! -v VLLM_EXTRA_ARGS ]]; then
  VLLM_EXTRA_ARGS=""
  VLLM_EXTRA_ARGS_USER_SET=0
fi
AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:-}"
AFD_PID=""
VLLM_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --vllm-env) VLLM_ENV_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --vllm-model) VLLM_MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --prompt-repeat) PROMPT_REPEAT="$2"; shift 2 ;;
    --batch-size) AFD_BATCH_SIZE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --cuda-graph-max-bs) AFD_CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --decode-graph-bs) AFD_DECODE_GRAPH_BS="$2"; shift 2 ;;
    --afd-num-mb) AFD_NUM_MB="$2"; shift 2 ;;
    --max-batched-tokens) AFD_MAX_BATCHED_TOKENS="$2"; shift 2 ;;
    --afd-max-running-requests) AFD_MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --cache-type) CACHE_TYPE="$2"; shift 2 ;;
    --page-size) PAGE_SIZE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --afd-attn-dp-size) ATTN_DP_SIZE="$2"; shift 2 ;;
    --afd-mlp-dp-size) MLP_DP_SIZE="$2"; shift 2 ;;
    --attn-tp-size) ATTN_TP_SIZE="$2"; shift 2 ;;
    --mlp-tp-size) MLP_TP_SIZE="$2"; shift 2 ;;
    --afd-mlp-ep-size) AFD_MLP_EP_SIZE="$2"; shift 2 ;;
    --afd-moe-a2a-backend) AFD_MOE_A2A_BACKEND="$2"; shift 2 ;;
    --afd-moe-runner-backend) AFD_MOE_RUNNER_BACKEND="$2"; shift 2 ;;
    --vllm-tp-size) VLLM_TP_SIZE="$2"; shift 2 ;;
    --ray-address) RAY_ADDRESS="$2"; shift 2 ;;
    --ray-nsys) RAY_NSYS=1; shift ;;
    --ray-nsys-output-prefix) RAY_NSYS_OUTPUT_PREFIX="$2"; shift 2 ;;
    --host) SERVER_HOST="$2"; shift 2 ;;
    --afd-port) AFD_PORT="$2"; shift 2 ;;
    --vllm-port) VLLM_PORT="$2"; shift 2 ;;
    --prompt-logprobs) PROMPT_LOGPROBS="$2"; shift 2 ;;
    --sample-concurrency) SAMPLE_CONCURRENCY="$2"; shift 2 ;;
    --batch-submit) BATCH_SUBMIT=1; shift ;;
    --no-batch-submit) BATCH_SUBMIT=0; shift ;;
    --ignore-eos) IGNORE_EOS=1; shift ;;
    --max-seq-len-override) MAX_SEQ_LEN_OVERRIDE="$2"; shift 2 ;;
    --vllm-max-model-len) VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --keep-artifacts) KEEP_ARTIFACTS=1; shift ;;
    --afd-device-comm-num-sms) AFD_DEVICE_COMM_NUM_SMS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$AFD_CUDA_GRAPH_MAX_BS" ]]; then
  AFD_CUDA_GRAPH_MAX_BS="$AFD_BATCH_SIZE"
fi
if (( AFD_NUM_MB < 1 )); then
  echo "--afd-num-mb must be >= 1, got: $AFD_NUM_MB" >&2
  exit 1
fi

if [[ -z "$MODEL" || -z "$PROMPT_FILE" ]]; then
  echo "--model and --prompt-file are required" >&2
  usage >&2
  exit 1
fi
if [[ -z "$VLLM_MODEL" ]]; then
  VLLM_MODEL="$MODEL"
fi
if [[ -z "$VLLM_ENV_NAME" ]]; then
  VLLM_ENV_NAME="$ENV_NAME"
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

# Size the AFD running table to the whole workload so batch-submit never
# deadlocks: AFD prefill is globally prioritized, so if submitted prompts
# exceed max_running_req the unadmitted prompts keep the scheduler in "prefill"
# forever and decode never runs. When --afd-max-running-requests is left at its
# 0 default, derive it = num_prompts (non-blank prompt lines x --prompt-repeat,
# matching load_prompts()). Pass an explicit --afd-max-running-requests to pin
# a different table size.
if [[ "$AFD_MAX_RUNNING_REQUESTS" -le 0 ]]; then
  _prompt_count="$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")"
  AFD_MAX_RUNNING_REQUESTS="$(( _prompt_count * PROMPT_REPEAT ))"
fi

if [[ "$VLLM_EXTRA_ARGS_USER_SET" -eq 0 ]]; then
  VLLM_EXTRA_ARGS="--enforce-eager"
  if [[ "$MODEL" == *FP8* || "$MODEL" == *fp8* ]]; then
    # vLLM 0.19 may auto-select FLASHINFER_TRTLLM for Qwen3-MoE FP8 and
    # hang during the startup profile run on this environment. TRITON is
    # slower but deterministic for the alignment baseline.
    VLLM_EXTRA_ARGS+=" --moe-backend triton"
  fi
fi
if [[ "$MODEL" == *FP8* || "$MODEL" == *fp8* ]]; then
  export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
  export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
fi

if [[ -z "$VLLM_TP_SIZE" ]]; then
  VLLM_TP_SIZE="$(count_csv_items "$GPUS")"
fi

if [[ -z "$AFD_PORT" ]]; then
  # AFD uses server_port plus several adjacent ZMQ ports.
  AFD_PORT="$(pick_free_port 8)"
fi
if [[ -z "$VLLM_PORT" ]]; then
  VLLM_PORT="$(pick_free_port)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"
REPORT_DIR="$CALIB_DIR/reports"
mkdir -p "$REPORT_DIR"
export MINISGL_LOCAL_CACHE_BASE="${MINISGL_LOCAL_CACHE_BASE:-$CALIB_DIR/cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$MINISGL_LOCAL_CACHE_BASE/flashinfer}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/tvm_ffi}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_jit}"
export N2M_M2N_GIN_BUILD_DIR="${N2M_M2N_GIN_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/gin_comm}"
export MINISGL_DEEPEP_BUILD_DIR="${MINISGL_DEEPEP_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_moe}"
export MINISGL_DEEPGEMM_BUILD_DIR="${MINISGL_DEEPGEMM_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepgemm}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
TMP_DIR="${TMP_DIR:-$REPORT_DIR/afd_online_alignment_run_${TIMESTAMP}}"
mkdir -p "$TMP_DIR"
SAMPLE_JSON="$TMP_DIR/sample.json"
SAMPLE_LOG="$TMP_DIR/sample.log"
AFD_LOG="$TMP_DIR/afd.log"
VLLM_LOG="$TMP_DIR/vllm.log"
REPORT_JSON="${OUTPUT_JSON:-$TMP_DIR/alignment.json}"
AFD_RAY_LOG_DIR="$TMP_DIR/ray_logs"
mkdir -p "$AFD_RAY_LOG_DIR"
if [[ "$RAY_NSYS" -eq 1 && -z "$RAY_NSYS_OUTPUT_PREFIX" ]]; then
  RAY_NSYS_OUTPUT_PREFIX="$AFD_RAY_LOG_DIR/minisgl"
fi

cleanup() {
  set +e
  if [[ -n "$AFD_PID" ]] && process_is_running "$AFD_PID"; then
    stop_process_tree "$AFD_PID"
    wait "$AFD_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$VLLM_PID" ]] && process_is_running "$VLLM_PID"; then
    stop_process_tree "$VLLM_PID"
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$KEEP_ARTIFACTS" -eq 0 ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

ensure_active_env "$ENV_NAME"
cd "$CALIB_DIR"

log_step "Starting online AFD server"
echo "  model: $MODEL"
echo "  vllm_model: $VLLM_MODEL"
echo "  vllm_env: $VLLM_ENV_NAME"
echo "  prompt_file: $PROMPT_FILE"
echo "  prompt_repeat: $PROMPT_REPEAT"
echo "  afd_batch_size: $AFD_BATCH_SIZE"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  afd_cuda_graph_max_bs: $AFD_CUDA_GRAPH_MAX_BS"
echo "  afd_decode_graph_bs: ${AFD_DECODE_GRAPH_BS:-<auto>}"
echo "  afd_num_mb: $AFD_NUM_MB"
echo "  afd_max_batched_tokens: $AFD_MAX_BATCHED_TOKENS"
echo "  afd_max_running_requests: ${AFD_MAX_RUNNING_REQUESTS:-0}"
echo "  afd_mlp_ep_size: $AFD_MLP_EP_SIZE"
echo "  afd_moe_a2a_backend: $AFD_MOE_A2A_BACKEND"
echo "  afd_moe_runner_backend: $AFD_MOE_RUNNER_BACKEND"
echo "  vllm_extra_args: ${VLLM_EXTRA_ARGS:-<none>}"
echo "  vllm_use_flashinfer_moe_fp8: ${VLLM_USE_FLASHINFER_MOE_FP8:-<unset>}"
echo "  vllm_use_deep_gemm: ${VLLM_USE_DEEP_GEMM:-<unset>}"
echo "  vllm_moe_use_deep_gemm: ${VLLM_MOE_USE_DEEP_GEMM:-<unset>}"
echo "  cache_type: $CACHE_TYPE"
echo "  page_size: $PAGE_SIZE"
echo "  afd_device_comm_num_sms: $AFD_DEVICE_COMM_NUM_SMS"
echo "  afd_attn_dp_size: $ATTN_DP_SIZE"
echo "  afd_mlp_dp_size: $MLP_DP_SIZE"
echo "  cache_root: $MINISGL_LOCAL_CACHE_BASE"
echo "  afd_log: $AFD_LOG"
echo "  ray_log_dir: $AFD_RAY_LOG_DIR"
echo "  batch_submit: $BATCH_SUBMIT"
echo "  ignore_eos: $IGNORE_EOS"
if [[ "$RAY_NSYS" -eq 1 ]]; then
  echo "  ray_nsys: 1"
  echo "  ray_nsys_output_prefix: $RAY_NSYS_OUTPUT_PREFIX"
fi

afd_start_cmd=(
  "$SCRIPT_ROOT/serve/fastafd_server.sh"
  --env "$ENV_NAME"
  --model "$MODEL"
  --host "$SERVER_HOST"
  --port "$AFD_PORT"
  --launch-backend ray
  --ray-address "$RAY_ADDRESS"
  --ray-log-dir "$AFD_RAY_LOG_DIR"
  --afd-report-dir "$AFD_RAY_LOG_DIR"
)
if [[ "$RAY_NSYS" -eq 1 ]]; then
  if [[ -z "${MINISGL_RAY_NSYS_BIN:-}" ]]; then
    MINISGL_RAY_NSYS_BIN="$(command -v nsys || true)"
    export MINISGL_RAY_NSYS_BIN
  fi
  if [[ -z "${MINISGL_RAY_NSYS_BIN:-}" ]]; then
    echo "Failed to resolve nsys; set MINISGL_RAY_NSYS_BIN explicitly." >&2
    exit 1
  fi
  afd_start_cmd+=(
    --ray-nsys
    --ray-nsys-output-prefix "$RAY_NSYS_OUTPUT_PREFIX"
  )
fi
afd_start_cmd+=(
  --afd-attn-dp-size "$ATTN_DP_SIZE"
  --afd-mlp-dp-size "$MLP_DP_SIZE"
  --afd-attn-tp-size "$ATTN_TP_SIZE"
  --afd-mlp-tp-size "$MLP_TP_SIZE"
  --afd-mlp-ep-size "$AFD_MLP_EP_SIZE"
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND"
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND"
  --afd-batch-size "$AFD_BATCH_SIZE"
  --afd-max-batched-tokens "$AFD_MAX_BATCHED_TOKENS"
  --afd-max-running-requests "$AFD_MAX_RUNNING_REQUESTS"
  --cache-type "$CACHE_TYPE"
  --page-size "$PAGE_SIZE"
  --cuda-graph-max-bs "$AFD_CUDA_GRAPH_MAX_BS"
  --afd-num-mb "$AFD_NUM_MB"
  --afd-device-comm-num-sms "$AFD_DEVICE_COMM_NUM_SMS"
)
if [[ -n "$AFD_DECODE_GRAPH_BS" ]]; then
  afd_start_cmd+=(--afd-decode-graph-bs "$AFD_DECODE_GRAPH_BS")
fi
if [[ -n "$MAX_SEQ_LEN_OVERRIDE" ]]; then
  afd_start_cmd+=(--max-seq-len-override "$MAX_SEQ_LEN_OVERRIDE")
fi
if [[ -n "$AFD_SERVER_EXTRA_ARGS" ]]; then
  afd_start_cmd+=(--extra-args "$AFD_SERVER_EXTRA_ARGS")
fi

"${afd_start_cmd[@]}" >"$AFD_LOG" 2>&1 &
AFD_PID=$!

if ! wait_for_ready \
  "http://${SERVER_HOST}:${AFD_PORT}/v1/models" \
  "AFD" \
  "$AFD_LOG" \
  "$AFD_PID" \
  "${AFD_READY_TIMEOUT:-900}"; then
  status="$?"
  if [[ "$status" == "2" ]]; then
    fail_with_logs "AFD exited before becoming ready" 1 "$AFD_LOG"
  fi
  fail_with_logs "AFD failed to become ready" 1 "$AFD_LOG"
fi

log_step "Sampling through online AFD"
sample_cmd=(
  python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" sample
  --model "$MODEL"
  --minisgl-url "http://${SERVER_HOST}:${AFD_PORT}"
  --prompt-file "$PROMPT_FILE"
  --prompt-repeat "$PROMPT_REPEAT"
  --max-tokens "$MAX_NEW_TOKENS"
  --temperature 0
  --top-p 1
  --top-k 1
  --sample-concurrency "$SAMPLE_CONCURRENCY"
  --timeout "${AFD_SAMPLE_TIMEOUT:-600}"
  --sample-json "$SAMPLE_JSON"
)
if [[ "$BATCH_SUBMIT" -eq 1 ]]; then
  sample_cmd+=(--batch-submit)
fi
if [[ "$IGNORE_EOS" -eq 1 ]]; then
  sample_cmd+=(--ignore-eos)
fi
if ! "${sample_cmd[@]}" >"$SAMPLE_LOG" 2>&1; then
  fail_with_logs "Online AFD sampling failed" 1 "$SAMPLE_LOG" "$AFD_LOG"
fi

if [[ ! -f "$SAMPLE_JSON" ]]; then
  fail_with_logs "Expected sample bundle was not written: $SAMPLE_JSON" 1 "$SAMPLE_LOG" "$AFD_LOG"
fi

stop_process_tree "$AFD_PID"
wait "$AFD_PID" >/dev/null 2>&1 || true
AFD_PID=""

log_step "Starting vLLM"
vllm_start_cmd=(
  "$SCRIPT_ROOT/serve/vllm_server.sh"
  --env "$VLLM_ENV_NAME"
  --model "$VLLM_MODEL"
  --gpus "$GPUS"
  --port "$VLLM_PORT"
  --host "$SERVER_HOST"
  --tp-size "$VLLM_TP_SIZE"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --extra-args "$VLLM_EXTRA_ARGS"
)
if [[ "$VLLM_ENV_NAME" != "$ENV_NAME" ]]; then
  vllm_start_cmd=(
    conda run --no-capture-output -n "$VLLM_ENV_NAME"
    "${vllm_start_cmd[@]}"
  )
fi
"${vllm_start_cmd[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

if ! wait_for_ready \
  "http://${SERVER_HOST}:${VLLM_PORT}/v1/models" \
  "vLLM" \
  "$VLLM_LOG" \
  "$VLLM_PID" \
  "${VLLM_READY_TIMEOUT:-900}"; then
  status="$?"
  if [[ "$status" == "2" ]]; then
    fail_with_logs "vLLM exited before becoming ready" 1 "$VLLM_LOG"
  fi
  fail_with_logs "vLLM failed to become ready" 1 "$VLLM_LOG"
fi

log_step "Scoring under vLLM"
if ! python "$SCRIPT_ROOT/validate/compare_minisgl_vllm.py" score \
  --sample-json "$SAMPLE_JSON" \
  --vllm-url "http://${SERVER_HOST}:${VLLM_PORT}" \
  --prompt-logprobs "$PROMPT_LOGPROBS" \
  --timeout "${VLLM_SCORE_TIMEOUT:-600}" \
  --output-json "$REPORT_JSON" \
  >"$TMP_DIR/score.log" 2>&1; then
  fail_with_logs "vLLM scoring failed" 1 "$TMP_DIR/score.log" "$VLLM_LOG"
fi

python - <<'PY' "$REPORT_JSON" "${ALIGN_MIN_TOP1_RATE:-}" "${ALIGN_MIN_TOP10_RATE:-1.0}"
import json
import sys

path, min_top1_raw, min_top10_raw = sys.argv[1:4]
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
summary_payload = report.get("summary", {})

summary = {
    "overall_top1_agreement_rate": summary_payload.get("overall_top1_agreement_rate"),
    "overall_top10_hit_rate": summary_payload.get("overall_top10_hit_rate"),
    "overall_top100_hit_rate": summary_payload.get("overall_top100_hit_rate"),
    "overall_avg_rank_in_vllm": summary_payload.get("overall_avg_rank_in_vllm"),
    "overall_max_rank_in_vllm": summary_payload.get("overall_max_rank_in_vllm"),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))

if min_top1_raw:
    min_top1 = float(min_top1_raw)
    if (summary_payload.get("overall_top1_agreement_rate") or 0.0) < min_top1:
        raise SystemExit(
            f"overall_top1_agreement_rate={summary_payload.get('overall_top1_agreement_rate')} < {min_top1}"
        )

min_top10 = float(min_top10_raw)
if (summary_payload.get("overall_top10_hit_rate") or 0.0) < min_top10:
    raise SystemExit(
        f"overall_top10_hit_rate={summary_payload.get('overall_top10_hit_rate')} < {min_top10}"
    )
PY

log_step "Completed"
echo "sample_json: $SAMPLE_JSON"
echo "report_json: $REPORT_JSON"
echo "afd_log: $AFD_LOG"
echo "ray_log_dir: $AFD_RAY_LOG_DIR"
echo "vllm_log: $VLLM_LOG"

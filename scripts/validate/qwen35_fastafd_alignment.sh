#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  MODEL_REPO=... MODEL_REVISION=... MODEL_DISPLAY_NAME=... \
    qwen35_fastafd_alignment.sh

Shared driver for the Qwen3.5 two-node FastAFD/vLLM alignment. It does not name
a checkpoint; a preset supplies the model identity and any model-specific
overrides, then execs this script. See:
  qwen35_122b_a10b_fp8_fastafd_alignment.sh
  qwen35_397b_a17b_fp8_fastafd_alignment.sh

Configuration is supplied through the environment. The principal controls are
MAX_TOKENS, AFD_BATCH_SIZE, AFD_NUM_MB, MINISGL_CUDA_GRAPH_MAX_BS,
AFD_DECODE_GRAPH_BS, AFD_BATCH_SUBMIT, and AFD_SERVER_EXTRA_ARGS.

The default is the eager graph-off mb1 correctness baseline. Set
AFD_SERVER_EXTRA_ARGS to an explicit empty string to enable both AFD decode
graphs once graph-off correctness passes.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"

# Model identity has no default on purpose: a wrong default would silently
# download and align the wrong checkpoint, and the pinned revision is what makes
# an alignment result reproducible.
for _required in MODEL_REPO MODEL_REVISION MODEL_DISPLAY_NAME; do
  if [[ -z "${!_required:-}" ]]; then
    echo "$_required is required. Run a preset such as" >&2
    echo "  scripts/validate/qwen35_122b_a10b_fp8_fastafd_alignment.sh" >&2
    echo "or export MODEL_REPO, MODEL_REVISION and MODEL_DISPLAY_NAME yourself." >&2
    exit 1
  fi
done
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$(python - "$MODEL_REPO" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
fi

# Both engines read the same snapshot by default. A preset overrides this when a
# checkpoint's quantization metadata asks vLLM for something FastAFD does not
# want, in which case it points the reference at its own view of the snapshot.
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$MODEL_PATH}"

REPORT_DIR="$CALIB_DIR/reports"
mkdir -p "$REPORT_DIR"
PROMPT_FILE="${PROMPT_FILE:-$CALIB_DIR/prompts/qwen35_122b_alignment.txt}"
ENV_NAME="${ENV_NAME:-minisgl-cuda130-m2n}"
VLLM_ENV_NAME="${VLLM_ENV_NAME:-vllm-cuda130}"
GPUS="${GPUS:-0,1,2,3}"
MINISGL_TP_SIZE="${MINISGL_TP_SIZE:-8}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-4}"
MLP_TP_SIZE="${MLP_TP_SIZE:-4}"
ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
MLP_DP_SIZE="${MLP_DP_SIZE:-1}"
AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-none}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MINISGL_PORT="${MINISGL_PORT:-}"
VLLM_PORT="${VLLM_PORT:-}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-10}"
PROMPT_REPEAT="${PROMPT_REPEAT:-1}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-1}"
AFD_BATCH_SUBMIT="${AFD_BATCH_SUBMIT:-1}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-1}"
AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-}"
AFD_NUM_MB="${AFD_NUM_MB:-1}"
MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-2048}"
MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-1}"
MINISGL_RAY_ADDRESS="${MINISGL_RAY_ADDRESS:-auto}"
AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-1}"
AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-2048}"
AFD_CACHE_TYPE="${AFD_CACHE_TYPE:-naive}"
AFD_DEVICE_COMM_NUM_SMS="${AFD_DEVICE_COMM_NUM_SMS:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
# The reference runs eager and unfused so that a mismatch points at a kernel
# rather than at a fusion pass, and `--language-model-only` keeps the comparison
# to the weights FastAFD actually loads.
if [[ ! -v VLLM_EXTRA_ARGS ]]; then
  VLLM_EXTRA_ARGS='--enforce-eager --language-model-only --moe-backend triton --compilation-config {"pass_config":{"fuse_allreduce_rms":false}}'
fi
if [[ ! -v AFD_SERVER_EXTRA_ARGS ]]; then
  AFD_SERVER_EXTRA_ARGS='--afd-disable-attention-decode-graph --afd-disable-model-decode-graph'
fi
AFD_READY_TIMEOUT="${AFD_READY_TIMEOUT:-1800}"
AFD_SAMPLE_TIMEOUT="${AFD_SAMPLE_TIMEOUT:-600}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
VLLM_SCORE_TIMEOUT="${VLLM_SCORE_TIMEOUT:-600}"
export VLLM_EXTRA_ARGS AFD_SERVER_EXTRA_ARGS AFD_READY_TIMEOUT AFD_SAMPLE_TIMEOUT
export VLLM_READY_TIMEOUT VLLM_SCORE_TIMEOUT

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi
if [[ -z "$MINISGL_MAX_RUNNING_REQUESTS" ]]; then
  _prompt_count="$(grep -cve '^[[:space:]]*$' "$PROMPT_FILE")"
  MINISGL_MAX_RUNNING_REQUESTS="$(( _prompt_count * PROMPT_REPEAT ))"
fi
case "$AFD_BATCH_SUBMIT" in
  0) BATCH_SUBMIT_ARG="--no-batch-submit" ;;
  1) BATCH_SUBMIT_ARG="--batch-submit" ;;
  *) echo "AFD_BATCH_SUBMIT must be 0 or 1, got: $AFD_BATCH_SUBMIT" >&2; exit 1 ;;
esac

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen35_afd_serve_alignment_tp8_ray_${TIMESTAMP}.json}"

echo "Running ${MODEL_DISPLAY_NAME} mb${AFD_NUM_MB} AFD alignment"
echo "  model_repo: $MODEL_REPO"
echo "  model_revision: $MODEL_REVISION"
echo "  model_path: $MODEL_PATH"
echo "  vllm_model_path: $VLLM_MODEL_PATH"
echo "  prompts: $PROMPT_FILE"
echo "  max_tokens: $MAX_TOKENS"
echo "  afd_topology: AG-TP${ATTN_TP_SIZE} + EG-TP/EP${MLP_TP_SIZE}"
echo "  afd_moe_runner_backend: $AFD_MOE_RUNNER_BACKEND"
echo "  afd_cuda_graph_max_bs: $MINISGL_CUDA_GRAPH_MAX_BS"
echo "  afd_decode_graph_bs: ${AFD_DECODE_GRAPH_BS:-<auto>}"
echo "  afd_num_mb: $AFD_NUM_MB"
echo "  afd_batch_submit: $AFD_BATCH_SUBMIT"
echo "  sample_concurrency: $SAMPLE_CONCURRENCY"
echo "  afd_server_extra_args: ${AFD_SERVER_EXTRA_ARGS:-<none>}"
echo "  afd_ready_timeout: $AFD_READY_TIMEOUT"
echo "  vllm_tp_size: $VLLM_TP_SIZE"
echo "  vllm_env: $VLLM_ENV_NAME"
echo "  vllm_ready_timeout: $VLLM_READY_TIMEOUT"
echo "  output_json: $OUTPUT_JSON"

exec "$SCRIPT_ROOT/validate/fastafd_vllm_alignment.sh" \
  --env "$ENV_NAME" \
  --vllm-env "$VLLM_ENV_NAME" \
  --model "$MODEL_PATH" \
  --vllm-model "$VLLM_MODEL_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-repeat "$PROMPT_REPEAT" \
  --batch-size "$AFD_BATCH_SIZE" \
  --max-new-tokens "$MAX_TOKENS" \
  --cuda-graph-max-bs "$MINISGL_CUDA_GRAPH_MAX_BS" \
  --decode-graph-bs "$AFD_DECODE_GRAPH_BS" \
  --afd-num-mb "$AFD_NUM_MB" \
  --max-batched-tokens "$AFD_MAX_BATCHED_TOKENS" \
  --cache-type "$AFD_CACHE_TYPE" \
  --afd-max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS" \
  --gpus "$GPUS" \
  --afd-attn-dp-size "$ATTN_DP_SIZE" \
  --afd-mlp-dp-size "$MLP_DP_SIZE" \
  --attn-tp-size "$ATTN_TP_SIZE" \
  --mlp-tp-size "$MLP_TP_SIZE" \
  --afd-mlp-ep-size "$AFD_MLP_EP_SIZE" \
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND" \
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND" \
  --vllm-tp-size "$VLLM_TP_SIZE" \
  --ray-address "$MINISGL_RAY_ADDRESS" \
  --host "$SERVER_HOST" \
  --afd-port "$MINISGL_PORT" \
  --vllm-port "$VLLM_PORT" \
  --prompt-logprobs "$PROMPT_LOGPROBS" \
  --sample-concurrency "$SAMPLE_CONCURRENCY" \
  "$BATCH_SUBMIT_ARG" \
  --max-seq-len-override "$MINISGL_MAX_SEQ_LEN" \
  --page-size "$MINISGL_PAGE_SIZE" \
  --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
  --output-json "$OUTPUT_JSON" \
  --afd-device-comm-num-sms "$AFD_DEVICE_COMM_NUM_SMS" \
  --keep-artifacts

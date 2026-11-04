#!/usr/bin/env bash
# Shared Qwen3.5 FP8 M:1 AFD throughput driver: M attention nodes (AG DP4/TP1 each) + one FFN node (EG DP4/TP1/EP4),
# MegaMoE M:N transport. The node policy is the Qwen3-235B dynamic-node preset's:
# AFD_NODE_LIST names every node (IPs), the last one (or AFD_MLP_NODE) is the
# FFN node, every other node runs 4 attention DP ranks; PER_ATTN_GPU_BSZ (when AFD_ACTIVE_GLOBAL is unset) sets
# the global batch as PER_ATTN_GPU_BSZ x attention GPUs. A preset supplies MODEL_PATH, the batch and the
# microbatch count, then execs this script.
#
# Four properties this wrapper enforces rather than assumes:
#   1. tensor parallelism is forbidden on both roles, so a retained decode window
#      contains zero TP all-reduce kernels;
#   2. AFD_NUM_MB must be explicit; MB1 and MB2 both keep overlap enabled,
#      and microbatch count must not silently choose a serialized baseline;
#   3. the workload is the pinned prompt pool, deterministically cycled by request
#      index, so AFD and a vLLM reference decode byte-identical prompts;
#   4. batches are named globally (AFD_ACTIVE_GLOBAL) and the per-rank and
#      per-lane shapes are derived and echoed rather than implied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"

RUNNER="$SCRIPT_ROOT/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must name a pinned Qwen3.5 FP8 snapshot." >&2
  exit 1
fi

export ENV_NAME="${ENV_NAME:-minisgl-cuda130-m2n}"
export PROMPT_LEN="${PROMPT_LEN:-8192}"
# A longer decode plateau makes the retained window robust to prefill-duration
# estimation error; decode steps are identical regardless of how many run.
export MAX_TOKENS="${MAX_TOKENS:-48}"
export AFD_MAX_NEW_TOKENS="${AFD_MAX_NEW_TOKENS:-$MAX_TOKENS}"
export WARMUP_REPLAYS="${WARMUP_REPLAYS:-5}"

# ---- nodes: M attention nodes + one FFN node ---------------------------------
export AFD_GPUS_PER_NODE="${AFD_GPUS_PER_NODE:-4}"
if [[ -z "${AFD_NODE_LIST:-}" ]]; then
  echo "AFD_NODE_LIST (comma-separated node IPs) must be set; this wrapper never picks nodes itself." >&2
  exit 1
fi
IFS=',' read -r -a _AFD_NODES <<< "$AFD_NODE_LIST"
if (( ${#_AFD_NODES[@]} < 2 )); then
  echo "AFD_NODE_LIST must contain at least 2 nodes: one FFN node and at least one attention node" >&2
  exit 1
fi
export AFD_MLP_NODE="${AFD_MLP_NODE:-${_AFD_NODES[$((${#_AFD_NODES[@]} - 1))]}}"
_ATTN_NODES=()
_MLP_SEEN=0
for node in "${_AFD_NODES[@]}"; do
  if [[ "$node" == "$AFD_MLP_NODE" && "$_MLP_SEEN" == "0" ]]; then
    _MLP_SEEN=1
    continue
  fi
  _ATTN_NODES+=("$node")
done
if [[ "$_MLP_SEEN" == "0" ]]; then
  echo "AFD_MLP_NODE=$AFD_MLP_NODE is not present in AFD_NODE_LIST=$AFD_NODE_LIST" >&2
  exit 1
fi
_repeat_nodes() {  # _repeat_nodes N node... -> each node N times, comma-separated
  local repeat="$1" node i
  shift
  local out=()
  for node in "$@"; do
    for ((i = 0; i < repeat; i++)); do out+=("$node"); done
  done
  local IFS=,
  echo "${out[*]}"
}
export MINISGL_AFD_ATTN_NODES="${MINISGL_AFD_ATTN_NODES:-$(_repeat_nodes "$AFD_GPUS_PER_NODE" "${_ATTN_NODES[@]}")}"
export MINISGL_AFD_MLP_NODES="${MINISGL_AFD_MLP_NODES:-$(_repeat_nodes "$AFD_GPUS_PER_NODE" "$AFD_MLP_NODE")}"
export AFD_TOTAL_NODES="${#_AFD_NODES[@]}"

# ---- no tensor parallelism on either role -----------------------------------
export ATTN_DP_SIZE="$(( ${#_ATTN_NODES[@]} * AFD_GPUS_PER_NODE ))"
export AFD_ATTN_DP_SIZE="$ATTN_DP_SIZE"
export ATTN_TP_SIZE="${ATTN_TP_SIZE:-1}"
export AFD_ATTN_TP_SIZE="$ATTN_TP_SIZE"
export MLP_DP_SIZE="$AFD_GPUS_PER_NODE"
export MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
export AFD_MLP_EP_SIZE="$AFD_GPUS_PER_NODE"
if ((ATTN_TP_SIZE != 1 || MLP_TP_SIZE != 1)); then
  echo "This wrapper forbids tensor parallelism; got ATTN_TP_SIZE=$ATTN_TP_SIZE MLP_TP_SIZE=$MLP_TP_SIZE." >&2
  exit 1
fi

# ---- microbatch lanes must be explicit --------------------------------------
if [[ -z "${AFD_NUM_MB+x}" ]]; then
  echo "AFD_NUM_MB must be set explicitly (1 or 2; overlap is enabled for both)." >&2
  exit 1
fi
export AFD_NUM_MB
if ((AFD_NUM_MB < 1)); then
  echo "AFD_NUM_MB must be >= 1; got $AFD_NUM_MB." >&2
  exit 1
fi

# ---- global batch namespace --------------------------------------------------
# AFD_ACTIVE_GLOBAL is the target number of concurrently decoding requests across
# all attention DP replicas. Per-rank and per-lane shapes are derived from it.
if [[ -z "${AFD_ACTIVE_GLOBAL:-}" && -z "${PER_ATTN_GPU_BSZ:-}" ]]; then
  echo "Set PER_ATTN_GPU_BSZ (per attention GPU) or AFD_ACTIVE_GLOBAL (all attention GPUs)." >&2
  exit 1
fi
export AFD_ACTIVE_GLOBAL="${AFD_ACTIVE_GLOBAL:-$((PER_ATTN_GPU_BSZ * ATTN_DP_SIZE))}"
_GRAPH_LANES=$((ATTN_DP_SIZE * AFD_NUM_MB))
if ((AFD_ACTIVE_GLOBAL % _GRAPH_LANES != 0)); then
  echo "AFD_ACTIVE_GLOBAL=$AFD_ACTIVE_GLOBAL must be divisible by ATTN_DP_SIZE*AFD_NUM_MB=$_GRAPH_LANES." >&2
  exit 1
fi
_B_PER_AG_RANK=$((AFD_ACTIVE_GLOBAL / ATTN_DP_SIZE))
_B_PER_LANE=$((AFD_ACTIVE_GLOBAL / _GRAPH_LANES))

export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-2048}"
# Chunked prefill leaves one request per attention DP replica behind the first
# decode wave, so submit one spare per replica to reach the target active wave.
if ((PROMPT_LEN > AFD_MAX_BATCHED_TOKENS)); then
  _SPARE_PROMPTS="$ATTN_DP_SIZE"
else
  _SPARE_PROMPTS=0
fi
export NUM_PROMPTS="${NUM_PROMPTS:-$((AFD_ACTIVE_GLOBAL + _SPARE_PROMPTS))}"

# ---- overlap policy ----------------------------------------------------------
export AFD_DISABLE_OVERLAP="${AFD_DISABLE_OVERLAP:-0}"
# M2N's lane-owned buffers support graph overlap without bypassing the DeepEP
# safety guard. An explicit force switch is reserved for diagnostic use.
export AFD_FORCE_MULTI_MB_GRAPH_OVERLAP="${AFD_FORCE_MULTI_MB_GRAPH_OVERLAP:-0}"
export AFD_ASYNC_PENDING_STEPS="${AFD_ASYNC_PENDING_STEPS:-5}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"

export AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-none}"
export AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"
export MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-megamoe_m2n}"

export AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
export MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
# MINISGL_MAX_SEQ_LEN is deliberately NOT set here: it depends on the generated
# token count, which is derived from the prefill bracket further down. Sizing it
# as PROMPT_LEN+64 caps decode at ~64 tokens, so requests retire at staggered
# points, the graph replays a fixed batch full of PADDING, and attention looks
# ~38x cheap while the real active batch has drained. See the export below.
export MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-64}"

_GRAPH_MB_BS="$_B_PER_LANE"
if [[ -z "${AFD_DECODE_GRAPH_BS:-}" ]]; then
  if ((_GRAPH_MB_BS > 1)); then
    # Prime the shared M:N runtime with the smallest graph before capturing the
    # target graph; a cold first dispatch at the target shape can fail readiness.
    export AFD_DECODE_GRAPH_BS="1,$_GRAPH_MB_BS"
  else
    export AFD_DECODE_GRAPH_BS="1"
  fi
fi
export MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-$_GRAPH_MB_BS}"
export AFD_MEMORY_RATIO="${AFD_MEMORY_RATIO:-0.78}"
export AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend trtllm --memory-ratio $AFD_MEMORY_RATIO}"

# ---- pinned workload, deterministically cycled ------------------------------
# The vLLM reference harness cycles the pool by request_id % pool_rows.
# Materialize the same sequence so both frameworks decode identical prompts.
_PROMPT_POOL="${PROMPT_POOL:-$CALIB_DIR/prompts/prompts_512x${PROMPT_LEN}_seed20260527.txt}"
if [[ ! -f "$_PROMPT_POOL" ]]; then
  echo "Prompt pool not found: $_PROMPT_POOL" >&2
  exit 1
fi
_PROMPT_CACHE="$CALIB_DIR/reports/prompts_cycled_${NUM_PROMPTS}x${PROMPT_LEN}.txt"
if [[ -z "${PROMPT_FILE:-}" ]]; then
  if [[ ! -f "$_PROMPT_CACHE" ]]; then
    mkdir -p "$(dirname "$_PROMPT_CACHE")"
    awk -v want="$NUM_PROMPTS" '
      { pool[n++] = $0 }
      END {
        if (n == 0) { print "empty prompt pool" > "/dev/stderr"; exit 1 }
        for (i = 0; i < want; i++) print pool[i % n]
      }' "$_PROMPT_POOL" > "$_PROMPT_CACHE"
  fi
  export PROMPT_FILE="$_PROMPT_CACHE"
fi
export PROMPT_KIND="${PROMPT_KIND:-existing}"
_PROMPT_POOL_SHA="$(sha256sum "$_PROMPT_POOL" | awk '{print $1}')"

# ---- Nsight window -----------------------------------------------------------
_PREFILL_CHUNK_TOKENS="$AFD_MAX_BATCHED_TOKENS"
if ((_PREFILL_CHUNK_TOKENS > PROMPT_LEN)); then
  _PREFILL_CHUNK_TOKENS="$PROMPT_LEN"
fi
_PREFILL_CHUNKS_PER_REQ=$(((PROMPT_LEN + _PREFILL_CHUNK_TOKENS - 1) / _PREFILL_CHUNK_TOKENS))
_PREFILL_CHUNKS_PER_STEP=$((AFD_MAX_BATCHED_TOKENS / _PREFILL_CHUNK_TOKENS))
if ((_PREFILL_CHUNKS_PER_STEP < 1)); then
  _PREFILL_CHUNKS_PER_STEP=1
fi
_PREFILL_PROMPTS_PER_DP=$(((NUM_PROMPTS + ATTN_DP_SIZE - 1) / ATTN_DP_SIZE))
# Prefill duration is bracketed, not point-estimated. Dividing prompt tokens by
# the token cap assumes no decode contention and is a LOWER bound; requests that
# finish prefill immediately consume a decode-token slot on every later
# iteration, so the per-step prefill budget shrinks toward (cap - active), which
# gives an UPPER bound. The Nsight window spans the bracket and the analyzer
# isolates the pure-decode tail by kernel timestamp.
_PREFILL_STEPS_LOW=$(((_PREFILL_PROMPTS_PER_DP * _PREFILL_CHUNKS_PER_REQ + _PREFILL_CHUNKS_PER_STEP - 1) / _PREFILL_CHUNKS_PER_STEP))
_PREFILL_TOKENS_PER_DP=$((_PREFILL_PROMPTS_PER_DP * PROMPT_LEN))
_PREFILL_BUDGET=$((AFD_MAX_BATCHED_TOKENS - _B_PER_AG_RANK))
if ((_PREFILL_BUDGET < 1)); then
  _PREFILL_BUDGET=1
fi
_PREFILL_STEPS_HIGH=$(((_PREFILL_TOKENS_PER_DP + _PREFILL_BUDGET - 1) / _PREFILL_BUDGET))
if ((_PREFILL_STEPS_HIGH < _PREFILL_STEPS_LOW)); then
  _PREFILL_STEPS_HIGH="$_PREFILL_STEPS_LOW"
fi
# Scheduler-step model. The harness runs a warmup pass and then the formal pass,
# each of which performs a full prefill:
#   formal decode starts at : 2*prefill_real + WARMUP_REPLAYS
#   total steps             : 2*prefill_real + WARMUP_REPLAYS + MAX_TOKENS - 1
# with prefill_real in [LOW, HIGH]. The profiler must START at or after the LATEST
# possible decode start, and STOP at or before the EARLIEST possible total, or the
# worker sessions never finalize and only coordinator traces are written.
_DECODE_START_LATEST=$((2 * _PREFILL_STEPS_HIGH + WARMUP_REPLAYS))
_NSYS_DECODE_STEPS="${NSYS_DECODE_STEPS:-40}"
_NSYS_START_DEFAULT="$_DECODE_START_LATEST"
if ((_NSYS_START_DEFAULT < 1)); then
  _NSYS_START_DEFAULT=1
fi
_NSYS_STOP_DEFAULT=$((_NSYS_START_DEFAULT + _NSYS_DECODE_STEPS))
# Generate enough tokens that the stop step is reachable even in the fastest
# (LOW-bound) prefill case. Decode steps are identical regardless of how many run.
_MIN_TOKENS_FOR_STOP=$((_NSYS_STOP_DEFAULT - 2 * _PREFILL_STEPS_LOW - WARMUP_REPLAYS + 1))
_TOKEN_MARGIN="${TOKEN_MARGIN:-32}"
_REQUIRED_TOKENS=$((_MIN_TOKENS_FOR_STOP + _TOKEN_MARGIN))
if ((_REQUIRED_TOKENS > MAX_TOKENS)); then
  MAX_TOKENS="$_REQUIRED_TOKENS"
  export MAX_TOKENS
  export AFD_MAX_NEW_TOKENS="$MAX_TOKENS"
fi
_DECODE_REPLAYS=$((MAX_TOKENS > 0 ? MAX_TOKENS - 1 : 0))
_TOTAL_STEPS_MIN=$((2 * _PREFILL_STEPS_LOW + WARMUP_REPLAYS + _DECODE_REPLAYS))

# Every request must be able to generate all MAX_TOKENS without hitting the
# sequence ceiling, otherwise requests retire at staggered points and the decode
# graph replays a mostly-padded batch. PROMPT_TOKENS_MAX is the tokenizer-measured
# ceiling of the pinned pool (8090-8187 for the 8K lane), not the nominal name.
export PROMPT_TOKENS_MAX="${PROMPT_TOKENS_MAX:-$PROMPT_LEN}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-$((PROMPT_TOKENS_MAX + MAX_TOKENS + 64))}"
export NSYS_START_STEP="${NSYS_START_STEP:-$_NSYS_START_DEFAULT}"
export NSYS_STOP_STEP="${NSYS_STOP_STEP:-$_NSYS_STOP_DEFAULT}"
if ((NSYS_STOP_STEP > _TOTAL_STEPS_MIN)); then
  echo "Nsight stop step $NSYS_STOP_STEP exceeds the minimum reachable total step $_TOTAL_STEPS_MIN; worker traces would never finalize." >&2
  exit 1
fi

export NSYS="${NSYS:-1}"
export NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
export RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-0}"
export CLEAN_FIRST="${CLEAN_FIRST:-0}"
export CLEAN_AFTER="${CLEAN_AFTER:-0}"
export MINISGL_AFD_SHUTDOWN_TIMEOUT_S="${MINISGL_AFD_SHUTDOWN_TIMEOUT_S:-30}"
export AFD_STOP_TIMEOUT_S="${AFD_STOP_TIMEOUT_S:-90}"

export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/qwen35_fp8_${PROMPT_LEN}_B${AFD_ACTIVE_GLOBAL}_attn${ATTN_DP_SIZE}dp_ep${AFD_MLP_EP_SIZE}_${AFD_TOTAL_NODES}node_notp_mb${AFD_NUM_MB}_${TIMESTAMP}}"

echo "Qwen3.5 FP8 dynamic-node no-TP replay workload"
echo "  model: $MODEL_PATH"
echo "  total_nodes: $AFD_TOTAL_NODES (attention ${#_ATTN_NODES[@]} : FFN 1)"
echo "  mlp_node: $AFD_MLP_NODE"
echo "  attn_nodes: ${_ATTN_NODES[*]}"
echo "  topology: AG-DP${ATTN_DP_SIZE}/TP${ATTN_TP_SIZE} + EG-DP${MLP_DP_SIZE}/TP${MLP_TP_SIZE}/EP${AFD_MLP_EP_SIZE} (tensor_parallel=none)"
echo "  prompt_len: $PROMPT_LEN"
echo "  prompt_pool: $_PROMPT_POOL"
echo "  prompt_pool_sha256: $_PROMPT_POOL_SHA"
echo "  prompt_file: $PROMPT_FILE (kind=$PROMPT_KIND, cycled)"
echo "  submitted_prompts: $NUM_PROMPTS"
echo "  active_global: $AFD_ACTIVE_GLOBAL"
echo "  active_per_ag_rank: $_B_PER_AG_RANK"
echo "  active_per_lane: $_B_PER_LANE"
echo "  microbatches: $AFD_NUM_MB (explicit)"
echo "  overlap_policy: disable=$AFD_DISABLE_OVERLAP force_multi_mb=$AFD_FORCE_MULTI_MB_GRAPH_OVERLAP pending=$AFD_ASYNC_PENDING_STEPS cuda_connections=$CUDA_DEVICE_MAX_CONNECTIONS"
echo "  afd_moe_backend: $MINISGL_AFD_MOE_BACKEND"
echo "  target_graph_microbatch: $_GRAPH_MB_BS"
echo "  graph_warmup_batches: $AFD_DECODE_GRAPH_BS"
echo "  server_extra_args: $AFD_SERVER_EXTRA_ARGS"
echo "  prefill_step_bracket: $_PREFILL_STEPS_LOW..$_PREFILL_STEPS_HIGH"
echo "  generated_tokens: $MAX_TOKENS (auto-raised so the stop step is reachable)"
echo "  min_reachable_total_step: $_TOTAL_STEPS_MIN"
echo "  prompt_tokens_max: $PROMPT_TOKENS_MAX"
echo "  max_seq_len: $MINISGL_MAX_SEQ_LEN (prompt_tokens_max + generated + 64)"
echo "  nsys_step_window: $NSYS_START_STEP..$NSYS_STOP_STEP"
echo "  run_dir: $RUN_DIR"

if [[ "${QWEN35_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec bash "$RUNNER"

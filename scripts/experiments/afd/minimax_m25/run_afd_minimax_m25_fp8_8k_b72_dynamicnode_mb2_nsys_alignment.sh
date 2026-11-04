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
RUNNER_DIR="$CALIB_DIR/scripts"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# MiniMax M2.5 FP8 AFD AG/EG 8k decode replay workload.
#
# Default workload:
#   - 4 Ray GPU nodes total
#   - 1 MLP node: MLP dp=4, ep=4
#   - 3 attention nodes: attention dp=12, tp=1
#   - 72 requests per attention GPU = 864 global requests
#   - mb=2, so replay bucket is 36 requests per attention GPU per MB
#
# Node policy:
#   - AFD_TOTAL_NODES controls how many Ray GPU nodes to use; default 4.
#   - Exactly one node is used for MLP.
#   - Every remaining node is used for attention.
#   - By default, the last node in AFD_NODE_LIST is the MLP node.
#
# Override examples:
#   AFD_NODE_LIST=ip0,ip1,ip2,ip3 AFD_MLP_NODE=ip3 bash scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh
#   RUN_VLLM_ALIGNMENT=0 bash scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh

DEFAULT_MODEL="$HOME/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/snapshots/f710177d938eff80b684d42c5aa84b382612f21f"
if [[ ! -d "$DEFAULT_MODEL" ]]; then
  DEFAULT_MODEL="MiniMaxAI/MiniMax-M2.5"
fi
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

export PROMPT_LEN="${PROMPT_LEN:-8192}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-72}"
export MAX_TOKENS="${MAX_TOKENS:-16}"

export AFD_TOTAL_NODES="${AFD_TOTAL_NODES:-${NUM_NODES:-4}}"
export AFD_GPUS_PER_NODE="${AFD_GPUS_PER_NODE:-4}"
export AFD_NUM_MB="${AFD_NUM_MB:-2}"

if [[ -z "${AFD_NODE_LIST:-}" ]]; then
  export AFD_NODE_LIST="$(
    python - <<'PY'
import os
import ray

total = int(os.environ["AFD_TOTAL_NODES"])
ray.init(
    address=os.environ.get("RAY_ADDRESS", "auto"),
    ignore_reinit_error=True,
    logging_level="ERROR",
)
nodes = [
    node["NodeManagerAddress"]
    for node in ray.nodes()
    if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0
]
nodes = sorted(dict.fromkeys(nodes))
if len(nodes) < total:
    raise SystemExit(f"need {total} Ray GPU nodes, found {len(nodes)}: {nodes}")
print(",".join(nodes[:total]))
PY
  )"
fi

IFS=',' read -r -a _AFD_NODES <<< "$AFD_NODE_LIST"
if (( ${#_AFD_NODES[@]} < 2 )); then
  echo "AFD_NODE_LIST must contain at least 2 nodes: one MLP node and at least one attention node" >&2
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
if (( ${#_ATTN_NODES[@]} < 1 )); then
  echo "No attention nodes remain after selecting AFD_MLP_NODE=$AFD_MLP_NODE" >&2
  exit 1
fi

_repeat_nodes() {
  local repeat="$1"
  shift
  python - "$repeat" "$@" <<'PY'
import sys
repeat = int(sys.argv[1])
nodes = sys.argv[2:]
expanded = []
for node in nodes:
    expanded.extend([node] * repeat)
print(",".join(expanded))
PY
}

export MINISGL_AFD_ATTN_NODES="${MINISGL_AFD_ATTN_NODES:-$(_repeat_nodes "$AFD_GPUS_PER_NODE" "${_ATTN_NODES[@]}")}"
export MINISGL_AFD_MLP_NODES="${MINISGL_AFD_MLP_NODES:-$(_repeat_nodes "$AFD_GPUS_PER_NODE" "$AFD_MLP_NODE")}"

export AFD_ATTN_DP_SIZE="$(( ${#_ATTN_NODES[@]} * AFD_GPUS_PER_NODE ))"
export ATTN_DP_SIZE="$AFD_ATTN_DP_SIZE"
export AFD_ATTN_TP_SIZE="${AFD_ATTN_TP_SIZE:-1}"
export ATTN_TP_SIZE="$AFD_ATTN_TP_SIZE"
export MLP_DP_SIZE="${MLP_DP_SIZE:-4}"
export MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
export AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"

export NUM_PROMPTS="${NUM_PROMPTS:-$(( PER_ATTN_GPU_BSZ * AFD_ATTN_DP_SIZE ))}"
export AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
export MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
# Reused Qwen 8k prompts tokenize slightly longer under MiniMax (observed max
# 8248), so 8320 leaves room for 16 decode tokens without scheduler clipping.
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-8320}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"

export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-512}"
_DEFAULT_DECODE_GRAPH_BS=$(( (NUM_PROMPTS + ATTN_DP_SIZE * AFD_NUM_MB - 1) / (ATTN_DP_SIZE * AFD_NUM_MB) ))
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-$_DEFAULT_DECODE_GRAPH_BS}"

export AFD_MEMORY_RATIO="${AFD_MEMORY_RATIO:-0.79}"
export AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend trtllm --memory-ratio $AFD_MEMORY_RATIO}"

export MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-megamoe_m2n}"
export MINISGL_MEGAMOE_AG_SMS="${MINISGL_MEGAMOE_AG_SMS:-24}"

export NSYS="${NSYS:-1}"
export NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
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
export VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
export VLLM_SCORE_CONCURRENCY="${VLLM_SCORE_CONCURRENCY:-16}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code --enforce-eager --moe-backend triton --gpu-memory-utilization 0.70 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
# MiniMax M2.5 AFD uses TP1 attention while the local vLLM baseline uses TP4.
# A few boundary tokens can land at rank 11/12 while top100 remains 1.0.
export ALIGN_MIN_TOP10_RATE="${ALIGN_MIN_TOP10_RATE:-0.999}"

export CLEAN_FIRST="${CLEAN_FIRST:-0}"
export CLEAN_AFTER="${CLEAN_AFTER:-0}"
export POST_SHUTDOWN_SLEEP="${POST_SHUTDOWN_SLEEP:-20}"
export AFD_STOP_TIMEOUT_S="${AFD_STOP_TIMEOUT_S:-90}"

PROMPT_BASE_FILE="${PROMPT_BASE_FILE:-$CALIB_DIR/prompts/prompts_512x8192_seed20260527.txt}"
DEFAULT_PROMPT_FILE="$CALIB_DIR/reports/prompts_${NUM_PROMPTS}x${PROMPT_LEN}_from512_seed20260527.txt"
if [[ -z "${PROMPT_FILE:-}" ]]; then
  export PROMPT_FILE="$DEFAULT_PROMPT_FILE"
  if [[ ! -f "$PROMPT_FILE" ]]; then
    python - "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$NUM_PROMPTS" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
count = int(sys.argv[3])
lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"no prompts in {src}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(lines[i % len(lines)] for i in range(count)) + "\n", encoding="utf-8")
print(f"wrote {count} prompts to {dst} from {len(lines)} source prompts")
PY
  fi
fi
export PROMPT_KIND="${PROMPT_KIND:-existing}"

# MiniMax tokenizes the reused Qwen 8k prompt file slightly longer than 8192
# tokens, while chunked prefill is intentionally small to keep large-node runs
# below the memory ceiling. Start nsys at the conservative 8192-token estimate
# and stop at the max-seq estimate so decode replay is not missed.
_PREFILL_CHUNK_TOKENS="$AFD_MAX_BATCHED_TOKENS"
if (( _PREFILL_CHUNK_TOKENS > PROMPT_LEN )); then
  _PREFILL_CHUNK_TOKENS="$PROMPT_LEN"
fi
_PREFILL_MIN_CHUNKS_PER_REQ=$(( (PROMPT_LEN + _PREFILL_CHUNK_TOKENS - 1) / _PREFILL_CHUNK_TOKENS ))
_PREFILL_TOKEN_BUDGET=$(( MINISGL_MAX_SEQ_LEN - MAX_TOKENS ))
if (( _PREFILL_TOKEN_BUDGET < PROMPT_LEN )); then
  _PREFILL_TOKEN_BUDGET="$PROMPT_LEN"
fi
_PREFILL_MAX_CHUNKS_PER_REQ=$(( (_PREFILL_TOKEN_BUDGET + _PREFILL_CHUNK_TOKENS - 1) / _PREFILL_CHUNK_TOKENS ))
_PREFILL_CHUNKS_PER_DP_STEP=$(( AFD_MAX_BATCHED_TOKENS / _PREFILL_CHUNK_TOKENS ))
if (( _PREFILL_CHUNKS_PER_DP_STEP < 1 )); then
  _PREFILL_CHUNKS_PER_DP_STEP=1
fi
_PREFILL_START_STEP=$(( (NUM_PROMPTS * _PREFILL_MIN_CHUNKS_PER_REQ + ATTN_DP_SIZE * _PREFILL_CHUNKS_PER_DP_STEP - 1) / (ATTN_DP_SIZE * _PREFILL_CHUNKS_PER_DP_STEP) ))
_PREFILL_STOP_STEP=$(( (NUM_PROMPTS * _PREFILL_MAX_CHUNKS_PER_REQ + ATTN_DP_SIZE * _PREFILL_CHUNKS_PER_DP_STEP - 1) / (ATTN_DP_SIZE * _PREFILL_CHUNKS_PER_DP_STEP) ))
_DECODE_GRAPH_STEPS=$(( MAX_TOKENS > 0 ? MAX_TOKENS - 1 : 0 ))
_NSYS_START_DEFAULT=$(( _PREFILL_START_STEP - PER_ATTN_GPU_BSZ ))
if (( _NSYS_START_DEFAULT < 1 )); then
  _NSYS_START_DEFAULT=1
fi
export NSYS_START_STEP="${NSYS_START_STEP:-$_NSYS_START_DEFAULT}"
export NSYS_STOP_STEP="${NSYS_STOP_STEP:-$((_PREFILL_STOP_STEP + _DECODE_GRAPH_STEPS + 2))}"

if [[ -z "${MINISGL_PORT:-}" || ( "$RUN_VLLM_ALIGNMENT" == "1" && -z "${VLLM_PORT:-}" ) ]]; then
  eval "$(
    python - <<'PY'
import socket

def block_free(base: int, width: int = 40) -> bool:
    sockets = []
    try:
        for port in range(base, base + width):
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()

for base in range(42100, 50000, 100):
    if block_free(base):
        print(f"export MINISGL_PORT=${{MINISGL_PORT:-{base}}}")
        print(f"export VLLM_PORT=${{VLLM_PORT:-{base + 20}}}")
        break
else:
    raise SystemExit("could not find a free 40-port block")
PY
  )"
fi

AFD_MEMORY_TAG="${AFD_MEMORY_RATIO/./}"
export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_minimax_m25_fp8_8k_b${PER_ATTN_GPU_BSZ}_peragpu_${#_AFD_NODES[@]}node_mem${AFD_MEMORY_TAG}_bucket${AFD_MAX_BATCHED_TOKENS}_${TIMESTAMP}}"

echo "AFD dynamic-node MiniMax M2.5 b${PER_ATTN_GPU_BSZ}/attention-GPU"
echo "  total_nodes: ${#_AFD_NODES[@]}"
echo "  mlp_node: $AFD_MLP_NODE"
echo "  attn_nodes: ${_ATTN_NODES[*]}"
echo "  attn_dp/tp: $ATTN_DP_SIZE/$ATTN_TP_SIZE"
echo "  mlp_dp/tp/ep: $MLP_DP_SIZE/$MLP_TP_SIZE/$AFD_MLP_EP_SIZE"
echo "  num_prompts: $NUM_PROMPTS"
echo "  per_attn_gpu_bsz: $PER_ATTN_GPU_BSZ"
echo "  per_attn_gpu_mb_bsz: $AFD_DECODE_GRAPH_BS"
echo "  max_batched_tokens: $AFD_MAX_BATCHED_TOKENS"
echo "  prefill_step_window_estimate: $_PREFILL_START_STEP..$_PREFILL_STOP_STEP"
echo "  nsys_step_window: $NSYS_START_STEP..$NSYS_STOP_STEP"
echo "  run_vllm_alignment: $RUN_VLLM_ALIGNMENT"

exec bash "$RUNNER_DIR/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh"

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

# Qwen3.5-397B-A17B-FP8 AFD AG/EG two-node 8k decode replay workload.
#
# A thin wrapper around the shared no-TP driver, which owns the topology, the
# MegaMoE M:N settings, the prompt cycling, the Nsight window and cleanup. This
# file only names the model and the workload shape.
#
# 320 concurrently decoding requests per attention GPU across four attention DP
# replicas. With mb=2 that replays a 160-request per-microbatch graph. 397B is
# 60 layers against 122B's 48 and routes 512 experts at top-k 10, so the same
# GPUs hold fewer requests than the 122B preset's 256. Results were measured
# against revision ea5b4f81096f3901c91dea97f81324302495781d.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-397B-A17B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-8192}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-320}"
export AFD_NUM_MB="${AFD_NUM_MB:-2}"

export ATTN_DP_SIZE="${ATTN_DP_SIZE:-4}"
export AFD_ACTIVE_GLOBAL="${AFD_ACTIVE_GLOBAL:-$((PER_ATTN_GPU_BSZ * ATTN_DP_SIZE))}"

export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen35_397b_a17b_fp8_8k_b${PER_ATTN_GPU_BSZ}_peragpu_2node_mb${AFD_NUM_MB}_${TIMESTAMP}}"

exec bash "$SCRIPT_DIR/run_qwen35_fp8_afd_dp4_ep4_notp_2node.sh"

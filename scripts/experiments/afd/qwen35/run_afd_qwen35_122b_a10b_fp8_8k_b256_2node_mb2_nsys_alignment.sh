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

# Qwen3.5-122B-A10B-FP8 AFD AG/EG two-node 8k decode replay workload.
#
# A thin wrapper around the shared no-TP driver, which owns the topology, the
# MegaMoE M:N settings, the prompt cycling, the Nsight window and cleanup. This
# file only names the model and the workload shape.
#
# 256 concurrently decoding requests per attention GPU across four attention DP
# replicas. With mb=2 that replays a 128-request per-microbatch graph. Results
# were measured against revision a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-122B-A10B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-8192}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-256}"
export AFD_NUM_MB="${AFD_NUM_MB:-2}"

export ATTN_DP_SIZE="${ATTN_DP_SIZE:-4}"
export AFD_ACTIVE_GLOBAL="${AFD_ACTIVE_GLOBAL:-$((PER_ATTN_GPU_BSZ * ATTN_DP_SIZE))}"

export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen35_122b_a10b_fp8_8k_b${PER_ATTN_GPU_BSZ}_peragpu_2node_mb${AFD_NUM_MB}_${TIMESTAMP}}"

exec bash "$SCRIPT_DIR/run_qwen35_fp8_afd_dp4_ep4_notp_2node.sh"

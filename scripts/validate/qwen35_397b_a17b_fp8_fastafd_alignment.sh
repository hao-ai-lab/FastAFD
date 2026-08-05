#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  qwen35_397b_a17b_fp8_fastafd_alignment.sh

Run the pinned official Qwen3.5-397B-A17B-FP8 two-node FastAFD/vLLM alignment.
Same driver as the 122B preset, differing in model identity, report name, and
cold-start timeouts: loading 379 GB of weights on both engines does not fit the
default 1800 s readiness window. Every control is documented in
qwen35_fastafd_alignment.sh --help.
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

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-397B-A17B-FP8}"
MODEL_REVISION="${MODEL_REVISION:-ea5b4f81096f3901c91dea97f81324302495781d}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-Qwen3.5-397B-A17B-FP8}"

REPORT_DIR="$CALIB_DIR/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen35_397b_a17b_fp8_afd_serve_alignment_tp8_ray_${TIMESTAMP}.json}"
AFD_READY_TIMEOUT="${AFD_READY_TIMEOUT:-3600}"
AFD_SAMPLE_TIMEOUT="${AFD_SAMPLE_TIMEOUT:-1200}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-3600}"
VLLM_SCORE_TIMEOUT="${VLLM_SCORE_TIMEOUT:-1200}"

export MODEL_REPO MODEL_REVISION MODEL_DISPLAY_NAME OUTPUT_JSON
export AFD_READY_TIMEOUT AFD_SAMPLE_TIMEOUT VLLM_READY_TIMEOUT VLLM_SCORE_TIMEOUT

exec "$SCRIPT_DIR/qwen35_fastafd_alignment.sh" "$@"

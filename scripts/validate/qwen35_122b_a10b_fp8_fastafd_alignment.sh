#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  qwen35_122b_a10b_fp8_fastafd_alignment.sh

Run the pinned official Qwen3.5-122B-A10B-FP8 two-node FastAFD/vLLM alignment.
This preset supplies only the model identity and report name; every control is
documented in the shared driver, qwen35_fastafd_alignment.sh --help.
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

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-122B-A10B-FP8}"
MODEL_REVISION="${MODEL_REVISION:-a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-Qwen3.5-122B-A10B-FP8}"

REPORT_DIR="$CALIB_DIR/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen35_122b_a10b_fp8_afd_serve_alignment_tp8_ray_${TIMESTAMP}.json}"

export MODEL_REPO MODEL_REVISION MODEL_DISPLAY_NAME OUTPUT_JSON

exec "$SCRIPT_DIR/qwen35_fastafd_alignment.sh" "$@"

#!/usr/bin/env bash
# Run one benchmark configuration (configs.json) and check it against its reference result.
#
# usage: run.sh CONFIG_ID NODE_IPS
#   NODE_IPS: comma-separated Ray node IPs, attention nodes first and the FFN node last (the configuration's "nodes").
# Environment:
#   FASTAFD_MODEL_ROOT   directory holding the checkpoints of models.json (or FASTAFD_MODEL_<KEY> per model)
#   FASTAFD_POOL_DIR     directory with the prompt pools (build_prompts.py)
#   FASTAFD_OUT          output root: runs/<config>-<UTC>/ (logs, driver_env.txt, config.json, result.json), prompts/
#   MINISGL_RAY_ADDRESS  the Ray cluster (GCS address) the nodes belong to; run this on one of the configuration's nodes
#   optional: FASTAFD_CACHE_DIR (JIT caches, default $FASTAFD_OUT/cache), MINISGL_WORKER_PYTHONPATH_PREPEND
#   (e.g. a FlashInfer 0.7.0.post1 install for the workers), ENV_NAME (the active conda env or venv path).
# Exit: the runner's exit code if it failed, else 0 for PASS and 1 for FAIL / NOT_COMPARABLE.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG="$1"
NODES="$2"
: "${FASTAFD_POOL_DIR:?}" "${FASTAFD_OUT:?}" "${MINISGL_RAY_ADDRESS:?}"

eval "$(python3 "$HERE/benchmark.py" env "$CONFIG")"
IFS=, read -r -a NL <<< "$NODES"
if (( ${#NL[@]} != FASTAFD_CONFIG_NODES )); then
  echo "configuration $CONFIG needs $FASTAFD_CONFIG_NODES nodes, got ${#NL[@]}: $NODES" >&2
  exit 2
fi
export AFD_NODE_LIST="$NODES" AFD_MLP_NODE="${NL[${#NL[@]}-1]}" AFD_TOTAL_NODES="${#NL[@]}"

# The measurement settings of every reference result: no profiler, no vLLM alignment pass, per-step timing logs.
export NSYS=0 RUN_VLLM_ALIGNMENT=0 CLEAN_FIRST=0 CLEAN_AFTER=0 MINISGL_AFD_STEP_TIMES=1 MINISGL_AFD_LOG_STEPS=1
export MINISGL_PORT="${MINISGL_PORT:-23100}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ENV_NAME="${ENV_NAME:-${CONDA_DEFAULT_ENV:-}}" PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
C="${FASTAFD_CACHE_DIR:-$FASTAFD_OUT/cache}"
export MINISGL_LOCAL_CACHE_BASE="$C" FLASHINFER_WORKSPACE_BASE="$C/flashinfer"
for v in TVM_FFI_CACHE_DIR EP_JIT_CACHE_DIR N2M_M2N_GIN_BUILD_DIR MINISGL_DEEPEP_BUILD_DIR MINISGL_DEEPGEMM_BUILD_DIR \
         DG_JIT_CACHE_DIR TRITON_CACHE_DIR; do
  [[ -n "${!v:-}" ]] || export "$v=$C/private/${v,,}"
done

export RUN_DIR="$FASTAFD_OUT/runs/$CONFIG-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
env | LC_ALL=C sort > "$RUN_DIR/driver_env.txt"
git -c safe.directory='*' -C "$REPO" rev-parse HEAD > "$RUN_DIR/code_head.txt"
git -c safe.directory='*' -C "$REPO" status --porcelain --untracked-files=no > "$RUN_DIR/code_dirty.txt"
python3 "$HERE/benchmark.py" show "$CONFIG" --nodes "$NODES" > "$RUN_DIR/config.json"

echo "RUN $CONFIG nodes=$NODES run_dir=$RUN_DIR start $(date -u +%FT%TZ)"
set +e
bash "$REPO/$PRESET" > "$RUN_DIR/driver.log" 2>&1
rc=$?
set -e
echo "RUNNER_EXIT rc=$rc $(date -u +%FT%TZ)"
python3 "$HERE/benchmark.py" check "$RUN_DIR" > "$RUN_DIR/result.json" || true
cat "$RUN_DIR/result.json"
(( rc == 0 )) || exit "$rc"
grep -q '"verdict": "PASS"' "$RUN_DIR/result.json"

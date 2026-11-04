#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  enter_clean.sh CONDA_ENV_NAME

Example:
  ./scripts/enter_clean.sh minisgl-cuda130
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

ENV_NAME="$1"
# Honor an explicit CONDA_SH; otherwise pick whichever conda install exists.
if [[ -z "${CONDA_SH:-}" ]]; then
  for _cand in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -f "$_cand/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="$_cand/etc/profile.d/conda.sh"
      break
    fi
  done
fi
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
START_DIR="${PWD}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-${START_DIR}/cache}"
USER_NAME="${USER:-$(id -un)}"
LOGNAME_VALUE="${LOGNAME:-$USER_NAME}"
TERM_VALUE="${TERM:-xterm-256color}"
SHELL_VALUE="${SHELL:-/bin/bash}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

exec env -i \
  HOME="${HOME}" \
  USER="${USER_NAME}" \
  LOGNAME="${LOGNAME_VALUE}" \
  TERM="${TERM_VALUE}" \
  SHELL="${SHELL_VALUE}" \
  PATH="/usr/bin:/bin" \
  LANG="C.UTF-8" \
  LC_ALL="C.UTF-8" \
  BASH_ENV="" \
  ENV="" \
  PROMPT_COMMAND="" \
  bash --noprofile --norc -lc "
    set -eo pipefail
    cd \"${WORKDIR}\"
    source \"${CONDA_SH}\"
    conda activate \"${ENV_NAME}\"
    mkdir -p \"${CACHE_ROOT}/flashinfer\"
    mkdir -p \"${CACHE_ROOT}/tvm-ffi\"
    mkdir -p \"${CACHE_ROOT}/deepep-jit\"
    mkdir -p \"${CACHE_ROOT}/gin-comm\"
    mkdir -p \"${CACHE_ROOT}/deepep-moe\"
    mkdir -p \"${CACHE_ROOT}/deepgemm\"
    NCCL_PIP_LIB=\$(python - <<'PY'
import site
from pathlib import Path
for root in site.getsitepackages():
    cand = Path(root) / 'nvidia' / 'nccl' / 'lib'
    if cand.exists():
        print(cand)
        break
PY
)
    export CUDA_HOME=\"\${CONDA_PREFIX}\"
    export CUDA_PATH=\"\${CONDA_PREFIX}\"
    export CUDA_NVCC_EXECUTABLE=\"\${CONDA_PREFIX}/bin/nvcc\"
    export PATH=\"\${CONDA_PREFIX}/bin:\${PATH}\"
    export TRITON_PTXAS_BLACKWELL_PATH=\"\${CONDA_PREFIX}/bin/ptxas\"
    export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${CONDA_PREFIX}/targets/sbsa-linux/lib\${NCCL_PIP_LIB:+:\${NCCL_PIP_LIB}}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
    export LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${CONDA_PREFIX}/targets/sbsa-linux/lib\${NCCL_PIP_LIB:+:\${NCCL_PIP_LIB}}\${LIBRARY_PATH:+:\${LIBRARY_PATH}}\"
    # JIT builds (FlashInfer: -L\$CUDA_HOME/lib64/stubs -lcuda) link against the
    # driver stub. CUDA 13 Conda envs keep it in lib/stubs, not lib64/stubs, so
    # a cold cache fails with 'cannot find -lcuda'. Search it last: link time
    # only, and the real libcuda.so.1 still comes from the driver at run time.
    if [[ -e \"\${CONDA_PREFIX}/lib/stubs/libcuda.so\" ]]; then
      export LIBRARY_PATH=\"\${LIBRARY_PATH}:\${CONDA_PREFIX}/lib/stubs\"
    fi
    export FLASHINFER_WORKSPACE_BASE=\"${CACHE_ROOT}/flashinfer\"
    export TVM_FFI_CACHE_DIR=\"${CACHE_ROOT}/tvm-ffi\"
    export EP_JIT_CACHE_DIR=\"${CACHE_ROOT}/deepep-jit\"
    export N2M_M2N_GIN_BUILD_DIR=\"${CACHE_ROOT}/gin-comm\"
    export MINISGL_DEEPEP_BUILD_DIR=\"${CACHE_ROOT}/deepep-moe\"
    export MINISGL_DEEPGEMM_BUILD_DIR=\"${CACHE_ROOT}/deepgemm\"
    echo \"[clean-shell] env=\${CONDA_DEFAULT_ENV:-} prefix=\${CONDA_PREFIX:-}\"
    echo \"[clean-shell] inherited PYTHONPATH=\${PYTHONPATH:-<unset>} CUDA_HOME=\${CUDA_HOME:-<unset>} LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-<unset>} LIBRARY_PATH=\${LIBRARY_PATH:-<unset>} FLASHINFER_WORKSPACE_BASE=\${FLASHINFER_WORKSPACE_BASE:-<unset>} TVM_FFI_CACHE_DIR=\${TVM_FFI_CACHE_DIR:-<unset>} EP_JIT_CACHE_DIR=\${EP_JIT_CACHE_DIR:-<unset>} N2M_M2N_GIN_BUILD_DIR=\${N2M_M2N_GIN_BUILD_DIR:-<unset>} MINISGL_DEEPEP_BUILD_DIR=\${MINISGL_DEEPEP_BUILD_DIR:-<unset>} MINISGL_DEEPGEMM_BUILD_DIR=\${MINISGL_DEEPGEMM_BUILD_DIR:-<unset>} TRITON_PTXAS_BLACKWELL_PATH=\${TRITON_PTXAS_BLACKWELL_PATH:-<unset>}\"
    exec bash --noprofile --norc -i
  "

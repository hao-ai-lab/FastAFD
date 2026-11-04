#!/usr/bin/env bash
# Build the pinned vLLM reference that FastAFD's alignment validators score against.
#
# The same base commit, patches, precompiled extensions, hash-pinned wheels and
# install commands as the reference the alignment results were scored with. The source is a git
# checkout instead of an extracted archive, so every alignment report's
# reference_identity shows the commit and the patch diff.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/reference/build_vllm_reference.sh --prefix DIR [options]

Creates a venv at DIR with vLLM d4004455 (0.21.1rc1.dev278) plus
scripts/reference/patches, editable from DIR/src/vllm with the compiled
extensions of vLLM's precompiled wheel for that commit, and the wheels pinned
by scripts/reference/requirements.lock (torch 2.11.0+cu130, FlashInfer
0.6.11.post2, ...). Linux aarch64, CPython 3.12 and a CUDA 13.0 toolkit only.
Then use it with VLLM_ENV_NAME=DIR VLLM_ENV_SETUP=DIR/vllm-reference.env.

Options:
  --prefix DIR       New install directory. Required; must not exist.
  --python PATH      CPython 3.12 interpreter for the venv. Default: python3.12
  --cuda-home DIR    CUDA 13.0 toolkit (bin/nvcc, include/). Default: $CUDA_HOME
  --wheel-dir DIR    Install the locked wheels from DIR only (--no-index); each
                     wheel is still checked against the lock's sha256.
  -h, --help         Show this help.
EOF
}

die() {
  echo "build_vllm_reference: $*" >&2
  exit 1
}

VLLM_REPO_URL="https://github.com/vllm-project/vllm"
VLLM_COMMIT="d4004455d2357985830af10e432709b42c820455"
VLLM_VERSION="0.21.1rc1.dev278+gd4004455d"
VLLM_WHEEL_NAME="vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_28_aarch64.whl"
VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-0.21.1rc1.dev278%2Bgd4004455d-cp38-abi3-manylinux_2_28_aarch64.whl"
VLLM_WHEEL_SHA256="067f4458fbab21c712dfee73e8f353f86bf4dfc7c6730870ee0799046c51593b"
# Identity of the result as scripts/validate/reference_identity.py records it:
# the patch diff over d4004455 and the installed vllm package tree.
SOURCE_DIFF_SHA256="057a5a20f91ef5ade6bf73e3296f3b22947830eb3b6a3d9991e2b214cabef0fd"
PACKAGE_TREE_SHA256="5ea2f83bd056a08eaa7f804493cc17e88ea82e19f2cf464d10c22fdcc9ddb82a"
TORCH_VERSION="2.11.0+cu130"
FLASHINFER_VERSION="0.6.11.post2"

PREFIX=""
PYTHON_BIN="python3.12"
CUDA_HOME_DIR="${CUDA_HOME:-}"
WHEEL_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME_DIR="$2"; shift 2 ;;
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ -n "$PREFIX" ]] || { usage >&2; die "--prefix is required"; }
[[ ! -e "$PREFIX" ]] || die "refusing to reuse existing $PREFIX; choose a new directory"
[[ "$(uname -m)" == aarch64 ]] || die "the lock pins Linux aarch64 wheels; this machine is $(uname -m)"
"$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null \
  || die "$PYTHON_BIN is not CPython 3.12 (the lock pins cp312 wheels)"
[[ -x "$CUDA_HOME_DIR/bin/nvcc" ]] || die "no CUDA toolkit: pass --cuda-home DIR with bin/nvcc"
"$CUDA_HOME_DIR/bin/nvcc" --version | grep -q 'release 13\.0' || die "$CUDA_HOME_DIR is not CUDA 13.0"
for tool in git curl sha256sum; do
  command -v "$tool" >/dev/null || die "$tool is required"
done
if [[ -n "$WHEEL_DIR" ]]; then
  [[ -d "$WHEEL_DIR" ]] || die "no wheel directory $WHEEL_DIR"
  WHEEL_DIR="$(cd "$WHEEL_DIR" && pwd)"
fi
CUDA_HOME_DIR="$(cd "$CUDA_HOME_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/requirements.lock"
PROBE="$SCRIPT_DIR/../validate/reference_identity.py"
mkdir -p "$PREFIX/src"
PREFIX="$(cd "$PREFIX" && pwd)"
SRC="$PREFIX/src/vllm"
WHEEL="$PREFIX/src/$VLLM_WHEEL_NAME"

# Run a command without inherited compiler, include/library, Python-path, vLLM
# or pip settings, as the recipe's build did.
clean_env() {
  local unset_args=() name
  for name in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH GCC_EXEC_PREFIX \
      COMPILER_PATH LIBRARY_PATH LD_LIBRARY_PATH CONDA_BUILD_SYSROOT _PYTHON_SYSCONFIGDATA_NAME \
      _CONDA_PYTHON_SYSCONFIGDATA_NAME PYTHONPATH PYTHONHOME $(compgen -e | grep -E '^(VLLM|PIP)_' || true); do
    unset_args+=(-u "$name")
  done
  env "${unset_args[@]}" PATH="$PREFIX/bin:$CUDA_HOME_DIR/bin:$PATH" CUDA_HOME="$CUDA_HOME_DIR" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$@"
}

# check_identity FILE DOTTED.KEY=VALUE...: compare fields of reference_identity.py's JSON.
check_identity() {
  "$PYTHON_BIN" - "$@" <<'PY' || die "identity mismatch in $1"
import json
import sys

identity = json.load(open(sys.argv[1], encoding="utf-8"))
mismatches = []
for check in sys.argv[2:]:
    key, expected = check.split("=", 1)
    value = identity
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value != expected:
        mismatches.append(f"{key} = {value!r}, expected {expected!r}")
for mismatch in mismatches:
    print(mismatch, file=sys.stderr)
sys.exit(1 if mismatches else 0)
PY
}

echo "==> Source: $VLLM_COMMIT + patches -> $SRC"
git init -q "$SRC"
git -C "$SRC" fetch -q --depth 1 "$VLLM_REPO_URL" "$VLLM_COMMIT"
git -C "$SRC" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$VLLM_COMMIT" ]] || die "fetched HEAD is not $VLLM_COMMIT"
for patch in "$SCRIPT_DIR"/patches/*.patch; do
  git -C "$SRC" apply --check "$patch"
  git -C "$SRC" apply "$patch"
done
"$PYTHON_BIN" -B "$PROBE" --package-dir "$SRC/vllm" --output "$PREFIX/src/source-identity.json"
check_identity "$PREFIX/src/source-identity.json" "git.head=$VLLM_COMMIT" "git.diff_sha256=$SOURCE_DIFF_SHA256"

echo "==> Precompiled extensions: $VLLM_WHEEL_NAME"
if [[ -n "$WHEEL_DIR" && -f "$WHEEL_DIR/$VLLM_WHEEL_NAME" ]]; then
  cp "$WHEEL_DIR/$VLLM_WHEEL_NAME" "$WHEEL"
else
  curl -fsSL --retry 3 -o "$WHEEL.part" "$VLLM_WHEEL_URL"
  mv "$WHEEL.part" "$WHEEL"
fi
echo "$VLLM_WHEEL_SHA256  $WHEEL" | sha256sum -c --quiet - || die "$WHEEL does not match sha256 $VLLM_WHEEL_SHA256"

echo "==> venv and locked dependencies"
"$PYTHON_BIN" -m venv "$PREFIX"
index_args=(--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130)
if [[ -n "$WHEEL_DIR" ]]; then
  index_args=(--no-index --find-links "$WHEEL_DIR")
fi
clean_env "$PREFIX/bin/python" -m pip install --require-hashes --no-deps "${index_args[@]}" -r "$LOCK"

echo "==> vLLM (editable, precompiled extensions)"
(
  cd "$SRC"
  # PYTHONDONTWRITEBYTECODE: vLLM's setup.py imports vllm/envs.py, which would
  # otherwise leave a __pycache__ in the source tree that the reference lacks.
  clean_env VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
    SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" MAX_JOBS=8 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=10.0 \
    TRITON_CUDACRT_PATH="$CUDA_HOME_DIR/include" TRITON_CUDART_PATH="$CUDA_HOME_DIR/include" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PREFIX/bin/python" -m pip install --no-deps --no-build-isolation -e "$SRC"
)

cat > "$PREFIX/vllm-reference.env" <<EOF
# Environment of the vLLM reference process: pass this file as VLLM_ENV_SETUP
# (scripts/serve/vllm_server.sh sources it for VLLM_ENV_NAME=$PREFIX).
# CUDA toolkit the reference was built with; FlashInfer JIT-compiles with it.
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CFLAGS CXXFLAGS CPPFLAGS LDFLAGS \\
  PYTHONPATH PYTHONHOME NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CUDA_HOME_DIR" CUDA_PATH="$CUDA_HOME_DIR" CUDA_NVCC_EXECUTABLE="$CUDA_HOME_DIR/bin/nvcc"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME_DIR/bin/ptxas"
export PATH="$CUDA_HOME_DIR/bin:\$PATH"
EOF

echo "==> Identity"
clean_env PYTHONDONTWRITEBYTECODE=1 "$PREFIX/bin/python" -B "$PROBE" --output "$PREFIX/reference-identity.json"
check_identity "$PREFIX/reference-identity.json" "vllm.version=$VLLM_VERSION" "vllm.git.head=$VLLM_COMMIT" \
  "vllm.git.diff_sha256=$SOURCE_DIFF_SHA256" "vllm.tree_sha256=$PACKAGE_TREE_SHA256" \
  "torch.version=$TORCH_VERSION" "flashinfer.version=$FLASHINFER_VERSION"

echo "Reference ready: VLLM_ENV_NAME=$PREFIX VLLM_ENV_SETUP=$PREFIX/vllm-reference.env"

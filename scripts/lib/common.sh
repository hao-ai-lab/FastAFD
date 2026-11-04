#!/usr/bin/env bash

fastafd_repo_root() {
  local start="${1:-.}"
  local dir

  if [[ -f "$start" ]]; then
    dir="$(cd "$(dirname "$start")" && pwd)"
  else
    dir="$(cd "$start" && pwd)"
  fi

  local git_root
  if git_root="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)"; then
    printf '%s\n' "$git_root"
    return 0
  fi

  while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/scripts" && -d "$dir/python" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  echo "Failed to locate FastAFD repository root from: $start" >&2
  return 1
}

fastafd_init_paths() {
  REPO_ROOT="$(fastafd_repo_root "${1:-.}")"
  CALIB_DIR="$REPO_ROOT"
  SCRIPT_ROOT="$REPO_ROOT/scripts"
  export REPO_ROOT CALIB_DIR SCRIPT_ROOT
}

# Resolve a Python environment given by path. A value containing '/' (conda
# env names cannot) must be a directory with an executable bin/python, such as
# a venv: print its absolute prefix. Print nothing for a bare name, which callers
# treat as a conda env name. Fail for a path that is not such a directory.
fastafd_python_env_prefix() {
  local env="$1"
  if [[ "$env" != */* ]]; then
    return 0
  fi
  if [[ ! -d "$env" || ! -x "$env/bin/python" ]]; then
    echo "Not a Python environment (need a directory with an executable bin/python): $env" >&2
    return 1
  fi
  (cd "$env" && pwd)
}

# Print, one word per line, the command prefix that runs a program in the vLLM
# env VLLM_ENV: nothing for a path (already resolved by
# fastafd_python_env_prefix) or for the active env ENV_NAME itself, otherwise
# `conda run --no-capture-output -n VLLM_ENV`.
fastafd_vllm_env_launcher() {
  local env_name="$1" vllm_env="$2"
  if [[ "$vllm_env" != */* && "$vllm_env" != "$env_name" ]]; then
    printf '%s\n' conda run --no-capture-output -n "$vllm_env"
  fi
}

# Print the vLLM env a preset scores with when its caller left VLLM_ENV_NAME
# unset: FASTAFD_VLLM_REFERENCE, the install prefix of the README's reproducible
# vLLM reference. Without it, fall back to the historical conda env name
# vllm-cuda130 and warn on stderr that this is not that reference.
fastafd_default_vllm_env() {
  if [[ -z "${FASTAFD_VLLM_REFERENCE:-}" ]]; then
    echo "WARNING: VLLM_ENV_NAME and FASTAFD_VLLM_REFERENCE are unset; scoring in conda env 'vllm-cuda130', which is not the reproducible vLLM reference (README.md, Reproducible vLLM Reference)." >&2
    printf '%s\n' vllm-cuda130
    return 0
  fi
  if [[ "$FASTAFD_VLLM_REFERENCE" != */* ]]; then
    echo "FASTAFD_VLLM_REFERENCE must be the install prefix of the vLLM reference, a path, got: '$FASTAFD_VLLM_REFERENCE'. To score in a conda env, set VLLM_ENV_NAME to its name instead." >&2
    return 1
  fi
  printf '%s\n' "$FASTAFD_VLLM_REFERENCE"
}

# When a preset scores in the FASTAFD_VLLM_REFERENCE prefix (VLLM_ENV_NAME, the
# first argument, names it) and the caller set no VLLM_ENV_SETUP, export the
# vllm-reference.env that build_vllm_reference.sh wrote there: without it the
# reference would run with this shell's CUDA toolkit. A missing file leaves
# VLLM_ENV_SETUP unset, with a note on stderr. Conda env names are untouched.
fastafd_default_vllm_env_setup() {
  local vllm_env="$1" prefix
  if [[ -n "${VLLM_ENV_SETUP:-}" || -z "${FASTAFD_VLLM_REFERENCE:-}" ||
        "$vllm_env" != "$FASTAFD_VLLM_REFERENCE" || "$vllm_env" != */* ]]; then
    return 0
  fi
  # Absolute, because the aligner changes directory before vLLM sources it.
  if prefix="$(cd "$FASTAFD_VLLM_REFERENCE" 2>/dev/null && pwd)" && [[ -f "$prefix/vllm-reference.env" ]]; then
    export VLLM_ENV_SETUP="$prefix/vllm-reference.env"
    return 0
  fi
  echo "NOTE: no vllm-reference.env in FASTAFD_VLLM_REFERENCE=$FASTAFD_VLLM_REFERENCE; VLLM_ENV_SETUP stays unset and the vLLM reference runs with this shell's CUDA toolkit." >&2
}

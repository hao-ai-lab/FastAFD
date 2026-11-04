#!/usr/bin/env bash
# Kill AFD and vLLM processes by command-line pattern on EVERY live Ray node.
#
# One Ray task per alive node in ray.nodes() SIGKILLs each process whose command
# line contains one of the patterns below (or the patterns given as arguments).
# It does not check which job, run or user started a process, so on a shared Ray
# cluster it also kills other people's jobs. Run it only on a Ray cluster you have
# to yourself. The experiment presets call it only with CLEAN_FIRST=1 or
# CLEAN_AFTER=1 (both default 0), and before their vLLM stage when
# RUN_VLLM_ALIGNMENT=1.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_PREFIX_DIR="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    CONDA_SH="$CONDA_PREFIX_DIR/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
  else
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
fi
CONDA_ENV_NAME="${CONDA_ENV_NAME:-minisgl-cuda130}"

if [[ $# -gt 0 ]]; then
  PATTERNS=("$@")
else
  PATTERNS=(
    "python -m minisgl --mode afd-serve"
    "compare_minisgl_vllm.py sample"
    "compare_minisgl_vllm.py score"
    "score_vllm_sharded_alignment.py"
    "vllm serve"
    "ray::AfdAttentionWorker"
    "ray::AfdExpertWorker"
    "ray::AfdDummyWorker"
    "ray::_VllmServerActor"
  )
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  echo "Set CONDA_SH=/path/to/conda.sh and retry." >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
set -u

cd "${REPO_ROOT}"

python - "${PATTERNS[@]}" <<'PY'
import os
import signal
import subprocess
import sys

import ray


patterns = sys.argv[1:]
ray.init(address="auto", ignore_reinit_error=True)


@ray.remote(num_cpus=0)
def cleanup(proc_patterns: list[str]):
    out = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    self_pid = os.getpid()
    parent_pid = os.getppid()
    killed: list[tuple[int, str]] = []
    for line in out:
        line = line.strip()
        if not line:
            continue
        pid_str, args = line.split(None, 1)
        pid = int(pid_str)
        if pid in {self_pid, parent_pid}:
            continue
        if (
            "python - " in args
            or "ps -eo pid=,args=" in args
            or "kill_afd_ray_tasks.sh" in args
        ):
            continue
        if any(pattern in args for pattern in proc_patterns):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append((pid, args))
            except ProcessLookupError:
                pass
    gpu_apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "node": os.uname().nodename,
        "killed": killed,
        "gpu_apps": gpu_apps,
    }


node_resource_keys: list[str] = []
for node in ray.nodes():
    if not node.get("Alive", False):
        continue
    for resource_name in node.get("Resources", {}):
        if resource_name.startswith("node:"):
            node_resource_keys.append(str(resource_name))
            break

results = ray.get(
    [
        cleanup.options(resources={resource_key: 0.01}).remote(patterns)
        for resource_key in node_resource_keys
    ]
)
seen: set[str] = set()
for item in results:
    node = item["node"]
    if node in seen:
        continue
    seen.add(node)
    print(node)
    print(f"killed: {len(item['killed'])}")
    for pid, args in item["killed"]:
        print(f"  {pid} {args}")
    print("gpu_apps:")
    print(item["gpu_apps"] or "<none>")
    print("---")
PY

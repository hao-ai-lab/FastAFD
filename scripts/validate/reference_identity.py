#!/usr/bin/env python3
"""Record which vLLM, torch and FlashInfer a reference process actually imports.

A version string does not identify a vLLM reference: an editable checkout keeps
reporting the version generated when its extensions were built, while its Python
source can be a later commit plus uncommitted edits. `scripts/serve/vllm_server.sh
--identity-json FILE` runs this file with the reference environment's interpreter,
right before it starts vLLM, and the alignment driver copies the result into the
report as `reference_identity`.

Recorded for vLLM: `__version__`, `__file__`, and the package source --
`git.head` plus, for a dirty checkout, the sha256 of
`git diff --full-index --binary HEAD` (independent of clone depth and diff
settings); `tree_sha256`, the sha256 of `sha256sum`-style lines
`<sha256>  <path relative to the package>` over every regular file of the package
except `__pycache__/`, `*.pyc` and `*.so`, sorted by path (this also identifies a
source tree that is not a git checkout); and the sha256 of every compiled
extension (`*.so`) inside the package. Also recorded: torch and FlashInfer versions
and locations, and `CUDA_HOME` (the toolkit FlashInfer JIT-compiles with).

The probe only reads files. It never raises for a missing module or a failed git
call; what it cannot determine is recorded as an `error` string, so the caller
can still start vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
# Fixed flags so the digest does not depend on the user's diff configuration
# (prefixes, context, colour, external drivers) or the clone's abbrev length.
GIT_DIFF_ARGS = (
    "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--full-index", "--binary",
    "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", "-U3", "HEAD", "--",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(package_dir: Path) -> dict[str, Any]:
    """Digest a package tree: source files as one tree hash, extensions one by one."""
    tree_lines = []
    extensions = []
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(name for name in dirs if name != "__pycache__")
        for name in sorted(files):
            path = Path(root) / name
            if path.is_symlink() or not path.is_file() or name.endswith(".pyc"):
                continue
            relative = path.relative_to(package_dir).as_posix()
            digest = sha256_file(path)
            if name.endswith(".so"):
                extensions.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
            else:
                tree_lines.append(f"{digest}  {relative}\n")
    tree_lines.sort(key=lambda line: line.split("  ", 1)[1])
    extensions.sort(key=lambda entry: entry["path"])
    return {
        "tree_sha256": hashlib.sha256("".join(tree_lines).encode("utf-8")).hexdigest(),
        "tree_files": len(tree_lines),
        "extensions": extensions,
    }


def _git(args: list[str] | tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, LC_ALL="C", GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(
        ["git", "--no-optional-locks", *args], cwd=cwd, env=env,
        capture_output=True, timeout=120, check=False,
    )


def _git_text(result: subprocess.CompletedProcess) -> str:
    return result.stdout.decode("utf-8", "replace").strip()


def git_source(package_dir: Path) -> dict[str, Any] | None:
    """Return HEAD and dirty-diff identity if the package is tracked by a git checkout.

    None means the package is not tracked by any enclosing repository (for example
    a source tree extracted from an archive, or one nested in an unrelated repo).
    """
    try:
        top = _git(["rev-parse", "--show-toplevel"], package_dir)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"git unavailable: {error}"}
    if top.returncode != 0:
        message = top.stderr.decode("utf-8", "replace").strip()
        if "not a git repository" in message:
            return None
        return {"error": message.splitlines()[-1] if message else f"git exit {top.returncode}"}
    toplevel = Path(_git_text(top))
    marker = (package_dir / "__init__.py").resolve()
    try:
        relative = marker.relative_to(toplevel.resolve()).as_posix()
    except ValueError:
        return None
    if _git(["ls-files", "--error-unmatch", "--", relative], toplevel).returncode != 0:
        return None
    head = _git(["rev-parse", "HEAD"], toplevel)
    diff = _git(GIT_DIFF_ARGS, toplevel)
    names = _git(["diff", "--name-only", "--no-renames", "HEAD", "--"], toplevel)
    if head.returncode != 0 or diff.returncode != 0 or names.returncode != 0:
        failed = next(r for r in (head, diff, names) if r.returncode != 0)
        return {"toplevel": str(toplevel), "error": failed.stderr.decode("utf-8", "replace").strip()}
    return {
        "toplevel": str(toplevel),
        "head": _git_text(head),
        "dirty": bool(diff.stdout),
        "diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.stdout else None,
        "diff_files": _git_text(names).splitlines(),
        "diff_command": "git " + " ".join(GIT_DIFF_ARGS),
    }


def module_identity(name: str) -> tuple[dict[str, Any], Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # a broken optional import must not stop vLLM from starting
        return {"error": f"{type(error).__name__}: {error}"}, None
    return {"version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None)}, module


def collect() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "python": {"executable": sys.executable, "prefix": sys.prefix, "version": sys.version.split()[0]},
        "cuda_home": os.environ.get("CUDA_HOME"),
    }
    vllm, _ = module_identity("vllm")
    if vllm.get("file"):
        package_dir = Path(vllm["file"]).resolve().parent
        vllm["git"] = git_source(package_dir)
        try:
            vllm.update(package_files(package_dir))
        except OSError as error:
            vllm["tree_error"] = f"{type(error).__name__}: {error}"
    identity["vllm"] = vllm
    torch, torch_module = module_identity("torch")
    if torch_module is not None:
        torch["cuda"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    identity["torch"] = torch
    identity["flashinfer"], _ = module_identity("flashinfer")
    identity["flashinfer_cubin"], _ = module_identity("flashinfer_cubin")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write.")
    parser.add_argument(
        "--package-dir", type=Path,
        help="Only digest the vLLM package source in this directory (git and tree); import nothing.",
    )
    args = parser.parse_args()
    if args.package_dir is not None:
        package_dir = args.package_dir.resolve()
        source = {"package_dir": str(package_dir), "git": git_source(package_dir), **package_files(package_dir)}
        args.output.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        print(f"source: git={source['git']} tree_sha256={source['tree_sha256']}", flush=True)
        return 0
    identity = collect()
    args.output.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    vllm = identity["vllm"]
    git = vllm.get("git") or {}
    print(
        "reference_identity: "
        f"vllm={vllm.get('version')} file={vllm.get('file')} git_head={git.get('head')} "
        f"diff_sha256={git.get('diff_sha256')} tree_sha256={vllm.get('tree_sha256')} "
        f"torch={identity['torch'].get('version')} flashinfer={identity['flashinfer'].get('version')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

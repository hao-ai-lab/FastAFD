"""Every JIT entry point a Python wrapper binds (load_jit cuda_wrappers: ("launch_x",
"Kernel::method")) names a static method its CUDA source defines, so a renamed or
missing entry fails here rather than at the first GPU compile."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import minisgl.core

_KERNEL = Path(minisgl.core.__file__).resolve().parent / "kernel"
_MODULES = ("moe_topk.py", "minimax_qk_norm.py")


def _load_jit_calls(path: Path):
    """(cuda_files, [(python name, "Kernel::method")]) of every load_jit(...) call."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_jit":
            kw = {k.arg: k.value for k in node.keywords}
            files = [ast.literal_eval(e) for e in kw["cuda_files"].elts]
            wrappers = [tuple(ast.literal_eval(e)) for e in kw["cuda_wrappers"].elts]
            yield files, wrappers


@pytest.mark.parametrize("module", _MODULES)
def test_wrapped_methods_exist_in_their_cuda_sources(module):
    calls = list(_load_jit_calls(_KERNEL / module))
    assert calls, f"{module}: no load_jit call found"
    for files, wrappers in calls:
        source = "\n".join((_KERNEL / "csrc" / "jit" / f).read_text() for f in files)
        for _, target in wrappers:
            kernel, method = target.split("::")
            assert re.search(rf"\bstruct {kernel}\b|\bclass {kernel}\b", source), f"{module}: {kernel} not in {files}"
            assert re.search(rf"\bstatic\b[^;{{]*\b{method}\s*\(", source), f"{module}: {target} not in {files}"

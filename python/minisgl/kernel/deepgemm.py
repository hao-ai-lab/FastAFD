from __future__ import annotations

import functools
import fcntl
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from .fp8_quant import per_token_cast_to_fp8 as _per_token_cast_to_fp8
from .build_utils import (
    nvcc_build_env as _build_nvcc_env,
    resolve_cuda_lib_dir as _resolve_cuda_lib_dir,
    resolve_nvcc as _resolve_nvcc,
)


_EXT_NAME = "_minisgl_deepgemm"


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepgemm"


def _library_root() -> Path:
    return _source_root() / "deep_gemm"


def _source_files(root: Path) -> list[Path]:
    return [root / "csrc" / "python_api.cpp"]


def _latest_source_mtime(root: Path) -> float:
    files: list[Path] = []
    files.extend(_source_files(root))
    for subdir in (
        root / "csrc",
        root / "deep_gemm" / "include",
        root / "third-party" / "cutlass" / "include",
        root / "third-party" / "fmt" / "include",
    ):
        files.extend(path for path in subdir.rglob("*") if path.is_file())
    return max(path.stat().st_mtime for path in files)


def _build_dir() -> Path:
    return Path(
        os.environ.get(
            "MINISGL_DEEPGEMM_BUILD_DIR",
            Path.home() / ".cache" / "minisgl" / "deepgemm",
        )
    )


def _built_so_path(build_dir: Path) -> Path | None:
    candidates = sorted((build_dir / "lib").glob(f"{_EXT_NAME}*.so"))
    return candidates[0] if candidates else None


@contextmanager
def _extension_build_lock(build_dir: Path):
    lock_path = build_dir / "build.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_setup_py(setup_path: Path, root: Path, cuda_home: Path, cuda_lib_dir: Path) -> None:
    sources = [str(path) for path in _source_files(root)]
    include_dirs = [
        str(cuda_home / "include"),
        str(cuda_home / "include" / "cccl"),
        str(root / "csrc"),
        str(root / "deep_gemm" / "include"),
        str(root / "third-party" / "cutlass" / "include"),
        str(root / "third-party" / "fmt" / "include"),
    ]
    cxx11_abi = int(torch.compiled_with_cxx11_abi())
    setup_path.write_text(
        f"""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = {sources!r}
include_dirs = {include_dirs!r}
library_dirs = {[str(cuda_lib_dir)]!r}
cxx_flags = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
]

setuptools.setup(
    name={_EXT_NAME!r},
    ext_modules=[
        CUDAExtension(
            name={_EXT_NAME!r},
            sources=sources,
            include_dirs=include_dirs,
            libraries=["cudart", "nvrtc"],
            library_dirs=library_dirs,
            extra_compile_args={{"cxx": cxx_flags}},
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
""",
        encoding="utf-8",
    )


def _find_cuda_home(nvcc: str) -> str:
    if cuda_home := os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return cuda_home
    return str(Path(nvcc).resolve().parent.parent)


def _build_extension_path(*, force_rebuild: bool = False) -> Path:
    root = _source_root()
    if not root.exists():
        raise RuntimeError(f"DeepGEMM source tree is missing: {root}")
    build_dir = _build_dir()
    lib_dir = build_dir / "lib"
    temp_root = build_dir / "temp"
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    setup_path = build_dir / "setup.py"

    existing = _built_so_path(build_dir)
    if (
        not force_rebuild
        and existing is not None
        and existing.stat().st_mtime >= _latest_source_mtime(root)
    ):
        return existing

    nvcc = _resolve_nvcc()
    cuda_home = Path(_find_cuda_home(nvcc))
    cuda_lib_dir = _resolve_cuda_lib_dir(nvcc)
    with _extension_build_lock(build_dir):
        existing = _built_so_path(build_dir)
        if (
            not force_rebuild
            and existing is not None
            and existing.stat().st_mtime >= _latest_source_mtime(root)
        ):
            return existing

        _write_setup_py(setup_path, root, cuda_home, cuda_lib_dir)

        for old_so in lib_dir.glob(f"{_EXT_NAME}*.so"):
            old_so.unlink()

        env = _build_nvcc_env(nvcc)
        env.setdefault("MAX_JOBS", os.environ.get("MAX_JOBS", "16"))
        with tempfile.TemporaryDirectory(
            dir=temp_root,
            prefix=f"build-{os.getpid()}-",
            ignore_cleanup_errors=True,
        ) as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(setup_path),
                    "build_ext",
                    "--build-lib",
                    str(lib_dir),
                    "--build-temp",
                    str(temp_dir),
                ],
                cwd=str(build_dir),
                env=env,
                check=True,
            )
        built = _built_so_path(build_dir)
        if built is None:
            raise RuntimeError(f"DeepGEMM build did not produce {_EXT_NAME}*.so under {lib_dir}")
        return built


@functools.cache
def _load_extension() -> ModuleType:
    import importlib.util

    try:
        so_path = _build_extension_path()
    except Exception:
        so_path = _build_extension_path(force_rebuild=True)
    module = sys.modules.get(_EXT_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load DeepGEMM extension from {so_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_EXT_NAME] = module
        spec.loader.exec_module(module)

    cuda_home = _find_cuda_home(_resolve_nvcc())
    module.init(str(_library_root()), cuda_home)
    return module


def ensure_deepgemm_built(*, force_rebuild: bool = False) -> Path:
    return _build_extension_path(force_rebuild=force_rebuild)


def __getattr__(name: str) -> Any:
    return getattr(_load_extension(), name)


def ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    if x.dtype != torch.float or x.size(-1) % 4 != 0:
        raise RuntimeError(f"UE8M0 packing expects float scales with last dim divisible by 4, got {x.shape} {x.dtype}")
    if validate and not bool((x.view(torch.int) & ((1 << 23) - 1) == 0).all().item()):
        raise RuntimeError("UE8M0 packing expects exponent-only float scales")
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _per_token_cast_to_fp8(
        x,
        use_ue8m0=use_ue8m0,
        gran_k=gran_k,
        use_packed_ue8m0=use_packed_ue8m0,
    )


def per_block_cast_to_fp8(
    x: torch.Tensor,
    *,
    use_ue8m0: bool,
    gran_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise RuntimeError(f"per_block_cast_to_fp8 expects 2D tensor, got {x.shape}")
    m, n = x.shape
    x_padded = torch.zeros((align(m, gran_k), align(n, gran_k)), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def set_mk_alignment_for_contiguous_layout(alignment: int) -> None:
    _load_extension().set_mk_alignment_for_contiguous_layout(int(alignment))


def get_mk_alignment_for_contiguous_layout() -> int:
    return int(_load_extension().get_mk_alignment_for_contiguous_layout())


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m: int | None = None) -> int:
    ext = _load_extension()
    if expected_m is None:
        return int(ext.get_theoretical_mk_alignment_for_contiguous_layout())
    return int(ext.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m)))


def transform_sf_into_required_layout(*args, **kwargs):
    return _load_extension().transform_sf_into_required_layout(*args, **kwargs)


def fp8_gemm_nt(*args, **kwargs):
    return _load_extension().fp8_gemm_nt(*args, **kwargs)


def m_grouped_bf16_gemm_nt_contiguous(*args, **kwargs):
    return _load_extension().m_grouped_bf16_gemm_nt_contiguous(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    return _load_extension().m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs)


def m_grouped_bf16_gemm_nt_masked(*args, **kwargs):
    return _load_extension().m_grouped_bf16_gemm_nt_masked(*args, **kwargs)


def m_grouped_fp8_gemm_nt_masked(*args, **kwargs):
    return _load_extension().m_grouped_fp8_gemm_nt_masked(*args, **kwargs)


def fp8_mqa_logits(*args, **kwargs):
    """``logits[i, j] = kv_sf[j] * sum_h weights[i, h] * relu(q[i, h] . kv[j])`` for
    ``cu_seq_len_k_start[i] <= j < cu_seq_len_k_end[i]`` (-inf elsewhere with
    ``clean_logits``): the lightning-indexer score. ``q [n, 32 | 64, d]`` e4m3,
    ``kv = (e4m3 [nk, d], fp32 [nk])``, ``weights [n, heads]`` fp32."""
    return _load_extension().fp8_mqa_logits(*args, **kwargs)


bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked


__all__ = [
    "ensure_deepgemm_built",
    "ceil_div",
    "align",
    "ceil_to_ue8m0",
    "pack_ue8m0_to_int",
    "per_token_cast_to_fp8",
    "per_block_cast_to_fp8",
    "set_mk_alignment_for_contiguous_layout",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "transform_sf_into_required_layout",
    "fp8_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "m_grouped_fp8_gemm_nt_masked",
    "fp8_mqa_logits",
    "bf16_m_grouped_gemm_nt_masked",
    "fp8_m_grouped_gemm_nt_masked",
]

from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch
import torch.distributed as dist

from .build_utils import nvcc_build_env as _build_nvcc_env, resolve_nvcc as _resolve_nvcc


_EXT_NAME = "_minisgl_deepep_moe"


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "deepep"


def _source_files(root: Path) -> list[Path]:
    return [
        root / "csrc" / "python_api.cpp",
        root / "csrc" / "kernels" / "backend" / "nccl.cu",
        root / "csrc" / "kernels" / "backend" / "cuda_driver.cu",
    ]


def _resolve_nccl_layout() -> tuple[Path, Path]:
    pkg_root = Path(distribution("nvidia-nccl-cu13").locate_file(""))
    nccl_root = pkg_root / "nvidia" / "nccl"
    include_dir = nccl_root / "include"
    lib_dir = nccl_root / "lib"
    host_lib = lib_dir / "libnccl.so.2"
    if not include_dir.exists():
        raise RuntimeError(f"Missing NCCL include directory: {include_dir}")
    if not host_lib.exists():
        raise RuntimeError(f"Missing NCCL host library: {host_lib}")
    return include_dir, lib_dir


def _latest_source_mtime(root: Path) -> float:
    files = _source_files(root)
    files.extend((root / "csrc").rglob("*.hpp"))
    files.extend((root / "csrc").rglob("*.cuh"))
    files.extend((root / "deep_ep" / "include").rglob("*.cuh"))
    return max(path.stat().st_mtime for path in files)


def _build_signature(root: Path) -> str:
    return "\n".join(
        [
            f"nccl={distribution('nvidia-nccl-cu13').version}",
            f"source_mtime={_latest_source_mtime(root)}",
        ]
    )


def _build_dir() -> Path:
    return Path(os.environ.get("MINISGL_DEEPEP_BUILD_DIR", Path.home() / ".cache" / "minisgl" / "deepep_moe"))


def _built_so_path(build_dir: Path) -> Path | None:
    candidates = sorted((build_dir / "lib").glob(f"{_EXT_NAME}*.so"))
    return candidates[0] if candidates else None


@contextmanager
def _extension_build_lock(build_dir: Path):
    import fcntl

    lock_path = build_dir / "build.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_setup_py(
    setup_path: Path,
    root: Path,
    *,
    nccl_include_dir: Path,
    nccl_lib_dir: Path,
) -> None:
    sources = [str(path) for path in _source_files(root)]
    include_dirs = [
        str(root / "deep_ep" / "include"),
        str(root / "csrc"),
        str(nccl_include_dir),
    ]
    setup_path.write_text(
        f"""
import os
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = {sources!r}
include_dirs = {include_dirs!r}
nccl_lib_dir = {str(nccl_lib_dir)!r}

cxx_flags = [
    "-O3",
    "-Wno-deprecated-declarations",
    "-Wno-unused-variable",
    "-Wno-sign-compare",
    "-Wno-reorder",
    "-Wno-attributes",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]
nvcc_flags = [
    "-O3",
    "-Xcompiler",
    "-O3",
    "--extended-lambda",
    "--diag-suppress=128,2417",
    "-rdc=true",
    "--ptxas-options=--register-usage-level=10",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
]

setuptools.setup(
    name={_EXT_NAME!r},
    ext_modules=[
        CUDAExtension(
            name={_EXT_NAME!r},
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=[nccl_lib_dir],
            extra_compile_args={{
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
                "nvcc_dlink": [
                    "-dlink",
                ],
            }},
            extra_link_args=[
                "-lcuda",
                "-l:libnccl.so.2",
                f"-Wl,-rpath,{{nccl_lib_dir}}",
            ],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
""",
        encoding="utf-8",
    )


def _build_extension_path(
    *,
    force_rebuild: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path:
    def _log(stage: str) -> None:
        if log is not None:
            log(f"build:{stage}")

    root = _source_root()
    if not root.exists():
        raise RuntimeError(f"DeepEP vendor source tree is missing: {root}")
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    signature_path = build_dir / "build.signature"
    signature = _build_signature(root)
    existing = _built_so_path(build_dir)
    if (
        not force_rebuild
        and existing is not None
        and signature_path.exists()
        and signature_path.read_text(encoding="utf-8") == signature
    ):
        _log(f"fast_path so_path={existing}")
        return existing

    _log("lock_wait")
    with _extension_build_lock(build_dir):
        _log("lock_acquired")
        lib_dir = build_dir / "lib"
        temp_root = build_dir / "temp"
        lib_dir.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        setup_path = build_dir / "setup.py"

        existing = _built_so_path(build_dir)
        if (
            not force_rebuild
            and existing is not None
            and signature_path.exists()
            and signature_path.read_text(encoding="utf-8") == signature
        ):
            _log(f"locked_fast_path so_path={existing}")
            return existing

        nccl_include_dir, nccl_lib_dir = _resolve_nccl_layout()
        _write_setup_py(
            setup_path,
            root,
            nccl_include_dir=nccl_include_dir,
            nccl_lib_dir=nccl_lib_dir,
        )

        nvcc = _resolve_nvcc()
        env = _build_nvcc_env(nvcc)
        major, minor = torch.cuda.get_device_capability()
        env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        env.setdefault("MAX_JOBS", os.environ.get("MAX_JOBS", "16"))

        for old_so in lib_dir.glob(f"{_EXT_NAME}*.so"):
            old_so.unlink()

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
                check=True,
                env=env,
            )
        _log("build_done")
        built = _built_so_path(build_dir)
        if built is None:
            raise RuntimeError(
                f"DeepEP extension build did not produce {_EXT_NAME}*.so under {lib_dir}"
            )
        signature_path.write_text(signature, encoding="utf-8")
        return built


def _cuda_home_from_nvcc(nvcc: str) -> Path:
    return Path(nvcc).resolve().parents[1]


def _init_deepep_jit(module: ModuleType) -> None:
    nvcc = _resolve_nvcc()
    _, nccl_lib_dir = _resolve_nccl_layout()
    module.init_jit(
        str(_source_root() / "deep_ep"),
        str(_cuda_home_from_nvcc(nvcc)),
        str(nccl_lib_dir.parent),
    )


def _load_extension_with_log(log: Callable[[str], None] | None = None) -> ModuleType:
    def _log(stage: str) -> None:
        if log is not None:
            log(f"deepep_extension:{stage}")

    _log("build_path_start")
    try:
        so_path = _build_extension_path(log=_log)
    except Exception as exc:
        _log(f"build_path_failed error={type(exc).__name__}:{exc}")
        _log("force_rebuild_start")
        so_path = _build_extension_path(force_rebuild=True, log=_log)
    _log(f"build_path_done so_path={so_path}")
    module = sys.modules.get(_EXT_NAME)
    if module is not None:
        _log("module_cache_hit")
        return module
    _log("spec_start")
    spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load DeepEP extension from {so_path}")
    _log("module_from_spec_start")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXT_NAME] = module
    _log("exec_module_start")
    spec.loader.exec_module(module)
    _log("exec_module_done")
    _log("init_jit_start")
    _init_deepep_jit(module)
    _log("init_jit_done")
    return module


@functools.cache
def _load_extension() -> ModuleType:
    return _load_extension_with_log()


def ensure_deepep_moe_built(*, force_rebuild: bool = False) -> Path:
    return _build_extension_path(force_rebuild=force_rebuild)


def deepep_topk_idx_dtype() -> torch.dtype:
    return _load_extension().topk_idx_t


def _group_rank(group: Any) -> int:
    if hasattr(group, "rank"):
        return int(group.rank())
    return int(dist.get_rank(group=group))


def _group_size(group: Any) -> int:
    if hasattr(group, "size"):
        return int(group.size())
    return int(dist.get_world_size(group=group))


def _group_barrier(group: Any) -> None:
    if hasattr(group, "barrier"):
        group.barrier()
    else:
        dist.barrier(group=group)


@dataclass
class _NCCLCommHandle:
    module: ModuleType
    nccl_comm: int
    managed: bool
    refcount: int = 1
    destroyed: bool = False

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        if self.managed:
            self.module.destroy_nccl_comm(int(self.nccl_comm))


_NCCL_COMM_CACHE: dict[int, _NCCLCommHandle] = {}


def _get_nccl_comm_handle(module: ModuleType, group: Any) -> _NCCLCommHandle:
    key = id(group)
    handle = _NCCL_COMM_CACHE.get(key)
    if handle is not None:
        handle.refcount += 1
        return handle

    rank = _group_rank(group)
    size = _group_size(group)
    unique_ids = [None] * size
    dist.all_gather_object(unique_ids, module.get_local_nccl_unique_id(), group=group)
    handle = _NCCLCommHandle(
        module=module,
        nccl_comm=int(module.create_nccl_comm(unique_ids[0], size, rank)),
        managed=True,
    )
    _NCCL_COMM_CACHE[key] = handle
    return handle


def _release_nccl_comm_handle(group: Any, handle: _NCCLCommHandle) -> None:
    key = id(group)
    cached = _NCCL_COMM_CACHE.get(key)
    if cached is not handle:
        return
    handle.refcount -= 1
    if handle.refcount <= 0:
        _NCCL_COMM_CACHE.pop(key, None)
        handle.destroy()


@dataclass(frozen=True)
class _ElasticRuntimeHandle:
    num_experts: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    token_metadata_at_forward: torch.Tensor | None
    channel_linked_list: torch.Tensor | None


@dataclass(frozen=True)
class DeepEPMoeElasticHandle:
    handle: Any
    recv_shape: tuple[int, ...]
    use_expanded_layout: bool = True


class DeepEPMoeElasticBuffer:
    """Minisgl-owned DeepEP V2 ElasticBuffer wrapper."""

    is_elastic = True

    def __init__(
        self,
        *,
        group: Any,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        num_allocated_qps: int = 0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        def _log(stage: str) -> None:
            if log is not None:
                log(f"deepep_elastic:{stage}")

        _log("load_extension_start")
        self._module = _load_extension_with_log(_log)
        _log("load_extension_done")
        self._group = group
        _log("group_info_start")
        self.rank_idx = _group_rank(group)
        self.num_ranks = _group_size(group)
        _log(f"group_info_done rank_idx={self.rank_idx} num_ranks={self.num_ranks}")
        self.requested_num_max_dispatch_tokens_per_rank = int(num_max_dispatch_tokens_per_rank)
        # DeepEP's runtime path can request one extra metadata slot over the
        # nominal per-rank token bucket. Keep the public bucket unchanged while
        # allocating the ElasticBuffer with the guard slot it needs.
        self.num_max_dispatch_tokens_per_rank = (
            self.requested_num_max_dispatch_tokens_per_rank + 1
        )
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        deterministic = False
        allow_hybrid_mode = True
        allow_multiple_reduction = True
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = True
        self.num_allocated_qps = int(num_allocated_qps) or (
            129 if allow_hybrid_mode else 17
        )
        sl_idx = int(os.environ.get("EP_OVERRIDE_RDMA_SL", "3"))
        _log(
            "config "
            f"requested_max_tokens={self.requested_num_max_dispatch_tokens_per_rank} "
            f"allocated_max_tokens={self.num_max_dispatch_tokens_per_rank} "
            f"hidden={self.hidden_size} num_experts={self.num_experts} top_k={self.top_k} "
            f"deterministic={deterministic} allow_hybrid={allow_hybrid_mode} "
            f"allow_multi_reduce={allow_multiple_reduction} "
            f"prefer_overlap={self.prefer_overlap_with_compute} "
            f"num_qps={self.num_allocated_qps} sl={sl_idx}"
        )
        _log("nccl_handle_start")
        self._nccl_handle = _get_nccl_comm_handle(self._module, group)
        _log(
            "nccl_handle_done "
            f"nccl_comm={int(self._nccl_handle.nccl_comm)} managed={self._nccl_handle.managed}"
        )
        _log("calculate_buffer_size_start")
        num_bytes = int(
            self._module.calculate_elastic_buffer_size(
                int(self._nccl_handle.nccl_comm),
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.top_k,
                False,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        )
        _log(f"calculate_buffer_size_done num_bytes={num_bytes}")
        _log("elastic_buffer_ctor_start")
        self._runtime = self._module.ElasticBuffer(
            self.rank_idx,
            self.num_ranks,
            int(self._nccl_handle.nccl_comm),
            num_bytes,
            deterministic,
            allow_hybrid_mode,
            allow_multiple_reduction,
            self.prefer_overlap_with_compute,
            sl_idx,
            self.num_allocated_qps,
            120,
            120,
            True,
        )
        _log("elastic_buffer_ctor_done")
        _log("domain_query_start")
        self.num_scaleout_ranks, self.num_scaleup_ranks = self._runtime.get_logical_domain_size()
        self.num_rdma_ranks, self.num_nvlink_ranks = self._runtime.get_physical_domain_size()
        _log(
            "domain_query_done "
            f"scaleout={self.num_scaleout_ranks} scaleup={self.num_scaleup_ranks} "
            f"rdma={self.num_rdma_ranks} nvlink={self.num_nvlink_ranks}"
        )
        _log("cuda_sync_after_ctor_start")
        torch.cuda.synchronize()
        _log("cuda_sync_after_ctor_done")
        _log("torch_group_barrier_start")
        _group_barrier(group)
        _log("torch_group_barrier_done")
        _log("runtime_barrier_start")
        self._runtime.barrier(True, False)
        _log("runtime_barrier_done")
        _log("cuda_sync_after_runtime_barrier_start")
        torch.cuda.synchronize()
        _log("cuda_sync_after_runtime_barrier_done")
        self.topk_idx_dtype = self._module.topk_idx_t
        self._destroyed = False

    @property
    def runtime(self) -> Any:
        return self._runtime

    @staticmethod
    def _is_cuda_graph_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _num_sms(self) -> int:
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if self.prefer_overlap_with_compute:
            return max(4, min(num_sms, 24))
        return max(4, min(num_sms, max(64, num_sms)))

    def _num_qps(self, num_sms: int) -> int:
        if self.allow_hybrid_mode:
            return min(num_sms * 16 + 1, self.num_allocated_qps)
        return min(num_sms + 1, self.num_allocated_qps)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        hidden_states_scale: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_max_dispatch_tokens_per_rank: int | None = None,
        do_cpu_sync: bool = False,
        use_expanded_layout: bool = True,
    ):
        if topk_ids.dtype != self.topk_idx_dtype:
            topk_ids = topk_ids.to(self.topk_idx_dtype)
        effective_max_tokens = (
            self.requested_num_max_dispatch_tokens_per_rank
            if num_max_dispatch_tokens_per_rank is None
            else int(num_max_dispatch_tokens_per_rank)
        )
        local_tokens = int(hidden_states.shape[0])
        if effective_max_tokens < local_tokens:
            raise RuntimeError(
                "DeepEP dispatch bucket is smaller than local token count: "
                f"bucket={effective_max_tokens} tokens={local_tokens}"
            )
        if effective_max_tokens > self.num_max_dispatch_tokens_per_rank:
            raise RuntimeError(
                "DeepEP dispatch bucket exceeds allocated runtime buffer: "
                f"bucket={effective_max_tokens} "
                f"allocated={self.num_max_dispatch_tokens_per_rank}"
            )
        # Decode graph capture/replay must avoid host synchronization, so the
        # graph path leaves this false and DeepEP allocates by worst-case bucket.
        # Prefill uses large buckets; enabling sync there lets DeepEP allocate
        # by actual receive counts and avoids multi-GiB dummy AG receive buffers.
        do_cpu_sync = bool(do_cpu_sync)
        num_sms = self._num_sms()
        num_qps = self._num_qps(num_sms)
        (
            recv_x,
            recv_sf,
            recv_topk_ids,
            recv_topk_weights,
            copied_topk_ids,
            _num_recv_tokens_per_expert_list,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            recv_src_metadata,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list,
            _event,
        ) = self._runtime.dispatch(
            hidden_states.contiguous(),
            hidden_states_scale,
            topk_ids.contiguous(),
            topk_weights.float().contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            effective_max_tokens,
            self.num_experts,
            int(expert_alignment),
            num_sms,
            num_qps,
            None,
            None,
            False,
            False,
            True,
            do_cpu_sync,
            bool(use_expanded_layout),
            True,
        )
        if hidden_states_scale is not None and recv_sf is None:
            raise RuntimeError("DeepEP FP8 dispatch did not return activation scales")
        runtime_handle = _ElasticRuntimeHandle(
            num_experts=self.num_experts,
            num_max_tokens_per_rank=effective_max_tokens,
            num_sms=num_sms,
            topk_idx=copied_topk_ids,
            psum_num_recv_tokens_per_scaleup_rank=psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert=psum_num_recv_tokens_per_expert,
            recv_src_metadata=recv_src_metadata,
            dst_buffer_slot_idx=dst_buffer_slot_idx,
            token_metadata_at_forward=token_metadata_at_forward,
            channel_linked_list=channel_linked_list,
        )
        wrapped = DeepEPMoeElasticHandle(
            handle=runtime_handle,
            recv_shape=tuple(int(x) for x in recv_x.shape),
            use_expanded_layout=bool(use_expanded_layout),
        )
        return (
            recv_x,
            recv_sf,
            recv_topk_ids,
            recv_topk_weights,
            wrapped,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor | None,
        handle: DeepEPMoeElasticHandle,
    ) -> torch.Tensor:
        runtime_handle = handle.handle
        if tuple(expert_output.shape) != tuple(handle.recv_shape):
            raise RuntimeError(
                "DeepEP combine expects expert output to match "
                f"recv_shape={handle.recv_shape}, got={tuple(expert_output.shape)}"
            )
        send = expert_output if expert_output.is_contiguous() else expert_output.contiguous()
        num_sms = int(runtime_handle.num_sms)
        num_qps = self._num_qps(num_sms)
        combined, _combined_topk_weights, _event = self._runtime.combine(
            send,
            topk_weights,
            None,
            None,
            runtime_handle.recv_src_metadata,
            runtime_handle.topk_idx,
            runtime_handle.psum_num_recv_tokens_per_scaleup_rank,
            runtime_handle.token_metadata_at_forward,
            runtime_handle.channel_linked_list,
            runtime_handle.num_experts,
            runtime_handle.num_max_tokens_per_rank,
            num_sms,
            num_qps,
            None,
            None,
            False,
            False,
            bool(handle.use_expanded_layout),
        )
        return combined

    def clean(self) -> None:
        return

    def reset_buffer_idx(self, idx: int = 0) -> None:
        del idx

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._runtime.destroy()
        _release_nccl_comm_handle(self._group, self._nccl_handle)


__all__ = [
    "DeepEPMoeElasticBuffer",
    "DeepEPMoeElasticHandle",
    "deepep_topk_idx_dtype",
    "ensure_deepep_moe_built",
]

from __future__ import annotations

import atexit
import os
import socket
from dataclasses import dataclass
from pathlib import Path
import threading
from collections.abc import Sequence
from typing import Any

import ray
from minisgl.ray_env_keys import AFD_WORKER_ENV_KEYS, AFD_WORKER_NODE_SCOPED_CACHE_KEYS
from minisgl.scheduler.config import SchedulerConfig
from minisgl.utils import div_ceil
from minisgl.utils import is_sm100_supported

PYTHON_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Ray runtime environment
# ---------------------------------------------------------------------------

_NODE_SCOPED_CACHE_ENV_KEYS = AFD_WORKER_NODE_SCOPED_CACHE_KEYS
_RUNTIME_ENV_KEYS = AFD_WORKER_ENV_KEYS


def _sanitize_node_tag(node_ip: str) -> str:
    return node_ip.replace(":", "_").replace(".", "_")


def _scope_cache_env(
    env_vars: dict[str, str],
    node_ip: str,
    *,
    rank_tag: str | None = None,
) -> None:
    tag = _sanitize_node_tag(node_ip)
    for key in _NODE_SCOPED_CACHE_ENV_KEYS:
        base_dir = env_vars.get(key)
        if base_dir:
            scoped = os.path.join(base_dir, f"ray-node-{tag}")
            if rank_tag:
                scoped = os.path.join(scoped, rank_tag)
            env_vars[key] = scoped


# ---------------------------------------------------------------------------
# Runtime config and sizing records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AfdRuntimeSizing:
    requested_graph_bs: tuple[int, ...]
    decode_graph_bs: tuple[int, ...]
    max_graph_bs: int
    max_running_req: int
    num_page_override: int | None
    max_comm_tokens: int
    num_mb: int
    # The per-MB size appended because the requested sizes (an explicit list,
    # or --cuda-graph-max-bs) stop below a lane at capacity; None otherwise.
    added_graph_bs: int | None = None


@dataclass(frozen=True)
class AfdDpNodePlacement:
    attn_nodes: list[str]
    mlp_nodes: list[str]
    attn_rank_nodes: tuple[tuple[str, ...], ...]
    mlp_rank_nodes: tuple[tuple[str, ...], ...]
    mode: str
    node_gpu_counts: dict[str, int]


@dataclass(frozen=True)
class AfdRuntimeConfig(SchedulerConfig):
    distributed_host: str = "127.0.0.1"
    distributed_port: int = 2333
    distributed_rank: int | None = None
    distributed_world_size: int | None = None
    distributed_tp_groups: tuple[tuple[int, ...], ...] | None = None
    ray_nsys: bool = False
    afd_num_mb: int = 1
    afd_attn_dp_size: int = 1
    afd_mlp_dp_size: int = 1
    afd_attn_tp_size: int = 1
    afd_mlp_tp_size: int = 1
    afd_moe_a2a_backend: str = "none"
    afd_moe_runner_backend: str = "auto"

    @property
    def distributed_addr(self) -> str:
        return f"tcp://{self.distributed_host}:{self.distributed_port}"


# ---------------------------------------------------------------------------
# Buffered logging
# ---------------------------------------------------------------------------

_LOG_FLUSH_LINES = 0
_LOG_FLUSH_BYTES = 0
_LOG_WRITERS: dict[str, "_BufferedLogWriter"] = {}
_LOG_WRITERS_LOCK = threading.Lock()


class _BufferedLogWriter:
    def __init__(self, log_path: str) -> None:
        self.log_path = str(log_path)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.log_path, "a", encoding="utf-8")
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._lock = threading.Lock()

    def write_line(self, message: str, *, flush: bool = False) -> None:
        line = message + "\n"
        with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line)
            should_flush = flush
            if _LOG_FLUSH_LINES > 0 and len(self._buffer) >= _LOG_FLUSH_LINES:
                should_flush = True
            if _LOG_FLUSH_BYTES > 0 and self._buffer_bytes >= _LOG_FLUSH_BYTES:
                should_flush = True
            if should_flush:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._handle.close()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        self._handle.write("".join(self._buffer))
        self._handle.flush()
        self._buffer.clear()
        self._buffer_bytes = 0


def _get_log_writer(log_path: str) -> _BufferedLogWriter:
    writer = _LOG_WRITERS.get(log_path)
    if writer is not None:
        return writer
    with _LOG_WRITERS_LOCK:
        writer = _LOG_WRITERS.get(log_path)
        if writer is None:
            writer = _BufferedLogWriter(log_path)
            _LOG_WRITERS[log_path] = writer
        return writer


def flush_log_lines(log_path: str | None = None) -> None:
    if log_path is not None:
        writer = _LOG_WRITERS.get(log_path)
        if writer is not None:
            writer.flush()
        return
    with _LOG_WRITERS_LOCK:
        writers = list(_LOG_WRITERS.values())
    for writer in writers:
        writer.flush()


def close_log_lines() -> None:
    with _LOG_WRITERS_LOCK:
        writers = list(_LOG_WRITERS.items())
        _LOG_WRITERS.clear()
    for _, writer in writers:
        try:
            writer.close()
        except Exception:
            pass


@atexit.register
def _close_log_lines_atexit() -> None:
    close_log_lines()


def log_line(log_path: str, message: str, *, flush: bool = False) -> None:
    _get_log_writer(log_path).write_line(message, flush=flush)


# ---------------------------------------------------------------------------
# Ray runtime env and placement helpers
# ---------------------------------------------------------------------------

def runtime_env(
    *,
    nsight: dict[str, str] | None = None,
    node_scoped_cache: bool = False,
    node_ip: str | None = None,
    cache_rank_tag: str | None = None,
) -> dict[str, Any]:
    env_vars: dict[str, str] = {"PYTHONPATH": str(PYTHON_DIR)}
    prepend = os.environ.get("MINISGL_WORKER_PYTHONPATH_PREPEND")   # unset = unchanged
    if prepend:
        env_vars["PYTHONPATH"] = prepend + os.pathsep + str(PYTHON_DIR)
    for key in _RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    if node_scoped_cache:
        if not node_ip:
            raise RuntimeError("node_ip is required when node_scoped_cache is enabled")
        _scope_cache_env(env_vars, node_ip, rank_tag=cache_rank_tag)
    payload: dict[str, Any] = {"env_vars": env_vars}
    if nsight is not None:
        payload["nsight"] = nsight
    return payload


# Live Ray node helpers are intentionally local to this module.  The launcher
# should consume the higher-level placement/resource functions below.
def _live_nodes() -> list[dict[str, Any]]:
    return [node for node in ray.nodes() if node.get("Alive", False)]


def node_resource_key(node_ip: str) -> str:
    for node in _live_nodes():
        if node.get("NodeManagerAddress") != node_ip:
            continue
        for resource_name in node.get("Resources", {}):
            if resource_name.startswith("node:"):
                return str(resource_name)
    raise RuntimeError(f"Could not find node resource key for {node_ip}")


def _node_gpu_count(node_ip: str) -> int:
    # A host can run a second, GPU-less raylet (e.g. a CPU-only entry node) under
    # the same IP; GPU tasks pinned by node:<ip> land on the one with GPUs.
    counts = [
        int(node.get("Resources", {}).get("GPU", 0))
        for node in _live_nodes()
        if node.get("NodeManagerAddress") == node_ip
    ]
    if counts:
        return max(counts)
    known_nodes = [
        str(node.get("NodeManagerAddress"))
        for node in _live_nodes()
        if node.get("NodeManagerAddress")
    ]
    raise RuntimeError(
        f"Could not find GPU count for {node_ip!r}. Ray exposes nodes by "
        f"resolved IP address, not hostname. Known Ray node IPs: {known_nodes}"
    )


def _parse_explicit_afd_dp_nodes(
    *,
    attn_dp_size: int,
    attn_tp_size: int,
    mlp_dp_size: int,
    mlp_tp_size: int,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]] | None:
    attn_env = os.environ.get("MINISGL_AFD_ATTN_NODES", "").strip()
    mlp_env = os.environ.get("MINISGL_AFD_MLP_NODES", "").strip()
    if not attn_env and not mlp_env:
        return None
    if not attn_env or not mlp_env:
        raise RuntimeError(
            "MINISGL_AFD_ATTN_NODES and MINISGL_AFD_MLP_NODES must be set together"
        )
    attn_nodes = [node.strip() for node in attn_env.split(",") if node.strip()]
    mlp_nodes = [node.strip() for node in mlp_env.split(",") if node.strip()]
    if len(attn_nodes) != int(attn_dp_size):
        raise RuntimeError(
            f"MINISGL_AFD_ATTN_NODES={attn_nodes} length must match attn_dp_size={attn_dp_size}"
        )
    if len(mlp_nodes) != int(mlp_dp_size):
        raise RuntimeError(
            f"MINISGL_AFD_MLP_NODES={mlp_nodes} length must match mlp_dp_size={mlp_dp_size}"
        )

    required: dict[str, int] = {}
    for node in attn_nodes:
        required[node] = required.get(node, 0) + int(attn_tp_size)
    for node in mlp_nodes:
        required[node] = required.get(node, 0) + int(mlp_tp_size)
    for node, count in required.items():
        available = _node_gpu_count(node)
        if available < count:
            raise RuntimeError(
                f"Explicit AFD placement overfills node {node}: required_gpus={count} "
                f"available_gpus={available} attn_nodes={attn_nodes} mlp_nodes={mlp_nodes}"
            )
    return (
        _rank_nodes_from_dp_nodes(attn_nodes, attn_tp_size),
        _rank_nodes_from_dp_nodes(mlp_nodes, mlp_tp_size),
    )


def _rank_nodes_from_dp_nodes(
    dp_nodes: list[str],
    tp_size: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(node for _ in range(int(tp_size))) for node in dp_nodes)


def _dp_leader_nodes(rank_nodes: tuple[tuple[str, ...], ...]) -> list[str]:
    return [ranks[0] for ranks in rank_nodes]


@dataclass(frozen=True)
class _DpTpPlacementChunk:
    role: str
    dp_rank: int
    tp_start: int
    tp_end: int

    @property
    def size(self) -> int:
        return self.tp_end - self.tp_start


def afd_dp_node_layout(
    *,
    attn_dp_size: int,
    attn_tp_size: int,
    mlp_dp_size: int,
    mlp_tp_size: int,
) -> AfdDpNodePlacement:
    """Return AFD worker placement.

    `attn_nodes`/`mlp_nodes` are the DP leader/control nodes.  The
    `*_rank_nodes` matrices are the actual Ray placement for each TP rank and
    may split one DP group across multiple nodes when TP size exceeds per-node
    GPU capacity.
    """
    attn_dp_size = int(attn_dp_size)
    attn_tp_size = int(attn_tp_size)
    mlp_dp_size = int(mlp_dp_size)
    mlp_tp_size = int(mlp_tp_size)
    nodes = sorted(
        str(node.get("NodeManagerAddress"))
        for node in _live_nodes()
        if node.get("NodeManagerAddress")
    )
    if not nodes:
        raise RuntimeError("No live Ray nodes found for AFD placement")
    node_gpu_counts = {node: _node_gpu_count(node) for node in nodes}

    explicit = _parse_explicit_afd_dp_nodes(
        attn_dp_size=attn_dp_size,
        attn_tp_size=attn_tp_size,
        mlp_dp_size=mlp_dp_size,
        mlp_tp_size=mlp_tp_size,
    )
    if explicit is not None:
        attn_rank_nodes, mlp_rank_nodes = explicit
        return AfdDpNodePlacement(
            attn_nodes=_dp_leader_nodes(attn_rank_nodes),
            mlp_nodes=_dp_leader_nodes(mlp_rank_nodes),
            attn_rank_nodes=attn_rank_nodes,
            mlp_rank_nodes=mlp_rank_nodes,
            mode="explicit_env",
            node_gpu_counts=node_gpu_counts,
        )

    max_node_gpus = max(node_gpu_counts.values(), default=0)
    if max_node_gpus <= 0:
        raise RuntimeError(f"No GPU resources found on live Ray nodes: {node_gpu_counts}")

    def _split_dp_tp_group(
        role: str,
        dp_rank: int,
        tp_size: int,
    ) -> list[_DpTpPlacementChunk]:
        chunks: list[_DpTpPlacementChunk] = []
        tp_start = 0
        while tp_start < int(tp_size):
            tp_end = min(tp_start + max_node_gpus, int(tp_size))
            chunks.append(_DpTpPlacementChunk(role, dp_rank, tp_start, tp_end))
            tp_start = tp_end
        return chunks

    chunks = (
        [
            chunk
            for dp_rank in range(attn_dp_size)
            for chunk in _split_dp_tp_group("attn", dp_rank, attn_tp_size)
        ]
        + [
            chunk
            for dp_rank in range(mlp_dp_size)
            for chunk in _split_dp_tp_group("mlp", dp_rank, mlp_tp_size)
        ]
    )
    best: tuple[tuple[int, int, int, tuple[int, ...]], list[str]] | None = None

    def _search(index: int, placement: list[str], used: dict[str, int]) -> None:
        nonlocal best
        if index == len(chunks):
            used_nodes = {node: count for node, count in used.items() if count > 0}
            used_counts = tuple(sorted(used_nodes.values()))
            if not used_counts:
                return
            attn_node_set = {
                placement[i]
                for i, chunk in enumerate(chunks)
                if chunk.role == "attn"
            }
            mlp_node_set = {
                placement[i]
                for i, chunk in enumerate(chunks)
                if chunk.role == "mlp"
            }
            score = (
                len(attn_node_set & mlp_node_set),
                len(used_counts),
                max(used_counts),
                tuple(nodes.index(node) for node in placement),
            )
            if best is None or score < best[0]:
                best = (score, list(placement))
            return

        chunk = chunks[index]
        for node in nodes:
            next_count = used[node] + chunk.size
            if next_count > node_gpu_counts[node]:
                continue
            used[node] = next_count
            placement.append(node)
            _search(index + 1, placement, used)
            placement.pop()
            used[node] -= chunk.size

    _search(0, [], {node: 0 for node in nodes})
    if best is None:
        raise RuntimeError(
            "Failed to place AFD DP groups on available Ray GPUs: "
            f"attn_dp_size={attn_dp_size} attn_tp_size={attn_tp_size} "
            f"mlp_dp_size={mlp_dp_size} mlp_tp_size={mlp_tp_size} "
            f"node_gpu_counts={node_gpu_counts}. Use explicit "
            "MINISGL_AFD_ATTN_NODES/MINISGL_AFD_MLP_NODES."
        )

    placement = best[1]
    attn_rank_node_lists = [["" for _ in range(attn_tp_size)] for _ in range(attn_dp_size)]
    mlp_rank_node_lists = [["" for _ in range(mlp_tp_size)] for _ in range(mlp_dp_size)]
    for node, chunk in zip(placement, chunks, strict=True):
        if chunk.role == "attn":
            rank_nodes = attn_rank_node_lists[chunk.dp_rank]
        else:
            rank_nodes = mlp_rank_node_lists[chunk.dp_rank]
        for tp_rank in range(chunk.tp_start, chunk.tp_end):
            rank_nodes[tp_rank] = node
    attn_rank_nodes = tuple(tuple(row) for row in attn_rank_node_lists)
    mlp_rank_nodes = tuple(tuple(row) for row in mlp_rank_node_lists)
    if any(not node for row in attn_rank_nodes + mlp_rank_nodes for node in row):
        raise RuntimeError(
            f"Incomplete AFD rank placement: attn={attn_rank_nodes} mlp={mlp_rank_nodes}"
        )
    return AfdDpNodePlacement(
        attn_nodes=_dp_leader_nodes(attn_rank_nodes),
        mlp_nodes=_dp_leader_nodes(mlp_rank_nodes),
        attn_rank_nodes=attn_rank_nodes,
        mlp_rank_nodes=mlp_rank_nodes,
        mode="balanced",
        node_gpu_counts=node_gpu_counts,
    )


@ray.remote(num_cpus=0, max_retries=0)
def pick_free_ports(count: int) -> list[int]:
    """Reserve ephemeral TCP ports on the executing node.

    The sockets are closed before return, so there is a short TOCTOU window
    before the caller binds. ZMQ paths avoid this with tcp://*:* +
    LAST_ENDPOINT; torch.distributed/NCCL master ports still need this helper.
    """
    sockets: list[socket.socket] = []
    try:
        for _ in range(int(count)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            sock.listen(1)
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


# num_gpus=0.01 keeps the probe on a GPU node; it only needs the local CUDA
# capability/toolchain environment, not GPU compute.
@ray.remote(num_cpus=1, num_gpus=0.01, max_retries=0)
def resolve_attention_backend_page_size(
    attention_backend: str,
    page_size: int,
) -> dict[str, int | str]:
    backend, normalized_page_size = normalize_attention_backend_page_size(
        attention_backend,
        page_size,
    )
    return {
        "attention_backend": backend,
        "page_size": int(normalized_page_size),
    }


# ---------------------------------------------------------------------------
# Attention backend and decode-graph sizing
# ---------------------------------------------------------------------------

def _supports_decode_graph_capture(
    attention_backend: str,
    page_size: int,
    cache_type: str,
) -> bool:
    if str(cache_type) != "naive":
        return False
    backend = str(attention_backend)
    decode_backend = backend.split(",", 1)[1] if "," in backend else backend
    if decode_backend == "fi":
        return int(page_size) == 1
    if decode_backend == "trtllm":
        return int(page_size) in (16, 32, 64)
    return False


def normalize_attention_backend_page_size(
    attention_backend: str,
    page_size: int,
) -> tuple[str, int]:
    backend = str(attention_backend)
    normalized_page_size = int(page_size)
    if backend == "auto":
        backend = "trtllm" if is_sm100_supported() else "fi"
    if "trtllm" in backend and normalized_page_size not in (16, 32, 64):
        normalized_page_size = 64
    if normalized_page_size != 1 and any(part == "fi" for part in backend.split(",")):
        raise RuntimeError(
            "AFD page_size > 1 currently requires a pure TensorRT-LLM attention backend; "
            f"got attention_backend={backend!r}, page_size={normalized_page_size}"
        )
    return backend, normalized_page_size


def _normalize_decode_graph_bs(
    explicit_bs: tuple[int, ...],
    cuda_graph_max_bs: int,
) -> tuple[int, ...]:
    """Bucket list aligned with vLLM/sglang: gap=8 in [8,256], gap=16 above."""
    if explicit_bs:
        return tuple(sorted({int(bs) for bs in explicit_bs if int(bs) > 0}))
    if cuda_graph_max_bs < 1:
        return ()
    sizes = [1, 2, 4]
    sizes.extend(range(8, min(cuda_graph_max_bs, 256) + 1, 8))
    if cuda_graph_max_bs > 256:
        sizes.extend(range(272, cuda_graph_max_bs + 1, 16))
    return tuple(sorted({s for s in sizes if s <= cuda_graph_max_bs}))


def compute_decode_graph_sizes(
    max_per_lane: int,
    explicit: Sequence[int] | None = None,
    *,
    max_capture_size: int | None = None,
) -> tuple[int, ...]:
    """Per-microbatch decode graph sizes, always covering a lane at capacity.

    Follows vLLM's VllmConfig._set_cudagraph_sizes (vllm/config/vllm.py): an
    explicit list is kept as given, sorted and de-duplicated (1669-1680);
    otherwise 1, 2, 4, then multiples of 8 below 256 and of 16 from 256 up to
    the capture limit (1688-1708). vLLM appends max_num_batched_tokens so its
    largest batch is captured off-stride (1701-1706); the largest AFD decode
    batch is ``max_per_lane`` and is appended the same way. Unlike vLLM, which
    leaves batches above its largest size to eager or piecewise execution
    (cudagraph_dispatcher.py:282), AFD appends it to an explicit list too: all
    lanes of a collective step replay or none does (as vLLM syncs the
    cudagraph mode across DP ranks, v1/worker/dp_utils.py:92-98), so one lane
    without a graph runs every lane eager, and AFD has no piecewise fallback.

    Args:
        max_per_lane: The largest per-microbatch decode batch of one attention-DP
            lane: ceil(its max_running_req / num_mb).
        explicit: ``--afd-decode-graph-bs``; None selects the default list.
        max_capture_size: ``--cuda-graph-max-bs``, bounding the default list the
            way vLLM's max_cudagraph_capture_size does; None or < 1 means
            ``max_per_lane``.
    """
    max_per_lane = max(1, int(max_per_lane))
    if explicit is not None:
        sizes = tuple(sorted({int(bs) for bs in explicit if int(bs) > 0}))
    else:
        limit = max_per_lane
        if max_capture_size is not None and int(max_capture_size) >= 1:
            limit = min(int(max_capture_size), max_per_lane)
        sizes = _normalize_decode_graph_bs((), limit)
    if not sizes or sizes[-1] < max_per_lane:
        sizes = (*sizes, max_per_lane)
    return sizes


def select_decode_graph_bs(active_req_count: int, buckets: tuple[int, ...]) -> int:
    """Pick the smallest captured bucket that covers `active_req_count`.

    `buckets` must be sorted ascending; `build_runtime_sizing` constructs them
    with the same bucket policy used for graph capture.
    """
    for bucket in buckets:
        if bucket >= active_req_count:
            return int(bucket)
    if not buckets:
        return int(active_req_count)
    raise RuntimeError(
        f"No decode graph bucket covers active_req_count={active_req_count}; buckets={list(buckets)}"
    )


def build_runtime_sizing(
    *,
    explicit_decode_graph_bs: tuple[int, ...],
    cuda_graph_max_bs: int,
    attention_backend: str,
    page_size: int,
    cache_type: str,
    afd_batch_size: int,
    afd_max_running_req: int,
    max_batched_tokens: int,
    afd_num_mb: int,
    attn_dp_size: int = 1,
) -> AfdRuntimeSizing:
    num_mb = max(1, int(afd_num_mb))
    requested_graph_bs = _normalize_decode_graph_bs(
        explicit_decode_graph_bs,
        cuda_graph_max_bs,
    )
    max_total_batch = (
        int(afd_max_running_req)
        if int(afd_max_running_req) > 0
        else max(int(afd_batch_size), 1)
    )
    # Graph sizes come from the scheduler's limit, as vLLM derives them from
    # max_num_seqs (vllm/config/vllm.py:1657-1659): each attention-DP lane holds
    # ceil(max_total_batch / attn_dp_size) requests (the coordinator's
    # attn_max_running_req) and may decode all of them at once -- a lane whose
    # last chunked prefill ends while other lanes still prefill does.
    lane_graph_bs = div_ceil(div_ceil(max_total_batch, max(1, int(attn_dp_size))), num_mb)
    decode_graph_bs: tuple[int, ...] = ()
    added_graph_bs = None
    if _supports_decode_graph_capture(attention_backend, page_size, cache_type):
        explicit = None
        if explicit_decode_graph_bs:
            explicit = tuple(int(bs) for bs in explicit_decode_graph_bs if int(bs) > 0)
            if num_mb > 1:
                # A per-MB size above ceil(max_total_batch / num_mb) never fits.
                max_per_mb = max(1, div_ceil(max_total_batch, num_mb))
                explicit = tuple(bs for bs in explicit if bs <= max_per_mb)
        decode_graph_bs = compute_decode_graph_sizes(
            lane_graph_bs,
            explicit,
            max_capture_size=int(cuda_graph_max_bs),
        )
        if explicit is not None:
            requested_limit = max(explicit, default=0)
        elif int(cuda_graph_max_bs) >= 1:
            requested_limit = int(cuda_graph_max_bs)
        else:
            requested_limit = lane_graph_bs
        if requested_limit < lane_graph_bs:
            added_graph_bs = lane_graph_bs
    max_graph_bs = max(decode_graph_bs) if decode_graph_bs else 0
    max_running_req = (
        int(afd_max_running_req)
        if int(afd_max_running_req) > 0
        else max(int(afd_batch_size), max_graph_bs, 1)
    )
    # vLLM-aligned: leave KV cache memory-bound by default. Explicit
    # `--num-pages` still flows through EngineConfig.num_page_override.
    num_page_override = None
    # max_comm_tokens sizes the AFD n2m staging buffer in tokens. Prefill can
    # send up to max_batched_tokens; decode graph can send up to max_graph_bs
    # one-token requests. Some grouped-routing models can send more route copies
    # to one expert rank than hidden rows in the local prefill chunk, so allow a
    # model script to reserve a larger communication bucket without changing the
    # scheduler's prefill token budget.
    env_comm_tokens = int(os.environ.get("MINISGL_AFD_MAX_COMM_TOKENS", "0") or 0)
    max_comm_tokens = max(int(max_batched_tokens), max_graph_bs, env_comm_tokens, 1)
    return AfdRuntimeSizing(
        requested_graph_bs=requested_graph_bs,
        decode_graph_bs=decode_graph_bs,
        max_graph_bs=max_graph_bs,
        max_running_req=max_running_req,
        num_page_override=num_page_override,
        max_comm_tokens=max_comm_tokens,
        num_mb=num_mb,
        added_graph_bs=added_graph_bs,
    )


__all__ = [
    "AfdRuntimeConfig",
    "afd_dp_node_layout",
    "build_runtime_sizing",
    "compute_decode_graph_sizes",
    "flush_log_lines",
    "log_line",
    "node_resource_key",
    "normalize_attention_backend_page_size",
    "pick_free_ports",
    "resolve_attention_backend_page_size",
    "runtime_env",
    "select_decode_graph_bs",
]

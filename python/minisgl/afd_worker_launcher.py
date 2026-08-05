from __future__ import annotations

import time
from typing import Any

import ray

from minisgl.env import afd_env
from minisgl.server.supervisor_ray import _build_ray_nsight_options

from .afd_attention_worker import AfdAttentionWorker
from .afd_expert_worker import AfdExpertWorker
from .afd_profiler import start_nsys_runtime_capture
from .afd_protocol import AfdWorkerReadyReply
from .afd_support import (
    afd_dp_node_layout,
    log_line,
    node_resource_key,
    pick_free_ports,
    resolve_attention_backend_page_size,
    runtime_env,
)


def startup_afd_workers(coord: Any) -> None:
    """Start Ray workers and their hot RPC loops for one AFD coordinator."""
    if not ray.is_initialized():
        ray.init(address=coord.server_args.ray_address, log_to_driver=False)

    # Stage 1: map logical AG/EG DP/TP ranks to Ray nodes.  The coordinator owns
    # the global topology; this placement object is the only source of truth for
    # where each worker actor should be created.
    layout = coord._layout
    placement = afd_dp_node_layout(
        attn_dp_size=layout.attn_dp_size,
        attn_tp_size=layout.attn_tp_size,
        mlp_dp_size=layout.mlp_dp_size,
        mlp_tp_size=layout.mlp_tp_size,
    )
    attn_nodes = placement.attn_nodes
    mlp_nodes = placement.mlp_nodes
    attn_rank_nodes = placement.attn_rank_nodes
    mlp_rank_nodes = placement.mlp_rank_nodes
    primary_attn_node = attn_nodes[0]
    node_scoped_cache = bool(coord.server_args.ray_node_scoped_cache)

    def _runtime_env_for_node(
        node_ip: str,
        *,
        nsight: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return runtime_env(
            nsight=nsight,
            node_scoped_cache=node_scoped_cache,
            node_ip=node_ip,
        )

    primary_attn_runtime_env = _runtime_env_for_node(primary_attn_node)
    log_line(
        coord.log_path,
        f"[afd-coordinator] placement mode={placement.mode} "
        f"attn_nodes={attn_nodes} mlp_nodes={mlp_nodes} "
        f"attn_rank_nodes={attn_rank_nodes} mlp_rank_nodes={mlp_rank_nodes} "
        f"node_gpu_counts={placement.node_gpu_counts}",
        flush=True,
    )

    # Stage 2: resolve backend-dependent sizing once on an attention GPU.  Some
    # attention backends choose their page size from the local CUDA/runtime
    # environment, so this cannot be decided purely on the coordinator process.
    if coord._needs_gpu_backend_resolution:
        resolved = ray.get(
            resolve_attention_backend_page_size.options(
                resources={node_resource_key(primary_attn_node): 0.01},
                runtime_env=primary_attn_runtime_env,
            ).remote(
                coord._requested_attention_backend,
                coord._requested_page_size,
            )
        )
        coord.attention_backend = str(resolved["attention_backend"])
        coord.page_size = int(resolved["page_size"])
        coord._refresh_runtime_sizing()

    # Stage 3: allocate the single TCP rendezvous port for the AG+EG union torch
    # distributed world.  Older per-DP process groups are no longer used here; all
    # workers join this one union world and form TP subgroups inside it.
    afd_union_port = int(
        ray.get(
            pick_free_ports.options(
                resources={node_resource_key(primary_attn_node): 0.01},
                runtime_env=primary_attn_runtime_env,
            ).remote(1)
        )[0]
    )

    # Stage 4: build the per-worker Ray runtime env.  NSYS injection and cache
    # directory scoping are rank-local so concurrent workers do not fight over
    # generated kernel/JIT/cache files.
    def _worker_runtime_env(ray_rank: int, node_ip: str) -> dict[str, object]:
        nsight = (
            _build_ray_nsight_options(coord.server_args, ray_rank)
            if coord.server_args.ray_nsys
            else None
        )
        cache_rank_tag = (
            f"rank{int(ray_rank)}"
            if node_scoped_cache
            else None
        )
        return runtime_env(
            nsight=nsight,
            node_scoped_cache=node_scoped_cache,
            node_ip=node_ip,
            cache_rank_tag=cache_rank_tag,
        )

    common_worker_kwargs: dict[str, Any] = {
        "attn_dp_size": layout.attn_dp_size,
        "mlp_dp_size": layout.mlp_dp_size,
        "attn_tp_size": layout.attn_tp_size,
        "mlp_tp_size": layout.mlp_tp_size,
        "model_path": coord.server_args.model_path,
        "attention_backend": coord.attention_backend,
        "max_seq_len": coord.max_seq_len,
        "max_comm_tokens": coord.max_comm_tokens,
        "num_page_override": coord.num_page_override,
        "cuda_graph_max_bs": coord.max_graph_bs,
        "memory_ratio": float(coord.server_args.memory_ratio),
        "decode_graph_bs": tuple(coord.decode_graph_bs),
        "enable_attention_decode_graph": coord.enable_attention_decode_graph,
        "enable_model_decode_graph": coord.enable_model_decode_graph,
        "page_size": coord.page_size,
        "cache_type": coord.server_args.cache_type,
        "log_dir": str(coord.report_dir),
        "device_comm_num_sms": coord.device_comm_num_sms,
        "ray_nsys": bool(coord.server_args.ray_nsys),
        "disable_overlap": bool(coord.server_args.afd_disable_overlap),
        "async_pending_steps": int(coord.async_pending_steps),
        "afd_num_mb": coord.afd_num_mb,
        "rpc_timeout_ms": coord._rpc_timeout_ms,
        "moe_a2a_backend": coord.server_args.afd_moe_a2a_backend,
        "moe_runner_backend": coord.server_args.afd_moe_runner_backend,
        "ep_size": coord.server_args.afd_mlp_ep_size,
    }

    # Stage 5: describe the AG+EG union distributed world.  MegaMoE/DeepEP M2N
    # collectives span both attention-side and expert-side workers, while TP
    # collectives only run inside the per-DP TP subgroups below.
    afd_union_host = primary_attn_node
    afd_union_world = layout.total_workers
    afd_union_tp_groups = layout.union_tp_groups()

    def _afd_union_dist_kwargs(union_rank: int) -> dict[str, Any]:
        return {
            "distributed_host": afd_union_host,
            "distributed_port": afd_union_port,
            "distributed_rank": int(union_rank),
            "distributed_world_size": int(afd_union_world),
            "distributed_tp_groups": afd_union_tp_groups,
        }

    AfdAttentionWorkerActor = ray.remote(
        num_cpus=1,
        num_gpus=1,
        max_restarts=0,
        max_task_retries=0,
    )(AfdAttentionWorker)
    AfdExpertWorkerActor = ray.remote(
        num_cpus=1,
        num_gpus=1,
        max_restarts=0,
        max_task_retries=0,
    )(AfdExpertWorker)

    # Stage 6: create attention-side (AG) actors.  Each AG worker owns request/KV
    # state for one attention DP/TP rank and is assigned to the MLP DP rank that
    # will process its routed tokens.
    for dp_rank in range(layout.attn_dp_size):
        assigned_mlp_dp = layout.mlp_dp_for_attn_dp(dp_rank)
        for rank in range(layout.attn_tp_size):
            attn_rank_node = attn_rank_nodes[dp_rank][rank]
            global_rank = layout.attn_worker_rank(dp_rank, rank)
            worker_kwargs = {
                **common_worker_kwargs,
                "tp_rank": rank,
                "tp_size": layout.attn_tp_size,
                "attn_dp_rank": dp_rank,
                "mlp_dp_rank": assigned_mlp_dp,
                "comm_rank": global_rank,
                "node_ip": attn_rank_node,
                "max_extend_tokens": coord.max_batched_tokens,
                "max_running_req": coord.attn_max_running_req,
                "use_dummy_weight": False,
            }
            worker_kwargs.update(_afd_union_dist_kwargs(global_rank))
            coord.attn_workers.append(
                AfdAttentionWorkerActor.options(
                    resources={node_resource_key(attn_rank_node): 0.01},
                    runtime_env=_worker_runtime_env(global_rank + 1, attn_rank_node),
                ).remote(**worker_kwargs)
            )

    # Stage 7: create expert-side (EG) actors.  EG workers do not own request
    # scheduling state; their slot capacity is sized for the AG DP lanes that can
    # send routed tokens to this MLP DP rank.
    for mlp_dp_rank in range(layout.mlp_dp_size):
        for rank in range(layout.mlp_tp_size):
            mlp_rank_node = mlp_rank_nodes[mlp_dp_rank][rank]
            global_rank = layout.mlp_worker_rank(mlp_dp_rank, rank)
            peer_attn_dp = layout.attn_dp_for_mlp_dp(mlp_dp_rank)
            worker_kwargs = {
                **common_worker_kwargs,
                "tp_rank": rank,
                "tp_size": layout.mlp_tp_size,
                "attn_dp_rank": peer_attn_dp,
                "mlp_dp_rank": mlp_dp_rank,
                "comm_rank": global_rank,
                "node_ip": mlp_rank_node,
                "max_extend_tokens": (
                    coord.max_batched_tokens
                    * layout.attn_fanin_per_mlp_dp
                ),
                # MLP page-table slots are partitioned by attention DP.
                "max_running_req": (
                    (coord.attn_max_running_req + 1) * layout.attn_dp_size - 1
                ),
                "use_dummy_weight": coord.server_args.use_dummy_weight,
            }
            worker_kwargs.update(
                _afd_union_dist_kwargs(global_rank)
            )
            coord.mlp_workers.append(
                AfdExpertWorkerActor.options(
                    resources={node_resource_key(mlp_rank_node): 0.01},
                    runtime_env=_worker_runtime_env(global_rank + 1, mlp_rank_node),
                ).remote(**worker_kwargs)
            )

    # Stage 8: run a cheap readiness hook on every actor before opening the hot
    # ZMQ path.  Heavy model/adapter initialization happens inside each hot loop,
    # but this catches actor construction failures early.
    ray.get(
        [
            worker.prepare_communication_runtime.remote()
            for worker in coord._all_workers()
        ]
    )

    # Stage 9: initialize coordinator-side scheduling/profiling/queues.  From
    # this point on the coordinator talks to workers through ZMQ command/result
    # queues instead of per-step Ray RPC.
    coord._init_centralized_schedulers()
    if coord.server_args.ray_nsys:
        if coord._nsys_start_step <= 0:
            start_nsys_runtime_capture(coord, sync=True, reason="startup")
        else:
            log_line(
                coord.log_path,
                "[afd-coordinator] nsys profiler:defer "
                f"start_step={coord._nsys_start_step} stop_step={coord._nsys_stop_step}",
                flush=True,
            )
    coord._init_worker_queues()
    if coord._worker_inbox is None or coord._worker_inbox_addr is None:
        raise RuntimeError("Failed to initialize AFD worker ZMQ queues")

    # Stage 10: enter each worker's hot RPC loop.  Workers initialize their AFD
    # runtime, warm up/capture decode graphs, and then send AfdWorkerReadyReply.
    coord._worker_hot_loop_refs = []
    for worker_rank, (worker, cmd_addr) in enumerate(
        zip(coord._all_workers(), coord._worker_cmd_addrs, strict=True)
    ):
        coord._worker_hot_loop_refs.append(
            worker.start_hot_rpc_loop.remote(
                cmd_addr=cmd_addr,
                result_addr=coord._worker_inbox_addr,
                detokenizer_addr=None,
                create_detokenizer_link=False,
            )
        )

    # Stage 11: wait until every worker reports that its hot loop is ready.  If a
    # worker exits early, surface that actor failure instead of waiting until the
    # global startup timeout expires.
    expected_ready = len(coord._all_workers())
    startup_timeout_s = float(afd_env("STARTUP_TIMEOUT_S", "1800"))
    startup_deadline = time.monotonic() + startup_timeout_s
    ready_replies = []
    while len(ready_replies) < expected_ready:
        try:
            reply = coord._worker_inbox.get(timeout_ms=5000)
            ready_replies.append(reply)
        except TimeoutError:
            ready_refs, _ = ray.wait(coord._worker_hot_loop_refs, num_returns=1, timeout=0)
            if ready_refs:
                try:
                    result = ray.get(ready_refs[0])
                except Exception as exc:
                    raise RuntimeError(f"AFD worker failed before ready: {exc}") from exc
                raise RuntimeError(f"AFD worker hot loop exited before ready: {result}")
            if time.monotonic() >= startup_deadline:
                raise RuntimeError(
                    "Timed out waiting for AFD workers to become ready: "
                    f"ready={len(ready_replies)} expected={expected_ready} "
                    f"timeout_s={startup_timeout_s}"
                )
            continue
    for reply in ready_replies:
        if not isinstance(reply, AfdWorkerReadyReply):
            raise RuntimeError(f"Expected AFD worker ready reply, got {reply}")


__all__ = ["startup_afd_workers"]

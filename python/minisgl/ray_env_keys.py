"""Environment variables forwarded into Ray actors: the one registry both Ray hops read."""

from __future__ import annotations

# A Ray actor does not see the launching shell's environment; each hop copies these
# keys (when set and non-empty) into the actor's runtime_env `env_vars`:
#   driver -> Ray actor         server/supervisor_ray.py `_build_ray_runtime_env`
#                               (the AFD coordinator and the plain Ray backend's ranks)
#   AFD coordinator -> worker   afd_support.py `runtime_env` (AG/EG worker actors)
# Ray merges a child actor's env_vars over its parent's, so a key forwarded only by
# the first hop still reaches AFD workers through the coordinator. Keys a worker
# reads belong in both hops, so that path is explicit rather than inherited.
_BOTH_HOPS = (
    # A path list prepended to the AFD workers' PYTHONPATH (afd_support.runtime_env), e.g. a dir holding
    # another flashinfer build for the workers; unset = unchanged behaviour
    "MINISGL_WORKER_PYTHONPATH_PREPEND",
    # Qwen3.5-397B block-FP8 attention-side dense decode fusions, each default off;
    # read by the AG workers' model and kernel adapters.
    "MINISGL_QWEN35_FP8_FUSED_DECODE",
    "MINISGL_QWEN35_MERGED_QKVZ",
    "MINISGL_QWEN35_FUSED_ROUTER",
    "MINISGL_QWEN35_FUSED_GATED_NORM",
    "MINISGL_QWEN35_FUSED_SHARED_GATE",
    # vLLM db9527a4 attention-side kernels for Qwen3.5 (kernel/vllm_*.py), default off.
    "MINISGL_QWEN35_VLLM_KERNELS",
    "MINISGL_QWEN35_VLLM_SHARED_GATE",
    "MINISGL_QWEN35_VLLM_QUANT",
    "MINISGL_QWEN35_VLLM_ROUTER",
    "MINISGL_QWEN35_VLLM_FULLATTN",
    "MINISGL_QWEN35_VLLM_NORM",
    "MINISGL_QWEN35_VLLM_SILU",
    # CAD decode mode (minisgl.cad.config.CAD_ENV_KEYS, the same tuple): read by the coordinator,
    # the attention and the model workers. CUDA_MODULE_LOADING=EAGER: no lazy module load behind a pending CAD flag wait.
    "MINISGL_AFD_MODE",
    "MINISGL_AFD_CAD_O_DTYPE",
    "MINISGL_AFD_CAD_MOE",
    "MINISGL_AFD_CAD_HOP",
    "MINISGL_AFD_CAD_HANG_TIMEOUT_NS",
    "MINISGL_AFD_CAD_INIT_TIMEOUT_S",
    "CUDA_MODULE_LOADING",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "MINISGL_RAY_NSYS_BIN",
    "MINISGL_RAY_NSYS_CUDA_GRAPH_TRACE",
    "MINISGL_RAY_NSYS_START_STEP",
    "MINISGL_RAY_NSYS_STOP_STEP",
    "MINISGL_AFD_CONTROL_PROFILE_START_STEP",
    "MINISGL_AFD_CONTROL_PROFILE_STOP_STEP",
    "MINISGL_NVTX_CPU_TRACE",
    "NSYS",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_SL",
    "NCCL_NET_GDR_LEVEL",
    "MINISGL_AFD_ATTN_NODES",
    "MINISGL_AFD_MLP_NODES",
    # DeepEP runtime knobs (kernel/deepep_moe.py): GPU / CPU collective timeouts and DeepEP's own GIN
    # switch (read in its csrc).
    "MINISGL_DEEPEP_CPU_TIMEOUT_SECS",
    "MINISGL_DEEPEP_GPU_TIMEOUT_SECS",
    "EP_DISABLE_GIN",
    "MINISGL_DEEPEP_BUILD_DIR",
    "MINISGL_DEEPGEMM_BUILD_DIR",
    "MINISGL_QK_NORM_ROPE_BACKEND",
    "MINISGL_AFD_PAGE_AUDIT",
    # Batched page bookkeeping (afd_scheduler / afd_attention_runtime): coordinator and AG workers.
    "MINISGL_AFD_PAGE_FAST",
    "MINISGL_AFD_LOG_STEPS",
    # Speculative decoding (spec.config): read by the coordinator and the AG workers.
    "MINISGL_AFD_SPEC",
    "MINISGL_AFD_SPEC_BLOCK",
    "MINISGL_AFD_SPEC_DRAFT",
    "MINISGL_AFD_SPEC_EAGLE3_LAYERS",
    "MINISGL_AFD_SPEC_SIM_ACCEPT",
    "MINISGL_AFD_SPEC_SIM_SEED",
    "MINISGL_AFD_SPEC_TIMING",
    "MINISGL_AFD_SPEC_DEVICE_INPUTS",
    "MINISGL_AFD_DRAFT_ATTN_SMS",
    "MINISGL_AFD_ROUTE_ON_LANE",
    "MINISGL_AFD_LANE_COMPUTE_STREAMS",
    "MINISGL_DENSE_GEMM_SMS",
    "MINISGL_AFD_M2N_BARRIER_TRACE",
    "MINISGL_AFD_MOE_BACKEND",
    "MINISGL_RAY_PY_EXECUTABLE",
    "MINISGL_RAY_PY_EXECUTABLE_RANKS",
    "MINISGL_AFD_MAX_COMM_TOKENS",
    "MINISGL_AFD_DISABLE_DEEPEP_CPU_SYNC",
    "MINISGL_AFD_LOG_M2N_ALLOC",
    "MINISGL_MEGAMOE_AG_SMS",
    "MINISGL_MEGAMOE_P2P",
    "MINISGL_MEGAMOE_AG_SPLIT",
    "MINISGL_MEGAMOE_AG_SMS_DISPATCH",
    "MINISGL_MEGAMOE_AG_SMS_COMBINE",
    "MINISGL_MEGAMOE_AG_WAITED_COMBINE",
    "MINISGL_MEGAMOE_AG_WAIT_MONO",
    "MINISGL_PDL",
    "MINISGL_MEGAMOE_EG_EARLY",
    "MINISGL_MEGAMOE_PUSH_DISPATCH",
    "MINISGL_MEGAMOE_EXACT_FP8_SCALES",
    "MINISGL_MEGAMOE_HANG_GUARD",
    "MINISGL_MEGAMOE_EG_PREFETCH_MB",
    "MINISGL_AFD_DIST_TIMEOUT",
    "MINISGL_AFD_STEP_TIMES",
    "MINISGL_MEGAMOE_CHECK_INPUTS",
    "MINISGL_MEGAMOE_CHECK_EG_OUTPUT",
    "MINISGL_MEGAMOE_AG_TIMING",
    "MINISGL_MEGAMOE_EG_TIMING",
    "MINISGL_MEGAMOE_EG_TIMING_DUMP",
    "MINISGL_MEGAMOE_EG_TIMING_DUMP_STEPS",
    "MINISGL_MEGAMOE_LOG_LAUNCH",
    "CUDA_LAUNCH_BLOCKING",
    "MINISGL_MEGAMOE_EG_EXPECTED_TOKENS",
    "MINISGL_MEGAMOE_EG_EXPECTED_FLOOR",
    "MINISGL_MEGAMOE_EG_PRECOMPILE",
    "DG_PRINT_CONFIGS",
    "MINISGL_DG_GEMM_WARMUP",
    "MINISGL_AFD_PREFIX_CACHE",
    "DG_JIT_WITH_LINEINFO",
    "EP_NCCL_ROOT_DIR",
    "EP_SUPPRESS_NCCL_CHECK",
    "EP_JIT_CACHE_DIR",
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_NVCC_EXECUTABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "MINISGL_QWEN35_CONV1D_EAGER",
    "MINISGL_QWEN35_GATE_EAGER",
    "DG_JIT_CACHE_DIR",
    "DG_JIT_NVCC_COMPILER",
    "DG_JIT_USE_NVRTC",
    "TRITON_PTXAS_BLACKWELL_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    # Defaulted to 1 by minisgl/env_override.py; forwarded so an explicit 0 reaches the workers.
    "FLASHINFER_USE_CUDA_NORM",
    "TVM_FFI_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "MAX_JOBS",
)

# Forwarded by the driver hop only. `_build_ray_runtime_env` rewrites PYTHONPATH
# itself; `runtime_env` sets the workers' PYTHONPATH to this checkout.
_DRIVER_HOP_ONLY = (
    "PYTHONPATH",
    "MINISGL_FP8_QUANT_BACKEND",
    "MINISGL_AFD_STARTUP_TIMEOUT_S",
    "N2M_M2N_GIN_BUILD_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "CC",
    "CXX",
    "LOG_LEVEL",
    "LOG_PID",
)

# Forwarded by the coordinator hop only.
_WORKER_HOP_ONLY = (
    "FLASHINFER_NVCC",
    "TORCH_USE_CUDA_DSA",
)

RAY_ACTOR_ENV_KEYS = _BOTH_HOPS + _DRIVER_HOP_ONLY
AFD_WORKER_ENV_KEYS = _BOTH_HOPS + _WORKER_HOP_ONLY

# Cache and build directories suffixed per Ray node (AFD workers: per node and rank)
# on multi-node runs, so JIT builds on shared storage do not collide.
AFD_WORKER_NODE_SCOPED_CACHE_KEYS = (
    "FLASHINFER_WORKSPACE_BASE",
    "TVM_FFI_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "EP_JIT_CACHE_DIR",
    "DG_JIT_CACHE_DIR",
    "MINISGL_DEEPEP_BUILD_DIR",
    "MINISGL_DEEPGEMM_BUILD_DIR",
)
RAY_ACTOR_NODE_SCOPED_CACHE_KEYS = AFD_WORKER_NODE_SCOPED_CACHE_KEYS + (
    "N2M_M2N_GIN_BUILD_DIR",
)
